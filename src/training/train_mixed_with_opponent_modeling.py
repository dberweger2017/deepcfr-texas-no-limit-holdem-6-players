"""Internal mixed-opponent training implementation.

Use ``python -m src.training.train_opponent_modeling --mixed`` as the public CLI entrypoint.
"""

import pokers as pkrs
import torch
import numpy as np
import os
import random
import time
import argparse
from src.opponent_modeling.deep_cfr_with_opponent_modeling import DeepCFRAgentWithOpponentModeling
from src.core.model import set_verbose
from src.agents.random_agent import RandomAgent
from src.utils.agents import (
    CheckpointAgent,
    checkpoint_uses_opponent_modeling,
)
from src.utils.evaluation import evaluate_agent_matchup
from src.utils.checkpoints import find_checkpoints, load_checkpoint
from src.utils.settings import set_strict_checking

ModelAgent = CheckpointAgent

def evaluate_against_opponents(agent, opponents, num_games=100, iteration=0, notifier=None):
    """Evaluate the trained agent against a set of opponents with enhanced error tracking."""
    metrics = evaluate_agent_matchup(
        agent,
        opponents,
        num_games=num_games,
        seed_start=10000,
        strict=True,
        label="opponent-modeling evaluation vs opponents",
        record_opponent_history=True,
        print_warnings=True,
    )
    
    # If too many zero reward games, send summary alert
    if metrics["zero_reward_games"] > num_games * 0.2 and notifier:  # More than 20% zero rewards
        notifier.alert_zero_reward_games(
            iteration,
            metrics["zero_reward_games"],
            metrics["completed_games"]
        )
    
    # Print detailed statistics
    completed_games = metrics["completed_games"]
    zero_reward_games = metrics["zero_reward_games"]
    print(f"Games completed: {completed_games}/{num_games} ({completed_games/num_games*100:.1f}%)")
    print(f"Zero reward games: {zero_reward_games}/{completed_games} ({zero_reward_games/max(1,completed_games)*100:.1f}%)")
    print("Illegal actions by AGENT: 0")
    print("Illegal actions by OPPONENTS: 0")
    print(f"State errors: {metrics['invalid_state_count']}, Game crashes: {metrics['game_crashes']}")
    print(f"Total actions: {metrics['total_actions']}, Avg actions per game: {metrics['total_actions']/max(1,num_games):.1f}")

    if completed_games == 0 and notifier:
        notifier.send_message(f"⚠️ <b>CRITICAL ERROR</b>: No games completed in iteration {iteration}")
    if completed_games == 0:
        raise RuntimeError("No games completed during opponent-modeling evaluation vs opponents")

    return metrics["avg_profit"]

def train_mixed_with_opponent_modeling(
    checkpoint_dir,
    num_iterations=20000, 
    traversals_per_iteration=200,
    refresh_interval=1000,
    num_opponents=5,
    save_dir="models/opponent_modeling/mixed",
    log_dir="logs/opponent_modeling/mixed",
    player_id=0,
    model_prefix="*checkpoint_iter_",
    verbose=False,
    checkpoint_path=None,
    allow_random_fallback=False,
):
    """
    Train a Deep CFR agent with opponent modeling against a mix of opponents
    loaded from checkpoints, refreshing the opponent pool periodically.
    """
    # Import required modules
    from torch.utils.tensorboard import SummaryWriter
    from scripts.telegram_notifier import TelegramNotifier
    
    # Set verbosity
    set_verbose(verbose)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize Telegram notifier (reads from .env file)
    try:
        notifier = TelegramNotifier()
    except Exception as exc:
        print(f"Warning: Could not initialize Telegram notifier: {exc}")
        notifier = None
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if notifier:
        notifier.send_message(
            f"🚀 <b>TRAINING STARTED</b>\nDevice: {device}\nIterations: {num_iterations}\nRefresh interval: {refresh_interval}"
        )
    
    # Initialize the agent with opponent modeling
    agent = DeepCFRAgentWithOpponentModeling(
        player_id=player_id, 
        num_players=6,
        device=device
    )
    
    # Load from checkpoint if provided
    starting_iteration = 1
    if checkpoint_path:
        print(f"Loading agent from checkpoint: {checkpoint_path}")
        try:
            agent.load_model(checkpoint_path)
            checkpoint_state = load_checkpoint(checkpoint_path, map_location=device)
            starting_iteration = checkpoint_state.get("iteration", agent.iteration_count) + 1
            agent.iteration_count = starting_iteration - 1
            print(f"Continuing from iteration {starting_iteration}")
            if notifier:
                notifier.send_message(
                    f"📥 <b>LOADED CHECKPOINT</b>\nContinuing from iteration {starting_iteration}"
                )
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            if notifier:
                notifier.send_message(
                    f"⚠️ <b>CHECKPOINT LOADING ERROR</b>\n{str(e)}"
                )
            raise
    
    # For tracking progress
    advantage_losses = []
    strategy_losses = []
    opponent_model_losses = []
    profits = []
    profits_vs_models = []
    
    # Checkpoint frequency
    checkpoint_frequency = 100
    
    # Function to select random model opponents
    def select_random_models():
        # Find all model checkpoint files
        checkpoint_files = [str(path) for path in find_checkpoints(checkpoint_dir, model_prefix)]
        
        if not checkpoint_files:
            message = f"No checkpoint files found matching '{model_prefix}' in {checkpoint_dir}"
            if not allow_random_fallback:
                raise ValueError(message)
            print(f"WARNING: {message}. Using random agents as opponents")
            return [None if i == player_id else RandomAgent(i) for i in range(6)]
        
        # Select random models
        selected_files = random.sample(checkpoint_files, min(num_opponents, len(checkpoint_files)))
        print(f"Selected model opponents:")
        opponent_names = []
        for i, f in enumerate(selected_files):
            filename = os.path.basename(f)
            opponent_names.append(filename)
            print(f"  {i+1}. {filename}")
        
        # Log selected opponents to Telegram
        if notifier:
            notifier.send_message(f"📊 <b>SELECTED OPPONENTS</b>\n- " + "\n- ".join(opponent_names))
        
        # Load the selected models
        model_opponents = []
        for pos, file_path in enumerate(selected_files, start=1):
            # Skip player_id position
            if pos == player_id:
                pos = (pos + 1) % 6
            
            is_om_model = checkpoint_uses_opponent_modeling(file_path)
            
            model_opponents.append(ModelAgent(
                player_id=pos,
                model_path=file_path,
                device=device,
                with_opponent_modeling=is_om_model
            ))
        
        # Create the full list of opponents
        opponents = [None] * 6
        
        # Add random agents to fill remaining positions
        for i in range(6):
            if i != player_id:
                # Check if we already have a model opponent for this position
                if not any(opp.player_id == i for opp in model_opponents):
                    opponents[i] = RandomAgent(i)
        
        # Add model opponents
        for model_opp in model_opponents:
            opponents[model_opp.player_id] = model_opp
        
        return opponents
    
    # Select initial opponents
    opponents = select_random_models()
    
    # Training loop
    final_iteration = starting_iteration + num_iterations - 1
    for iteration in range(starting_iteration, final_iteration + 1):
        local_iteration = iteration - starting_iteration + 1
        agent.iteration_count = iteration
        agent.local_training_iteration = local_iteration
        start_time = time.time()
        
        # Refresh opponents at specified intervals
        if iteration % refresh_interval == 1 and iteration > starting_iteration:
            print(f"\n=== Refreshing opponent pool at iteration {iteration} ===")
            if notifier:
                notifier.send_message(f"🔄 <b>REFRESHING OPPONENTS</b> at iteration {iteration}")
            opponents = select_random_models()
        
        print(f"Iteration {iteration}/{num_iterations}")
        
        # Run traversals to collect data
        print("  Collecting data...")
        for t in range(traversals_per_iteration):
            # Create a new poker game
            state = pkrs.State.from_seed(
                n_players=6,
                button=random.randint(0, 5),
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 10000)
            )
            
            # Perform CFR traversal
            try:
                agent.cfr_traverse(state, local_iteration, opponents)
            except Exception as e:
                error_msg = f"Error during traversal: {e}"
                print(error_msg)
                if notifier and iteration % 100 == 0:  # Don't flood with error messages
                    notifier.send_message(f"⚠️ <b>TRAVERSAL ERROR</b>\n{error_msg}")
                writer.flush()
                raise
        
        # Track traversal time
        traversal_time = time.time() - start_time
        writer.add_scalar('Time/Traversal', traversal_time, iteration)
        
        # Train advantage network
        print("  Training advantage network...")
        adv_loss = agent.train_advantage_network()
        advantage_losses.append(adv_loss)
        print(f"  Advantage network loss: {adv_loss:.6f}")
        writer.add_scalar('Loss/Advantage', adv_loss, iteration)
        
        # Every few iterations, train the strategy network
        if iteration % 5 == 0 or iteration == final_iteration:
            print("  Training strategy network...")
            strat_loss = agent.train_strategy_network()
            strategy_losses.append(strat_loss)
            print(f"  Strategy network loss: {strat_loss:.6f}")
            writer.add_scalar('Loss/Strategy', strat_loss, iteration)
        
        # Train opponent modeling periodically
        if iteration % 10 == 0 or iteration == final_iteration:
            print("  Training opponent modeling...")
            try:
                opp_loss = agent.train_opponent_modeling()
                opponent_model_losses.append(opp_loss)
                print(f"  Opponent modeling loss: {opp_loss:.6f}")
                writer.add_scalar('Loss/OpponentModeling', opp_loss, iteration)
            except Exception as e:
                error_msg = f"Error training opponent modeling: {e}"
                print(error_msg)
                if notifier:
                    notifier.send_message(f"⚠️ <b>OPPONENT MODELING ERROR</b>\n{error_msg}")
                writer.flush()
                raise
        
        # Evaluate periodically
        if iteration % 20 == 0 or iteration == final_iteration:
            # Evaluate against random agents (baseline)
            print("  Evaluating against random agents...")
            random_opponents = [RandomAgent(i) for i in range(6) if i != player_id]
            test_opponents = [None] * 6
            for opp in random_opponents:
                test_opponents[opp.player_id] = opp
                
            avg_profit_random = evaluate_against_opponents(
                agent, 
                test_opponents, 
                num_games=100, 
                iteration=iteration, 
                notifier=notifier
            )
            profits.append(avg_profit_random)
            print(f"  Average profit vs random: {avg_profit_random:.2f}")
            writer.add_scalar('Performance/ProfitVsRandom', avg_profit_random, iteration)
            
            # Evaluate against model opponents
            print("  Evaluating against model opponents...")
            avg_profit_models = evaluate_against_opponents(
                agent, 
                opponents, 
                num_games=100, 
                iteration=iteration, 
                notifier=notifier
            )
            profits_vs_models.append(avg_profit_models)
            print(f"  Average profit vs models: {avg_profit_models:.2f}")
            writer.add_scalar('Performance/ProfitVsModels', avg_profit_models, iteration)
            
            # Check for suspicious zero profit
            if abs(avg_profit_models) < 0.01:
                if notifier:
                    notifier.send_message(
                        f"⚠️ <b>ZERO PROFIT ALERT</b> at iteration {iteration}\n"
                        f"This suggests games may not be completing properly."
                    )
            
            # Send progress update every 100 iterations or if a significant change occurs
            if notifier and iteration % 100 == 0:
                notifier.send_training_progress(iteration, avg_profit_models, avg_profit_random)
        
        # Save checkpoint
        if iteration % checkpoint_frequency == 0 or iteration == final_iteration:
            checkpoint_path = f"{save_dir}/mixed_checkpoint_iter_{iteration}.pt"
            agent.save_model(checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
            
            # Notify on checkpoint save
            if iteration % 500 == 0:  # Less frequent notifications for checkpoints
                if notifier:
                    notifier.send_message(f"💾 <b>CHECKPOINT SAVED</b> at iteration {iteration}")
        
        # Log memory sizes
        writer.add_scalar('Memory/Advantage', len(agent.advantage_memory), iteration)
        writer.add_scalar('Memory/Strategy', len(agent.strategy_memory), iteration)
        
        # Log opponent model size
        num_tracked_opponents = len(agent.opponent_modeling.opponent_histories)
        total_history_entries = sum(len(h) for h in agent.opponent_modeling.opponent_histories.values())
        writer.add_scalar('OpponentModeling/NumOpponents', num_tracked_opponents, iteration)
        writer.add_scalar('OpponentModeling/TotalHistoryEntries', total_history_entries, iteration)
        
        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        print(f"  Iteration completed in {elapsed:.2f} seconds")
        print(f"  Tracking data for {num_tracked_opponents} unique opponents")
        print()
    
    # Final evaluation with more games
    print("Final evaluation...")
    
    # Against random agents
    random_opponents = [RandomAgent(i) for i in range(6) if i != player_id]
    test_opponents = [None] * 6
    for opp in random_opponents:
        test_opponents[opp.player_id] = opp
        
    avg_profit_random = evaluate_against_opponents(
        agent, 
        test_opponents, 
        num_games=500, 
        iteration=final_iteration, 
        notifier=notifier
    )
    print(f"Final performance vs random: Average profit per game: {avg_profit_random:.2f}")
    writer.add_scalar('Performance/FinalProfitVsRandom', avg_profit_random, 0)
    
    # Against model opponents
    avg_profit_models = evaluate_against_opponents(
        agent, 
        opponents, 
        num_games=500, 
        iteration=final_iteration, 
        notifier=notifier
    )
    print(f"Final performance vs models: Average profit per game: {avg_profit_models:.2f}")
    writer.add_scalar('Performance/FinalProfitVsModels', avg_profit_models, 0)
    
    # Final notification
    if notifier:
        notifier.send_message(
            f"✅ <b>TRAINING COMPLETED</b>\n"
            f"Final iteration: {final_iteration}\n"
            f"Final profit vs random: {avg_profit_random:.2f}\n"
            f"Final profit vs models: {avg_profit_models:.2f}"
        )
    
    # Close the tensorboard writer
    writer.close()
    
    return agent, advantage_losses, strategy_losses, opponent_model_losses, profits, profits_vs_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Deep CFR agent with opponent modeling against mixed opponents')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory containing opponent model checkpoints')
    parser.add_argument('--model-prefix', type=str, default="*checkpoint_iter_", help='Checkpoint prefix or glob fragment for model selection')
    parser.add_argument('--iterations', type=int, default=20000, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=200, help='Traversals per iteration')
    parser.add_argument('--refresh-interval', type=int, default=1000, help='How often to refresh opponent models')
    parser.add_argument('--num-opponents', type=int, default=5, help='Number of model opponents to select')
    parser.add_argument('--save-dir', type=str, default='models/opponent_modeling/mixed', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs/opponent_modeling/mixed', help='Directory for logs')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to continue training from')
    parser.add_argument('--strict', action='store_true', help='Raise exceptions for invalid game states')
    parser.add_argument('--allow-random-fallback', action='store_true', help='Use random opponents when the checkpoint pool is empty')
    args = parser.parse_args()

    set_strict_checking(args.strict)
    
    print(f"Starting mixed opponent training with models from: {args.checkpoint_dir}")
    print(f"Training for {args.iterations} additional iterations")
    print(f"Using {args.traversals} traversals per iteration")
    print(f"Refreshing opponents every {args.refresh_interval} iterations")
    print(f"Selecting {args.num_opponents} model opponents for each training phase")
    print(f"Logs will be saved to: {args.log_dir}")
    print(f"Models will be saved to: {args.save_dir}")
    
    if args.checkpoint:
        print(f"Continuing training from checkpoint: {args.checkpoint}")
    
    # Train the agent
    agent, adv_losses, strat_losses, om_losses, profits, profits_vs_models = train_mixed_with_opponent_modeling(
        checkpoint_dir=args.checkpoint_dir,
        num_iterations=args.iterations,
        traversals_per_iteration=args.traversals,
        refresh_interval=args.refresh_interval,
        num_opponents=args.num_opponents,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        model_prefix=args.model_prefix,
        verbose=args.verbose,
        checkpoint_path=args.checkpoint,
        allow_random_fallback=args.allow_random_fallback,
    )
    
    print("\nTraining Summary:")
    if adv_losses:
        print(f"Final advantage network loss: {adv_losses[-1]:.6f}")
    if strat_losses:
        print(f"Final strategy network loss: {strat_losses[-1]:.6f}")
    if om_losses:
        print(f"Final opponent modeling loss: {om_losses[-1]:.6f}")
    if profits:
        print(f"Final average profit vs random: {profits[-1]:.2f}")
    if profits_vs_models:
        print(f"Final average profit vs models: {profits_vs_models[-1]:.2f}")
    
    print("\nTo view training progress:")
    print(f"Run: tensorboard --logdir={args.log_dir}")
    print("Then open http://localhost:6006 in your browser")
