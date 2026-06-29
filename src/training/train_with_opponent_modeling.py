"""Internal opponent-model training implementation.

Use ``python -m src.training.train_opponent_modeling`` as the public CLI entrypoint.
"""

import pokers as pkrs
import torch
import numpy as np
import os
import random
import time
import argparse
import sys
from tqdm import tqdm
from src.opponent_modeling.deep_cfr_with_opponent_modeling import DeepCFRAgentWithOpponentModeling
from src.core.model import set_verbose
from src.agents.random_agent import RandomAgent
from src.utils.evaluation import evaluate_agent_matchup
from src.utils.settings import set_strict_checking
from src.utils.checkpoints import load_checkpoint

def evaluate_against_random(agent, num_games=500, num_players=6, iteration=0, notifier=None):
    """Evaluate the trained agent against random opponents, tracking opponent history."""
    random_agents = [RandomAgent(i) for i in range(num_players)]
    metrics = evaluate_agent_matchup(
        agent,
        random_agents,
        num_games=num_games,
        seed_start=0,
        num_players=num_players,
        label="opponent-modeling evaluation vs random",
        record_opponent_history=True,
        print_warnings=True,
    )
    
    # Report if too many crashes or zero reward games
    if metrics["game_crashes"] > 0 and notifier:
        notifier.send_message(f"⚠️ <b>EVALUATION ISSUES</b>\nIteration: {iteration}\nCrashed games: {metrics['game_crashes']}/{num_games}")
    
    if metrics["zero_reward_games"] > 0.2 * metrics["completed_games"] and notifier:
        notifier.send_message(f"⚠️ <b>SUSPICIOUS REWARDS</b>\nIteration: {iteration}\nZero reward games: {metrics['zero_reward_games']}/{metrics['completed_games']}")
    
    if metrics["completed_games"] == 0:
        if notifier:
            notifier.send_message(f"🚨 <b>CRITICAL ERROR</b>\nIteration: {iteration}\nNo games completed!")
        return 0
        
    return metrics["avg_profit"]

def train_deep_cfr_with_opponent_modeling(
    num_iterations=1000, 
    traversals_per_iteration=200,
    num_players=6, 
    player_id=0, 
    save_dir="models", 
    log_dir="logs/deepcfr_opponent_modeling", 
    verbose=False,
    checkpoint_path=None,
    progress_interval=100,
):
    """Train a Deep CFR agent with opponent modeling."""
    from torch.utils.tensorboard import SummaryWriter
    from scripts.telegram_notifier import TelegramNotifier

    # Set verbosity
    set_verbose(verbose)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Initialize Telegram notifier
    try:
        notifier = TelegramNotifier()
        notifier.send_message(f"🚀 <b>BASIC OPPONENT MODELING TRAINING STARTED</b>\nDevice: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\nIterations: {num_iterations}\nTraversals: {traversals_per_iteration}")
    except Exception as e:
        print(f"Warning: Could not initialize Telegram notifier: {e}")
        notifier = None
    
    # Initialize the agent with opponent modeling
    agent = DeepCFRAgentWithOpponentModeling(
        player_id=player_id, 
        num_players=num_players,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    starting_iteration = 1

    if checkpoint_path:
        print(f"Loading agent from checkpoint: {checkpoint_path}")
        agent.load_model(checkpoint_path)
        try:
            checkpoint_state = load_checkpoint(checkpoint_path, map_location=agent.device)
            starting_iteration = checkpoint_state.get("iteration", agent.iteration_count) + 1
        except Exception:
            starting_iteration = agent.iteration_count + 1
    
    # Create random agents for opponents
    random_agents = [RandomAgent(i) for i in range(num_players)]
    
    # For tracking progress
    advantage_losses = []
    strategy_losses = []
    opponent_model_losses = []
    profits = []
    
    # Checkpoint frequency
    checkpoint_frequency = 50  # Save more frequently due to opponent modeling
    
    # Training loop
    final_iteration = starting_iteration + num_iterations - 1
    progress = tqdm(
        range(starting_iteration, final_iteration + 1),
        desc="OM training",
        unit="iter",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    last_strat_loss = None
    last_opp_loss = None
    last_profit = None

    for iteration in progress:
        agent.iteration_count = iteration
        start_time = time.time()

        # Run traversals to collect data
        for t in range(traversals_per_iteration):
            # Create a new poker game
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=random.randint(0, num_players-1),
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 10000)
            )
            
            # Perform CFR traversal with opponent modeling
            try:
                agent.cfr_traverse(state, iteration, random_agents)
            except Exception as e:
                tqdm.write(f"Error in traversal: {e}")
                if notifier and t % 50 == 0:  # Don't send too many error messages
                    notifier.send_message(f"⚠️ <b>TRAVERSAL ERROR</b>\nIteration: {iteration}\nError: {str(e)}")
        
        # Track traversal time
        traversal_time = time.time() - start_time
        writer.add_scalar('Time/Traversal', traversal_time, iteration)
        
        # Train advantage network
        adv_loss = agent.train_advantage_network()
        advantage_losses.append(adv_loss)
        writer.add_scalar('Loss/Advantage', adv_loss, iteration)
        
        # Every few iterations, train the strategy network
        if iteration % 5 == 0 or iteration == final_iteration:
            last_strat_loss = agent.train_strategy_network()
            strategy_losses.append(last_strat_loss)
            writer.add_scalar('Loss/Strategy', last_strat_loss, iteration)
        
        # Train opponent modeling periodically
        if iteration % 10 == 0 or iteration == final_iteration:
            try:
                last_opp_loss = agent.train_opponent_modeling()
                opponent_model_losses.append(last_opp_loss)
                writer.add_scalar('Loss/OpponentModeling', last_opp_loss, iteration)
            except Exception as e:
                tqdm.write(f"Error training opponent model: {e}")
                if notifier:
                    notifier.send_message(f"⚠️ <b>OPPONENT MODEL ERROR</b>\nIteration: {iteration}\nError: {str(e)}")
        
        # Evaluate periodically
        if iteration % 20 == 0 or iteration == final_iteration:
            last_profit = evaluate_against_random(agent, num_games=200, iteration=iteration, notifier=notifier)
            profits.append(last_profit)
            writer.add_scalar('Performance/Profit', last_profit, iteration)
            
            # Send progress update
            if notifier:
                num_opponents = len(agent.opponent_modeling.opponent_histories)
                notifier.send_training_progress(
                    iteration=iteration,
                    profit_vs_models=last_profit,
                    profit_vs_random=last_profit
                )
        
        # Save checkpoint
        if iteration % checkpoint_frequency == 0 or iteration == final_iteration:
            checkpoint_path = f"{save_dir}/checkpoint_iter_{iteration}.pt"
            agent.save_model(checkpoint_path)
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")
            
            # Add Telegram notification
            if notifier and iteration % 100 == 0:  # Less frequent notifications
                notifier.send_message(f"💾 <b>CHECKPOINT SAVED</b> at iteration {iteration}")
        
        # Log memory sizes
        writer.add_scalar('Memory/Advantage', len(agent.advantage_memory), iteration)
        writer.add_scalar('Memory/Strategy', len(agent.strategy_memory), iteration)
        
        # Log opponent model size
        num_opponents = len(agent.opponent_modeling.opponent_histories)
        total_history_entries = sum(len(h) for h in agent.opponent_modeling.opponent_histories.values())
        writer.add_scalar('OpponentModeling/NumOpponents', num_opponents, iteration)
        writer.add_scalar('OpponentModeling/TotalHistoryEntries', total_history_entries, iteration)
        
        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        progress.set_postfix(
            adv=f"{adv_loss:.3f}",
            strat="-" if last_strat_loss is None else f"{last_strat_loss:.3f}",
            om="-" if last_opp_loss is None else f"{last_opp_loss:.3f}",
            profit="-" if last_profit is None else f"{last_profit:.2f}",
            opp=num_opponents,
        )

        should_report = (
            progress_interval > 0
            and (iteration % progress_interval == 0 or iteration == final_iteration)
        )
        if should_report:
            tqdm.write(
                f"iter {iteration}/{final_iteration} | "
                f"adv={adv_loss:.4f} | "
                f"strat={'-' if last_strat_loss is None else f'{last_strat_loss:.4f}'} | "
                f"om={'-' if last_opp_loss is None else f'{last_opp_loss:.4f}'} | "
                f"profit={'-' if last_profit is None else f'{last_profit:.2f}'} | "
                f"adv_mem={len(agent.advantage_memory)} | "
                f"strat_mem={len(agent.strategy_memory)} | "
                f"opponents={num_opponents} | "
                f"{elapsed:.2f}s"
            )
    
    # Final evaluation with more games
    print("Final evaluation...")
    avg_profit = evaluate_against_random(agent, num_games=1000, iteration=num_iterations, notifier=notifier)
    print(f"Final performance: Average profit per game: {avg_profit:.2f}")
    writer.add_scalar('Performance/FinalProfit', avg_profit, 0)
    
    # Send final notification
    if notifier:
        num_opponents = len(agent.opponent_modeling.opponent_histories)
        notifier.send_message(
            f"✅ <b>TRAINING COMPLETED</b>\n"
            f"Total iterations: {num_iterations}\n"
            f"Final profit: {avg_profit:.2f}\n"
            f"Tracked opponents: {num_opponents}"
        )
    
    # Close the tensorboard writer
    writer.close()
    
    return agent, advantage_losses, strategy_losses, opponent_model_losses, profits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Deep CFR agent with opponent modeling')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=200, help='Traversals per iteration')
    parser.add_argument('--save-dir', type=str, default='models_om', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs/deepcfr_om', help='Directory for logs')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to continue training from')
    parser.add_argument('--strict', action='store_true', help='Raise exceptions for invalid game states')
    parser.add_argument('--progress-interval', type=int, default=100, help='Print compact training summaries every N iterations; set 0 to disable')
    args = parser.parse_args()

    set_strict_checking(args.strict)
    
    print(f"Starting Deep CFR training with opponent modeling for {args.iterations} iterations")
    print(f"Using {args.traversals} traversals per iteration")
    print(f"Logs will be saved to: {args.log_dir}")
    print(f"Models will be saved to: {args.save_dir}")
    
    # Train the agent
    agent, adv_losses, strat_losses, om_losses, profits = train_deep_cfr_with_opponent_modeling(
        num_iterations=args.iterations,
        traversals_per_iteration=args.traversals,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        verbose=args.verbose,
        checkpoint_path=args.checkpoint,
        progress_interval=args.progress_interval,
    )
    
    print("\nTraining Summary:")
    if adv_losses:
        print(f"Final advantage network loss: {adv_losses[-1]:.6f}")
    if strat_losses:
        print(f"Final strategy network loss: {strat_losses[-1]:.6f}")
    if om_losses:
        print(f"Final opponent modeling loss: {om_losses[-1]:.6f}")
    if profits:
        print(f"Final average profit: {profits[-1]:.2f}")
    
    print("\nTo view training progress:")
    print(f"Run: tensorboard --logdir={args.log_dir}")
    print("Then open http://localhost:6006 in your browser")
