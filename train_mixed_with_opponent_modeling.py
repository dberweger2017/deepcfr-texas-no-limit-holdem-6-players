# train_mixed_with_opponent_modeling.py
import pokers as pkrs
import torch
import numpy as np
import os
import random
import time
import argparse
import glob
from torch.utils.tensorboard import SummaryWriter
from deep_cfr_with_opponent_modeling import DeepCFRAgentWithOpponentModeling
from model import encode_state, set_verbose

class RandomAgent:
    """Simple random agent for poker."""
    def __init__(self, player_id):
        self.player_id = player_id
        self.name = f"Player {player_id}"
        
    def choose_action(self, state):
        if not state.legal_actions:
            raise ValueError(f"No legal actions available for player {self.player_id}")
        
        action_enum = random.choice(state.legal_actions)
        
        if action_enum in (pkrs.ActionEnum.Fold, pkrs.ActionEnum.Check, pkrs.ActionEnum.Call):
            return pkrs.Action(action_enum)
        
        elif action_enum == pkrs.ActionEnum.Raise:
            player_state = state.players_state[state.current_player]
            current_bet = player_state.bet_chips
            
            lower_bound = current_bet + state.min_bet
            upper_bound = current_bet + player_state.stake
            
            if player_state.stake < state.min_bet:
                return pkrs.Action(pkrs.ActionEnum.Call)
            
            candidate_half = current_bet + state.pot * 0.5
            candidate_full = current_bet + state.pot
            
            candidates = []
            if lower_bound <= candidate_half <= upper_bound:
                candidates.append(candidate_half)
            if lower_bound <= candidate_full <= upper_bound:
                candidates.append(candidate_full)
            
            if candidates:
                chosen_total = random.choice(candidates)
            else:
                chosen_total = random.uniform(lower_bound, upper_bound)
            
            desired_raise = chosen_total - current_bet
            final_raise = min(desired_raise, player_state.stake)
            
            return pkrs.Action(action_enum, final_raise)
        
        raise ValueError(f"Unexpected action type: {action_enum}")

class ModelAgent:
    """Wrapper for a DeepCFRAgent or DeepCFRAgentWithOpponentModeling loaded from a checkpoint."""
    def __init__(self, player_id, model_path, device='cpu', with_opponent_modeling=False):
        self.player_id = player_id
        self.name = f"Model Agent {player_id} ({os.path.basename(model_path)})"
        self.model_path = model_path
        self.device = device
        self.with_opponent_modeling = with_opponent_modeling
        
        # Load the appropriate agent type
        if with_opponent_modeling:
            from deep_cfr_with_opponent_modeling import DeepCFRAgentWithOpponentModeling
            self.agent = DeepCFRAgentWithOpponentModeling(player_id=player_id, device=device)
        else:
            from deep_cfr import DeepCFRAgent  # Assuming original DeepCFRAgent is available
            self.agent = DeepCFRAgent(player_id=player_id, device=device)
            
        # Load model weights
        self.agent.load_model(model_path)
    
    def choose_action(self, state):
        """Choose an action for the given state."""
        # If the agent supports opponent modeling, we could add more sophisticated logic here
        return self.agent.choose_action(state)

def evaluate_against_opponents(agent, opponents, num_games=100):
    """Evaluate the trained agent against a set of opponents."""
    total_profit = 0
    num_players = 6
    
    for game in range(num_games):
        # Create a new poker game
        state = pkrs.State.from_seed(
            n_players=num_players,
            button=game % num_players,
            sb=1,
            bb=2,
            stake=200.0,
            seed=game
        )
        
        # Play until the game is over
        while not state.final_state:
            current_player = state.current_player
            
            if current_player == agent.player_id:
                # Use opponent modeling for the current opponent
                action = agent.choose_action(state, opponent_id=current_player)
            else:
                action = opponents[current_player].choose_action(state)
                
                # Record this opponent's action
                if hasattr(agent, 'record_opponent_action'):
                    if action.action == pkrs.ActionEnum.Fold:
                        action_id = 0
                    elif action.action == pkrs.ActionEnum.Check or action.action == pkrs.ActionEnum.Call:
                        action_id = 1
                    elif action.action == pkrs.ActionEnum.Raise:
                        if action.amount <= state.pot * 0.75:
                            action_id = 2
                        else:
                            action_id = 3
                    else:
                        action_id = 1
                        
                    agent.record_opponent_action(state, action_id, current_player)
                
            state = state.apply_action(action)
        
        # Record end of game
        if hasattr(agent, 'end_game_recording'):
            agent.end_game_recording(state)
        
        # Add the profit for this game
        profit = state.players_state[agent.player_id].reward
        total_profit += profit
    
    return total_profit / num_games

def train_mixed_with_opponent_modeling(
    checkpoint_dir,
    num_iterations=20000,
    traversals_per_iteration=200,
    refresh_interval=1000,
    num_opponents=5,
    save_dir="models_mixed_om",
    log_dir="logs/deepcfr_mixed_om",
    player_id=0,
    model_prefix="*",  # Default to include all models
    verbose=False
):
    """
    Train a Deep CFR agent with opponent modeling against a mix of opponents
    loaded from checkpoints, refreshing the opponent pool periodically.
    """
    # Set verbosity
    set_verbose(verbose)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize the agent with opponent modeling
    agent = DeepCFRAgentWithOpponentModeling(
        player_id=player_id, 
        num_players=6,
        device=device
    )
    
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
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"{model_prefix}.pt"))
        
        if not checkpoint_files:
            print(f"WARNING: No checkpoint files found matching pattern '{model_prefix}' in {checkpoint_dir}")
            print("Using random agents as opponents")
            return [RandomAgent(i) for i in range(6) if i != player_id]
        
        # Select random models
        selected_files = random.sample(checkpoint_files, min(num_opponents, len(checkpoint_files)))
        print(f"Selected model opponents:")
        for i, f in enumerate(selected_files):
            print(f"  {i+1}. {os.path.basename(f)}")
        
        # Load the selected models
        model_opponents = []
        for pos, file_path in enumerate(selected_files, start=1):
            # Skip player_id position
            if pos == player_id:
                pos = (pos + 1) % 6
            
            # Check if this is an opponent modeling checkpoint or regular checkpoint
            # This is a simplistic way to detect - you might need a more robust method
            is_om_model = "om" in os.path.basename(file_path).lower()
            
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
    for iteration in range(1, num_iterations + 1):
        agent.iteration_count = iteration
        start_time = time.time()
        
        # Refresh opponents at specified intervals
        if iteration % refresh_interval == 1:
            print(f"\n=== Refreshing opponent pool at iteration {iteration} ===")
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
            # NOTE: We need to make sure cfr_traverse can handle our model opponents
            # This might require some adaptation in the DeepCFRAgentWithOpponentModeling class
            try:
                agent.cfr_traverse(state, iteration, opponents)
            except Exception as e:
                print(f"Error during traversal: {e}")
                continue
        
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
        if iteration % 5 == 0 or iteration == num_iterations:
            print("  Training strategy network...")
            strat_loss = agent.train_strategy_network()
            strategy_losses.append(strat_loss)
            print(f"  Strategy network loss: {strat_loss:.6f}")
            writer.add_scalar('Loss/Strategy', strat_loss, iteration)
        
        # Train opponent modeling periodically
        if iteration % 10 == 0 or iteration == num_iterations:
            print("  Training opponent modeling...")
            opp_loss = agent.train_opponent_modeling()
            opponent_model_losses.append(opp_loss)
            print(f"  Opponent modeling loss: {opp_loss:.6f}")
            writer.add_scalar('Loss/OpponentModeling', opp_loss, iteration)
        
        # Evaluate periodically
        if iteration % 20 == 0 or iteration == num_iterations:
            # Evaluate against random agents (baseline)
            print("  Evaluating against random agents...")
            random_opponents = [RandomAgent(i) for i in range(6) if i != player_id]
            test_opponents = [None] * 6
            for opp in random_opponents:
                test_opponents[opp.player_id] = opp
                
            avg_profit_random = evaluate_against_opponents(agent, test_opponents, num_games=100)
            profits.append(avg_profit_random)
            print(f"  Average profit vs random: {avg_profit_random:.2f}")
            writer.add_scalar('Performance/ProfitVsRandom', avg_profit_random, iteration)
            
            # Evaluate against model opponents
            print("  Evaluating against model opponents...")
            avg_profit_models = evaluate_against_opponents(agent, opponents, num_games=100)
            profits_vs_models.append(avg_profit_models)
            print(f"  Average profit vs models: {avg_profit_models:.2f}")
            writer.add_scalar('Performance/ProfitVsModels', avg_profit_models, iteration)
        
        # Save checkpoint
        if iteration % checkpoint_frequency == 0 or iteration == num_iterations:
            checkpoint_path = f"{save_dir}/mixed_om_checkpoint_iter_{iteration}.pt"
            agent.save_model(checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
        
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
        
    avg_profit_random = evaluate_against_opponents(agent, test_opponents, num_games=500)
    print(f"Final performance vs random: Average profit per game: {avg_profit_random:.2f}")
    writer.add_scalar('Performance/FinalProfitVsRandom', avg_profit_random, 0)
    
    # Against model opponents
    avg_profit_models = evaluate_against_opponents(agent, opponents, num_games=500)
    print(f"Final performance vs models: Average profit per game: {avg_profit_models:.2f}")
    writer.add_scalar('Performance/FinalProfitVsModels', avg_profit_models, 0)
    
    # Close the tensorboard writer
    writer.close()
    
    return agent, advantage_losses, strategy_losses, opponent_model_losses, profits, profits_vs_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Deep CFR agent with opponent modeling against mixed opponents')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory containing opponent model checkpoints')
    parser.add_argument('--model-prefix', type=str, default="*", help='File pattern for selecting model checkpoints')
    parser.add_argument('--iterations', type=int, default=20000, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=200, help='Traversals per iteration')
    parser.add_argument('--refresh-interval', type=int, default=1000, help='How often to refresh opponent models')
    parser.add_argument('--num-opponents', type=int, default=5, help='Number of model opponents to select')
    parser.add_argument('--save-dir', type=str, default='models_mixed_om', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs/deepcfr_mixed_om', help='Directory for logs')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    print(f"Starting mixed opponent training with models from: {args.checkpoint_dir}")
    print(f"Training for {args.iterations} iterations")
    print(f"Using {args.traversals} traversals per iteration")
    print(f"Refreshing opponents every {args.refresh_interval} iterations")
    print(f"Selecting {args.num_opponents} model opponents for each training phase")
    print(f"Logs will be saved to: {args.log_dir}")
    print(f"Models will be saved to: {args.save_dir}")
    
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
        verbose=args.verbose
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