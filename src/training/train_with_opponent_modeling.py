# train_with_opponent_modeling.py
import pokers as pkrs
import torch
import numpy as np
import os
import random
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from src.opponent_modeling.deep_cfr_with_opponent_modeling import DeepCFRAgentWithOpponentModeling
from src.core.model import set_verbose

class RandomAgent:
    """Simple random agent for poker (unchanged from original)."""
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

def evaluate_against_random(agent, num_games=500, num_players=6):
    """Evaluate the trained agent against random opponents, tracking opponent history."""
    random_agents = [RandomAgent(i) for i in range(num_players)]
    total_profit = 0
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
                action = random_agents[current_player].choose_action(state)
                
                # Record this opponent's action
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
        agent.end_game_recording(state)
        
        # Add the profit for this game
        profit = state.players_state[agent.player_id].reward
        total_profit += profit
    
    return total_profit / num_games

def train_deep_cfr_with_opponent_modeling(
    num_iterations=1000, 
    traversals_per_iteration=200,
    num_players=6, 
    player_id=0, 
    save_dir="models", 
    log_dir="logs/deepcfr_opponent_modeling", 
    verbose=False
):
    """Train a Deep CFR agent with opponent modeling."""
    # Set verbosity
    set_verbose(verbose)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Initialize the agent with opponent modeling
    agent = DeepCFRAgentWithOpponentModeling(
        player_id=player_id, 
        num_players=num_players,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
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
    for iteration in range(1, num_iterations + 1):
        agent.iteration_count = iteration
        start_time = time.time()
        
        print(f"Iteration {iteration}/{num_iterations}")
        
        # Run traversals to collect data
        print("  Collecting data...")
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
            agent.cfr_traverse(state, iteration, random_agents)
        
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
            print("  Evaluating agent...")
            avg_profit = evaluate_against_random(agent, num_games=200)
            profits.append(avg_profit)
            print(f"  Average profit per game: {avg_profit:.2f}")
            writer.add_scalar('Performance/Profit', avg_profit, iteration)
        
        # Save checkpoint
        if iteration % checkpoint_frequency == 0 or iteration == num_iterations:
            checkpoint_path = f"{save_dir}/om_checkpoint_iter_{iteration}.pt"
            agent.save_model(checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
        
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
        print(f"  Iteration completed in {elapsed:.2f} seconds")
        print(f"  Tracking data for {num_opponents} unique opponents")
        print()
    
    # Final evaluation with more games
    print("Final evaluation...")
    avg_profit = evaluate_against_random(agent, num_games=1000)
    print(f"Final performance: Average profit per game: {avg_profit:.2f}")
    writer.add_scalar('Performance/FinalProfit', avg_profit, 0)
    
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
    args = parser.parse_args()
    
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
        print(f"Final average profit: {profits[-1]:.2f}")
    
    print("\nTo view training progress:")
    print(f"Run: tensorboard --logdir={args.log_dir}")
    print("Then open http://localhost:6006 in your browser")