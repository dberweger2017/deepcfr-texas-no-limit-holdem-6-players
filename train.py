# train.py
import pokers as pkrs
import numpy as np
import random
import torch
import time
import os
import matplotlib.pyplot as plt
import argparse
from deep_cfr import DeepCFRAgent
from model import set_verbose

class RandomAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        
    def choose_action(self, state):
        """Choose a random legal action."""
        if not state.legal_actions:
            # Default action if no legal actions (shouldn't happen)
            return pkrs.Action(pkrs.ActionEnum.Call)
        
        # Select a random legal action
        action_enum = random.choice(state.legal_actions)
        
        # For raises, select a random amount between min and max
        if action_enum == pkrs.ActionEnum.Raise:
            player_state = state.players_state[state.current_player]
            min_amount = state.min_bet
            max_amount = player_state.stake  # All-in
            
            # Choose between 0.5x pot, 1x pot, or a random amount
            pot_amounts = [state.pot * 0.5, state.pot]
            valid_amounts = [amt for amt in pot_amounts if min_amount <= amt <= max_amount]
            
            if valid_amounts:
                amount = random.choice(valid_amounts)
            else:
                amount = random.uniform(min_amount, max_amount)
                
            return pkrs.Action(action_enum, amount)
        else:
            return pkrs.Action(action_enum)

def evaluate_against_random(agent, num_games=100, num_players=6):
    """Evaluate the trained agent against random opponents."""
    random_agents = [RandomAgent(i) for i in range(num_players)]
    
    total_profit = 0
    for game in range(num_games):
        # Create a new poker game
        state = pkrs.State.from_seed(
            n_players=num_players,
            button=game % num_players,  # Rotate button for fairness
            sb=1,
            bb=2,
            stake=200.0,
            seed=game
        )
        
        # Play until the game is over
        while not state.final_state:
            current_player = state.current_player
            
            if current_player == agent.player_id:
                action = agent.choose_action(state)
            else:
                action = random_agents[current_player].choose_action(state)
                
            state = state.apply_action(action)
        
        # Add the profit for this game
        profit = state.players_state[agent.player_id].reward
        total_profit += profit
    
    return total_profit / num_games

def train_deep_cfr(num_iterations=1000, traversals_per_iteration=200, 
                   num_players=6, player_id=0, save_dir="models", 
                   log_dir="logs/deepcfr", verbose=False):
    """
    Train a Deep CFR agent in a 6-player No-Limit Texas Hold'em game
    against 5 random opponents.
    """
    # Import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    
    # Set verbosity
    set_verbose(verbose)
    
    # Create the directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Initialize the agent
    agent = DeepCFRAgent(player_id=player_id, num_players=num_players, 
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create random agents for the opponents
    random_agents = [RandomAgent(i) for i in range(num_players)]
    
    # For tracking learning progress
    losses = []
    profits = []
    
    # Checkpoint frequency
    checkpoint_frequency = 100  # Save a checkpoint every 100 iterations
    
    # Training loop
    for iteration in range(1, num_iterations + 1):
        agent.iteration_count = iteration
        start_time = time.time()
        
        print(f"Iteration {iteration}/{num_iterations}")
        
        # Run traversals to collect data
        print("  Collecting data...")
        for _ in range(traversals_per_iteration):
            # Create a new poker game
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=random.randint(0, num_players-1),
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 10000)
            )
            
            # Perform CFR traversal
            agent.cfr_traverse(state, iteration, random_agents)
        
        # Track traversal time
        traversal_time = time.time() - start_time
        writer.add_scalar('Time/Traversal', traversal_time, iteration)
        
        # Train advantage network
        print("  Training advantage network...")
        adv_loss = agent.train_advantage_network()
        losses.append(adv_loss)
        print(f"  Advantage network loss: {adv_loss:.6f}")
        
        # Log the loss to tensorboard
        writer.add_scalar('Loss/Advantage', adv_loss, iteration)
        writer.add_scalar('Memory/Advantage', len(agent.advantage_memory), iteration)
        
        # Every few iterations, train the strategy network and evaluate
        if iteration % 10 == 0 or iteration == num_iterations:
            print("  Training strategy network...")
            strat_loss = agent.train_strategy_network()
            print(f"  Strategy network loss: {strat_loss:.6f}")
            writer.add_scalar('Loss/Strategy', strat_loss, iteration)
            
            # Evaluate the agent
            print("  Evaluating agent...")
            avg_profit = evaluate_against_random(agent, num_games=100, num_players=num_players)
            profits.append(avg_profit)
            print(f"  Average profit per game: {avg_profit:.2f}")
            writer.add_scalar('Performance/Profit', avg_profit, iteration)
            
            # Save the model
            model_path = f"{save_dir}/deep_cfr_iter_{iteration}.pt"
            agent.save_model(model_path)
            print(f"  Model saved to {model_path}")
        
        # Save checkpoint periodically
        if iteration % checkpoint_frequency == 0:
            checkpoint_path = f"{save_dir}/checkpoint_iter_{iteration}.pt"
            torch.save({
                'iteration': iteration,
                'advantage_net': agent.advantage_net.state_dict(),
                'strategy_net': agent.strategy_net.state_dict(),
                'losses': losses,
                'profits': profits
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
        
        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        print(f"  Iteration completed in {elapsed:.2f} seconds")
        print(f"  Advantage memory size: {len(agent.advantage_memory)}")
        print(f"  Strategy memory size: {len(agent.strategy_memory)}")
        writer.add_scalar('Memory/Strategy', len(agent.strategy_memory), iteration)
        
        # Commit the tensorboard logs
        writer.flush()
        print()
    
    # Final evaluation
    print("Final evaluation...")
    avg_profit = evaluate_against_random(agent, num_games=1000)
    print(f"Final performance: Average profit per game: {avg_profit:.2f}")
    writer.add_scalar('Performance/FinalProfit', avg_profit, 0)
    
    # Close the tensorboard writer
    writer.close()
    
    return agent, losses, profits

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a Deep CFR agent for poker')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=200, help='Traversals per iteration')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs/deepcfr', help='Directory for tensorboard logs')
    args = parser.parse_args()
    
    print(f"Starting Deep CFR training for {args.iterations} iterations")
    print(f"Using {args.traversals} traversals per iteration")
    print(f"Logs will be saved to: {args.log_dir}")
    print(f"Models will be saved to: {args.save_dir}")
    
    # Train the Deep CFR agent
    agent, losses, profits = train_deep_cfr(
        num_iterations=args.iterations,
        traversals_per_iteration=args.traversals,
        num_players=6,
        player_id=0,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        verbose=args.verbose
    )
    
    print("\nTraining Summary:")
    print(f"Final loss: {losses[-1]:.6f}")
    if profits:
        print(f"Final average profit: {profits[-1]:.2f}")
    
    print("\nTo view training progress:")
    print(f"Run: tensorboard --logdir={args.log_dir}")
    print("Then open http://localhost:6006 in your browser")