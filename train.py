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

def continue_training(checkpoint_path, additional_iterations=1000, 
                     traversals_per_iteration=200, save_dir="models", 
                     log_dir="logs/deepcfr_continued", verbose=False):
    """
    Continue training a Deep CFR agent from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the saved checkpoint
        additional_iterations: Number of additional iterations to train
        traversals_per_iteration: Number of traversals per iteration
        save_dir: Directory to save new models
        log_dir: Directory for tensorboard logs
        verbose: Whether to print verbose output
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
    
    # Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize the agent
    num_players = 6  # Assuming 6 players as in the original training
    player_id = 0    # Assuming player_id 0 as in the original training
    agent = DeepCFRAgent(player_id=player_id, num_players=num_players, 
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model weights
    agent.advantage_net.load_state_dict(checkpoint['advantage_net'])
    agent.strategy_net.load_state_dict(checkpoint['strategy_net'])
    
    # Set the iteration count from the checkpoint
    start_iteration = checkpoint['iteration'] + 1
    agent.iteration_count = start_iteration - 1
    
    # Load the training history if available
    losses = checkpoint.get('losses', [])
    profits = checkpoint.get('profits', [])
    
    print(f"Loaded model from iteration {start_iteration-1}")
    print(f"Continuing training for {additional_iterations} more iterations")
    
    # Create random agents for the opponents
    random_agents = [RandomAgent(i) for i in range(num_players)]
    
    # Checkpoint frequency
    checkpoint_frequency = 100  # Save a checkpoint every 100 iterations
    
    # Training loop
    for iteration in range(start_iteration, start_iteration + additional_iterations):
        agent.iteration_count = iteration
        start_time = time.time()
        
        print(f"Iteration {iteration}/{start_iteration + additional_iterations - 1}")
        
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
        if iteration % 10 == 0 or iteration == start_iteration + additional_iterations - 1:
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

def train_against_checkpoint(checkpoint_path, additional_iterations=1000, 
                           traversals_per_iteration=200, save_dir="models", 
                           log_dir="logs/deepcfr_selfplay", verbose=False):
    """
    Train a new Deep CFR agent against a fixed agent loaded from a checkpoint.
    
    Args:
        checkpoint_path: Path to the saved checkpoint to use as an opponent
        additional_iterations: Number of iterations to train
        traversals_per_iteration: Number of traversals per iteration
        save_dir: Directory to save new models
        log_dir: Directory for tensorboard logs
        verbose: Whether to print verbose output
    """
    from torch.utils.tensorboard import SummaryWriter
    
    # Set verbosity
    set_verbose(verbose)
    
    # Create the directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    print(f"Loading opponent from checkpoint: {checkpoint_path}")
    
    # Load the opponent agent from checkpoint
    opponent_agent = DeepCFRAgent(player_id=0, num_players=6, 
                                 device='cuda' if torch.cuda.is_available() else 'cpu')
    opponent_agent.load_model(checkpoint_path)
    
    # Create opponent agents for each player position
    opponent_agents = []
    for player_pos in range(6):
        # Create a new agent for each position
        if player_pos == 0:
            # For player 0, use the main opponent agent
            opponent_agents.append(opponent_agent)
        else:
            # For other positions, create copies of the opponent
            pos_agent = DeepCFRAgent(player_id=player_pos, num_players=6,
                                     device='cuda' if torch.cuda.is_available() else 'cpu')
            pos_agent.load_model(checkpoint_path)
            # We need to change the player_id for each agent
            pos_agent.player_id = player_pos
            opponent_agents.append(pos_agent)
    
    # Initialize the new learning agent
    learning_agent = DeepCFRAgent(player_id=0, num_players=6, 
                                  device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # For tracking learning progress
    losses = []
    profits = []
    
    # Checkpoint frequency
    checkpoint_frequency = 20  # Save more frequently for self-play
    
    class AgentWrapper:
        """Simple wrapper to match the interface expected by the CFR traverse function"""
        def __init__(self, agent):
            self.agent = agent
            self.player_id = agent.player_id
            
        def choose_action(self, state):
            return self.agent.choose_action(state)
    
    # Training loop
    for iteration in range(1, additional_iterations + 1):
        learning_agent.iteration_count = iteration
        start_time = time.time()
        
        print(f"Self-play Iteration {iteration}/{additional_iterations}")
        
        # Run traversals to collect data
        print("  Collecting data...")
        for t in range(traversals_per_iteration):
            # Rotate the button position for fairness
            button_pos = t % 6
            
            # Create a new poker game
            state = pkrs.State.from_seed(
                n_players=6,
                button=button_pos,
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 10000)
            )
            
            # Set up the opponent agents for this traversal
            # The learning agent always plays as player 0
            # We need to wrap the opponents to match the interface
            opponent_wrappers = [None] * 6
            for pos in range(6):
                if pos != 0:  # Not the learning agent's position
                    # Use the opponent agent for this position
                    opponent_wrappers[pos] = AgentWrapper(opponent_agents[pos])
            
            # Perform CFR traversal with the learning agent as player 0
            # and opponent agents in other positions
            learning_agent.cfr_traverse(state, iteration, opponent_wrappers)
        
        # Track traversal time
        traversal_time = time.time() - start_time
        writer.add_scalar('Time/Traversal', traversal_time, iteration)
        
        # Train advantage network
        print("  Training advantage network...")
        adv_loss = learning_agent.train_advantage_network()
        losses.append(adv_loss)
        print(f"  Advantage network loss: {adv_loss:.6f}")
        
        # Log the loss to tensorboard
        writer.add_scalar('Loss/Advantage', adv_loss, iteration)
        writer.add_scalar('Memory/Advantage', len(learning_agent.advantage_memory), iteration)
        
        # Every few iterations, train the strategy network and evaluate
        if iteration % 10 == 0 or iteration == additional_iterations:
            print("  Training strategy network...")
            strat_loss = learning_agent.train_strategy_network()
            print(f"  Strategy network loss: {strat_loss:.6f}")
            writer.add_scalar('Loss/Strategy', strat_loss, iteration)
            
            # First evaluate against original checkpoint agent
            print("  Evaluating against checkpoint agent...")
            avg_profit_vs_checkpoint = evaluate_against_agent(
                learning_agent, opponent_agent, num_games=100)
            print(f"  Average profit vs checkpoint: {avg_profit_vs_checkpoint:.2f}")
            writer.add_scalar('Performance/ProfitVsCheckpoint', 
                              avg_profit_vs_checkpoint, iteration)
            
            # Also evaluate against random for comparison
            print("  Evaluating against random agents...")
            avg_profit_random = evaluate_against_random(
                learning_agent, num_games=100, num_players=6)
            profits.append(avg_profit_random)
            print(f"  Average profit vs random: {avg_profit_random:.2f}")
            writer.add_scalar('Performance/ProfitVsRandom', 
                              avg_profit_random, iteration)
            
            # Save the model
            model_path = f"{save_dir}/deep_cfr_selfplay_iter_{iteration}"
            learning_agent.save_model(model_path)
            print(f"  Model saved to {model_path}")
        
        # Save checkpoint periodically
        if iteration % checkpoint_frequency == 0:
            checkpoint_path = f"{save_dir}/selfplay_checkpoint_iter_{iteration}.pt"
            torch.save({
                'iteration': iteration,
                'advantage_net': learning_agent.advantage_net.state_dict(),
                'strategy_net': learning_agent.strategy_net.state_dict(),
                'losses': losses,
                'profits': profits
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
        
        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        print(f"  Iteration completed in {elapsed:.2f} seconds")
        print(f"  Advantage memory size: {len(learning_agent.advantage_memory)}")
        print(f"  Strategy memory size: {len(learning_agent.strategy_memory)}")
        writer.add_scalar('Memory/Strategy', len(learning_agent.strategy_memory), iteration)
        
        # Commit the tensorboard logs
        writer.flush()
        print()
    
    # Final evaluation with more games
    print("Final evaluation...")
    avg_profit_vs_checkpoint = evaluate_against_agent(
        learning_agent, opponent_agent, num_games=500)
    print(f"Final performance vs checkpoint: Average profit per game: {avg_profit_vs_checkpoint:.2f}")
    writer.add_scalar('Performance/FinalProfitVsCheckpoint', avg_profit_vs_checkpoint, 0)
    
    avg_profit_random = evaluate_against_random(learning_agent, num_games=500)
    print(f"Final performance vs random: Average profit per game: {avg_profit_random:.2f}")
    writer.add_scalar('Performance/FinalProfitVsRandom', avg_profit_random, 0)
    
    writer.close()
    
    return learning_agent, losses, profits

def evaluate_against_agent(agent, opponent_agent, num_games=100):
    """Evaluate the trained agent against an opponent agent."""
    total_profit = 0
    
    class AgentWrapper:
        def __init__(self, agent, player_id):
            self.agent = agent
            self.player_id = player_id
            
        def choose_action(self, state):
            # We need to map the game's current_player to agent's perspective
            return self.agent.choose_action(state)
    
    # Create opponent wrappers for each position
    opponent_wrappers = []
    for pos in range(6):
        if pos == agent.player_id:
            opponent_wrappers.append(None)  # Will be filled with the main agent
        else:
            # Create a copy of the opponent agent for this position
            pos_agent = DeepCFRAgent(player_id=pos, num_players=6,
                                     device=opponent_agent.device)
            pos_agent.advantage_net.load_state_dict(opponent_agent.advantage_net.state_dict())
            pos_agent.strategy_net.load_state_dict(opponent_agent.strategy_net.state_dict())
            opponent_wrappers.append(AgentWrapper(pos_agent, pos))
    
    for game in range(num_games):
        # Create a new poker game with rotating button
        state = pkrs.State.from_seed(
            n_players=6,
            button=game % 6,  # Rotate button for fairness
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
                action = opponent_wrappers[current_player].choose_action(state)
                
            state = state.apply_action(action)
        
        # Add the profit for this game
        profit = state.players_state[agent.player_id].reward
        total_profit += profit
    
    return total_profit / num_games

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a Deep CFR agent for poker')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=200, help='Traversals per iteration')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs/deepcfr', help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to continue training from')
    args = parser.parse_args()
    
    if args.checkpoint:
        print(f"Continuing training from checkpoint: {args.checkpoint}")
        agent, losses, profits = continue_training(
            checkpoint_path=args.checkpoint,
            additional_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=args.log_dir + "_continued",
            verbose=args.verbose
        )
    else:
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