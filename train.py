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
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create opponent agents for all positions
    opponent_agents = []
    for pos in range(6):
        # Create a new agent for each position
        pos_agent = DeepCFRAgent(player_id=pos, num_players=6, device=device)
        pos_agent.load_model(checkpoint_path)
        opponent_agents.append(pos_agent)
    
    # Initialize the new learning agent for position 0
    learning_agent = DeepCFRAgent(player_id=0, num_players=6, device=device)
    
    # Optionally: initialize learning agent from checkpoint weights
    # This would give it a better starting point
    # learning_agent.advantage_net.load_state_dict(opponent_agents[0].advantage_net.state_dict())
    # learning_agent.strategy_net.load_state_dict(opponent_agents[0].strategy_net.state_dict())
    
    # For tracking learning progress
    losses = []
    profits = []
    
    # Checkpoint frequency
    checkpoint_frequency = 20  # Save more frequently for self-play
    
    class AgentWrapper:
        """Wrapper to ensure agents receive observations from their own perspective"""
        def __init__(self, agent):
            self.agent = agent
            self.player_id = agent.player_id
            
        def choose_action(self, state):
            # This is critical - the agent views the state from its own perspective
            return self.agent.choose_action(state)
    
    # Save the original cfr_traverse method so we can restore it later
    original_cfr_traverse = DeepCFRAgent.cfr_traverse
    
    # Define a modified cfr_traverse method for self-play
    def self_play_cfr_traverse(self, state, iteration, opponent_agents, depth=0):
        """
        Modified CFR traverse method to ensure proper state perspective for each agent.
        """
        # Add recursion depth protection
        max_depth = 1000
        if depth > max_depth:
            if verbose:
                print(f"WARNING: Max recursion depth reached ({max_depth}). Returning zero value.")
            return 0
        
        if state.final_state:
            # Return payoff for the trained agent
            return state.players_state[self.player_id].reward
        
        current_player = state.current_player
        
        # Debug information for the current state
        if verbose and depth % 100 == 0:
            print(f"Depth: {depth}, Player: {current_player}, Stage: {state.stage}")
        
        # If it's the trained agent's turn
        if current_player == self.player_id:
            legal_action_ids = self.get_legal_action_ids(state)
            
            if not legal_action_ids:
                if verbose:
                    print(f"WARNING: No legal actions found for player {current_player} at depth {depth}")
                return 0
                
            # Encode state from this agent's perspective
            state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).to(self.device)
            
            # Get advantages from network
            with torch.no_grad():
                advantages = self.advantage_net(state_tensor.unsqueeze(0))[0]
                
            # Use regret matching to compute strategy
            advantages_np = advantages.cpu().numpy()
            advantages_masked = np.zeros(self.num_actions)
            for a in legal_action_ids:
                advantages_masked[a] = max(advantages_np[a], 0)
                
            # Choose an action based on the strategy
            if sum(advantages_masked) > 0:
                strategy = advantages_masked / sum(advantages_masked)
            else:
                strategy = np.zeros(self.num_actions)
                for a in legal_action_ids:
                    strategy[a] = 1.0 / len(legal_action_ids)
            
            # Choose actions and traverse
            action_values = np.zeros(self.num_actions)
            for action_id in legal_action_ids:
                try:
                    pokers_action = self.action_id_to_pokers_action(action_id, state)
                    new_state = state.apply_action(pokers_action)
                    
                    # Check if the action was valid
                    if new_state.status != pkrs.StateStatus.Ok:
                        if verbose:
                            print(f"WARNING: Invalid action {action_id} at depth {depth}. Status: {new_state.status}")
                        continue
                        
                    action_values[action_id] = self.cfr_traverse(new_state, iteration, opponent_agents, depth + 1)
                except Exception as e:
                    if verbose:
                        print(f"ERROR in traversal for action {action_id}: {e}")
                    action_values[action_id] = 0
            
            # Compute counterfactual regrets and add to memory
            ev = sum(strategy[a] * action_values[a] for a in legal_action_ids)
            
            # Calculate normalization factor
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)
            
            for action_id in legal_action_ids:
                # Calculate regret
                regret = action_values[action_id] - ev
                
                # Normalize and clip regret
                normalized_regret = regret / max_abs_val
                clipped_regret = np.clip(normalized_regret, -10.0, 10.0)
                
                # Apply scaling
                scale_factor = np.sqrt(iteration) if iteration > 1 else 1.0
                
                self.advantage_memory.append((
                    encode_state(state, self.player_id),  # Encode from this agent's perspective
                    action_id,
                    clipped_regret * scale_factor
                ))
            
            # Add to strategy memory
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_ids:
                strategy_full[a] = strategy[a]
            
            self.strategy_memory.append((
                encode_state(state, self.player_id),  # Encode from this agent's perspective
                strategy_full,
                iteration
            ))
            
            return ev
            
        # If it's another player's turn (opponent agent)
        else:
            try:
                # Let the appropriate opponent agent choose an action
                if opponent_agents[current_player] is not None:
                    action = opponent_agents[current_player].choose_action(state)
                    new_state = state.apply_action(action)
                    
                    # Check if the action was valid
                    if new_state.status != pkrs.StateStatus.Ok:
                        if verbose:
                            print(f"WARNING: Opponent agent made invalid action at depth {depth}. Status: {new_state.status}")
                        return 0
                        
                    return self.cfr_traverse(new_state, iteration, opponent_agents, depth + 1)
                else:
                    # This should not happen - all opponent positions should have agents
                    if verbose:
                        print(f"WARNING: No opponent agent for position {current_player}")
                    return 0
            except Exception as e:
                if verbose:
                    print(f"ERROR in opponent agent traversal: {e}")
                return 0
    
    # Replace the cfr_traverse method with our self-play version
    DeepCFRAgent.cfr_traverse = self_play_cfr_traverse
    
    try:
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
                opponent_wrappers = [None] * 6
                for pos in range(6):
                    if pos != 0:  # Not the learning agent's position
                        # Use the opponent agent for this position
                        opponent_wrappers[pos] = AgentWrapper(opponent_agents[pos])
                
                # Perform CFR traversal
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
                
                # Evaluate against checkpoint agents
                print("  Evaluating against checkpoint agent...")
                avg_profit_vs_checkpoint = evaluate_against_checkpoint_agents(
                    learning_agent, opponent_agents, num_games=100)
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
        avg_profit_vs_checkpoint = evaluate_against_checkpoint_agents(
            learning_agent, opponent_agents, num_games=500)
        print(f"Final performance vs checkpoint: Average profit per game: {avg_profit_vs_checkpoint:.2f}")
        writer.add_scalar('Performance/FinalProfitVsCheckpoint', avg_profit_vs_checkpoint, 0)
        
        avg_profit_random = evaluate_against_random(learning_agent, num_games=500)
        print(f"Final performance vs random: Average profit per game: {avg_profit_random:.2f}")
        writer.add_scalar('Performance/FinalProfitVsRandom', avg_profit_random, 0)
        
        writer.close()
        
        return learning_agent, losses, profits
    
    finally:
        # Restore the original cfr_traverse method to avoid side effects
        DeepCFRAgent.cfr_traverse = original_cfr_traverse

def evaluate_against_checkpoint_agents(agent, opponent_agents, num_games=100):
    """
    Evaluate the trained agent against opponent agents.
    Each agent will receive and process observations from its own perspective.
    """
    total_profit = 0
    
    class AgentWrapper:
        def __init__(self, agent):
            self.agent = agent
            self.player_id = agent.player_id
            
        def choose_action(self, state):
            # Each agent processes the state from its own perspective
            return self.agent.choose_action(state)
    
    # Wrap checkpoint agents
    opponent_wrappers = [None] * 6
    for pos in range(6):
        if pos != agent.player_id:
            opponent_wrappers[pos] = AgentWrapper(opponent_agents[pos])
    
    for game in range(num_games):
        # Create a new poker game with rotating button
        state = pkrs.State.from_seed(
            n_players=6,
            button=game % 6,  # Rotate button for fairness
            sb=1,
            bb=2,
            stake=200.0,
            seed=game + 10000  # Using different seeds than training
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

def evaluate_against_agent(agent, opponent_agent, num_games=100):
    """Evaluate the trained agent against an opponent agent."""
    # This is kept for backward compatibility
    # But we recommend using evaluate_against_checkpoint_agents instead
    
    # Create a list of opponent agents for all positions
    opponent_agents = []
    for pos in range(6):
        if pos == agent.player_id:
            opponent_agents.append(None)  # Will be filled with the main agent
        else:
            # Create a copy of the opponent agent for this position
            pos_agent = DeepCFRAgent(player_id=pos, num_players=6,
                                    device=opponent_agent.device)
            pos_agent.advantage_net.load_state_dict(opponent_agent.advantage_net.state_dict())
            pos_agent.strategy_net.load_state_dict(opponent_agent.strategy_net.state_dict())
            opponent_agents.append(pos_agent)
    
    return evaluate_against_checkpoint_agents(agent, opponent_agents, num_games)

# For importing model.encode_state in the cfr_traverse method
from model import encode_state

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a Deep CFR agent for poker')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=200, help='Traversals per iteration')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs/deepcfr', help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to continue training from')
    parser.add_argument('--self-play', action='store_true', help='Train against checkpoint instead of random agents')
    args = parser.parse_args()
    
    if args.checkpoint and args.self_play:
        print(f"Starting self-play training against checkpoint: {args.checkpoint}")
        agent, losses, profits = train_against_checkpoint(
            checkpoint_path=args.checkpoint,
            additional_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=args.log_dir + "_selfplay",
            verbose=args.verbose
        )
    elif args.checkpoint:
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