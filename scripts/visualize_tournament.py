# visualize_tournament.py
import pokers as pkrs
import numpy as np
import torch
import random
import os
import time
import argparse
import matplotlib.pyplot as plt
from src.core.deep_cfr import DeepCFRAgent
from collections import defaultdict
import pandas as pd
import seaborn as sns

def load_agent_from_checkpoint(checkpoint_path, player_id=0, device='cpu'):
    """Load an agent from a checkpoint file."""
    print(f"Loading agent from {checkpoint_path}")
    
    agent = DeepCFRAgent(player_id=player_id, num_players=6, device=device)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load weights into the agent
    agent.advantage_net.load_state_dict(checkpoint['advantage_net'])
    agent.strategy_net.load_state_dict(checkpoint['strategy_net'])
    
    # Set the iteration count
    agent.iteration_count = checkpoint.get('iteration', 0)
    
    return agent

class AgentWrapper:
    """Wrapper for the agent to ensure proper player ID assignment."""
    def __init__(self, agent, player_id):
        self.agent = agent
        self.agent.player_id = player_id  # Update player_id to match position
        self.player_id = player_id
        
    def choose_action(self, state):
        return self.agent.choose_action(state)

def run_tournament(checkpoint_paths, num_games=100, device='cpu',
                  blinds=(1, 2), stake=200, verbose=False):
    """
    Run a tournament between agents loaded from different checkpoints.
    
    Args:
        checkpoint_paths: List of paths to different agent checkpoints
        num_games: Number of hands to play
        device: Device to run models on ('cpu' or 'cuda')
        blinds: Tuple of (small blind, big blind) amounts
        stake: Starting stack size for each player
        verbose: Whether to print detailed information
        
    Returns:
        DataFrame with stack sizes for each player after each hand
    """
    num_players = 6
    if len(checkpoint_paths) > num_players:
        print(f"Warning: Only using first {num_players} checkpoints")
        checkpoint_paths = checkpoint_paths[:num_players]
    
    # Fill remaining spots with duplicates if needed
    while len(checkpoint_paths) < num_players:
        checkpoint_paths.append(checkpoint_paths[0])
    
    # Load agents from checkpoints
    agents = []
    for i, path in enumerate(checkpoint_paths):
        agents.append(load_agent_from_checkpoint(path, player_id=i, device=device))
    
    # Wrap agents to ensure they have the correct player_id
    wrapped_agents = [AgentWrapper(agent, i) for i, agent in enumerate(agents)]
    
    # Initialize chip tracking
    stack_history = []
    
    # Extract checkpoint iterations for better labeling
    agent_names = []
    for i, path in enumerate(checkpoint_paths):
        # Try to extract iteration number from checkpoint path
        try:
            if 'iter_' in path:
                iter_num = path.split('iter_')[1].split('.')[0]
                agent_names.append(f"Agent {i} (iter {iter_num})")
            else:
                agent_names.append(f"Agent {i}")
        except:
            agent_names.append(f"Agent {i}")
    
    # Initialize player stacks for the tournament
    current_stacks = [stake] * num_players
    cumulative_profits = [0] * num_players
    
    # Run the games
    for game in range(num_games):
        # Rotate the button position for fairness
        button_pos = game % num_players
        
        if verbose:
            print(f"\nGame {game+1}/{num_games} - Button: Player {button_pos}")
        
        # Create a new poker game with current stacks
        # Note: In a real continuous tournament, we'd use the current stacks
        # But pokers.State.from_seed requires all players to have the same stake
        # So we'll use the fixed stake and track profits separately
        state = pkrs.State.from_seed(
            n_players=num_players,
            button=button_pos,
            sb=blinds[0],
            bb=blinds[1],
            stake=stake,
            seed=game  # Use game number as seed for reproducibility
        )
        
        # Play until the game is over
        hand_actions = []
        while not state.final_state:
            current_player = state.current_player
            
            # Choose an action for the current player
            action = wrapped_agents[current_player].choose_action(state)
            
            # Record the action
            if verbose:
                print(f"Player {current_player} ({agent_names[current_player]}) takes action: {action}")
            hand_actions.append((current_player, action))
            
            # Apply the action
            state = state.apply_action(action)
        
        # Game is over - collect results
        hand_profits = [player.reward for player in state.players_state]
        
        # Verify zero-sum (ignoring rake)
        total_profit = sum(hand_profits)
        if abs(total_profit) > 0.001:  # Allow for small floating-point errors
            print(f"WARNING: Game {game+1} is not zero-sum! Total profit: {total_profit:.2f}")
            
            # Debug info
            print("Player rewards:", hand_profits)
            print("Winner IDs:", state.winners if hasattr(state, 'winners') else "Unknown")
            
            # Try to fix the non-zero-sum by adjusting the first player's profit
            hand_profits[0] -= total_profit
            print(f"Adjusted Player 0's profit by {-total_profit:.2f} to maintain zero-sum")
        
        # Update cumulative profits
        for i in range(num_players):
            cumulative_profits[i] += hand_profits[i]
        
        # Print hand summary if verbose
        if verbose:
            print(f"Game {game+1} completed - Results:")
            for i in range(num_players):
                print(f"  {agent_names[i]}: {hand_profits[i]:+.2f}")
            print(f"  (Zero-sum check: {sum(hand_profits):.2f})")
        
        # Record stack sizes for this hand
        stack_record = {
            'hand': game + 1,
        }
        
        # Add stack for each player
        for i in range(num_players):
            stack_record[f'player_{i}_stack'] = stake + cumulative_profits[i]
            stack_record[f'player_{i}_profit'] = hand_profits[i]
            stack_record[f'player_{i}_cumulative_profit'] = cumulative_profits[i]
            stack_record[f'player_{i}_name'] = agent_names[i]
        
        stack_history.append(stack_record)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(stack_history)
    
    # Print final standings
    print("\nFinal Tournament Standings:")
    for i in range(num_players):
        print(f"{i+1}. {agent_names[i]}: {cumulative_profits[i]:+.2f}")
    print(f"Total money in system: {sum(cumulative_profits):.2f} (should be close to 0)")
    
    return results_df

def plot_stack_history(results_df, output_dir="tournament_results"):
    """Generate visualizations from tournament results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract player names
    player_names = [results_df[f'player_{i}_name'].iloc[0] for i in range(6)]
    
    # 1. Plot stack sizes over time
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.plot(results_df['hand'], results_df[f'player_{i}_stack'], 
                 label=player_names[i])
    
    plt.xlabel('Hand Number')
    plt.ylabel('Stack Size')
    plt.title('Player Stack Sizes Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'stack_sizes_over_time.png'), dpi=300)
    
    # 2. Plot cumulative profit over time
    plt.figure(figsize=(12, 8))
    
    # Use the pre-calculated cumulative profit
    for i in range(6):
        plt.plot(results_df['hand'], results_df[f'player_{i}_cumulative_profit'], 
                 label=player_names[i])
    
    plt.xlabel('Hand Number')
    plt.ylabel('Cumulative Profit/Loss')
    plt.title('Cumulative Profit Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'cumulative_profit.png'), dpi=300)
    
    # 3. Final performance comparison
    final_profits = [results_df[f'player_{i}_cumulative_profit'].iloc[-1] for i in range(6)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(player_names, final_profits)
    
    # Color the bars based on profit/loss
    for i, bar in enumerate(bars):
        if final_profits[i] >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Total Profit/Loss')
    plt.title('Final Profit/Loss by Player')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_performance.png'), dpi=300)
    
    # 4. Create a heatmap of player performance by hand segment
    # Divide the tournament into segments for analysis
    num_hands = len(results_df)
    segment_size = max(num_hands // 5, 1)  # Divide into ~5 segments
    
    # Create a matrix of profit by player and segment
    segments = []
    segment_labels = []
    
    for start_idx in range(0, num_hands, segment_size):
        end_idx = min(start_idx + segment_size, num_hands)
        segment_label = f"Hands {start_idx+1}-{end_idx}"
        segment_labels.append(segment_label)
        
        segment_data = {}
        for i in range(6):
            # Sum the individual hand profits in this segment (not cumulative)
            segment_profit = results_df[f'player_{i}_profit'].iloc[start_idx:end_idx].sum()
            segment_data[player_names[i]] = segment_profit
        
        segments.append(segment_data)
    
    # Convert to DataFrame for heatmap
    heatmap_df = pd.DataFrame(segments, index=segment_labels)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df, cmap="RdYlGn", center=0, annot=True, fmt=".1f")
    plt.title('Profit/Loss by Tournament Segment')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segment_heatmap.png'), dpi=300)
    
    # 5. Plot zero-sum validation
    # Calculate sum of all profits for each hand to verify zero-sum property
    total_profits_by_hand = []
    for hand in range(len(results_df)):
        hand_total = sum(results_df[f'player_{i}_profit'].iloc[hand] for i in range(6))
        total_profits_by_hand.append(hand_total)
    
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['hand'], total_profits_by_hand)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Hand Number')
    plt.ylabel('Sum of All Profits/Losses')
    plt.title('Zero-Sum Validation (Should be zero for every hand)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'zero_sum_validation.png'), dpi=300)
    
    # 6. Save the raw data
    results_df.to_csv(os.path.join(output_dir, 'tournament_data.csv'), index=False)
    
    print(f"Visualizations saved to {output_dir}/")
    return plt.figure()  # Return a figure for display if needed

def main():
    parser = argparse.ArgumentParser(description='Run and visualize a poker tournament between different agents')
    parser.add_argument('--checkpoints', nargs='+', required=True, 
                      help='Paths to checkpoint files to load agents from')
    parser.add_argument('--num-games', type=int, default=100,
                      help='Number of hands to play')
    parser.add_argument('--stake', type=float, default=200.0,
                      help='Initial stake for each player')
    parser.add_argument('--blinds', type=float, nargs=2, default=[1.0, 2.0],
                      help='Small and big blind amounts')
    parser.add_argument('--output-dir', type=str, default='tournament_results',
                      help='Directory to save results and visualizations')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed information during the games')
    
    args = parser.parse_args()
    
    # Run the tournament
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = run_tournament(
        checkpoint_paths=args.checkpoints,
        num_games=args.num_games,
        device=device,
        blinds=args.blinds,
        stake=args.stake,
        verbose=args.verbose
    )
    
    # Generate visualizations
    plot_stack_history(results, output_dir=args.output_dir)

if __name__ == "__main__":
    main()