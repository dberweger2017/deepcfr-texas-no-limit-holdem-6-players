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
from scripts.telegram_notifier import TelegramNotifier

class RandomAgent:
    """Simple random agent for poker (unchanged from original)."""
    def __init__(self, player_id):
        self.player_id = player_id
        self.name = f"Player {player_id}"
        
    def choose_action(self, state):
        """Choose a random legal action with correctly calculated bet sizing."""
        if not state.legal_actions:
            raise ValueError(f"No legal actions available for player {self.player_id}")
        
        # Select a random legal action
        action_enum = random.choice(state.legal_actions)
        
        # For fold, check, and call, no amount is needed
        if action_enum == pkrs.ActionEnum.Fold:
            return pkrs.Action(action_enum)
        elif action_enum == pkrs.ActionEnum.Check:
            return pkrs.Action(action_enum)
        elif action_enum == pkrs.ActionEnum.Call:
            return pkrs.Action(action_enum)
        # For raises, carefully calculate a valid amount
        elif action_enum == pkrs.ActionEnum.Raise:
            player_state = state.players_state[state.current_player]
            current_bet = player_state.bet_chips
            available_stake = player_state.stake
            
            # Calculate call amount (needed to match current min_bet)
            call_amount = max(0, state.min_bet - current_bet)
            
            # If player can't even call, go all-in
            if available_stake <= call_amount:
                return pkrs.Action(action_enum, available_stake)
            
            # Calculate remaining stake after calling
            remaining_stake = available_stake - call_amount
            
            # If player can't raise at all, just call
            if remaining_stake <= 0:
                return pkrs.Action(pkrs.ActionEnum.Call)
            
            # Define minimum raise (typically 1 chip or the big blind)
            min_raise = 1.0
            if hasattr(state, 'bb'):
                min_raise = state.bb
            
            # Calculate potential additional raise amounts
            half_pot_raise = max(state.pot * 0.5, min_raise)
            full_pot_raise = max(state.pot, min_raise)
            
            # Create a list of valid additional raise amounts
            valid_amounts = []
            
            # Add half pot if affordable
            if half_pot_raise <= remaining_stake:
                valid_amounts.append(half_pot_raise)
            
            # Add full pot if affordable
            if full_pot_raise <= remaining_stake:
                valid_amounts.append(full_pot_raise)
            
            # Add minimum raise if none of the above is affordable
            if not valid_amounts and min_raise <= remaining_stake:
                valid_amounts.append(min_raise)
            
            # Small chance to go all-in
            if random.random() < 0.05 and remaining_stake > 0:  # 5% chance
                valid_amounts.append(remaining_stake)
            
            # If we can't afford any valid raise, fall back to call
            if not valid_amounts:
                return pkrs.Action(pkrs.ActionEnum.Call)
            
            # Choose a random additional raise amount
            additional_raise = random.choice(valid_amounts)
            
            # Ensure it doesn't exceed available stake
            additional_raise = min(additional_raise, remaining_stake)
            
            return pkrs.Action(action_enum, additional_raise)

def evaluate_against_random(agent, num_games=500, num_players=6, iteration=0, notifier=None):
    """Evaluate the trained agent against random opponents, tracking opponent history."""
    random_agents = [RandomAgent(i) for i in range(num_players)]
    total_profit = 0
    completed_games = 0
    game_crashes = 0
    zero_reward_games = 0
    
    for game in range(num_games):
        try:
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
            try:
                while not state.final_state:
                    current_player = state.current_player
                    
                    if current_player == agent.player_id:
                        # Use opponent modeling for the current opponent
                        action = agent.choose_action(state, opponent_id=current_player)
                    else:
                        action = random_agents[current_player].choose_action(state)
                        
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
                completed_games += 1
                
                # Check for zero rewards (suspicious)
                if abs(profit) < 0.001:
                    zero_reward_games += 1
            except Exception as e:
                if notifier and game % 20 == 0:  # Limit notification frequency
                    notifier.send_message(f"⚠️ <b>GAME ERROR</b>\nIteration: {iteration}, Hand: {game}\nError: {str(e)}")
                game_crashes += 1
                
        except Exception as e:
            game_crashes += 1
            if notifier and game % 20 == 0:  # Limit notification frequency
                notifier.send_message(f"⚠️ <b>GAME SETUP ERROR</b>\nIteration: {iteration}, Game: {game}\nError: {str(e)}")
    
    # Report if too many crashes or zero reward games
    if game_crashes > 0 and notifier:
        notifier.send_message(f"⚠️ <b>EVALUATION ISSUES</b>\nIteration: {iteration}\nCrashed games: {game_crashes}/{num_games}")
    
    if zero_reward_games > 0.2 * completed_games and notifier:
        notifier.send_message(f"⚠️ <b>SUSPICIOUS REWARDS</b>\nIteration: {iteration}\nZero reward games: {zero_reward_games}/{completed_games}")
    
    if completed_games == 0:
        if notifier:
            notifier.send_message(f"🚨 <b>CRITICAL ERROR</b>\nIteration: {iteration}\nNo games completed!")
        return 0
        
    return total_profit / completed_games

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
            try:
                agent.cfr_traverse(state, iteration, random_agents)
            except Exception as e:
                print(f"Error in traversal: {e}")
                if notifier and t % 50 == 0:  # Don't send too many error messages
                    notifier.send_message(f"⚠️ <b>TRAVERSAL ERROR</b>\nIteration: {iteration}\nError: {str(e)}")
        
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
            try:
                opp_loss = agent.train_opponent_modeling()
                opponent_model_losses.append(opp_loss)
                print(f"  Opponent modeling loss: {opp_loss:.6f}")
                writer.add_scalar('Loss/OpponentModeling', opp_loss, iteration)
            except Exception as e:
                print(f"Error training opponent model: {e}")
                if notifier:
                    notifier.send_message(f"⚠️ <b>OPPONENT MODEL ERROR</b>\nIteration: {iteration}\nError: {str(e)}")
        
        # Evaluate periodically
        if iteration % 20 == 0 or iteration == num_iterations:
            print("  Evaluating agent...")
            avg_profit = evaluate_against_random(agent, num_games=200, iteration=iteration, notifier=notifier)
            profits.append(avg_profit)
            print(f"  Average profit per game: {avg_profit:.2f}")
            writer.add_scalar('Performance/Profit', avg_profit, iteration)
            
            # Send progress update
            if notifier:
                num_opponents = len(agent.opponent_modeling.opponent_histories)
                notifier.send_training_progress(
                    iteration=iteration,
                    profit_vs_models=avg_profit,
                    profit_vs_random=avg_profit
                )
        
        # Save checkpoint
        if iteration % checkpoint_frequency == 0 or iteration == num_iterations:
            checkpoint_path = f"{save_dir}/om_checkpoint_iter_{iteration}.pt"
            agent.save_model(checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
            
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
        print(f"  Iteration completed in {elapsed:.2f} seconds")
        print(f"  Tracking data for {num_opponents} unique opponents")
        print()
    
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