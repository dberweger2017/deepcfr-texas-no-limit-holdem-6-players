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
from src.opponent_modeling.deep_cfr_with_opponent_modeling import DeepCFRAgentWithOpponentModeling
from src.core.model import encode_state, set_verbose

class RandomAgent:
    """Simple random agent for poker."""
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

class ModelAgent:
    """Wrapper for a DeepCFRAgent or DeepCFRAgentWithOpponentModeling loaded from a checkpoint.
    Only sanitizes the bet amounts without changing decision logic."""
    def __init__(self, player_id, model_path, device='cpu', with_opponent_modeling=False):
        self.player_id = player_id
        self.name = f"Model Agent {player_id} ({os.path.basename(model_path)})"
        self.model_path = model_path
        self.device = device
        self.with_opponent_modeling = with_opponent_modeling
        
        # Load the appropriate agent type
        if with_opponent_modeling:
            from src.opponent_modeling.deep_cfr_with_opponent_modeling import DeepCFRAgentWithOpponentModeling
            self.agent = DeepCFRAgentWithOpponentModeling(player_id=player_id, device=device)
        else:
            from src.core.deep_cfr import DeepCFRAgent
            self.agent = DeepCFRAgent(player_id=player_id, device=device)
            
        # Load model weights
        self.agent.load_model(model_path)
    
    def choose_action(self, state):
        """Choose an action while sanitizing bet amounts to legal values."""
        # Get the original action from the agent
        original_action = self.agent.choose_action(state)
        
        # Only process Raise actions
        if original_action.action != pkrs.ActionEnum.Raise:
            return original_action
            
        # Calculate legal bet bounds
        player_state = state.players_state[state.current_player]
        current_bet = player_state.bet_chips
        available_stake = player_state.stake
        
        # Calculate call amount (needed to match current min_bet)
        call_amount = state.min_bet - current_bet
        
        # If the player can't even call, return Call action
        if available_stake < call_amount:
            return pkrs.Action(pkrs.ActionEnum.Call)
        
        # Remaining stake after calling
        remaining_stake = available_stake - call_amount
        
        # Get the original amount (which might be calculated incorrectly by the model)
        additional_amount = original_action.amount
        
        # Ensure the additional amount doesn't exceed remaining stake
        additional_amount = min(additional_amount, remaining_stake)
        
        # Ensure the additional amount is non-negative
        additional_amount = max(0, additional_amount)
        
        # Create new action with sanitized amount
        return pkrs.Action(pkrs.ActionEnum.Raise, additional_amount)

def evaluate_against_opponents(agent, opponents, num_games=100, iteration=0, notifier=None):
    """Evaluate the trained agent against a set of opponents with enhanced error tracking."""
    total_profit = 0
    num_players = 6
    
    # Statistics tracking
    completed_games = 0
    zero_reward_games = 0
    illegal_actions_by_agent = 0
    illegal_actions_by_opponents = 0
    state_errors = 0
    game_crashes = 0
    total_actions = 0
    
    for game in range(num_games):
        # Create a new poker game
        state = pkrs.State.from_seed(
            n_players=num_players,
            button=game % num_players,
            sb=1,
            bb=2,
            stake=200.0,
            seed=game + 10000  # Using different seeds than training
        )
        
        # Track actions for this game
        action_count = 0
        game_actions = []  # Store recent actions for debugging
        
        # Play until the game is over
        try:
            while not state.final_state:
                current_player = state.current_player
                
                try:
                    # Store state before action for debugging
                    state_before = state
                    
                    # Choose action
                    if current_player == agent.player_id:
                        action = agent.choose_action(state, opponent_id=current_player)
                        is_training_agent = True
                    else:
                        action = opponents[current_player].choose_action(state)
                        is_training_agent = False
                    
                    # Track this action
                    action_desc = f"P{current_player} {action.action}"
                    if action.action == pkrs.ActionEnum.Raise:
                        action_desc += f" {action.amount:.1f}"
                    game_actions.append(action_desc)
                    
                    # Check for illegal actions (especially raise amounts)
                    if action.action == pkrs.ActionEnum.Raise:
                        player_state = state.players_state[current_player]
                        current_bet = player_state.bet_chips
                        available_stake = player_state.stake
                        
                        # Check if this would be a legal raise
                        lower_bound = current_bet + state.min_bet
                        upper_bound = current_bet + available_stake
                        
                        if action.amount > available_stake:
                            if is_training_agent:
                                illegal_actions_by_agent += 1
                            else:
                                illegal_actions_by_opponents += 1
                                
                            if notifier:
                                notifier.send_message(
                                    f"⚠️ <b>ILLEGAL BET DETECTED</b>\n"
                                    f"{'OUR AGENT' if is_training_agent else 'OPPONENT'} tried to bet "
                                    f"{action.amount} with only {available_stake} available\n"
                                    f"Player: {current_player}, Iteration: {iteration}"
                                )
                                
                            # This should be fixed by your implementation, but let's double-check
                            if action.amount > available_stake:
                                action = pkrs.Action(pkrs.ActionEnum.Raise, available_stake)
                    
                    # Record opponent action for modeling
                    if current_player != agent.player_id and hasattr(agent, 'record_opponent_action'):
                        # Convert action to action_id
                        if action.action == pkrs.ActionEnum.Fold:
                            action_id = 0
                        elif action.action == pkrs.ActionEnum.Check or action.action == pkrs.ActionEnum.Call:
                            action_id = 1
                        elif action.action == pkrs.ActionEnum.Raise:
                            # 0.5x pot or 1x pot
                            action_id = 2 if action.amount <= state.pot * 0.75 else 3
                        else:
                            action_id = 1
                            
                        agent.record_opponent_action(state, action_id, current_player)
                    
                    # Apply the action
                    new_state = state.apply_action(action)
                    action_count += 1
                    total_actions += 1
                    
                    # Check if the action was valid
                    if new_state.status != pkrs.StateStatus.Ok:
                        state_errors += 1
                        
                        # Identify which agent caused the error
                        error_source = "TRAINING AGENT" if is_training_agent else f"OPPONENT (Player {current_player})"
                        
                        # Error details for debug
                        error_details = f"Error Source: {error_source}\n"
                        if new_state.status == pkrs.StateStatus.HighBet:
                            player_state = state_before.players_state[current_player]
                            error_details += f"Player Stake: {player_state.stake}\n"
                            error_details += f"Current Bet: {player_state.bet_chips}\n"
                            error_details += f"Min Bet: {state_before.min_bet}\n"
                            if hasattr(action, 'amount'):
                                error_details += f"Attempted Raise: {action.amount}\n"
                        
                        print(f"STATE ERROR: {new_state.status} by {error_source}")
                        print(error_details)
                        
                        if notifier:
                            notifier.alert_state_error(
                                iteration,
                                new_state.status,
                                state_before,
                                is_training_agent=is_training_agent
                            )
                        break  # Stop this game
                    
                    # Update state
                    state = new_state
                    
                except Exception as e:
                    print(f"ERROR in game {game}, player {current_player}: {e}")
                    if notifier and game % 10 == 0:  # Don't send too many error messages
                        notifier.send_message(
                            f"⚠️ <b>ACTION ERROR</b>\n"
                            f"Game {game}, Player {current_player}\n"
                            f"Error: {str(e)}"
                        )
                    break  # Stop this game
            
            # Check if game completed successfully
            if state.final_state:
                completed_games += 1
                
                # Record end of game for opponent modeling
                if hasattr(agent, 'end_game_recording'):
                    agent.end_game_recording(state)
                
                # Get reward and add to total
                profit = state.players_state[agent.player_id].reward
                total_profit += profit
                
                # Check for zero rewards (suspicious)
                if abs(profit) < 0.001:
                    zero_reward_games += 1
                    if zero_reward_games >= 5 and notifier and game % 10 == 0:
                        # Get the last few actions for context
                        recent_actions = game_actions[-min(10, len(game_actions)):]
                        notifier.send_message(
                            f"⚠️ <b>ZERO REWARD GAME</b>\n"
                            f"Iteration {iteration}, Game {game}\n"
                            f"Action count: {action_count}\n"
                            f"Recent actions: {' → '.join(recent_actions)}"
                        )
            else:
                game_crashes += 1
                
        except Exception as e:
            game_crashes += 1
            print(f"GAME CRASH: Game {game} crashed with error: {e}")
            if notifier and game % 20 == 0:  # Limit notification frequency
                notifier.send_message(
                    f"🚨 <b>GAME CRASH</b>\n"
                    f"Iteration {iteration}, Game {game}\n"
                    f"Error: {str(e)}"
                )
    
    # If too many zero reward games, send summary alert
    if zero_reward_games > num_games * 0.2 and notifier:  # More than 20% zero rewards
        notifier.alert_zero_reward_games(
            iteration,
            zero_reward_games,
            completed_games
        )
    
    # Print detailed statistics
    print(f"Games completed: {completed_games}/{num_games} ({completed_games/num_games*100:.1f}%)")
    print(f"Zero reward games: {zero_reward_games}/{completed_games} ({zero_reward_games/max(1,completed_games)*100:.1f}%)")
    print(f"Illegal actions by AGENT: {illegal_actions_by_agent}")
    print(f"Illegal actions by OPPONENTS: {illegal_actions_by_opponents}")
    print(f"State errors: {state_errors}, Game crashes: {game_crashes}")
    print(f"Total actions: {total_actions}, Avg actions per game: {total_actions/max(1,num_games):.1f}")
    
    # Calculate average profit (only from completed games)
    if completed_games > 0:
        avg_profit = total_profit / completed_games
    else:
        avg_profit = 0
        if notifier:
            notifier.send_message(f"⚠️ <b>CRITICAL ERROR</b>: No games completed in iteration {iteration}")
    
    return avg_profit

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
    verbose=False,
    checkpoint_path=None  # New parameter to continue from checkpoint
):
    """
    Train a Deep CFR agent with opponent modeling against a mix of opponents
    loaded from checkpoints, refreshing the opponent pool periodically.
    """
    # Import required modules
    from torch.utils.tensorboard import SummaryWriter
    import glob
    import os
    import time
    from scripts.telegram_notifier import TelegramNotifier
    import traceback
    
    # Set verbosity
    set_verbose(verbose)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize Telegram notifier (reads from .env file)
    notifier = TelegramNotifier()
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    notifier.send_message(f"🚀 <b>TRAINING STARTED</b>\nDevice: {device}\nIterations: {num_iterations}\nRefresh interval: {refresh_interval}")
    
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
            
            # Extract iteration number from filename if possible
            try:
                iteration_str = checkpoint_path.split('iter_')[1].split('.')[0]
                starting_iteration = int(iteration_str) + 1
                agent.iteration_count = starting_iteration - 1
                print(f"Continuing from iteration {starting_iteration}")
                notifier.send_message(f"📥 <b>LOADED CHECKPOINT</b>\nContinuing from iteration {starting_iteration}")
            except Exception as e:
                print(f"Could not determine iteration from checkpoint filename: {e}")
                print("Starting from iteration 1")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            notifier.send_message(f"⚠️ <b>CHECKPOINT LOADING ERROR</b>\n{str(e)}\nStarting from scratch.")
    
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
        opponent_names = []
        for i, f in enumerate(selected_files):
            filename = os.path.basename(f)
            opponent_names.append(filename)
            print(f"  {i+1}. {filename}")
        
        # Log selected opponents to Telegram
        notifier.send_message(f"📊 <b>SELECTED OPPONENTS</b>\n- " + "\n- ".join(opponent_names))
        
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
    for iteration in range(starting_iteration, num_iterations + 1):
        agent.iteration_count = iteration
        start_time = time.time()
        
        # Refresh opponents at specified intervals
        if iteration % refresh_interval == 1 and iteration > starting_iteration:
            print(f"\n=== Refreshing opponent pool at iteration {iteration} ===")
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
                agent.cfr_traverse(state, iteration, opponents)
            except Exception as e:
                error_msg = f"Error during traversal: {e}"
                print(error_msg)
                if iteration % 100 == 0:  # Don't flood with error messages
                    notifier.send_message(f"⚠️ <b>TRAVERSAL ERROR</b>\n{error_msg}")
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
            try:
                opp_loss = agent.train_opponent_modeling()
                opponent_model_losses.append(opp_loss)
                print(f"  Opponent modeling loss: {opp_loss:.6f}")
                writer.add_scalar('Loss/OpponentModeling', opp_loss, iteration)
            except Exception as e:
                error_msg = f"Error training opponent modeling: {e}"
                print(error_msg)
                notifier.send_message(f"⚠️ <b>OPPONENT MODELING ERROR</b>\n{error_msg}")
        
        # Evaluate periodically
        if iteration % 20 == 0 or iteration == num_iterations:
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
                notifier.send_message(
                    f"⚠️ <b>ZERO PROFIT ALERT</b> at iteration {iteration}\n"
                    f"This suggests games may not be completing properly."
                )
            
            # Send progress update every 100 iterations or if a significant change occurs
            if iteration % 100 == 0:
                notifier.send_training_progress(iteration, avg_profit_models, avg_profit_random)
        
        # Save checkpoint
        if iteration % checkpoint_frequency == 0 or iteration == num_iterations:
            checkpoint_path = f"{save_dir}/mixed_om_checkpoint_iter_{iteration}.pt"
            agent.save_model(checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
            
            # Notify on checkpoint save
            if iteration % 500 == 0:  # Less frequent notifications for checkpoints
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
        iteration=num_iterations, 
        notifier=notifier
    )
    print(f"Final performance vs random: Average profit per game: {avg_profit_random:.2f}")
    writer.add_scalar('Performance/FinalProfitVsRandom', avg_profit_random, 0)
    
    # Against model opponents
    avg_profit_models = evaluate_against_opponents(
        agent, 
        opponents, 
        num_games=500, 
        iteration=num_iterations, 
        notifier=notifier
    )
    print(f"Final performance vs models: Average profit per game: {avg_profit_models:.2f}")
    writer.add_scalar('Performance/FinalProfitVsModels', avg_profit_models, 0)
    
    # Final notification
    notifier.send_message(
        f"✅ <b>TRAINING COMPLETED</b>\n"
        f"Total iterations: {num_iterations}\n"
        f"Final profit vs random: {avg_profit_random:.2f}\n"
        f"Final profit vs models: {avg_profit_models:.2f}"
    )
    
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
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to continue training from')
    args = parser.parse_args()
    
    print(f"Starting mixed opponent training with models from: {args.checkpoint_dir}")
    print(f"Training for {args.iterations} iterations")
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
        checkpoint_path=args.checkpoint
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