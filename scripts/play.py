#play.py
import pokers as pkrs
import torch
import numpy as np
import argparse
import os
import random
import glob
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import set_verbose
from src.utils import log_game_error
from src.utils.settings import STRICT_CHECKING, set_strict_checking

def get_action_description(action):
    """Convert a pokers action to a human-readable string."""
    if action.action == pkrs.ActionEnum.Fold:
        return "Fold"
    elif action.action == pkrs.ActionEnum.Check:
        return "Check"
    elif action.action == pkrs.ActionEnum.Call:
        return "Call"
    elif action.action == pkrs.ActionEnum.Raise:
        return f"Raise to {action.amount:.2f}"
    else:
        return f"Unknown action: {action.action}"

def card_to_string(card):
    """Convert a poker card to a readable string."""
    suits = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
    ranks = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 
             7: "9", 8: "10", 9: "J", 10: "Q", 11: "K", 12: "A"}
    
    return f"{ranks[int(card.rank)]}{suits[int(card.suit)]}"

def display_game_state(state, player_id=0):
    """Display the current game state in a human-readable format."""
    print("\n" + "="*70)
    
    # Fix for Stage enum - convert to string properly
    stage_names = {
        0: "PreFlop",
        1: "Flop", 
        2: "Turn", 
        3: "River", 
        4: "Showdown"
    }
    stage_name = stage_names.get(int(state.stage), str(state.stage))
    print(f"Stage: {stage_name}")
    
    print(f"Pot: ${state.pot:.2f}")
    print(f"Button position: Player {state.button}")
    
    # Show community cards
    community_cards = " ".join([card_to_string(card) for card in state.public_cards])
    print(f"Community cards: {community_cards if community_cards else 'None'}")
    
    # Show player's hand
    hand = " ".join([card_to_string(card) for card in state.players_state[player_id].hand])
    print(f"Your hand: {hand}")
    
    # Show all players' states
    print("\nPlayers:")
    for i, p in enumerate(state.players_state):
        status = "YOU" if i == player_id else "AI"
        active = "Active" if p.active else "Folded"
        print(f"Player {i} ({status}): ${p.stake:.2f} - Bet: ${p.bet_chips:.2f} - {active}")
    
    # Show legal actions for human player if it's their turn
    if state.current_player == player_id:
        print("\nLegal actions:")
        for action_enum in state.legal_actions:
            if action_enum == pkrs.ActionEnum.Fold:
                print("  f: Fold")
            elif action_enum == pkrs.ActionEnum.Check:
                print("  c: Check")
            elif action_enum == pkrs.ActionEnum.Call:
                # Calculate call amount
                call_amount = max(0, state.min_bet - state.players_state[player_id].bet_chips)
                print(f"  c: Call ${call_amount:.2f}")
            elif action_enum == pkrs.ActionEnum.Raise:
                min_raise = state.min_bet
                max_raise = state.players_state[player_id].stake
                print(f"  r: Raise (min: ${min_raise:.2f}, max: ${max_raise:.2f})")
                print("    h: Raise half pot")
                print("    p: Raise pot")
                print("    m: Custom raise amount")
    
    print("="*70)

def get_human_action(state, player_id=0):
    """Get action from human player via console input."""
    while True:
        action_input = input("Your action (f=fold, c=check/call, r=raise, h=half pot, p=pot, m=custom): ").strip().lower()
        
        # Process fold
        if action_input == 'f' and pkrs.ActionEnum.Fold in state.legal_actions:
            return pkrs.Action(pkrs.ActionEnum.Fold)
        
        # Process check/call
        elif action_input == 'c':
            if pkrs.ActionEnum.Check in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Check)
            elif pkrs.ActionEnum.Call in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Call)
        
        # Process raise shortcuts
        elif action_input in ['r', 'h', 'p', 'm'] and pkrs.ActionEnum.Raise in state.legal_actions:
            player_state = state.players_state[player_id]
            current_bet = player_state.bet_chips
            available_stake = player_state.stake
            
            # Calculate call amount
            call_amount = state.min_bet - current_bet
            
            # If player can't even call, go all-in
            if available_stake <= call_amount:
                return pkrs.Action(pkrs.ActionEnum.Raise, available_stake)
            
            # Calculate remaining stake after calling
            remaining_stake = available_stake - call_amount
            
            # If player can't raise after calling, just call
            if remaining_stake <= 0:
                print("You don't have enough chips to raise. Calling instead.")
                return pkrs.Action(pkrs.ActionEnum.Call)
            
            # Define minimum raise (typically 1 chip or the big blind)
            min_raise = 1.0
            if hasattr(state, 'bb'):
                min_raise = state.bb
                
            if action_input == 'h':  # Half pot
                bet_amount = max(state.pot * 0.5, min_raise)  # Ensure minimum raise
                bet_amount = min(bet_amount, remaining_stake)  # Ensure we don't exceed stake
                
                # If we can't meet minimum raise, fall back to call
                if bet_amount < min_raise:
                    print(f"Cannot raise by the minimum required amount. Calling instead.")
                    return pkrs.Action(pkrs.ActionEnum.Call)
                    
                return pkrs.Action(pkrs.ActionEnum.Raise, bet_amount)
            
            elif action_input == 'p':  # Full pot
                bet_amount = max(state.pot, min_raise)  # Ensure minimum raise
                bet_amount = min(bet_amount, remaining_stake)  # Ensure we don't exceed stake
                
                # If we can't meet minimum raise, fall back to call
                if bet_amount < min_raise:
                    print(f"Cannot raise by the minimum required amount. Calling instead.")
                    return pkrs.Action(pkrs.ActionEnum.Call)
                    
                return pkrs.Action(pkrs.ActionEnum.Raise, bet_amount)
            
            elif action_input == 'm' or action_input == 'r':  # Custom amount
                while True:
                    try:
                        amount_str = input(f"Enter raise amount (min: {min_raise:.2f}, max: {remaining_stake:.2f}): ")
                        amount = float(amount_str)
                        
                        # Check if amount meets minimum raise
                        if amount >= min_raise and amount <= remaining_stake:
                            return pkrs.Action(pkrs.ActionEnum.Raise, amount)
                        else:
                            print(f"Amount must be between {min_raise:.2f} and {remaining_stake:.2f}")
                    except ValueError:
                        print("Please enter a valid number")
        
        print("Invalid action. Please try again.")

def select_random_models(models_dir, num_models=5, model_pattern="*.pt"):
    """
    Select random model checkpoint files from a directory.
    
    Args:
        models_dir: Directory containing model checkpoint files
        num_models: Number of models to select
        model_pattern: File pattern to match model files
        
    Returns:
        List of paths to selected model files
    """
    # Get all model checkpoint files in the directory
    model_files = glob.glob(os.path.join(models_dir, model_pattern))
    
    if not model_files:
        print(f"No model files found in {models_dir} matching pattern '{model_pattern}'")
        return []
    
    # Select random models
    selected_models = random.sample(model_files, min(num_models, len(model_files)))
    return selected_models

def play_against_models(models_dir=None, model_pattern="*.pt", num_models=5, 
                        player_position=0, initial_stake=200.0, small_blind=1.0, 
                        big_blind=2.0, verbose=False, shuffle_models=True):
    """
    Play against randomly selected AI models from a directory.
    
    Args:
        models_dir: Directory containing model checkpoint files
        model_pattern: File pattern to match model files
        num_models: Number of models to select
        player_position: Position of the human player (0-5)
        initial_stake: Starting chip count for all players
        small_blind: Small blind amount
        big_blind: Big blind amount
        verbose: Whether to show detailed output
        shuffle_models: Whether to select new random models for each game
    """
    set_verbose(verbose)
    
    # Check if models directory exists
    if models_dir and not os.path.isdir(models_dir):
        print(f"Warning: Models directory {models_dir} not found.")
        models_dir = None
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Track game statistics
    num_games = 0
    total_profit = 0
    player_stake = initial_stake
    
    # Main game loop
    while True:
        if player_stake <= 0:
            print("\nYou're out of chips! Game over.")
            break
        
        # Ask if player wants to continue after the first game
        if num_games > 0:
            choice = input("\nContinue playing? (y/n): ").strip().lower()
            if choice != 'y':
                print("Thanks for playing!")
                break
        
        # Select new random models for this game if shuffling is enabled or first game
        if (shuffle_models or num_games == 0) and models_dir:
            model_paths = select_random_models(models_dir, num_models, model_pattern)
            print(f"Selected {len(model_paths)} random models for this game:")
            for i, path in enumerate(model_paths):
                print(f"  Model {i+1}: {os.path.basename(path)}")
        elif not models_dir:
            model_paths = []
            print("No models directory specified, using random agents.")
        
        # Create agents for each position
        agents = []
        for i in range(6):
            if i == player_position:
                # Human player
                agents.append(None)
            else:
                # Determine which model to use
                model_idx = (i - 1) if i > player_position else i
                if models_dir and model_idx < len(model_paths):
                    # Load model
                    try:
                        agent = DeepCFRAgent(player_id=i, num_players=6, device=device)
                        agent.load_model(model_paths[model_idx])
                        agents.append(agent)
                        print(f"Loaded model for Player {i}: {os.path.basename(model_paths[model_idx])}")
                    except Exception as e:
                        print(f"Error loading model for Player {i}: {e}")
                        print("Using random agent instead")
                        agents.append(RandomAgent(i))
                else:
                    # Use random agent
                    agents.append(RandomAgent(i))
                    print(f"Using random agent for Player {i}")
        
        num_games += 1
        print(f"\n--- Game {num_games} ---")
        print(f"Your current balance: ${player_stake:.2f}")
        
        # Rotate button position for fairness
        button_pos = (num_games - 1) % 6
        
        # Create a new poker game
        state = pkrs.State.from_seed(
            n_players=6,
            button=button_pos,
            sb=small_blind,
            bb=big_blind,
            stake=initial_stake,
            seed=random.randint(0, 10000)
        )
        
        # Play until the game is over
        while not state.final_state:
            current_player = state.current_player
            
            # Display game state before human acts
            if current_player == player_position:
                display_game_state(state, player_position)
                action = get_human_action(state, player_position)
                print(f"You chose: {get_action_description(action)}")
            else:
                # Abbreviated state display for AI turns
                print(f"\nPlayer {current_player}'s turn")
                action = agents[current_player].choose_action(state)
                print(f"Player {current_player} chose: {get_action_description(action)}")
            
            # Apply the action
            new_state = state.apply_action(action)
            if new_state.status != pkrs.StateStatus.Ok:
                log_file = log_game_error(state, action, f"State status not OK ({new_state.status})")
                if STRICT_CHECKING:
                    raise ValueError(f"State status not OK ({new_state.status}). Details logged to {log_file}")
                else:
                    print(f"WARNING: State status not OK ({new_state.status}). Details logged to {log_file}")
                    break  # Skip this game in non-strict mode
            
            state = new_state
        
        # Game is over, show results
        print("\n--- Game Over ---")
        
        # Show all players' hands
        print("Final hands:")
        for i, p in enumerate(state.players_state):
            if p.active:
                # Check if the hand attribute exists and has cards
                if hasattr(p, 'hand') and p.hand:
                    hand = " ".join([card_to_string(card) for card in p.hand])
                    print(f"Player {i}: {hand}")
                else:
                    print(f"Player {i}: Hand data unavailable")
        
        # Show community cards
        community_cards = " ".join([card_to_string(card) for card in state.public_cards])
        print(f"Community cards: {community_cards}")
        
        # Show results
        print("\nResults:")
        for i, p in enumerate(state.players_state):
            player_type = "YOU" if i == player_position else "AI"
            print(f"Player {i} ({player_type}): ${p.reward:.2f}")
        
        # Update player's stake
        game_profit = state.players_state[player_position].reward
        total_profit += game_profit
        player_stake += game_profit
        
        print(f"\nThis game: {'Won' if game_profit > 0 else 'Lost'} ${abs(game_profit):.2f}")
        print(f"Running total: ${total_profit:.2f}")
        print(f"Current balance: ${player_stake:.2f}")
    
    # Show overall statistics
    print("\n--- Overall Statistics ---")
    print(f"Games played: {num_games}")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"Average profit per game: ${total_profit/num_games if num_games > 0 else 0:.2f}")
    print(f"Final balance: ${player_stake:.2f}")

class RandomAgent:
    """Simple random agent for poker that ensures valid bet sizing."""
    def __init__(self, player_id):
        self.player_id = player_id
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play poker against random AI models')
    parser.add_argument('--models-dir', type=str, default=None, help='Directory containing model checkpoint files')
    parser.add_argument('--model-pattern', type=str, default="*.pt", help='File pattern to match model files')
    parser.add_argument('--num-models', type=int, default=5, help='Number of models to select')
    parser.add_argument('--position', type=int, default=0, help='Your position at the table (0-5)')
    parser.add_argument('--stake', type=float, default=200.0, help='Initial stake')
    parser.add_argument('--sb', type=float, default=1.0, help='Small blind amount')
    parser.add_argument('--bb', type=float, default=2.0, help='Big blind amount')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    parser.add_argument('--no-shuffle', action='store_true', help='Do not select new random models for each game')
    parser.add_argument('--strict', action='store_true', help='Enable strict error checking that raises exceptions for invalid game states')
    args = parser.parse_args()
    
    set_strict_checking(args.strict)

    # Start the game
    play_against_models(
        models_dir=args.models_dir,
        model_pattern=args.model_pattern,
        num_models=args.num_models,
        player_position=args.position,
        initial_stake=args.stake,
        small_blind=args.sb,
        big_blind=args.bb,
        verbose=args.verbose,
        shuffle_models=not args.no_shuffle
    )