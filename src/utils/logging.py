"""
Logging utilities for the DeepCFR Poker AI project.
Provides functionality for detailed error logging during gameplay and training.
"""

import os
import time
import traceback
import pokers as pkrs
from typing import Callable, Optional

ALL_IN_EPSILON = 1e-6


def _is_stalled_all_in_check_state(state: pkrs.State) -> bool:
    """Detect pokers states where all remaining players are all-in but only check cycles."""
    if state.final_state:
        return False

    if list(state.legal_actions) != [pkrs.ActionEnum.Check]:
        return False

    active_players = [player for player in state.players_state if player.active]
    if len(active_players) < 2:
        return False

    return all(float(player.stake) <= ALL_IN_EPSILON for player in active_players)


def resolve_stalled_all_in_showdown(state: pkrs.State) -> pkrs.State:
    """
    Advance a known pokers all-in/check cycle to showdown.

    In six-player side-pot hands, pokers can leave all remaining active players
    with no chips, only Check legal, and repeated checks do not advance streets.
    The engine can still settle the hand if we explicitly move through the board.
    """
    for _ in range(6):
        if not _is_stalled_all_in_check_state(state):
            return state

        if state.stage == pkrs.Stage.Showdown:
            next_state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Check))
            if next_state.status != pkrs.StateStatus.Ok:
                return state
            state = next_state
        else:
            state.to_next_stage()

    return state


def card_to_string(card):
    """Convert a poker card to a readable string."""
    suits = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
    ranks = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 
             7: "9", 8: "10", 9: "J", 10: "Q", 11: "K", 12: "A"}
    
    return f"{ranks[int(card.rank)]}{suits[int(card.suit)]}"

def log_game_error(state: pkrs.State, action: pkrs.Action, error_msg: str, 
                 card_converter: Callable = None) -> Optional[str]:
    """
    Log detailed error information to a file when a game state error occurs.
    
    Args:
        state: The game state before the action was applied
        action: The action that caused the error
        error_msg: The error message or status
        card_converter: Optional function to convert cards to strings 
                      (defaults to internal card_to_string)
    
    Returns:
        The path to the created log file, or None if logging failed
    """
    # Use provided card converter or default
    if card_converter is None:
        card_converter = card_to_string
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create a timestamp for the filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    microseconds = int((time.time() % 1) * 1_000_000)
    log_filename = os.path.join(logs_dir, f"poker_error_{timestamp}-{microseconds:06d}.txt")
    
    try:
        with open(log_filename, "w", encoding="utf-8") as f:
            stack_trace = traceback.format_exc()
            if stack_trace.strip() == "NoneType: None":
                stack_trace = "No active exception.\n"

            # Write error summary
            f.write(f"=== POKER GAME ERROR LOG ===\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {error_msg}\n\n")
            f.write(f"Stack trace:\n{stack_trace}\n")
            
            # Write game state information
            f.write(f"=== GAME STATE ===\n")
            f.write(f"Stage: {state.stage}\n")
            f.write(f"Pot: ${state.pot:.2f}\n")
            f.write(f"Button position: Player {state.button}\n")
            f.write(f"Current player: Player {state.current_player}\n")
            f.write(f"Min bet: ${state.min_bet:.2f}\n\n")
            
            # Write community cards
            community_cards = " ".join([card_converter(card) for card in state.public_cards])
            f.write(f"Community cards: {community_cards if community_cards else 'None'}\n\n")
            
            # Write all players' states
            f.write(f"=== PLAYER STATES ===\n")
            for i, p in enumerate(state.players_state):
                hand = "Unknown"
                if hasattr(p, 'hand') and p.hand:
                    hand = " ".join([card_converter(card) for card in p.hand])
                    
                f.write(f"Player {i}: ${p.stake:.2f} - Bet: ${p.bet_chips:.2f} - {'Active' if p.active else 'Folded'}\n")
                f.write(f"  Hand: {hand}\n")
            f.write("\n")
            
            # Write the action that caused the error
            f.write(f"=== ERROR-CAUSING ACTION ===\n")
            if action.action == pkrs.ActionEnum.Raise:
                f.write(f"Action: {action.action} ${action.amount:.2f}\n")
            else:
                f.write(f"Action: {action.action}\n")
            f.write("\n")
            
            # Write legal actions
            f.write(f"=== LEGAL ACTIONS ===\n")
            f.write(f"{state.legal_actions}\n\n")
            
            # Write any previous action if available
            if hasattr(state, 'from_action') and state.from_action:
                f.write(f"=== PREVIOUS ACTION ===\n")
                prev_action = state.from_action.action
                if prev_action.action == pkrs.ActionEnum.Raise:
                    f.write(f"Action: {prev_action.action} ${prev_action.amount:.2f}\n")
                else:
                    f.write(f"Action: {prev_action.action}\n")
        
        print(f"Error details logged to {log_filename}")
        return log_filename
    except Exception as log_error:
        print(f"Failed to write error log: {log_error}")
        return None


def apply_action_with_logging(
    state: pkrs.State,
    action: pkrs.Action,
    *,
    strict: bool = False,
    error_prefix: str = "State status not OK",
    card_converter: Callable = None,
):
    """
    Apply an action and log invalid state transitions consistently.

    Returns:
        Tuple of (new_state_or_none, log_file_or_none, status)
    """
    new_state = state.apply_action(action)
    if new_state.status == pkrs.StateStatus.Ok:
        new_state = resolve_stalled_all_in_showdown(new_state)
        return new_state, None, new_state.status

    log_file = log_game_error(
        state,
        action,
        f"{error_prefix} ({new_state.status})",
        card_converter=card_converter,
    )

    if strict:
        raise ValueError(f"{error_prefix} ({new_state.status}). Details logged to {log_file}")

    return None, log_file, new_state.status
