# src/agents/random_agent.py
import random
import pokers as pkrs
from src.utils.settings import STRICT_CHECKING
from src.utils.logging import log_game_error

class RandomAgent:
    """
    Simple random agent for poker that chooses a random legal action
    and ensures valid bet sizing, especially for Raises vs Calls.
    """
    def __init__(self, player_id):
        self.player_id = player_id
        self.name = f"RandomAgent_{player_id}" # Added name for clarity

    def choose_action(self, state):
        """Choose a random legal action with correctly calculated bet sizing."""
        if not state.legal_actions:
            # This should ideally not happen in a valid game state
            print(f"WARNING: No legal actions available for player {self.player_id}. Attempting Fold.")
            # Attempt Fold as fallback, though it might also be illegal
            return pkrs.Action(pkrs.ActionEnum.Fold)

        # Select a random legal action type from the available ones
        action_enum = random.choice(state.legal_actions)

        # Handle non-Raise actions first
        if action_enum == pkrs.ActionEnum.Fold:
            return pkrs.Action(action_enum)
        elif action_enum == pkrs.ActionEnum.Check:
            return pkrs.Action(action_enum)
        elif action_enum == pkrs.ActionEnum.Call:
            return pkrs.Action(action_enum)

        # Handle Raise action
        elif action_enum == pkrs.ActionEnum.Raise:
            player_state = state.players_state[state.current_player]
            current_bet = player_state.bet_chips
            available_stake = player_state.stake

            # Calculate call amount (needed to match current min_bet)
            call_amount = max(0, state.min_bet - current_bet)

            # Check if the player can actually make a valid raise beyond the call amount.
            # A raise requires putting in *more* than the call amount.
            # The minimum *additional* amount for a raise is typically the big blind or 1 chip.
            min_raise_increment = 1.0 # A small default minimum increment
            # Attempt to get BB for a more standard minimum raise size
            if state.min_bet > 0 and state.pot > 0: # Heuristic: BB is likely related to initial bets/pot
                 # Find the likely BB size (often state.min_bet in preflop after BB post)
                 # This is imperfect, might need a direct BB value passed to state if available
                 likely_bb = state.min_bet if state.stage == pkrs.Stage.Preflop and state.pot <= 3 * state.min_bet else state.min_bet / 2
                 min_raise_increment = max(1.0, likely_bb)
            elif hasattr(state, 'bb'): # If bb is explicitly available
                 min_raise_increment = max(1.0, state.bb)


            if available_stake <= call_amount + min_raise_increment:
                # Player cannot make a valid raise (not enough chips beyond the call amount)
                # OR the player is going all-in just to meet the call amount.
                # This action should be treated as a Call, not a Raise.
                # print(f"RandomAgent: Raise chosen, but cannot make valid raise. Stake={available_stake}, CallAmt={call_amount}, MinIncr={min_raise_increment}. Switching to Call.") # Optional debug
                # Ensure Call is legal before returning it
                if pkrs.ActionEnum.Call in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                else:
                    # Fallback: If Call isn't legal (e.g., already all-in matching bet), Fold.
                    # print(f"RandomAgent WARNING: Cannot Call (not legal), falling back to Fold.")
                    # Ensure Fold is legal if possible
                    if pkrs.ActionEnum.Fold in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Fold)
                    else:
                        # Last resort if Fold is also illegal (highly unlikely)
                        # print(f"RandomAgent CRITICAL WARNING: Cannot Call or Fold!")
                        # Return Call anyway, let Rust handle the error state
                        return pkrs.Action(pkrs.ActionEnum.Call)

            # If we reach here, a valid raise *is* possible.
            remaining_stake_after_call = available_stake - call_amount

            # Define potential raise amounts (as *additional* chips beyond the call)
            # Ensure amounts are at least the minimum increment and no more than remaining stake
            valid_additional_amounts = []

            # Minimum possible raise increment
            valid_additional_amounts.append(min_raise_increment)

            # Half pot raise (additional amount)
            half_pot_additional = max(state.pot * 0.5, min_raise_increment)
            if half_pot_additional <= remaining_stake_after_call:
                 valid_additional_amounts.append(half_pot_additional)

            # Full pot raise (additional amount)
            full_pot_additional = max(state.pot, min_raise_increment)
            if full_pot_additional <= remaining_stake_after_call:
                 valid_additional_amounts.append(full_pot_additional)

            # All-in raise (additional amount is the remaining stake after calling)
            # Only add if it's strictly greater than other options already present
            # and meets the minimum increment requirement.
            if remaining_stake_after_call >= min_raise_increment and remaining_stake_after_call not in valid_additional_amounts:
                 valid_additional_amounts.append(remaining_stake_after_call)

            # Filter amounts again just to be safe (should be redundant)
            possible_additional_amounts = [
                amount for amount in valid_additional_amounts
                if min_raise_increment <= amount <= remaining_stake_after_call
            ]

            # If somehow no valid raise amount is possible (should be covered by initial check), Call.
            if not possible_additional_amounts:
                 print(f"RandomAgent WARNING: No valid additional raise amounts found after filtering. Falling back to Call.")
                 if pkrs.ActionEnum.Call in state.legal_actions:
                     return pkrs.Action(pkrs.ActionEnum.Call)
                 else: # Fallback to Fold if Call isn't legal
                     if pkrs.ActionEnum.Fold in state.legal_actions:
                         return pkrs.Action(pkrs.ActionEnum.Fold)
                     else: # Last resort
                         return pkrs.Action(pkrs.ActionEnum.Call)


            # Choose a random *additional* raise amount from the valid options
            additional_raise = random.choice(possible_additional_amounts)

            # Create the final Raise action
            action = pkrs.Action(action_enum, additional_raise)

            # Optional: Strict checking (if enabled)
            if STRICT_CHECKING:
                # Temporarily apply action to check Rust status
                test_state = state.apply_action(action)
                if test_state.status != pkrs.StateStatus.Ok:
                    log_file = log_game_error(state, action, f"Random agent created invalid action: {test_state.status}")
                    # Fallback to Call if the generated Raise is invalid
                    print(f"RandomAgent STRICT CHECK FAILED: Invalid Raise({additional_raise}). Status: {test_state.status}. Falling back to Call. Log: {log_file}")
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    else: # Fallback to Fold if Call isn't legal
                        if pkrs.ActionEnum.Fold in state.legal_actions:
                            return pkrs.Action(pkrs.ActionEnum.Fold)
                        else: # Last resort
                            return pkrs.Action(pkrs.ActionEnum.Call)

            return action
        else:
            # Should not happen if action_enum is from legal_actions
            print(f"WARNING: RandomAgent encountered unexpected action enum: {action_enum}. Falling back to Fold.")
            if pkrs.ActionEnum.Fold in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Fold)
            else: # Last resort
                return pkrs.Action(pkrs.ActionEnum.Check) # Or Check if Fold is illegal
