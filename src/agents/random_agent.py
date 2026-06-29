# src/agents/random_agent.py
import random
import pokers as pkrs
from src.utils import settings
from src.utils.logging import log_game_error
from src.utils.actions import preset_raise_action

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

        elif action_enum == pkrs.ActionEnum.Raise:
            action = preset_raise_action(
                state,
                random.choice(["min", "half_pot", "pot", "all_in"]),
            )

            # Optional: Strict checking (if enabled)
            if settings.is_strict_checking() and action.action == pkrs.ActionEnum.Raise:
                # Temporarily apply action to check Rust status
                test_state = state.apply_action(action)
                if test_state.status != pkrs.StateStatus.Ok:
                    log_file = log_game_error(state, action, f"Random agent created invalid action: {test_state.status}")
                    # Fallback to Call if the generated Raise is invalid
                    print(f"RandomAgent STRICT CHECK FAILED: Invalid Raise({action.amount}). Status: {test_state.status}. Falling back to Call. Log: {log_file}")
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
