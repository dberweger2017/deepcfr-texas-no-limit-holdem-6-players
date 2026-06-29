"""Shared seeded hand-running and evaluation helpers."""

from typing import Any, Dict, Optional, Sequence

import pokers as pkrs

from src.utils import settings
from src.utils.logging import apply_action_with_logging


def action_history_id(action: pkrs.Action, pot: float) -> int:
    """Map a pokers action to the compact opponent-modeling history id."""
    if action.action == pkrs.ActionEnum.Fold:
        return 0
    if action.action in (pkrs.ActionEnum.Check, pkrs.ActionEnum.Call):
        return 1
    if action.action == pkrs.ActionEnum.Raise:
        return 2 if action.amount <= pot * 0.75 else 3
    return 1


def choose_agent_action(agent, state, *, opponent_id=None):
    """Call an agent's choose_action with optional OM opponent context when supported."""
    if opponent_id is not None:
        try:
            return agent.choose_action(state, opponent_id=opponent_id)
        except TypeError:
            pass
    return agent.choose_action(state)


def empty_evaluation_metrics(requested_games: int) -> Dict[str, Any]:
    """Return the canonical metrics shape for a skipped evaluation."""
    return {
        "avg_profit": 0.0,
        "completed_games": 0,
        "invalid_state_games": 0,
        "invalid_state_count": 0,
        "non_zero_sum_games": 0,
        "setup_errors": 0,
        "requested_games": requested_games,
        "zero_reward_games": 0,
        "game_crashes": 0,
        "total_actions": 0,
    }


def evaluate_agent_matchup(
    agent,
    opponents: Sequence[Optional[Any]],
    *,
    num_games: int,
    seed_start: int = 0,
    button_start: int = 0,
    num_players: int = 6,
    stake: float = 200.0,
    sb: float = 1.0,
    bb: float = 2.0,
    strict: Optional[bool] = None,
    label: str = "evaluation",
    record_opponent_history: bool = False,
    print_warnings: bool = False,
) -> Dict[str, Any]:
    """Play seeded hands and return stable evaluation metrics."""
    if strict is None:
        strict = settings.is_strict_checking()

    total_profit = 0.0
    completed_games = 0
    invalid_state_games = 0
    invalid_state_count = 0
    non_zero_sum_games = 0
    setup_errors = 0
    zero_reward_games = 0
    total_actions = 0

    for game in range(num_games):
        state = None
        try:
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=(button_start + game) % num_players,
                sb=sb,
                bb=bb,
                stake=stake,
                seed=seed_start + game,
            )

            game_had_invalid_state = False

            while not state.final_state:
                current_player = state.current_player
                if current_player == agent.player_id:
                    action = choose_agent_action(agent, state, opponent_id=None)
                else:
                    opponent = opponents[current_player]
                    if opponent is None:
                        raise ValueError(f"No opponent configured for player {current_player}")
                    action = choose_agent_action(opponent, state)

                    if record_opponent_history and hasattr(agent, "record_opponent_action"):
                        agent.record_opponent_action(
                            state,
                            action_history_id(action, state.pot),
                            current_player,
                        )

                new_state, log_file, status = apply_action_with_logging(
                    state,
                    action,
                    strict=strict,
                    error_prefix=f"State status not OK during {label}",
                )
                total_actions += 1

                if new_state is None:
                    invalid_state_count += 1
                    game_had_invalid_state = True
                    if print_warnings:
                        print(
                            f"WARNING: State status not OK ({status}) in game {game}. "
                            f"Details logged to {log_file}"
                        )
                    break

                state = new_state

            if state.final_state:
                completed_games += 1
                profit = state.players_state[agent.player_id].reward
                total_profit += profit
                if abs(profit) < 0.001:
                    zero_reward_games += 1
                zero_sum_delta = abs(sum(player.reward for player in state.players_state))
                if zero_sum_delta > 1e-9:
                    non_zero_sum_games += 1
                if record_opponent_history and hasattr(agent, "end_game_recording"):
                    agent.end_game_recording(state)
            elif game_had_invalid_state:
                invalid_state_games += 1
        except Exception as exc:
            if strict:
                raise
            setup_errors += 1
            if print_warnings:
                print(f"Error in game {game}: {exc}")
        finally:
            if (
                state is not None
                and record_opponent_history
                and hasattr(agent, "end_game_recording")
                and not state.final_state
            ):
                agent.current_game_history = {}

    avg_profit = total_profit / completed_games if completed_games else 0.0
    return {
        "avg_profit": avg_profit,
        "completed_games": completed_games,
        "invalid_state_games": invalid_state_games,
        "invalid_state_count": invalid_state_count,
        "non_zero_sum_games": non_zero_sum_games,
        "setup_errors": setup_errors,
        "requested_games": num_games,
        "zero_reward_games": zero_reward_games,
        "game_crashes": setup_errors + invalid_state_games,
        "total_actions": total_actions,
    }
