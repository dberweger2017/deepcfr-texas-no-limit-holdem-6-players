"""Shared action helpers for mapping model decisions to legal pokers actions."""

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pokers as pkrs


ACTION_TYPE_FOLD = 0
ACTION_TYPE_CHECK_CALL = 1
ACTION_TYPE_RAISE = 2


class ActionMappingFailure(ValueError):
    """Raised when an action cannot be mapped without falling back."""


@dataclass(frozen=True)
class RaiseBounds:
    """Additional chips a player may add after calling the current bet."""

    call_amount: float
    min_raise: float
    max_raise: float
    current_bet: float
    available_stake: float

    @property
    def can_raise(self) -> bool:
        return self.max_raise >= self.min_raise


def safe_fallback_action(
    legal_actions: Iterable[pkrs.ActionEnum],
    *,
    prefer_call: bool = True,
    strict: bool = False,
    reason: str = "No legal fallback action available",
    attempted_action: pkrs.Action = None,
    fallback_recorder=None,
) -> pkrs.Action:
    """Return the least disruptive legal non-raise fallback action."""
    legal_actions = list(legal_actions)
    ordered_actions = (
        (pkrs.ActionEnum.Call, pkrs.ActionEnum.Check, pkrs.ActionEnum.Fold)
        if prefer_call
        else (pkrs.ActionEnum.Check, pkrs.ActionEnum.Call, pkrs.ActionEnum.Fold)
    )
    for action_enum in ordered_actions:
        if action_enum in legal_actions:
            fallback = pkrs.Action(action_enum)
            break
    else:
        fallback = pkrs.Action(pkrs.ActionEnum.Fold)

    if strict:
        raise ActionMappingFailure(
            f"{reason}. Legal actions: {[str(action) for action in legal_actions]}"
        )

    _record_fallback(
        fallback_recorder,
        reason=reason,
        attempted_action=attempted_action,
        fallback_action=fallback,
    )
    return fallback


def legal_action_types(state) -> List[int]:
    """Map pokers legal actions onto the project action abstraction."""
    action_types = []
    if pkrs.ActionEnum.Fold in state.legal_actions:
        action_types.append(ACTION_TYPE_FOLD)
    if pkrs.ActionEnum.Check in state.legal_actions or pkrs.ActionEnum.Call in state.legal_actions:
        action_types.append(ACTION_TYPE_CHECK_CALL)
    if pkrs.ActionEnum.Raise in state.legal_actions:
        action_types.append(ACTION_TYPE_RAISE)
    return action_types


def min_raise_increment(state) -> float:
    """Best available estimate for a valid additional raise amount."""
    bb = getattr(state, "bb", None)
    if bb is not None and float(bb) > 0:
        return max(1.0, float(bb))

    current_player_state = state.players_state[state.current_player]
    to_call = max(0.0, float(state.min_bet) - float(current_player_state.bet_chips))
    if float(state.min_bet) > 0:
        return max(1.0, to_call if to_call > 0 else float(state.min_bet))
    return 1.0


def raise_bounds(state) -> RaiseBounds:
    """Return legal additional-raise bounds for the current player."""
    player_state = state.players_state[state.current_player]
    current_bet = float(player_state.bet_chips)
    available_stake = float(player_state.stake)
    call_amount = max(0.0, float(state.min_bet) - current_bet)
    max_raise = max(0.0, available_stake - call_amount)
    return RaiseBounds(
        call_amount=call_amount,
        min_raise=min_raise_increment(state),
        max_raise=max_raise,
        current_bet=current_bet,
        available_stake=available_stake,
    )


def build_raise_action(
    state,
    additional_amount: Optional[float],
    *,
    strict: bool = False,
    fallback_recorder=None,
) -> pkrs.Action:
    """
    Build a legal raise from an intended additional amount.

    ``pokers`` expects the raise amount to be the additional chips committed
    after matching the current bet, so this clamps within those bounds.
    """
    if pkrs.ActionEnum.Raise not in state.legal_actions:
        return safe_fallback_action(
            state.legal_actions,
            strict=strict,
            reason="Raise requested when raise is not legal",
            attempted_action=pkrs.Action(pkrs.ActionEnum.Raise, additional_amount or 0.0),
            fallback_recorder=fallback_recorder,
        )

    bounds = raise_bounds(state)
    if not bounds.can_raise and not strict:
        return safe_fallback_action(
            state.legal_actions,
            reason="Raise requested but computed raise bounds are invalid",
            attempted_action=pkrs.Action(pkrs.ActionEnum.Raise, additional_amount or 0.0),
            fallback_recorder=fallback_recorder,
        )

    if additional_amount is None:
        additional_amount = bounds.min_raise if bounds.can_raise else bounds.max_raise

    additional_amount = float(additional_amount)
    if bounds.can_raise:
        additional_amount = min(max(additional_amount, bounds.min_raise), bounds.max_raise)
    else:
        additional_amount = min(max(0.0, additional_amount), bounds.max_raise)

    total_commit = bounds.call_amount + additional_amount
    epsilon = 1e-5
    if total_commit > bounds.available_stake + epsilon:
        additional_amount = max(0.0, bounds.available_stake - bounds.call_amount)

    if additional_amount + epsilon < bounds.min_raise and not strict:
        return safe_fallback_action(
            state.legal_actions,
            reason="Raise requested below the minimum legal raise",
            attempted_action=pkrs.Action(pkrs.ActionEnum.Raise, additional_amount),
            fallback_recorder=fallback_recorder,
        )

    return _validated_raise_or_fallback(
        state,
        bounds,
        additional_amount,
        strict=strict,
        fallback_recorder=fallback_recorder,
    )


def _engine_accepts_action(state, action: pkrs.Action) -> bool:
    """Validate against pokers when a real state object is available."""
    apply_action = getattr(state, "apply_action", None)
    if apply_action is None:
        return True

    try:
        return apply_action(action).status == pkrs.StateStatus.Ok
    except Exception:
        return False


def _validated_raise_or_fallback(
    state,
    bounds: RaiseBounds,
    additional_amount: float,
    *,
    strict: bool = False,
    fallback_recorder=None,
) -> pkrs.Action:
    """Return the closest engine-accepted raise, or a non-raise fallback."""
    candidates = [max(0.0, additional_amount)]

    # Some all-in edges are rejected by the engine at exact float equality.
    # Try tiny reductions before abandoning the raise decision.
    for epsilon in (1e-6, 1e-4, 1e-2):
        stepped_down = additional_amount - epsilon
        if stepped_down + 1e-9 >= bounds.min_raise:
            candidates.append(stepped_down)

    if bounds.min_raise not in candidates:
        candidates.append(bounds.min_raise)
    if bounds.max_raise not in candidates:
        candidates.append(bounds.max_raise)

    for candidate in candidates:
        action = pkrs.Action(pkrs.ActionEnum.Raise, candidate)
        if _engine_accepts_action(state, action):
            return action

    return safe_fallback_action(
        state.legal_actions,
        strict=strict,
        reason="No candidate raise amount was accepted by the engine",
        attempted_action=pkrs.Action(pkrs.ActionEnum.Raise, additional_amount),
        fallback_recorder=fallback_recorder,
    )


def sanitize_action(
    state,
    action: pkrs.Action,
    *,
    strict: bool = False,
    fallback_recorder=None,
) -> pkrs.Action:
    """Preserve an action choice while normalizing illegal raise amounts."""
    if action.action == pkrs.ActionEnum.Raise:
        return build_raise_action(
            state,
            action.amount,
            strict=strict,
            fallback_recorder=fallback_recorder,
        )

    if action.action in state.legal_actions:
        return action

    return safe_fallback_action(
        state.legal_actions,
        strict=strict,
        reason=f"Action {action.action} is not legal",
        attempted_action=action,
        fallback_recorder=fallback_recorder,
    )


def action_type_to_pokers_action(
    action_type: int,
    state,
    *,
    bet_size_multiplier: Optional[float] = None,
    min_bet_size: float = 0.1,
    max_bet_size: float = 3.0,
    strict: bool = False,
    fallback_recorder=None,
) -> pkrs.Action:
    """Convert the model's 3-way action abstraction to a legal pokers action."""
    if action_type == ACTION_TYPE_FOLD:
        if pkrs.ActionEnum.Fold in state.legal_actions:
            return pkrs.Action(pkrs.ActionEnum.Fold)
        return safe_fallback_action(
            state.legal_actions,
            prefer_call=False,
            strict=strict,
            reason="Fold action type requested when fold is not legal",
            attempted_action=pkrs.Action(pkrs.ActionEnum.Fold),
            fallback_recorder=fallback_recorder,
        )

    if action_type == ACTION_TYPE_CHECK_CALL:
        if pkrs.ActionEnum.Check in state.legal_actions:
            return pkrs.Action(pkrs.ActionEnum.Check)
        if pkrs.ActionEnum.Call in state.legal_actions:
            return pkrs.Action(pkrs.ActionEnum.Call)
        return safe_fallback_action(
            state.legal_actions,
            strict=strict,
            reason="Check/call action type requested when neither is legal",
            attempted_action=pkrs.Action(pkrs.ActionEnum.Call),
            fallback_recorder=fallback_recorder,
        )

    if action_type == ACTION_TYPE_RAISE:
        if bet_size_multiplier is None:
            bet_size_multiplier = 1.0
        bet_size_multiplier = max(min_bet_size, min(max_bet_size, float(bet_size_multiplier)))
        desired_additional_raise = max(1.0, float(state.pot)) * bet_size_multiplier
        return build_raise_action(
            state,
            desired_additional_raise,
            strict=strict,
            fallback_recorder=fallback_recorder,
        )

    return safe_fallback_action(
        state.legal_actions,
        strict=strict,
        reason=f"Unknown action type {action_type}",
        fallback_recorder=fallback_recorder,
    )


def preset_raise_action(state, preset: str) -> pkrs.Action:
    """Build a raise action for common UI/random-agent presets."""
    bounds = raise_bounds(state)
    if preset == "min":
        amount = bounds.min_raise
    elif preset == "half_pot":
        amount = max(float(state.pot) * 0.5, bounds.min_raise)
    elif preset == "pot":
        amount = max(float(state.pot), bounds.min_raise)
    elif preset == "all_in":
        amount = bounds.max_raise
    else:
        raise ValueError(f"Unknown raise preset: {preset}")
    return build_raise_action(state, amount)


def _record_fallback(fallback_recorder, *, reason, attempted_action, fallback_action):
    if fallback_recorder is None:
        return
    fallback_recorder(
        reason=reason,
        attempted_action=attempted_action,
        fallback_action=fallback_action,
    )
