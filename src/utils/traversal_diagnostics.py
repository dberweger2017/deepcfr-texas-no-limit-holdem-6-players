"""Diagnostics for CFR traversal value paths."""

from collections import Counter, deque
from typing import Any, Dict, Iterable, Optional

import numpy as np


ACTION_NAMES = {
    0: "fold",
    1: "check_call",
    2: "raise",
}


class TraversalFailure(RuntimeError):
    """Raised when traversal reaches a branch that cannot be assigned a real EV."""

    def __init__(self, reason: str, event: Dict[str, Any]):
        self.reason = reason
        self.event = event
        super().__init__(format_traversal_failure(event))


class TraversalDiagnostics:
    """Small mutable recorder attached to training agents."""

    def __init__(self, max_samples: int = 20):
        self.max_samples = max_samples
        self.reset()

    def reset(self):
        self.agent_turns = 0
        self.action_value_count = 0
        self.action_value_zero_count = 0
        self.action_value_sum = 0.0
        self.action_value_min: Optional[float] = None
        self.action_value_max: Optional[float] = None
        self.ev_count = 0
        self.ev_sum = 0.0
        self.ev_min: Optional[float] = None
        self.ev_max: Optional[float] = None
        self.action_stats: Dict[int, Dict[str, float]] = {
            action_type: {"count": 0, "zero": 0, "sum": 0.0, "min": 0.0, "max": 0.0}
            for action_type in ACTION_NAMES
        }
        self.failure_counts: Counter[str] = Counter()
        self.recent_action_values = deque(maxlen=self.max_samples)
        self.recent_failures = deque(maxlen=self.max_samples)

    def record_action_values(
        self,
        *,
        state,
        depth: int,
        iteration: int,
        legal_action_types: Iterable[int],
        action_values,
        strategy,
        ev: float,
    ):
        self.agent_turns += 1
        legal_actions = list(legal_action_types)
        values = np.asarray(action_values, dtype=np.float64)
        strategy_values = np.asarray(strategy, dtype=np.float64)
        action_sample: Dict[str, Any] = {
            **state_context(state),
            "depth": depth,
            "iteration": iteration,
            "ev": float(ev),
            "actions": {},
        }

        for action_type in legal_actions:
            value = float(values[action_type])
            probability = float(strategy_values[action_type])
            self._record_value(action_type, value)
            action_sample["actions"][ACTION_NAMES.get(action_type, str(action_type))] = {
                "value": value,
                "strategy": probability,
            }

        self._record_ev(float(ev))
        self.recent_action_values.append(action_sample)

    def record_failure(self, reason: str, **context) -> Dict[str, Any]:
        self.failure_counts[reason] += 1
        event = {"reason": reason, **context}
        self.recent_failures.append(event)
        return event

    def snapshot(self, reset: bool = False) -> Dict[str, Any]:
        action_value_mean = (
            self.action_value_sum / self.action_value_count
            if self.action_value_count
            else 0.0
        )
        ev_mean = self.ev_sum / self.ev_count if self.ev_count else 0.0
        failure_total = sum(self.failure_counts.values())
        result: Dict[str, Any] = {
            "agent_turns": self.agent_turns,
            "action_value_count": self.action_value_count,
            "action_value_zero_count": self.action_value_zero_count,
            "action_value_zero_rate": (
                self.action_value_zero_count / self.action_value_count
                if self.action_value_count
                else 0.0
            ),
            "action_value_min": self.action_value_min or 0.0,
            "action_value_mean": action_value_mean,
            "action_value_max": self.action_value_max or 0.0,
            "ev_min": self.ev_min or 0.0,
            "ev_mean": ev_mean,
            "ev_max": self.ev_max or 0.0,
            "failure_total": failure_total,
            "recent_action_values": list(self.recent_action_values),
            "recent_failures": list(self.recent_failures),
        }

        for reason, count in self.failure_counts.items():
            result[f"failure_{reason}"] = count

        for action_type, stats in self.action_stats.items():
            name = ACTION_NAMES[action_type]
            count = int(stats["count"])
            result[f"action_value_{name}_count"] = count
            result[f"action_value_{name}_zero_rate"] = (
                stats["zero"] / count if count else 0.0
            )
            result[f"action_value_{name}_mean"] = stats["sum"] / count if count else 0.0
            result[f"action_value_{name}_min"] = stats["min"] if count else 0.0
            result[f"action_value_{name}_max"] = stats["max"] if count else 0.0

        if reset:
            self.reset()

        return result

    def _record_value(self, action_type: int, value: float):
        self.action_value_count += 1
        self.action_value_sum += value
        self.action_value_min = value if self.action_value_min is None else min(self.action_value_min, value)
        self.action_value_max = value if self.action_value_max is None else max(self.action_value_max, value)
        if abs(value) <= 1e-12:
            self.action_value_zero_count += 1

        stats = self.action_stats.setdefault(
            action_type,
            {"count": 0, "zero": 0, "sum": 0.0, "min": value, "max": value},
        )
        if stats["count"] == 0:
            stats["min"] = value
            stats["max"] = value
        else:
            stats["min"] = min(stats["min"], value)
            stats["max"] = max(stats["max"], value)
        stats["count"] += 1
        stats["sum"] += value
        if abs(value) <= 1e-12:
            stats["zero"] += 1

    def _record_ev(self, ev: float):
        self.ev_count += 1
        self.ev_sum += ev
        self.ev_min = ev if self.ev_min is None else min(self.ev_min, ev)
        self.ev_max = ev if self.ev_max is None else max(self.ev_max, ev)


def ensure_traversal_diagnostics(agent) -> TraversalDiagnostics:
    diagnostics = getattr(agent, "traversal_diagnostics", None)
    if diagnostics is None:
        diagnostics = TraversalDiagnostics()
        agent.traversal_diagnostics = diagnostics
    return diagnostics


def record_action_values(agent, **kwargs):
    ensure_traversal_diagnostics(agent).record_action_values(**kwargs)


def fail_traversal(agent, reason: str, *, exception: Exception = None, **context):
    diagnostics = ensure_traversal_diagnostics(agent)
    event = diagnostics.record_failure(reason, **context)
    failure = TraversalFailure(reason, event)
    if exception is not None:
        raise failure from exception
    raise failure


def state_context(state) -> Dict[str, Any]:
    current_player = getattr(state, "current_player", None)
    player_state = None
    players_state = getattr(state, "players_state", None)
    if players_state is not None and current_player is not None:
        try:
            player_state = players_state[current_player]
        except Exception:
            player_state = None

    return {
        "stage": str(getattr(state, "stage", "unknown")),
        "current_player": current_player,
        "button": getattr(state, "button", None),
        "pot": _safe_float(getattr(state, "pot", None)),
        "min_bet": _safe_float(getattr(state, "min_bet", None)),
        "legal_actions": [str(action) for action in getattr(state, "legal_actions", [])],
        "current_bet": _safe_float(getattr(player_state, "bet_chips", None)),
        "current_stake": _safe_float(getattr(player_state, "stake", None)),
    }


def action_context(action) -> Dict[str, Any]:
    if action is None:
        return {"action": None, "amount": None}
    return {
        "action": str(getattr(action, "action", "unknown")),
        "amount": _safe_float(getattr(action, "amount", None)),
    }


def failure_context(
    *,
    state,
    depth: int,
    iteration: int,
    player_id: int,
    action_type: int = None,
    action=None,
    status=None,
    log_file=None,
    message: str = None,
) -> Dict[str, Any]:
    return {
        **state_context(state),
        **action_context(action),
        "depth": depth,
        "iteration": iteration,
        "player_id": player_id,
        "action_type": action_type,
        "action_name": (
            ACTION_NAMES.get(action_type, str(action_type))
            if action_type is not None
            else None
        ),
        "status": str(status) if status is not None else None,
        "log_file": log_file,
        "message": message,
    }


def format_traversal_failure(event: Dict[str, Any]) -> str:
    parts = [
        "Traversal branch failed instead of being converted to value 0",
        f"reason={event.get('reason')}",
        f"iteration={event.get('iteration')}",
        f"depth={event.get('depth')}",
        f"stage={event.get('stage')}",
        f"player={event.get('current_player')}",
    ]
    if event.get("action_name") is not None:
        parts.append(f"action_type={event.get('action_name')}")
    if event.get("action") is not None:
        parts.append(f"action={event.get('action')}")
    if event.get("amount") is not None:
        parts.append(f"amount={event.get('amount')}")
    if event.get("status") is not None:
        parts.append(f"status={event.get('status')}")
    if event.get("log_file"):
        parts.append(f"log={event.get('log_file')}")
    if event.get("message"):
        parts.append(f"message={event.get('message')}")
    return "; ".join(parts)


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
