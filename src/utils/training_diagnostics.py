"""Small training diagnostics shared by training loops."""

from typing import Any, Dict

import numpy as np

from src.utils.traversal_diagnostics import ACTION_NAMES


def collect_training_diagnostics(
    agent,
    sample_size: int = 2048,
    reset_traversal: bool = False,
) -> Dict[str, Any]:
    """Summarize recent replay targets without mutating training RNG state."""
    diagnostics: Dict[str, Any] = {
        "strategy_samples": 0,
        "strategy_target_fold": 0.0,
        "strategy_target_check_call": 0.0,
        "strategy_target_raise": 0.0,
        "regret_samples": 0,
        "regret_target_min": 0.0,
        "regret_target_mean": 0.0,
        "regret_target_max": 0.0,
        "traversal": _empty_traversal_diagnostics(),
    }

    strategy_memory = list(getattr(agent, "strategy_memory", []))
    if strategy_memory:
        rows = strategy_memory[-sample_size:]
        strategies = np.array([row[2] for row in rows], dtype=np.float32)
        target_mean = np.mean(strategies, axis=0)
        diagnostics.update(
            {
                "strategy_samples": len(rows),
                "strategy_target_fold": float(target_mean[0]),
                "strategy_target_check_call": float(target_mean[1]),
                "strategy_target_raise": float(target_mean[2]),
            }
        )

    advantage_memory = getattr(agent, "advantage_memory", None)
    advantage_buffer = list(getattr(advantage_memory, "buffer", []))
    if advantage_buffer:
        rows = advantage_buffer[-sample_size:]
        regrets = np.array([row[4] for row in rows], dtype=np.float32)
        diagnostics.update(
            {
                "regret_samples": len(rows),
                "regret_target_min": float(np.min(regrets)),
                "regret_target_mean": float(np.mean(regrets)),
                "regret_target_max": float(np.max(regrets)),
            }
        )

    traversal_diagnostics = getattr(agent, "traversal_diagnostics", None)
    if traversal_diagnostics is not None:
        diagnostics["traversal"] = traversal_diagnostics.snapshot(reset=reset_traversal)

    return diagnostics


def write_training_diagnostics(writer, diagnostics: Dict[str, Any], iteration: int, prefix: str = "Diagnostics"):
    """Write replay target diagnostics to TensorBoard."""
    if writer is None or iteration is None:
        return
    writer.add_scalar(
        f"{prefix}/StrategyTargetFold",
        diagnostics["strategy_target_fold"],
        iteration,
    )
    writer.add_scalar(
        f"{prefix}/StrategyTargetCheckCall",
        diagnostics["strategy_target_check_call"],
        iteration,
    )
    writer.add_scalar(
        f"{prefix}/StrategyTargetRaise",
        diagnostics["strategy_target_raise"],
        iteration,
    )
    writer.add_scalar(f"{prefix}/RegretTargetMin", diagnostics["regret_target_min"], iteration)
    writer.add_scalar(f"{prefix}/RegretTargetMean", diagnostics["regret_target_mean"], iteration)
    writer.add_scalar(f"{prefix}/RegretTargetMax", diagnostics["regret_target_max"], iteration)
    _write_traversal_diagnostics(writer, diagnostics["traversal"], iteration, prefix)


def format_training_diagnostics(diagnostics: Dict[str, Any]) -> str:
    """Return a compact terminal summary of replay target diagnostics."""
    traversal = diagnostics.get("traversal", _empty_traversal_diagnostics())
    return (
        "targets "
        f"fold={diagnostics['strategy_target_fold']:.2f}, "
        f"check-call={diagnostics['strategy_target_check_call']:.2f}, "
        f"raise={diagnostics['strategy_target_raise']:.2f}; "
        "regret "
        f"min={diagnostics['regret_target_min']:.2f}, "
        f"mean={diagnostics['regret_target_mean']:.2f}, "
        f"max={diagnostics['regret_target_max']:.2f}; "
        "traversal "
        f"turns={traversal['agent_turns']}, "
        f"value_zero={traversal['action_value_zero_rate']:.1%}, "
        f"value_mean={traversal['action_value_mean']:.2f}, "
        f"ev_mean={traversal['ev_mean']:.2f}, "
        f"failures={traversal['failure_total']}"
    )


def _empty_traversal_diagnostics() -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "agent_turns": 0,
        "action_value_count": 0,
        "action_value_zero_count": 0,
        "action_value_zero_rate": 0.0,
        "action_value_min": 0.0,
        "action_value_mean": 0.0,
        "action_value_max": 0.0,
        "ev_min": 0.0,
        "ev_mean": 0.0,
        "ev_max": 0.0,
        "failure_total": 0,
        "recent_action_values": [],
        "recent_failures": [],
    }
    for action_name in ACTION_NAMES.values():
        diagnostics[f"action_value_{action_name}_count"] = 0
        diagnostics[f"action_value_{action_name}_zero_rate"] = 0.0
        diagnostics[f"action_value_{action_name}_mean"] = 0.0
        diagnostics[f"action_value_{action_name}_min"] = 0.0
        diagnostics[f"action_value_{action_name}_max"] = 0.0
    return diagnostics


def _write_traversal_diagnostics(writer, traversal: Dict[str, Any], iteration: int, prefix: str):
    writer.add_scalar(f"{prefix}/Traversal/AgentTurns", traversal["agent_turns"], iteration)
    writer.add_scalar(
        f"{prefix}/Traversal/ActionValueCount",
        traversal["action_value_count"],
        iteration,
    )
    writer.add_scalar(
        f"{prefix}/Traversal/ActionValueZeroRate",
        traversal["action_value_zero_rate"],
        iteration,
    )
    writer.add_scalar(f"{prefix}/Traversal/ActionValueMin", traversal["action_value_min"], iteration)
    writer.add_scalar(f"{prefix}/Traversal/ActionValueMean", traversal["action_value_mean"], iteration)
    writer.add_scalar(f"{prefix}/Traversal/ActionValueMax", traversal["action_value_max"], iteration)
    writer.add_scalar(f"{prefix}/Traversal/EVMin", traversal["ev_min"], iteration)
    writer.add_scalar(f"{prefix}/Traversal/EVMean", traversal["ev_mean"], iteration)
    writer.add_scalar(f"{prefix}/Traversal/EVMax", traversal["ev_max"], iteration)
    writer.add_scalar(f"{prefix}/Traversal/FailureTotal", traversal["failure_total"], iteration)

    for action_name in ACTION_NAMES.values():
        tensorboard_name = "".join(part.capitalize() for part in action_name.split("_"))
        writer.add_scalar(
            f"{prefix}/Traversal/{tensorboard_name}ValueMean",
            traversal[f"action_value_{action_name}_mean"],
            iteration,
        )
        writer.add_scalar(
            f"{prefix}/Traversal/{tensorboard_name}ValueZeroRate",
            traversal[f"action_value_{action_name}_zero_rate"],
            iteration,
        )

    for key, value in traversal.items():
        if key.startswith("failure_") and key != "failure_total":
            reason = key.removeprefix("failure_")
            writer.add_scalar(f"{prefix}/Traversal/Failures/{reason}", value, iteration)
