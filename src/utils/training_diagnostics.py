"""Small training diagnostics shared by training loops."""

from typing import Any, Dict

import numpy as np


def collect_training_diagnostics(agent, sample_size: int = 2048) -> Dict[str, Any]:
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


def format_training_diagnostics(diagnostics: Dict[str, Any]) -> str:
    """Return a compact terminal summary of replay target diagnostics."""
    return (
        "targets "
        f"fold={diagnostics['strategy_target_fold']:.2f}, "
        f"check-call={diagnostics['strategy_target_check_call']:.2f}, "
        f"raise={diagnostics['strategy_target_raise']:.2f}; "
        "regret "
        f"min={diagnostics['regret_target_min']:.2f}, "
        f"mean={diagnostics['regret_target_mean']:.2f}, "
        f"max={diagnostics['regret_target_max']:.2f}"
    )
