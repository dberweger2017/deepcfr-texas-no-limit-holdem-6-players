"""Shared checkpoint discovery helpers for training and evaluation workflows."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


_GLOB_CHARS = "*?[]"
CHECKPOINT_SCHEMA_VERSION = 1
AGENT_TYPE_STANDARD = "standard"
AGENT_TYPE_OPPONENT_MODELING = "opponent_modeling"


def checkpoint_pattern(prefix_or_pattern: str) -> str:
    """Normalize a checkpoint prefix or glob fragment into a usable .pt pattern."""
    if not prefix_or_pattern:
        return "*.pt"
    if prefix_or_pattern.endswith(".pt"):
        return prefix_or_pattern
    if any(char in prefix_or_pattern for char in _GLOB_CHARS):
        if prefix_or_pattern.endswith("*"):
            return f"{prefix_or_pattern}.pt"
        return f"{prefix_or_pattern}*.pt"
    return f"{prefix_or_pattern}*.pt"


def find_checkpoints(checkpoint_dir, prefix_or_pattern: str):
    """Recursively find checkpoint files under a directory."""
    root = Path(checkpoint_dir)
    pattern = checkpoint_pattern(prefix_or_pattern)
    return sorted(path for path in root.glob(f"**/{pattern}") if path.is_file())


def find_checkpoint_dirs(checkpoint_root: Path, *, relative_to: Optional[Path] = None) -> List[str]:
    """Return directories under ``checkpoint_root`` that contain at least one ``.pt`` file."""
    if not checkpoint_root.exists():
        return []
    base = relative_to or checkpoint_root
    dirs: List[str] = []
    for path in sorted(checkpoint_root.rglob("*")):
        if path.is_dir() and any(path.glob("*.pt")):
            dirs.append(str(path.relative_to(base)))
    return dirs


def select_random_checkpoints_in_dir(models_dir: Path | str, num_models: int = 5) -> List[Path]:
    """Pick up to ``num_models`` checkpoint files from a single directory (non-recursive)."""
    root = Path(models_dir)
    model_files = sorted(root.glob("*.pt"))
    if not model_files:
        return []
    return random.sample(model_files, min(num_models, len(model_files)))


def load_play_agent(path: Path | str, seat: int, device: str) -> Tuple[Any, Optional[dict]]:
    """
    Load a DeepCFR or opponent-modeling agent for interactive play.

    Returns ``(agent, warning)`` where ``warning`` is set when loading failed and a
    random fallback agent was used instead.
    """
    from src.agents.random_agent import RandomAgent
    from src.core.deep_cfr import DeepCFRAgent
    from src.opponent_modeling.deep_cfr_with_opponent_modeling import DeepCFRAgentWithOpponentModeling

    checkpoint_path = Path(path)
    model_name = checkpoint_path.name
    try:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        if checkpoint_uses_opponent_modeling_state(checkpoint):
            agent = DeepCFRAgentWithOpponentModeling(player_id=seat, device=device)
        else:
            agent = DeepCFRAgent(player_id=seat, num_players=6, device=device)
        agent.load_model(str(checkpoint_path))
        agent.model_name = model_name
        return agent, None
    except Exception as exc:
        logger.warning(
            "Failed to load checkpoint %s for seat %s: %s",
            checkpoint_path,
            seat,
            exc,
            exc_info=True,
        )
        agent = RandomAgent(seat)
        agent.model_name = f"Random (load failed: {model_name})"
        warning = {
            "seat": seat,
            "checkpoint": model_name,
            "error": str(exc),
        }
        return agent, warning


def checkpoint_path_uses_opponent_modeling(path: Path | str) -> bool:
    """Return whether a checkpoint file on disk is for the OM agent."""
    checkpoint = load_checkpoint(path, map_location="cpu")
    return checkpoint_uses_opponent_modeling_state(checkpoint)


def load_checkpoint(path, map_location=None):
    """Load a checkpoint with safer torch defaults when available."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_metadata(agent, agent_type: str, **extra):
    """Build metadata embedded in new checkpoint files."""
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "agent_type": agent_type,
        "num_players": getattr(agent, "num_players", 6),
        "player_id": getattr(agent, "player_id", 0),
        "iteration": getattr(agent, "iteration_count", 0),
    }
    metadata.update(extra)
    return metadata


def attach_checkpoint_metadata(checkpoint: dict, agent, agent_type: str, **extra):
    """Return a checkpoint dict with top-level and nested metadata fields."""
    metadata = checkpoint_metadata(agent, agent_type, **extra)
    checkpoint = dict(checkpoint)
    checkpoint.update(
        {
            "schema_version": metadata["schema_version"],
            "agent_type": metadata["agent_type"],
            "num_players": metadata["num_players"],
            "player_id": metadata["player_id"],
            "metadata": metadata,
        }
    )
    checkpoint.setdefault("iteration", metadata["iteration"])
    return checkpoint


def standard_checkpoint_state(agent, **extra):
    """Build a standard-agent checkpoint payload."""
    checkpoint = {
        "iteration": getattr(agent, "iteration_count", 0),
        "advantage_net": agent.advantage_net.state_dict(),
        "strategy_net": agent.strategy_net.state_dict(),
        "min_bet_size": getattr(agent, "min_bet_size", 0.1),
        "max_bet_size": getattr(agent, "max_bet_size", 3.0),
    }
    checkpoint.update(extra)
    return attach_checkpoint_metadata(checkpoint, agent, AGENT_TYPE_STANDARD)


def opponent_modeling_checkpoint_state(agent, **extra):
    """Build an opponent-modeling checkpoint payload."""
    checkpoint = {
        "iteration": getattr(agent, "iteration_count", 0),
        "advantage_net": agent.advantage_net.state_dict(),
        "strategy_net": agent.strategy_net.state_dict(),
        "history_encoder": agent.opponent_modeling.history_encoder.state_dict(),
        "opponent_model": agent.opponent_modeling.opponent_model.state_dict(),
    }
    checkpoint.update(extra)
    return attach_checkpoint_metadata(checkpoint, agent, AGENT_TYPE_OPPONENT_MODELING)


def infer_agent_type(checkpoint: dict) -> str:
    """Infer agent type from explicit metadata or legacy checkpoint keys."""
    agent_type = checkpoint.get("agent_type")
    if agent_type:
        return agent_type
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict) and metadata.get("agent_type"):
        return metadata["agent_type"]
    if "history_encoder" in checkpoint and "opponent_model" in checkpoint:
        return AGENT_TYPE_OPPONENT_MODELING
    return AGENT_TYPE_STANDARD


def checkpoint_uses_opponent_modeling_state(checkpoint: dict) -> bool:
    """Return whether an already-loaded checkpoint is for the OM agent."""
    return infer_agent_type(checkpoint) == AGENT_TYPE_OPPONENT_MODELING


def validate_checkpoint_compatibility(
    checkpoint: dict,
    *,
    expected_agent_type: str = None,
    expected_num_players: int = None,
):
    """Raise a clear error when checkpoint metadata conflicts with expectations."""
    agent_type = infer_agent_type(checkpoint)
    if expected_agent_type and agent_type != expected_agent_type:
        raise ValueError(
            f"Checkpoint agent_type '{agent_type}' is not compatible with "
            f"expected '{expected_agent_type}'."
        )

    num_players = checkpoint.get("num_players")
    metadata = checkpoint.get("metadata")
    if num_players is None and isinstance(metadata, dict):
        num_players = metadata.get("num_players")
    if expected_num_players is not None and num_players is not None and num_players != expected_num_players:
        raise ValueError(
            f"Checkpoint num_players '{num_players}' is not compatible with "
            f"expected '{expected_num_players}'."
        )

    return True
