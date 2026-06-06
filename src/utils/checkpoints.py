"""Shared checkpoint discovery helpers for training and evaluation workflows."""

from pathlib import Path

import torch


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
