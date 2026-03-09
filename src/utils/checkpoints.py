"""Shared checkpoint discovery helpers for training and evaluation workflows."""

from pathlib import Path


_GLOB_CHARS = "*?[]"


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
