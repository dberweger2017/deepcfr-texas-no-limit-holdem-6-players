"""Device resolution for training and inference."""

from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> str:
    """
    Resolve a device string for inference and general use.

    ``auto`` prefers CUDA when available, otherwise CPU. MPS is skipped on
    auto because single-sample poker workloads are faster on CPU for this
    project; pass ``mps`` explicitly to opt in.
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_training_device(device: str = "auto") -> str:
    """Resolve device for CFR training (same policy as ``resolve_device``)."""
    return resolve_device(device)


def checkpoint_map_location(device: str = "auto") -> str:
    """Device string for torch.load(map_location=...)."""
    return resolve_device(device)
