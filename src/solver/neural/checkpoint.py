"""Complete training snapshots at completed iteration boundaries."""

import os
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from src.solver.experiment import canonical
from src.solver.neural.network import new_network
from src.solver.neural.solver import Config, DeepCFR
from src.solver.tree import GameTree

FORMAT = "small-game-deep-cfr-training-v1"


def catalog_hash(solver: DeepCFR) -> str:
    return sha256(
        canonical([asdict(info) for info in solver.tree.information_sets]).encode()
        + solver.features.numpy().tobytes()
        + solver.mask.numpy().tobytes()
    ).hexdigest()


def atomic_write(path: Path, data: bytes) -> str:
    """Publish a complete immutable snapshot, without replacing an existing file."""
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        name = temporary.name
        try:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.link(name, path)
        finally:
            os.unlink(name)
    return sha256(data).hexdigest()


def state_dict(solver: DeepCFR) -> dict:
    if (
        solver.failed
        or solver.iterations < 1
        or len(solver.fits) != 2 * solver.iterations
    ):
        raise ValueError("Training snapshots require a completed valid iteration")
    return {
        "game": solver.tree.game,
        "config": asdict(solver.config),
        "iterations": solver.iterations,
        "catalog_sha256": catalog_hash(solver),
        "advantages": [net.state_dict() for net in solver.advantages],
        "strategy": None if solver.strategy is None else solver.strategy.state_dict(),
        "traversal_random": solver.traversal_random.getstate(),
        "memories": [
            {
                "seen": memory.seen,
                "random": memory.random.getstate(),
                **{
                    field: torch.from_numpy(
                        getattr(memory, field)[: memory.size].copy()
                    )
                    for field in ("infos", "iterations", "targets")
                },
            }
            for memory in solver.advantage_memories + [solver.strategy_memory]
        ],
        "played_strategy_sum": torch.from_numpy(solver.played_strategy_sum.copy()),
        "fits": solver.fits,
    }


def tensor(value, shape, dtype, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or tuple(value.shape) != tuple(shape)
        or value.dtype != dtype
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"Invalid checkpoint tensor: {name}")
    return value.numpy().copy()


def restore(data: dict) -> DeepCFR:
    config = Config(**data["config"])
    solver = DeepCFR(GameTree(data["game"]), config)
    iteration = data["iterations"]
    if type(iteration) is not int or iteration < 1:
        raise ValueError("Invalid checkpoint iteration")
    if data["catalog_sha256"] != catalog_hash(solver):
        raise ValueError("Checkpoint public information catalog differs")

    def network(weights, hidden):
        if not isinstance(weights, dict) or any(
            not isinstance(t, torch.Tensor)
            or t.dtype != torch.float32
            or not torch.isfinite(t).all()
            for t in weights.values()
        ):
            raise ValueError("Invalid checkpoint network weights")
        net = new_network(hidden, 0)
        net.load_state_dict(weights, strict=True)
        return net.eval().requires_grad_(False)

    if len(data["advantages"]) != 2 or len(data["memories"]) != 3:
        raise ValueError(
            "Checkpoint must contain two advantage networks and three memories"
        )
    solver.advantages = [
        network(weights, config.hidden) for weights in data["advantages"]
    ]
    solver.strategy = (
        None
        if data["strategy"] is None
        else network(data["strategy"], config.strategy_width)
    )
    solver.traversal_random.setstate(data["traversal_random"])
    for owner, (memory, saved) in enumerate(
        zip(solver.advantage_memories + [solver.strategy_memory], data["memories"])
    ):
        seen = saved["seen"]
        if type(seen) is not int or seen < 1:
            raise ValueError("Invalid checkpoint memory count")
        size = min(seen, config.capacity)
        ids = tensor(saved["infos"], (size,), torch.int64, "infos")
        times = tensor(saved["iterations"], (size,), torch.int64, "iterations")
        targets = tensor(saved["targets"], (size, 3), torch.float32, "targets")
        if (
            (ids < 0).any()
            or (ids >= len(solver.features)).any()
            or (times < 1).any()
            or (times > iteration).any()
        ):
            raise ValueError("Invalid checkpoint replay indices or iterations")
        if owner < 2 and (solver.tree.owners[ids] != owner).any():
            raise ValueError("Advantage memory contains another player's samples")
        legal = solver.mask.numpy()[ids]
        if (targets[~legal] != 0).any():
            raise ValueError("Checkpoint replay contains illegal-action targets")
        if owner == 2 and (
            (targets < 0).any()
            or not np.allclose(targets.sum(axis=1), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError("Invalid checkpoint strategy probabilities")
        memory.infos[:size], memory.iterations[:size], memory.targets[:size] = (
            ids,
            times,
            targets,
        )
        memory.size, memory.seen = size, seen
        memory.random.setstate(saved["random"])
    played = tensor(
        data["played_strategy_sum"], solver.mask.shape, torch.float64, "played average"
    )
    if (played < 0).any() or (played[~solver.mask.numpy()] != 0).any():
        raise ValueError("Invalid checkpoint diagnostic average")
    fits = data["fits"]
    if len(fits) != 2 * iteration or any(
        row["iteration"] != index // 2 + 1 or row["player"] != index % 2
        for index, row in enumerate(fits)
    ):
        raise ValueError("Checkpoint fit history does not match its iteration")
    canonical(fits)
    solver.played_strategy_sum, solver.fits, solver.iterations = played, fits, iteration
    return solver


def save_training(
    solver: DeepCFR, path: Path, *, manifest: dict, progress: dict
) -> str:
    payload = {
        "format": FORMAT,
        "manifest": manifest,
        "progress": progress,
        "solver": state_dict(solver),
    }
    buffer = BytesIO()
    torch.save(payload, buffer)
    return atomic_write(path, buffer.getvalue())


def load_training(path: Path, digest: str, *, manifest: dict) -> tuple[DeepCFR, dict]:
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("Training checkpoint hash mismatch")
    payload = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Unsupported training checkpoint")
    for field in ("version", "plan_sha256", "source_sha256", "environment", "protocol"):
        if canonical(payload["manifest"][field]) != canonical(manifest[field]):
            raise ValueError(f"Training checkpoint mismatch: {field}")
    return restore(payload["solver"]), payload["progress"]
