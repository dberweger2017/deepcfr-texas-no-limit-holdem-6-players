"""Portable diagnostic data, deliberately separate from resumable training state."""

import json
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np

from src.solver.experiment import canonical, write_json
from src.solver.neural.checkpoint import atomic_write, catalog_hash, load_training
from src.solver.neural.experiment import Plan, provenance
from src.solver.neural.memory import Reservoir
from src.solver.neural.solver import Config, DeepCFR
from src.solver.tree import GameTree

FORMAT = "frozen-strategy-replay-v1"


def export_replay(training: Path, output: Path) -> dict:
    original = json.loads((training / "manifest.json").read_text())
    plan = Plan.from_dict(original["plan"])
    descriptor = json.loads((training / "checkpoint.json").read_text())
    name = descriptor["file"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Invalid checkpoint filename")
    solver, progress = load_training(
        training / name, descriptor["sha256"], manifest=provenance(asdict(plan))
    )
    if solver.iterations != plan.iterations or solver.strategy is None:
        raise ValueError("Export requires the completed training plan")
    memory = solver.strategy_memory
    data = BytesIO()
    np.savez_compressed(
        data,
        **{
            field: getattr(memory, field)[: memory.size]
            for field in ("infos", "iterations", "targets")
        },
    )
    output.mkdir(parents=True, exist_ok=False)
    digest = atomic_write(output / "replay.npz", data.getvalue())
    manifest = {
        "format": FORMAT,
        "game": solver.tree.game,
        "seed": solver.config.seed,
        "iteration": solver.iterations,
        "size": memory.size,
        "seen": memory.seen,
        "capacity": memory.capacity,
        "catalog_sha256": catalog_hash(solver),
        "replay_sha256": digest,
        "checkpoint_sha256": descriptor["sha256"],
        "training_manifest": original,
        "original_evaluation": progress["report"]["evaluations"][-1],
        "purpose": "frozen replay diagnosis; not training recovery or model promotion",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_replay(path: Path) -> tuple[DeepCFR, Reservoir, dict]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["format"] != FORMAT:
        raise ValueError("Unsupported replay export")
    for field in ("iteration", "size", "seen", "capacity"):
        if type(manifest[field]) is not int or manifest[field] < 1:
            raise ValueError(f"Invalid replay {field}")
    size, seen, capacity = (manifest[k] for k in ("size", "seen", "capacity"))
    if size != min(seen, capacity) or capacity > 1_000_000:
        raise ValueError("Invalid replay counts")
    solver = DeepCFR(
        GameTree(manifest["game"]), Config(seed=manifest["seed"], capacity=size)
    )
    if manifest["catalog_sha256"] != catalog_hash(solver):
        raise ValueError("Replay public information catalog differs")
    original = manifest["training_manifest"]
    current = provenance(original["plan"])
    # Only the runtime may differ: the interpretation and origin of the data cannot.
    for field in ("source_sha256", "protocol", "plan_sha256"):
        if canonical(original[field]) != canonical(current[field]):
            raise ValueError(f"Replay training source mismatch: {field}")
    plan = Plan.from_dict(original["plan"])
    if (plan.game, plan.training.seed, plan.iterations, plan.training.capacity) != (
        manifest["game"],
        manifest["seed"],
        manifest["iteration"],
        capacity,
    ):
        raise ValueError("Replay metadata differs from the training plan")
    data = (path / "replay.npz").read_bytes()
    if sha256(data).hexdigest() != manifest["replay_sha256"]:
        raise ValueError("Replay hash mismatch")
    with np.load(BytesIO(data), allow_pickle=False) as arrays:
        if set(arrays.files) != {"infos", "iterations", "targets"}:
            raise ValueError("Unexpected replay arrays")
        ids, times, targets = (arrays[k] for k in ("infos", "iterations", "targets"))
    if (
        ids.shape != (size,)
        or ids.dtype != np.int64
        or times.shape != (size,)
        or times.dtype != np.int64
        or targets.shape != (size, 3)
        or targets.dtype != np.float32
        or not np.isfinite(targets).all()
        or (ids < 0).any()
        or (ids >= len(solver.features)).any()
        or (times < 1).any()
        or (times > manifest["iteration"]).any()
    ):
        raise ValueError("Invalid replay arrays")
    if (
        (targets[~solver.mask.numpy()[ids]] != 0).any()
        or (targets < 0).any()
        or not np.allclose(targets.sum(axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid replay strategy probabilities")
    memory = Reservoir(size, 0)
    memory.infos[:], memory.iterations[:], memory.targets[:] = ids, times, targets
    memory.size, memory.seen = size, seen
    return solver, memory, manifest
