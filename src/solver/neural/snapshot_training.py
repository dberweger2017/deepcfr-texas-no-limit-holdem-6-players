"""Training recovery owns the complete archive; inference exports cannot resume."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from weakref import ref

import torch

from src.solver.experiment import canonical
from src.solver.neural.average import archive_state, restore_archive
from src.solver.neural.checkpoint import atomic_write, restore, state_dict

FORMAT = "small-game-snapshot-training-v1"


def _validate(solver, archive):
    if (
        solver.failed
        or solver.strategy is not None
        or solver.iterations != archive.iterations
        or solver.tree.game != archive.game
        or solver.config.hidden != archive.hidden
        or not archive.iterations
    ):
        raise ValueError(
            "Solver and archive must describe one completed snapshot iteration"
        )
    current = solver.advantages[0].state_dict()
    saved = archive._snapshots[-1][0].state_dict()
    if any(not torch.equal(current[key], saved[key]) for key in current):
        raise ValueError(
            "Archive does not contain the current player-0 collection policy"
        )


def save_training(solver, archive, path: Path, *, manifest: dict, progress: dict):
    _validate(solver, archive)
    if archive._source is None or archive._source() is not solver:
        raise ValueError("Training requires the archive attached to this solver")
    payload = {
        "format": FORMAT,
        "manifest": manifest,
        "progress": progress,
        "solver": state_dict(solver),
        "archive": archive_state(archive),
    }
    buffer = BytesIO()
    torch.save(payload, buffer)
    return atomic_write(path, buffer.getvalue())


def load_training(path: Path, digest: str, *, manifest: dict):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("Snapshot training hash mismatch")
    payload = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(
            "Expected complete snapshot training state, not an inference export"
        )
    for field in ("version", "plan_sha256", "source_sha256", "environment", "protocol"):
        if canonical(payload["manifest"][field]) != canonical(manifest[field]):
            raise ValueError(f"Snapshot training mismatch: {field}")
    solver, archive = restore(payload["solver"]), restore_archive(payload["archive"])
    _validate(solver, archive)
    archive._source = ref(solver)
    return solver, archive, payload["progress"]
