"""Pinned v0.4.0 current policy and its artifact verification."""

import hashlib
from pathlib import Path

EXPECTED_NAME = "B100M-HU20-current-seed-2026093001.json.gz"
EXPECTED_BYTES = 40_144_034
EXPECTED_SHA256 = "4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf"


def verify(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Model must be a regular file")
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError("Model byte count does not match v0.4")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    if value != EXPECTED_SHA256:
        raise ValueError("Model SHA-256 does not match v0.4")
    return value


def load_policy(path: Path):
    from src.arena.catalog import Checkpoint
    from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
    from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT
    from src.blueprint.solver import HU20_UNCAPPED_GAME

    spec = Checkpoint("B100M", str(path), EXPECTED_SHA256, HU20_UNCAPPED_FORMAT)
    source = FrozenBlueprint(spec, path)
    if (source.game != HU20_UNCAPPED_GAME or source.abstraction != HU20_UNCAPPED_SCHEMA
            or source.players != 2 or source.raise_cap is not None
            or source.description["strategy"] != "current"
            or source.description["iteration"] < 1):
        raise ValueError("Artifact is not the pinned current native-reopening HU20 policy")
    return source
