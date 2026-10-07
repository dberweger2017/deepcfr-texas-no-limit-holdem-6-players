"""Pinned v0.4.1 opponent-sampled average; verification never publishes."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

NAME = "v0.4.1 · O1B · seed 2026100601"
ASSET_NAME = "O1B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz"
MODEL_BYTES = 142_677_367
MODEL_SHA256 = "571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d"
CHECKPOINT_SHA256 = "4ee91d3977d98c0b6b462ed310396c6a7cbba48ef51716bb0fdbf6609dfbc825"
SEED = 2026100601
ITERATION = 2_126_271


@dataclass(frozen=True, slots=True)
class _Identity:
    sha256: str


def verify(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("O model must be a regular file")
    if path.stat().st_size != MODEL_BYTES:
        raise ValueError("O model byte count differs")
    with path.open("rb") as source:
        value = hashlib.file_digest(source, "sha256").hexdigest()
    if value != MODEL_SHA256:
        raise ValueError("O model SHA-256 differs")
    return value


def load_policy(path: Path):
    from src.blueprint.average import AveragePolicy, EXTRACTIONS, FORMAT

    verify(path)
    # Reuse the exact arena reader and observation/menu inference, including
    # uniform zero-mass and missing-key behavior; never re-extract the export.
    policy = AveragePolicy(path, MODEL_SHA256)
    description = policy.description
    if (description["training_seed"] != SEED
            or description["iteration"] != ITERATION
            or description["source_checkpoint_sha256"] != CHECKPOINT_SHA256
            or description["strategy"] != EXTRACTIONS["opponent-sampled"]):
        raise ValueError("O model lineage or extraction differs")
    policy.spec = _Identity(MODEL_SHA256)
    policy.name = NAME
    policy.format_id = FORMAT
    return policy
