"""Pinned first-lineage CFR+ average for the Shield browser benchmark."""

from dataclasses import dataclass
from gzip import open as gzip_open
import json
from pathlib import Path

from src.diagnostics.cfr_average import DiagnosticAverage, EXTRACTION, FORMAT

MODEL_SHA256 = "a5e9d0fc6f4a448640f52f508187f779e43a0a8fd41158207f03de82adc219e1"
MODEL_BYTES = 143429027
CHECKPOINT_SHA256 = "1d162266e55d9098c83415f65e487a4f3eef796257b841a380c7b818deebdd55"
MODEL_SEED = 2026100601
MODEL_ITERATION = 2940243


@dataclass(frozen=True, slots=True)
class _Identity:
    sha256: str


class ShieldPolicy(DiagnosticAverage):
    name = "0.4.0-shield · seed 2026100601"
    adapter_id = "shield-traverser-average-v1"
    format_id = FORMAT
    benchmark_only = True

    def __init__(self, path: Path):
        if path.stat().st_size != MODEL_BYTES:
            raise ValueError("Shield artifact size differs")
        with gzip_open(path, "rt") as stream:
            metadata = json.loads(stream.readline())
        header = metadata["checkpoint_header"]
        if (header.get("training_options") != "regret-floor-0"
                or header["config"]["seed"] != MODEL_SEED
                or header["iteration"] != MODEL_ITERATION
                or header.get("average_rule", "traverser-reach") != "traverser-reach"
                or metadata.get("extraction") != EXTRACTION
                or metadata.get("source_checkpoint_sha256") != CHECKPOINT_SHA256):
            raise ValueError("Artifact is not the pinned Shield traverser-reach average")
        # Reuse the arena reader and its exact observation/menu probabilities.
        # Sampling and recovery randomness remain owned by PlayService.
        super().__init__(path, MODEL_SHA256)
        self.spec = _Identity(MODEL_SHA256)
