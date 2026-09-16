"""Archived Hold'em averages with a private per-hand sampler for every seat."""

from hashlib import sha256
from pathlib import Path

from src.arena.catalog import Checkpoint
from src.holdem.checkpoint import load_policy
from src.holdem.records import unpack


class FrozenAverage:
    def __init__(self, spec: Checkpoint, path: Path):
        self.spec = spec
        self.average, metadata = load_policy(path, spec.sha256)
        self.data = path.read_bytes()
        if sha256(self.data).hexdigest() != spec.sha256:
            raise ValueError("Snapshot changed while loading")
        table = unpack(metadata["table"])
        if table.capacity != len(table.stacks):
            raise ValueError("Arena snapshots require a complete physical-seat layout")
        self.players = table.capacity
        self.description = {
            "kind": spec.format,
            "weights_sha256": spec.sha256,
            "num_players": self.players,
            "iteration": metadata["iterations"],
            "training_seed": metadata["training_seed"],
            "source_revision": metadata["manifest"].get("revision"),
            "sampling": "linear-snapshot-per-hand-private-rng-v1",
        }

    def policy(self, seed: int):
        return self.average.player(seed)
