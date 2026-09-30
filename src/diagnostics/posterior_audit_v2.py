"""Frozen posterior-audit mathematics and durable deterministic row storage.

This is a diagnostic executor, not an inference policy or a trainer change.
Zero empirical evidence is unusable evidence, never a smoothed/prior posterior.
"""

import json
import os
from hashlib import sha256
from math import isfinite, log
from pathlib import Path

from src.arena.schedule import canonical, stream_seed

SELECTION_DIGEST = "578b8b67f21a15d2961e88cf498aeeb64831c9827e578e5ce1db62f85d43e322"
MAIN_ROOT = 202610050126
STABILITY_ROOTS = (202610050127, 202610050128, 202610050129)
HIGHER_ROOT = 202610050130
WORLD_ROOT = 202610050131


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DurableRows:
    """Append immutable ID rows, with fsync and an atomic hash index on commits.

    An interrupted/torn tail stays in its original segment. Recovery opens a
    fresh segment and skips every valid persisted ID, including failed rows.
    A failed completed row cannot become a retry with a favorable outcome.
    """
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = self.directory / "identity.json"
        if metadata.exists():
            if json.loads(metadata.read_text()) != identity:
                raise ValueError("Durable shard identity changed")
        else:
            atomic_json(metadata, identity)
        self.rows = {}
        self.recovery = []
        segments = sorted(self.directory.glob("segment-*.jsonl"))
        index_path = self.directory / "index.json"
        previous_index = json.loads(index_path.read_text()) if index_path.exists() else None
        for path in segments:
            data = path.read_bytes()
            if previous_index and path.name in previous_index["segments"]:
                seal = previous_index["segments"][path.name]
                if len(data) < seal["bytes"] or sha256(data[:seal["bytes"]]).hexdigest() != seal["sha256"]:
                    raise ValueError("Committed durable shard prefix changed")
            for number, line in enumerate(data.splitlines(keepends=True), 1):
                if not line.endswith(b"\n"):
                    self.recovery.append({"segment": path.name, "line": number,
                                          "status": "retained_uncommitted_torn_tail"})
                    break
                row = json.loads(line)
                if row["id"] in self.rows:
                    raise ValueError("Duplicate deterministic substream ID")
                self.rows[row["id"]] = row
        self.path = self.directory / f"segment-{len(segments):04d}.jsonl"
        self.handle = self.path.open("x")
        self.new_rows = 0

    def add(self, row):
        if row["id"] in self.rows:
            raise ValueError("Completed substream must not be rerun")
        self.handle.write(canonical(row) + "\n")
        self.handle.flush()
        self.rows[row["id"]] = row
        self.new_rows += 1

    def checkpoint(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())
        segments = {}
        for path in sorted(self.directory.glob("segment-*.jsonl")):
            data = path.read_bytes()
            segments[path.name] = {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
        atomic_json(self.directory / "index.json", {
            "completed_ids": len(self.rows), "segments": segments,
            "recovery": self.recovery, "failed_ids": [key for key, row in self.rows.items()
                                                      if row.get("status") == "failed"],
        })

    def sync(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self):
        self.checkpoint()
        self.handle.close()


def likelihood_seed(root, rank, event_index, pair, sample):
    return stream_seed(root, "test", "opponent", "reverse-lbr", rank,
                       event_index, tuple(pair), sample)


def posterior_from_counts(holdings, events, samples):
    """Consume *all* holding counts at every event, even after a finite zero."""
    if samples < 1 or not holdings:
        raise ValueError("Positive sample count and compatible holdings required")
    n = len(holdings)
    weights = [1 / n] * n
    zeros = []
    limited = 0
    for event in events:
        counts = event["counts"]
        if len(counts) != n or any(type(c) is not int or not 0 <= c <= samples for c in counts):
            raise ValueError("Incomplete/invalid raw holding counts")
        limited += event["limited_samples"]
        updated = [w * c / samples for w, c in zip(weights, counts)]
        mass = sum(updated)
        if mass == 0:
            zeros.append(event["public_event_index"])
            weights = [0.0] * n
        else:
            weights = [w / mass for w in updated]
    positive = [w for w in weights if w > 0]
    return {
        "holdings": [list(pair) for pair in holdings], "weights": weights,
        "positive_support": sum(w > 0 for w in weights),
        "zero_support_indices": [i for i, w in enumerate(weights) if w == 0],
        "ess": 1 / sum(w * w for w in weights) if positive else None,
        "entropy_nats": -sum(w * log(w) for w in positive) if positive else None,
        "maximum_weight": max(weights), "normalization": sum(weights),
        "zero_evidence_events": zeros, "limited_samples": limited,
        "samples_per_holding_action": samples,
        "status": "usable" if not zeros and not limited else "unusable",
    }


def stability_gate(four_estimates, higher):
    failures = []
    all_estimates = [*four_estimates, higher]
    if any(p["holdings"] != higher["holdings"] for p in all_estimates):
        raise ValueError("Posterior supports are not aligned")
    for i, p in enumerate(all_estimates):
        if p["zero_evidence_events"]:
            failures.append(f"estimate-{i}: zero evidence")
        if p["limited_samples"]:
            failures.append(f"estimate-{i}: limited LBR sample")
        if not p["ess"] or abs(p["normalization"] - 1) > 1e-10:
            failures.append(f"estimate-{i}: unusable normalization")
    tv = lambda a, b: .5 * sum(abs(x - y) for x, y in zip(a["weights"], b["weights"]))
    pairwise = [{"a": i, "b": j, "tv": tv(a, b)}
                for i, a in enumerate(four_estimates)
                for j, b in enumerate(four_estimates) if j > i]
    comparisons = []
    for i, p in enumerate(four_estimates):
        distance = tv(p, higher)
        ratio = p["ess"] / higher["ess"] if p["ess"] and higher["ess"] else None
        zero_mass = sum(w for w, small in zip(higher["weights"], p["weights"]) if small == 0)
        comparisons.append({"four_index": i, "tv_vs_16": distance,
                            "ess_ratio": ratio, "higher_mass_on_four_zero_support": zero_mass})
        if distance > .15:
            failures.append(f"four-{i}: TV versus 16 > 0.15")
        if ratio is None or not .5 <= ratio <= 2:
            failures.append(f"four-{i}: ESS ratio outside [0.5, 2]")
        if zero_mass > .10:
            failures.append(f"four-{i}: higher mass on zero support > 0.10")
    for row in pairwise:
        if row["tv"] > .20:
            failures.append(f"four-{row['a']}/four-{row['b']}: TV > 0.20")
    return {"passed": not failures, "failures": failures,
            "pairwise_four_tv": pairwise, "four_vs_higher": comparisons}


def holding_from_uniform(holdings, weights, uniform):
    if not 0 <= uniform < 1 or len(holdings) != len(weights) or not holdings:
        raise ValueError("Invalid holding CDF input")
    if any(not isfinite(w) or w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-10:
        raise ValueError("Conditional world needs a normalized usable posterior")
    total = 0.0
    for pair, weight in zip(holdings, weights):
        total += weight
        if uniform < total:
            return tuple(pair)
    return tuple(holdings[max(i for i, w in enumerate(weights) if w > 0)])
