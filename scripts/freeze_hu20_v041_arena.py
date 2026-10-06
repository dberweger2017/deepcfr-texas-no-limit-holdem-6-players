"""Freeze a v0.4.1 arena plan: every policy pinned by bytes, hash, lineage and iteration.

Panels, rules and contracts are #141's. `--blocks-lbr/--blocks-pressure/--blocks-other` set the
counts; the timing plan uses small counts and its own deal root, the final plan the frozen ones.
"""

import argparse
import gzip
import json
from pathlib import Path
from time import time

from src.arena.schedule import digest
from src.diagnostics.saved_hu20 import file_hash

ARMS = {"R": ("current", "R-{seed}.current.json.gz"), "O": ("average", "O-{seed}.average.jsonl.gz"),
        "C": ("current", "C-{seed}.current.json.gz"), "T": ("average", "T-{seed}.average.jsonl.gz")}
REFERENCE_SEEDS = (2026093001, 2026093002, 2026093003)
NATIVE_SEEDS = (2026100601, 2026100602, 2026100603)


def header(path, strategy):
    with gzip.open(path, "rt") as stream:
        first = stream.readline() if strategy == "average" else stream.read()
    document = json.loads(first)
    return document["checkpoint_header"] if strategy == "average" else document


def model(policies, arm, seed):
    strategy, pattern = ARMS[arm]
    path = policies / pattern.format(seed=seed)
    spec = {"name": f"{arm}-{seed}", "arm": arm, "seed": seed, "strategy": strategy, "path": path.name,
            "bytes": path.stat().st_size, "sha256": file_hash(path), "iteration": header(path, strategy)["iteration"]}
    if strategy == "average":
        with gzip.open(path, "rt") as stream:
            metadata = json.loads(stream.readline())
        spec.update(checkpoint_sha256=metadata["source_checkpoint_sha256"], extraction=metadata["extraction"])
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policies", type=Path, required=True)
    p.add_argument("--panels-from", type=Path, default=Path("configs/diagnostics/b500-cfr-average-comparison.json"))
    p.add_argument("--stage", choices=("timing", "frozen-final"), required=True)
    p.add_argument("--root", type=int, required=True)
    p.add_argument("--blocks-lbr", type=int, required=True)
    p.add_argument("--blocks-pressure", type=int, required=True)
    p.add_argument("--blocks-other", type=int, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    p.add_argument("--approval", default="https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/165#issuecomment-5994292073")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("Never overwrite a frozen plan")
    panels = []
    for panel in json.loads(a.panels_from.read_text())["panels"]:
        blocks = {"lbr": a.blocks_lbr, "native-pressure": a.blocks_pressure}.get(panel["name"], a.blocks_other)
        panels.append(dict(panel, blocks=blocks))
    models = [model(a.policies, arm, seed) for arm in ARMS
              for seed in (REFERENCE_SEEDS if arm == "R" else NATIVE_SEEDS)]
    plan = {"stage": a.stage, "root": a.root, "panels": panels, "models": models, "max_seconds": a.max_seconds,
            "started_at": time(), "expected_hands": len(models) * 2 * sum(p["blocks"] for p in panels),
            "approval": a.approval}
    a.out.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"sha256": digest(plan), "hands": plan["expected_hands"], "stage": a.stage}))


if __name__ == "__main__":
    main()
