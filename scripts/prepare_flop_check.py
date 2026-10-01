"""Prepare hash-verified native requests for the external flop diagnostic."""

import argparse
from collections import Counter
import json
from pathlib import Path

from src.arena.catalog import Checkpoint
from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT
from src.blueprint.hu20_river import public_ranges
from src.diagnostics.cfr_average import DiagnosticAverage
from src.diagnostics.flop_check import (atomic_json, compile_tree, fixture_root,
                                        gate_k, VERSION)
from src.diagnostics.saved_hu20 import file_hash
from src.game.types import Street


def load_policy(spec, inputs):
    path = Path(inputs) / spec["path"]
    if file_hash(path) != spec["sha256"]:
        raise ValueError("Policy SHA-256 differs before use")
    if spec["strategy"] == "current":
        policy = FrozenBlueprint(Checkpoint(spec["name"], str(path), spec["sha256"],
                                           HU20_UNCAPPED_FORMAT), path)
    elif spec["strategy"] == "stored-average":
        policy = DiagnosticAverage(path, spec["sha256"])
    else:
        raise ValueError("Unknown frozen extraction")
    if policy.description["training_seed"] != spec["seed"]:
        raise ValueError("Policy lineage differs")
    return policy


def prepare(spec, inputs, out, *, memory_bytes, cap=None, max_nodes=300_000):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    source = load_policy(spec, inputs)
    records = []
    for kind in ("limped", "min-raised", "3-bet"):
        root = fixture_root(kind)
        ranges, coverage = public_ranges(source, root)
        try:
            request, _ = compile_tree(root, raise_cap=cap, max_nodes=max_nodes)
            request.update(mode="estimate", memory_budget_bytes=memory_bytes,
                           ranges=[[{"hand": list(h), "weight": w} for h, w in ranges[s] if w > 0]
                                   for s in request["seat_map"]],
                           policy=spec, range_coverage=coverage)
            path = out / (kind + ".json"); atomic_json(path, request)
            records.append({"kind": kind, "status": "prepared", "request": path.name,
                            "request_sha256": file_hash(path), "nodes": len(request["nodes"]),
                            "by_street": dict(Counter(n.get("street", "terminal") for n in request["nodes"])),
                            "positive_holdings": [len(r) for r in request["ranges"]], "raise_cap": cap})
        except (MemoryError, TimeoutError) as exc:
            records.append({"kind": kind, "status": "compilation-oversize",
                            "failure": str(exc), "raise_cap": cap, "max_nodes": max_nodes})
        atomic_json(out / "manifest.json", {"format": VERSION, "policy": spec,
                    "memory_budget_bytes": memory_bytes, "records": records})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate-k", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--raise-cap", type=int)
    parser.add_argument("--policy-index", type=int, default=0)
    args = parser.parse_args()
    if args.gate_k:
        compilations = [compile_tree(fixture_root(kind, street=street),
                                    raise_cap=3, max_nodes=30_000)
                        for kind in ("limped", "min-raised", "3-bet")
                        for street in (Street.FLOP, Street.TURN, Street.RIVER)]
        result = gate_k(compilations)
        atomic_json(args.out, result)
        if not result["passed"]:
            raise SystemExit("Gate K failed")
    else:
        if not args.plan or not args.inputs:
            parser.error("Real requests require --plan and --inputs")
        specs = json.loads(args.plan.read_text())["policies"]
        prepare(specs[args.policy_index], args.inputs, args.out,
                memory_bytes=int(args.memory_gib * 1024**3), cap=args.raise_cap)


if __name__ == "__main__":
    main()
