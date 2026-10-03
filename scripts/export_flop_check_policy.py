"""Export a selected root's native lines, descriptors and frozen policy tables."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from scripts.prepare_flop_check import load_policy
from scripts.preflight_flop_check import prepare_guarded
from scripts.select_flop_check_spots import replay_root
from src.diagnostics.flop_check import atomic_json, compile_tree, export_descriptors, export_policy_tables
from src.diagnostics.flop_check_equity import build_equity_features
from src.diagnostics.flop_check_runtime import machine_snapshot
from src.diagnostics.saved_hu20 import file_hash


def export(record, spec, inputs, out, *, equity=False, check=lambda: None):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    root = replay_root(record); request, histories = compile_tree(root)
    del histories; check()
    atomic_json(out / "native-tree.json", request)
    arrays = out / "descriptors.npz"
    descriptors = export_descriptors(tuple(record["board"]), arrays, check)
    with np.load(arrays) as data:
        codes = data["codes"]
        by_street = {"flop": set(np.unique(codes[:1])), "turn": set(np.unique(codes[1:50])),
                     "river": set(np.unique(codes[50:]))}
        by_street = {s: {int(c) for c in values if c != 255} for s, values in by_street.items()}
    source = load_policy(spec, inputs); check()
    policy = export_policy_tables(request, source, by_street)
    atomic_json(out / "policy-tables.json", policy)
    del source; check()
    features = build_equity_features(record["board"], out / "equity.npz", check=check) if equity else None
    manifest = {"spot": record["spot"], "policy": spec, "descriptors": descriptors,
                "equity_features": features,
                "files": {p.name: file_hash(p) for p in out.iterdir() if p.is_file()}}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spots", "plan", "inputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--spot", required=True)
    parser.add_argument("--policy-index", type=int, default=0)
    parser.add_argument("--equity", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        watch = args.out.with_name(args.out.name + ".watchdog")
        watch.mkdir(parents=True, exist_ok=False)
        before = machine_snapshot(); atomic_json(watch / "machine-before.json", before)
        budget = min(6 * 1024**3, int(before["reclaimable_bytes"] * 0.8 / 1024**3) * 1024**3)
        if budget < 2 * 1024**3:
            raise MemoryError("Insufficient headroom for descriptor/policy export")
        atomic_json(watch / "admission.json", {"budget_bytes": budget,
                    "swap_baseline_bytes": before["swap_used_bytes"]})
        prepare_guarded([sys.executable, "-m", "scripts.export_flop_check_policy",
                         *sys.argv[1:], "--worker"], watch, budget, before["swap_used_bytes"])
        return
    records = json.loads(args.spots.read_text())["roots"]
    record = next(r for r in records if r["spot"] == args.spot)
    spec = json.loads(args.plan.read_text())["policies"][args.policy_index]
    export(record, spec, args.inputs, args.out, equity=args.equity)


if __name__ == "__main__":
    main()
