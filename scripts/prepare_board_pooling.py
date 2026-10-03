"""Export frozen roots and a shared equity codebook; production host only."""

import argparse
import json
from pathlib import Path
import signal
import numpy as np

from src.blueprint.hu20_river import public_ranges
from src.diagnostics.board_pooling import public_line
from src.diagnostics.board_pooling_policy import DiskAverage, build_index
from src.diagnostics.board_pooling_features import card_features, shared_codebook, add_pool_keys
from src.diagnostics.flop_check import atomic_json, compile_tree, export_policy_tables, gate_k
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.turn_check import replay_root


def prepare(plan_path, inputs, out, *, memory_bytes=5 * 1024**3):
    plan = json.loads(plan_path.read_text()); corpus_path = Path(plan["corpus"]["path"])
    if file_hash(corpus_path) != plan["corpus"]["sha256"]:
        raise ValueError("Frozen corpus hash differs")
    corpus = json.loads(corpus_path.read_text()); roots = corpus["roots"]
    if len(roots) != plan["boards"] or len({r["spot"] for r in roots}) != len(roots):
        raise ValueError("Frozen corpus count/uniqueness differs")
    out.mkdir(parents=True, exist_ok=False)
    compilations = []
    for record in roots:
        root = replay_root(record)
        if public_line(root, record["button"]) != corpus["chosen_line"]:
            raise ValueError("Frozen root belongs to another public line")
        compilations.append(compile_tree(root))
    key_gate = gate_k(compilations)
    atomic_json(out / "gate-k.json", key_gate)
    if not key_gate["passed"]:
        raise ValueError("Gate K failed")
    features, codebook = shared_codebook([card_features(r["board"], plan["equity_histogram_bins"])
                                         for r in roots], k=plan["equity_k"], seed=plan["equity_cluster_seed"])
    atomic_json(out / "codebook.json", codebook)
    jobs = []; exclusions = []
    for policy_index, spec in enumerate(plan["policies"]):
        index = out / f"policy-{policy_index}.sqlite"
        inventory = build_index(spec, inputs, index)
        atomic_json(out / f"policy-{policy_index}-index.json", inventory)
        source = DiskAverage(index, spec)
        for record, (request, _), feature in zip(roots, compilations, features, strict=True):
            job = f'{record["spot"]}-{policy_index}'
            leaf = out / "jobs" / job; leaf.mkdir(parents=True)
            ranges, coverage = public_ranges(source, replay_root(record))
            selected = [[{"hand": list(h), "weight": w} for h, w in ranges[s] if w > 0]
                        for s in request["seat_map"]]
            if any(not r for r in selected):
                exclusions.append({"job": job, "spot": record["spot"], "policy_index": policy_index,
                                   "reason": "zero policy support"})
                continue
            codes = np.asarray(feature["codes"])
            by_street = {"turn": set(map(int, np.unique(codes[:1]))) - {255},
                         "river": set(map(int, np.unique(codes[1:]))) - {255}}
            tables = export_policy_tables(request, source, by_street)
            data = dict(feature, **tables, pool_keys=add_pool_keys(request, feature, tables))
            atomic_json(leaf / "compact.json", data)
            exported = dict(request, mode="solve", compress=True, ranges=selected,
                            range_coverage=coverage, policy=spec, pooling_phase="collect",
                            compact_path=str((leaf / "compact.json").resolve()),
                            memory_budget_bytes=memory_bytes, max_iterations=plan["maximum_iterations"],
                            progress_every=plan["progress_every"], target_pct_pot=plan["target_pct_pot"],
                            seconds=plan["per_solve_seconds"])
            atomic_json(leaf / "request.json", exported)
            jobs.append({"job": job, "spot": record["spot"], "lineage": spec["seed"],
                         "policy_index": policy_index, "board_weight": record["board_weight"],
                         "request": str((leaf / "request.json").resolve()),
                         "request_sha256": file_hash(leaf / "request.json"),
                         "compact_sha256": file_hash(leaf / "compact.json")})
        source.db.close(); source.get.cache_clear()
    manifest = {"jobs": jobs, "support_exclusions": exclusions, "plan_sha256": file_hash(plan_path),
                "codebook_sha256": file_hash(out / "codebook.json"), "gate_k": key_gate}
    atomic_json(out / "manifest.json", manifest)
    return {"jobs": len(jobs), "support_exclusions": len(exclusions)}


def main():
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Owned exporter stopped")))
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "inputs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--memory-gib", type=float, default=5)
    a = p.parse_args()
    print(json.dumps(prepare(a.plan, a.inputs, a.out, memory_bytes=int(a.memory_gib * 1024**3))))


if __name__ == "__main__":
    main()
