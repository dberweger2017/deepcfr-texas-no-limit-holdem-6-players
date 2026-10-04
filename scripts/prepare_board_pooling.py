"""Export frozen roots and a shared equity codebook; production host only."""

import argparse
import json
import os
from pathlib import Path
import signal
from time import time
import numpy as np

from src.blueprint.hu20_river import public_ranges
from src.diagnostics.board_pooling import public_line
from src.diagnostics.board_pooling_policy import DiskAverage, build_index
from src.diagnostics.board_pooling_features import card_features, crossfit_codebooks, add_pool_keys
from src.diagnostics.flop_check import atomic_json, compile_tree, export_policy_tables, gate_k
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.turn_check import replay_root


def recovery_files(prior, manifest, expected_sha256):
    if file_hash(manifest) != expected_sha256:
        raise ValueError("Recovery inventory fingerprint differs")
    prior = prior.resolve()
    base = manifest.resolve().parent
    prior.relative_to(base)
    inventory = {r["path"]: r for r in json.loads(manifest.read_text())["files"]}
    def verified(name):
        path = prior / name
        row = inventory[str(path.relative_to(base))]
        if path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise ValueError(f"Recovery artifact differs: {path}")
        return path
    return verified


def prepare(plan_path, inputs, out, *, memory_bytes=5 * 1024**3,
            reuse_prepared=None, reuse_manifest=None, reuse_manifest_sha256=None):
    plan = json.loads(plan_path.read_text())
    # Shared working aliases can be archived while this campaign is idle.
    # Refuse a missing or changed lineage before expensive feature exports.
    for spec in plan["policies"]:
        source = inputs / spec["path"]
        if file_hash(source) != spec["sha256"]:
            raise ValueError(f"Average export hash differs: {source}")
    corpus_path = Path(plan["corpus"]["path"])
    if file_hash(corpus_path) != plan["corpus"]["sha256"]:
        raise ValueError("Frozen corpus hash differs")
    corpus = json.loads(corpus_path.read_text()); roots = corpus["roots"]
    split_path = Path(plan["crossfit"]["path"])
    if file_hash(split_path) != plan["crossfit"]["sha256"]:
        raise ValueError("Frozen crossfit split hash differs")
    split = json.loads(split_path.read_text())
    if len(roots) != plan["boards"] or len({r["spot"] for r in roots}) != len(roots):
        raise ValueError("Frozen corpus count/uniqueness differs")
    reuse = None
    if any(v is not None for v in (reuse_prepared, reuse_manifest, reuse_manifest_sha256)):
        if not all(v is not None for v in (reuse_prepared, reuse_manifest, reuse_manifest_sha256)):
            raise ValueError("Recovery requires the prior directory and pinned inventory")
        reuse = recovery_files(reuse_prepared, reuse_manifest, reuse_manifest_sha256)
        old_gate = json.loads(reuse("gate-k.json").read_text())
        if not old_gate["passed"] or old_gate["samples"] < 100000:
            raise ValueError("Recovery key gate did not pass")
    out.mkdir(parents=True, exist_ok=False)
    def status(stage, done, total):
        atomic_json(out / "status.json", {"stage": stage, "done": done, "total": total, "timestamp": time()})
    compilations = []
    status("tree-export", 0, len(roots))
    for record in roots:
        root = replay_root(record)
        if public_line(root, record["button"]) != corpus["chosen_line"]:
            raise ValueError("Frozen root belongs to another public line")
        compilations.append(compile_tree(root))
        status("tree-export", len(compilations), len(roots))
    key_gate = gate_k(compilations)
    atomic_json(out / "gate-k.json", key_gate)
    if not key_gate["passed"]:
        raise ValueError("Gate K failed")
    raw_features = []
    for record in roots:
        status("card-features", len(raw_features), len(roots))
        raw_features.append(card_features(record["board"], plan["equity_histogram_bins"]))
    status("frozen-codebooks", 0, 3)
    features, codebook = crossfit_codebooks(raw_features, [split["folds"][r["spot"]] for r in roots],
                                         k=plan["equity_k"], seed=plan["equity_cluster_seed"])
    atomic_json(out / "codebook.json", codebook)
    if reuse and file_hash(out / "codebook.json") != file_hash(reuse("codebook.json")):
        raise ValueError("Regenerated codebook differs from frozen recovery artifacts")
    jobs = []; exclusions = []
    for policy_index, spec in enumerate(plan["policies"]):
        status("policy-index", policy_index, len(plan["policies"]))
        index = out / f"policy-{policy_index}.sqlite"
        if reuse and (reuse_prepared / index.name).exists():
            inventory = json.loads(reuse(f"policy-{policy_index}-index.json").read_text())
            old_index = reuse(index.name)
            if inventory["source_sha256"] != spec["sha256"] or file_hash(old_index) != inventory["index_sha256"]:
                raise ValueError("Recovery index identity differs")
            os.link(old_index, index)
            inventory = dict(inventory, path=str(index))
        else:
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
            exported = dict(request, mode="solve", compress=True, ranges=selected,
                            range_coverage=coverage, policy=spec, pooling_phase="collect",
                            compact_path=str((leaf / "compact.json").resolve()),
                            memory_budget_bytes=memory_bytes, max_iterations=plan["maximum_iterations"],
                            progress_every=plan["progress_every"], target_pct_pot=plan["target_pct_pot"],
                            seconds=plan["per_solve_seconds"])
            previous = reuse_prepared / "jobs" / job if reuse else None
            if previous and (previous / "request.json").exists() and (previous / "compact.json").exists():
                old_request = json.loads(reuse(f"jobs/{job}/request.json").read_text())
                expected = json.loads(json.dumps(dict(exported,
                    compact_path=str((previous / "compact.json").resolve()))))
                if old_request != expected:
                    raise ValueError(f"Recovery request/ranges/menu differ: {job}")
                old_compact = reuse(f"jobs/{job}/compact.json")
                data = json.loads(old_compact.read_text())
                if any(data[k] != v for k, v in feature.items()):
                    raise ValueError(f"Recovery card features differ: {job}")
                # Only immutable, hash-checked files are shared; policy access
                # is read-only and all future outputs use fresh directories.
                os.link(old_compact, leaf / "compact.json")
            else:
                codes = np.asarray(feature["codes"])
                by_street = {"turn": set(map(int, np.unique(codes[:1]))) - {255},
                             "river": set(map(int, np.unique(codes[1:]))) - {255}}
                tables = export_policy_tables(request, source, by_street)
                data = dict(feature, **tables, pool_keys=add_pool_keys(request, feature, tables))
                atomic_json(leaf / "compact.json", data)
            atomic_json(leaf / "request.json", exported)
            status("policy-export", len(jobs)+1, plan["jobs_total"])
            jobs.append({"job": job, "spot": record["spot"], "lineage": spec["seed"],
                         "policy_index": policy_index, "board_weight": record["board_weight"],
                         "evaluation_fold": split["folds"][record["spot"]],
                         "replay_sample": record["spot"] in split["replay_boards"],
                         "request": str((leaf / "request.json").resolve()),
                         "request_sha256": file_hash(leaf / "request.json"),
                         "compact_sha256": file_hash(leaf / "compact.json")})
        source.db.close(); source.get.cache_clear()
    manifest = {"jobs": jobs, "support_exclusions": exclusions, "plan_sha256": file_hash(plan_path),
                "codebook_sha256": file_hash(out / "codebook.json"), "gate_k": key_gate}
    if reuse:
        manifest["recovery"] = {"prior": str(reuse_prepared), "inventory": str(reuse_manifest),
                                "inventory_sha256": reuse_manifest_sha256}
    atomic_json(out / "manifest.json", manifest)
    status("prepared", len(jobs)+len(exclusions), plan["jobs_total"])
    return {"jobs": len(jobs), "support_exclusions": len(exclusions)}


def main():
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Owned exporter stopped")))
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "inputs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--memory-gib", type=float, default=5)
    p.add_argument("--reuse-prepared", type=Path)
    p.add_argument("--reuse-manifest", type=Path)
    p.add_argument("--reuse-manifest-sha256")
    a = p.parse_args()
    print(json.dumps(prepare(a.plan, a.inputs, a.out, memory_bytes=int(a.memory_gib * 1024**3),
                            reuse_prepared=a.reuse_prepared, reuse_manifest=a.reuse_manifest,
                            reuse_manifest_sha256=a.reuse_manifest_sha256)))


if __name__ == "__main__":
    main()
