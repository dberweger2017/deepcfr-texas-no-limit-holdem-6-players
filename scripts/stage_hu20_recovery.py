"""Record an M4-local, versioned path mapping without resetting any lineage."""

import argparse
import json
from pathlib import Path
import subprocess

from scripts.hu20_scaling_runtime import validate_inputs
from scripts.train_hu20 import write_json
from src.blueprint.windowed import _hash
from scripts.report_hu20_reopening import verify_phase


def stage(root, original_plan):
    root = root.resolve(); source = Path(__file__).resolve().parents[1]
    original_plan = original_plan.resolve(); old = json.loads(original_plan.read_text())
    clock = json.loads((root/"clock.json").read_text())
    original_root = source/old["root"]
    if any(original_root.glob("evaluation-*")) or any(root.glob("campaign.json")):
        raise ValueError("Inspect existing confirmation/recovery before staging")
    audit = json.loads((source/"docs/reports/hu20-scaling-failure-artifacts/failure-audit.json").read_text())
    retained = {r["seed"]: r for r in audit["lineages"]}
    parents = {}
    for seed in old["training_seeds"]:
        if seed == 2026093001:
            folder = original_root/"training"/f"B-{seed}"
            cp = folder/"checkpoint-40000000.json.gz"
        elif seed == 2026093002:
            folder = root/"staged-m1"/f"B-{seed}"
            cp = folder/"partial-last-completed.json.gz"
        else:
            spec = old["parents"][str(seed)]
            parents[str(seed)] = {**spec, "checkpoint_path": str((source/spec["checkpoint_path"]).resolve())}
            continue
        result = json.loads((folder/"result.json").read_text())
        if verify_phase(folder) == 0:
            raise ValueError("Missing authoritative retained phase seal")
        row = retained[seed]
        if result["completed_nodes"] != row["completed_nodes"] or result["partial_checkpoint_sha256"] != row["partial_sha256"]:
            raise ValueError("Retained lineage audit mismatch")
        parents[str(seed)] = {"seed": seed, "completed_nodes": result["completed_nodes"],
            "entries": result["entries"], "iteration": result["completed_iterations"],
            "checkpoint_path": str(cp), "checkpoint_sha256": row["partial_sha256"],
            "previous_result_path": str(folder/"result.json"), "previous_iterations_path": str(folder/"iterations.jsonl")}
        if seed == 2026093001:
            parents[str(seed)]["recovered_milestones"] = [{"requested_total_nodes": 40000000,
                "checkpoint_path": str(cp), "checkpoint_sha256": row["partial_sha256"],
                "policy_path": str(folder/"current-40000000.json.gz"),
                "policy_sha256": "12d3f1ad67d08b350363ce57665347c2de5edc46fab67e474c5f175d6a5271b7"}]
    plan = {**old, **clock, "schema": "hu20-m4-recovery-v1", "root": str(root),
        "source": str(source), "original_root": str(original_root), "original_plan": str(original_plan),
        "original_parents": old["parents"], "parents": parents,
        "hosts": {"m4": old["hosts"]["m4"]}, "training_assignment": {"m4": old["training_seeds"]},
        "block_host_cycle": ["m4"], "coordinator_models": str(root/"models.json"),
        "independent_path": str(root/"inputs/independent-observations.jsonl.gz"),
        "runtime_inputs": [], "status": "staging", "path_mapping_version": 1,
        "recovery_preflight_root": 2026157001,
        "phase_order": ["validation", "training-fixed-seed-order", "primary-confirmation", "primary-native-audit", "diagnostics-if-reserve", "diagnostic-native-audit", "final-report"],
        "preflight_deadline": min(clock["deadline"]-7200, clock["started"]+5400),
        "report_reserve_seconds": 1800, "audit_reserve_seconds": 1200}
    # These guards are fixed before outcome inspection and refined by timing-only preflight.
    plan["training_deadline"] = clock["deadline"]-10800
    plan["evaluation_deadline"] = clock["deadline"]-3000
    for key in ("baseline_models", "reference_models"):
        plan[key] = [{**s, "path": str((source/s["path"]).resolve()),
                      "checkpoint_path": str((source/s["checkpoint_path"]).resolve())} for s in old[key]]
    seen = set()
    def add(path, expected=None, kind="bytes", schema=None, rows=None):
        path = Path(path).resolve()
        if path in seen: return
        seen.add(path)
        h = _hash(path)
        if expected and h != expected: raise ValueError(f"Staged immutable input hash: {path}")
        entry = {"path": str(path), "sha256": h, "bytes": path.stat().st_size, "kind": kind}
        if schema: entry["schema"] = schema
        if rows is not None: entry["rows"] = rows
        plan["runtime_inputs"].append(entry)
    add(plan["independent_path"], old["independent_sha256"], "gzip-jsonl", "independent-observations", 4248)
    for spec in parents.values():
        add(spec["checkpoint_path"], spec["checkpoint_sha256"], "gzip-jsonl", "training-checkpoint")
        for key in ("previous_result_path", "previous_iterations_path"):
            if key in spec: add(spec[key], kind="json" if key.endswith("result_path") else "bytes")
        for m in spec.get("recovered_milestones", []): add(m["policy_path"], m["policy_sha256"], "gzip-json")
    for s in plan["baseline_models"]+plan["reference_models"]:
        add(s["checkpoint_path"], s["checkpoint_sha256"], "gzip-jsonl", "training-checkpoint")
        add(s["path"], s["sha256"], "gzip-json")
    for relative, h in old["source_files_sha256"].items(): add(source/relative, h)
    # Include actual orchestration, schedule generators and references, not only the trainer.
    paths = subprocess.check_output(["git", "-C", str(source), "ls-files", "scripts", "src", "docs/reports/hu20-scaling-failure-artifacts", "configs/blueprint/hu20-scaling-both-macs.json"], text=True).splitlines()
    for relative in paths:
        path = source/relative
        if path.suffix in (".py", ".json"): add(path, kind="json" if path.suffix == ".json" else "bytes")
    for prior in old["prior_roots"]:
        path = Path(prior)
        if not path.is_dir(): raise FileNotFoundError(f"Required prior audit root: {path}")
        for p in sorted(path.rglob("*")):
            if p.is_file() and ("hands.jsonl" in p.name or p.name in ("campaign.json", "manifest.json")):
                add(p, kind="gzip-jsonl" if p.name.endswith("jsonl.gz") else "json" if p.suffix == ".json" else "bytes")
    for seed, spec in parents.items(): write_json(root/"inputs"/f"resume-{seed}.json", spec)
    output = source/"configs/blueprint/hu20-scaling-m4-recovery.json"
    write_json(output, plan)
    write_json(root/"runtime-path-mapping.json", {"version": 1, "inputs": plan["runtime_inputs"], "resume": parents})
    print(json.dumps({"plan": str(output), "runtime_inputs": len(plan["runtime_inputs"]),
                      "remaining_nodes": sum(100000000-s["completed_nodes"] for s in parents.values()),
                      "deadline": clock["deadline"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original-plan", type=Path, required=True); args = parser.parse_args()
    stage(args.root, args.original_plan)
