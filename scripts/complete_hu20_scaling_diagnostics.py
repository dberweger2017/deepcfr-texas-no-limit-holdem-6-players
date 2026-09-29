"""Finish #116's frozen diagnostic panels in a separate M4-only attempt.

This wrapper changes only operational deadlines and output locations. The
production evaluator, opponent rules, model files, deal roots and block counts
come from the sealed recovery attempt.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import time


FROZEN_PLAN_SHA256 = "55add291a54b1277128ee6e5c915433281ff19ef98d7794fc1cef1808c45b9b2"
SOURCE_REVISION = "874ba641fc6110a2d0998af986608633abef8898"
PARENT_NAME = "hu20-scaling-m4-recovery-20260929-1008"
ATTEMPT_NAME = "hu20-scaling-diagnostics-20260929"
SECONDS = 4 * 3600
EVALUATION_SECONDS = 2 * 3600 + 45 * 60


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def put(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def prepare(source):
    began = time()
    source = source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    from scripts.evaluate_hu20_scaling import tasks
    from scripts.train_hu20 import system

    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != SOURCE_REVISION:
        raise ValueError("Scientific source revision changed")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Scientific source is not clean")
    original = source / "configs/blueprint/hu20-scaling-m4-recovery.json"
    if digest(original) != FROZEN_PLAN_SHA256:
        raise ValueError("Frozen plan changed")
    parent_root = source / "results" / PARENT_NAME
    parent = json.loads((parent_root / "campaign.json").read_text())
    finished = json.loads((parent_root.with_name(PARENT_NAME + "-wrapper-finished.json")).read_text())
    if not parent["status"].startswith("reported-") or finished["seal_exit_code"] != 0:
        raise ValueError("Parent recovery is not sealed")
    if json.loads((parent_root / "audit-primary/results.json").read_text())["status"] != "complete":
        raise ValueError("Parent primary audit is incomplete")
    models_path = parent_root / "models.json"
    models = json.loads(models_path.read_text())
    model_hashes = {}
    for model in models:
        path = Path(model["path"])
        actual = digest(path)
        if actual != model["sha256"]:
            raise ValueError(f"Policy hash changed: {model['name']}")
        model_hashes[model["name"]] = actual
    plan = json.loads(original.read_text())
    root = source / "results" / ATTEMPT_NAME
    if root.exists():
        raise FileExistsError(f"Attempt already exists: {root}")
    deadline = began + SECONDS
    plan.update(attempt_id=ATTEMPT_NAME, root=str(root), started=began,
                deadline=deadline, evaluation_deadline=began + EVALUATION_SECONDS,
                panel_filter="diagnostic", coordinator_models=str(models_path),
                authorization="Owner requested completion of #116 frozen pending diagnostics on M4")
    plan["swap_baselines"]["m4"] = system(["sysctl", "vm.swapusage"])
    selected = list(tasks(plan, models))
    if len(selected) != 99 or len({(s["name"], label) for s, label, *_ in selected}) != 99:
        raise ValueError("Frozen diagnostic panel set changed")
    expected_hands = 2 * sum(count for _, _, _, _, count, _, _ in selected)
    if expected_hands != 608256 - 73728:
        raise ValueError("Frozen diagnostic hand count changed")
    root.mkdir(parents=True)
    plan_path = root / "diagnostic-plan.json"
    put(plan_path, plan)
    all_plan = {**plan, "panel_filter": "all"}
    put(root / "combined-plan.json", all_plan)
    put(root / "freeze.json", {
        "attempt_id": ATTEMPT_NAME,
        "authorization": plan["authorization"],
        "started": began,
        "deadline": deadline,
        "evaluation_deadline": plan["evaluation_deadline"],
        "parent_attempt": PARENT_NAME,
        "parent_plan_sha256": FROZEN_PLAN_SHA256,
        "parent_manifest_sha256": digest(parent_root.with_name(PARENT_NAME + "-final-manifest.json")),
        "parent_primary_audit_sha256": digest(parent_root / "audit-primary/results.json"),
        "source_revision": SOURCE_REVISION,
        "runner_sha256": digest(__file__),
        "diagnostic_plan_sha256": digest(plan_path),
        "model_list_sha256": digest(models_path),
        "model_policy_sha256": model_hashes,
        "panels": len(selected),
        "hands": expected_hands,
        "schedule": [{"policy": spec["name"], "attacker": label,
                      "blocks": count, "root": seed_root, "phase": phase}
                     for spec, label, _, _, count, seed_root, phase in selected],
        "resource_limits": plan["limits"],
        "swap_baseline": plan["swap_baselines"]["m4"],
    })
    return root


def run(source, root):
    source = source.resolve()
    root = root.resolve()
    os.chdir(source)
    freeze = json.loads((root / "freeze.json").read_text())
    plan_path = root / "diagnostic-plan.json"
    plan = json.loads(plan_path.read_text())
    if digest(__file__) != freeze["runner_sha256"] or digest(plan_path) != freeze["diagnostic_plan_sha256"]:
        raise ValueError("Frozen runner or diagnostic plan changed")
    if time() >= freeze["evaluation_deadline"] - 120:
        raise TimeoutError("Evaluation reserve expired before launch")
    python = Path(plan["hosts"]["m4"]["python"])
    jobs = [
        {"name": "evaluate-diagnostic", "deadline": freeze["evaluation_deadline"],
         "command": [str(python), "-m", "scripts.evaluate_hu20_scaling", "--plan", str(plan_path),
                     "--models", plan["coordinator_models"], "--host", "m4", "--out",
                     str(root / "evaluation-diagnostic"), "--deadline", str(freeze["evaluation_deadline"]),
                     "--swap-baseline", freeze["swap_baseline"]]},
        {"name": "audit-diagnostic", "deadline": freeze["deadline"] - 1800,
         "command": [str(python), "-m", "scripts.report_hu20_scaling", "--plan", str(plan_path),
                     "--evaluation", str(root / "evaluation-diagnostic"), "--out",
                     str(root / "audit-diagnostic")]},
    ]
    put(root / "jobs.json", jobs)
    status = {"status": "running", "started": time(), "pid": os.getpid()}
    put(root / "run-status.json", status)
    try:
        command = [str(python), "-m", "scripts.hu20_scaling_supervise", "--jobs", str(root / "jobs.json"),
                   "--out", str(root / "supervisor"), "--deadline", str(freeze["deadline"]),
                   "--swap-baseline", freeze["swap_baseline"], "--coordinator-pid", str(os.getpid()),
                   "--require-ac"]
        with (root / "supervisor-launcher.log").open("w") as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if child.returncode:
            raise RuntimeError("Diagnostic evaluator or native audit failed; inspect retained supervisor")
        if time() >= freeze["deadline"] - 600:
            raise TimeoutError("No safe report/seal reserve")
        combined = root / "combined-results.json"
        command = [str(python), "-m", "scripts.report_hu20_scaling", "--plan",
                   str(root / "combined-plan.json"), "--audits",
                   str(source / "results" / PARENT_NAME / "audit-primary/results.json"),
                   str(root / "audit-diagnostic/results.json"), "--out", str(combined)]
        with (root / "combine.log").open("w") as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if child.returncode:
            raise RuntimeError("Combined frozen-panel report failed")
        report = json.loads(combined.read_text())
        if report["status"] != "complete" or report["pending_panels"] or report["native_replayed_hands"] != 608256:
            raise ValueError("Frozen campaign remains incomplete")
        old = json.loads((source / "results" / PARENT_NAME / "report/results.json").read_text())
        if report["primary"] != old["primary"]:
            raise ValueError("Completed primary comparisons changed")
        status.update(status="complete", panels=freeze["panels"],
                      diagnostic_hands=freeze["hands"], combined_hands=report["native_replayed_hands"])
    except Exception as exc:
        status.update(status="incomplete", failure=f"{type(exc).__name__}: {exc}")
    status["finished"] = time()
    put(root / "run-status.json", status)
    manifest = {"attempt_id": ATTEMPT_NAME, "finished": time(), "deadline": freeze["deadline"], "files": {}}
    for path in sorted(root.rglob("*")):
        if time() >= freeze["deadline"]:
            raise TimeoutError("Final inventory reached diagnostic deadline")
        if path.is_file():
            manifest["files"][str(path.relative_to(root))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    put(root.with_name(root.name + "-final-manifest.json"), manifest)
    put(root.with_name(root.name + "-finished.json"), status)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = {"prepared_root": str(prepare(args.source))}
    else:
        result = run(args.source, args.source / "results" / ATTEMPT_NAME)
    print(json.dumps(result, sort_keys=True))
    return 0 if args.mode == "prepare" or result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
