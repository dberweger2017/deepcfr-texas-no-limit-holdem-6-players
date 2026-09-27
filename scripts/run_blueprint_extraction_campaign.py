"""Run the fixed M4 replay and paired evaluation sequentially, preserving attempts."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import time

from src.blueprint.windowed import _hash


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    plan = json.loads(args.plan.read_text())
    deadline = time() + plan["limits"]["max_campaign_seconds"]
    campaign = {"schema": "windowed-blueprint-campaign-v1", "started_unix_seconds": time(),
                "deadline_unix_seconds": deadline, "plan_sha256": _hash(args.plan),
                "parent": str(args.parent), "parent_sha256": _hash(args.parent),
                "original_root": str(args.original_root), "attempts": [], "status": "running"}
    save(args.out / "campaign.json", campaign)

    def launch(stage, name, command):
        attempt = {"stage": stage, "name": name, "command": command,
                   "started_unix_seconds": time(), "status": "running"}
        campaign["attempts"].append(attempt)
        save(args.out / "campaign.json", campaign)
        print(f"START {stage} {name}", flush=True)
        with (args.out / f"{stage}-{name}.log").open("w") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=max(1, deadline-time()))
                attempt["returncode"] = completed.returncode
                attempt["status"] = "complete" if completed.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                attempt.update(status="failed", error="Ten-hour campaign deadline")
        attempt["finished_unix_seconds"] = time()
        save(args.out / "campaign.json", campaign)
        print(f"END {stage} {name} {attempt['status']}", flush=True)
        if attempt["status"] != "complete":
            campaign["status"] = "failed"
            save(args.out / "campaign.json", campaign)
            raise RuntimeError(f"{stage} {name} failed; see retained log and result")

    try:
        for seed in plan["continuation_seeds"]:
            source = args.original_root / "training" / f"k1-{seed}"
            out = args.out / "extraction" / str(seed)
            out.parent.mkdir(exist_ok=True)
            launch("extraction", str(seed), [sys.executable, "-m", "scripts.run_blueprint_extraction",
                "--plan", str(args.plan), "--parent", str(args.parent),
                "--original", str(source / "checkpoint.json.gz"),
                "--lineage", str(source / "lineage.json"),
                "--out", str(out), "--seed", str(seed), "--deadline", str(deadline)])
        for arm in plan["arms"]:
            out = args.out / "evaluation" / arm
            out.parent.mkdir(exist_ok=True)
            if arm in ("U_safe", "parent", "TAG"):
                command = [sys.executable, "-m", "scripts.evaluate_postflop_replication",
                    "--plan", str(args.plan), "--arm", arm, "--out", str(out),
                    "--campaign-deadline", str(deadline)]
                if arm != "TAG":
                    command += ["--checkpoint", str(args.parent)]
            else:
                seed = int(arm[1:])
                source = args.out / "extraction" / str(seed)
                command = [sys.executable, "-m", "scripts.evaluate_blueprint_extraction",
                    "--plan", str(args.plan), "--arm", arm,
                    "--index", str(source / "policy-index.sqlite"),
                    "--manifest", str(source / "policy-manifest.json"),
                    "--out", str(out), "--deadline", str(deadline)]
            launch("evaluation", arm, command)
        campaign["status"] = "complete"
    except Exception as exc:
        campaign["status"] = "failed"
        campaign["error"] = f"{type(exc).__name__}: {exc}"
    campaign["finished_unix_seconds"] = time()
    save(args.out / "campaign.json", campaign)
    print(json.dumps({"status": campaign["status"], "error": campaign.get("error")}), flush=True)
    return 0 if campaign["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
