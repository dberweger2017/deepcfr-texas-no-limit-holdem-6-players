"""Bounded sequential M4 campaign with a development-to-confirmation freeze."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import time

from src.arena.schedule import digest
from src.blueprint.windowed import _hash


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "train-dev", "confirm"), required=True)
    parser.add_argument("--decision", type=Path)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("HU20 campaign source must be committed and clean")
    plan = json.loads(args.plan.read_text())
    record_path = args.root / "campaign.json"
    if args.stage == "preflight":
        if args.root.exists():
            raise FileExistsError(args.root)
        args.root.mkdir(parents=True)
        record = {"schema": "hu20-campaign-v2", "status": "preflight",
                  "started_unix_seconds": time(),
                  "deadline_unix_seconds": time()+plan["limits"]["max_campaign_seconds"],
                  "preflight_plan_sha256": digest(plan), "attempts": []}
    else:
        record = json.loads(record_path.read_text())
        if record["status"] not in (("awaiting_plan",) if args.stage == "train-dev"
                                    else ("awaiting_decision",)):
            raise ValueError("Campaign is not at the requested stage")
        if time() >= record["deadline_unix_seconds"]:
            raise TimeoutError("HU20 hard campaign deadline has expired")
        if args.stage == "train-dev":
            record["frozen_plan_sha256"] = digest(plan)
        elif record["frozen_plan_sha256"] != digest(plan):
            raise ValueError("Confirmation plan differs from frozen training plan")
    save(record_path, record)

    def launch(stage, name, command):
        attempt = {"stage": stage, "name": name, "command": command,
                   "source_revision": git_revision(),
                   "started_unix_seconds": time(), "status": "running"}
        record["attempts"].append(attempt)
        save(record_path, record)
        print(f"START {stage} {name}", flush=True)
        with (args.root / f"{stage}-{name}.log").open("w") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=max(1, record["deadline_unix_seconds"]-time()))
                attempt["returncode"] = result.returncode
                attempt["status"] = "complete" if result.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                attempt.update(status="failed", error="Ten-hour campaign deadline")
        attempt["finished_unix_seconds"] = time()
        save(record_path, record)
        print(f"END {stage} {name} {attempt['status']}", flush=True)
        if attempt["status"] != "complete":
            raise RuntimeError(f"{stage} {name} failed; retained log and partial result")

    try:
        if args.stage == "preflight":
            seed = plan["training_seeds"][0]
            launch("preflight", str(seed), [sys.executable, "-m", "scripts.train_hu20",
                "--plan", str(args.plan), "--seed", str(seed),
                "--out", str(args.root / "preflight"),
                "--deadline", str(record["deadline_unix_seconds"]), "--preflight"])
            record["status"] = "awaiting_plan"
        elif args.stage == "train-dev":
            for seed in plan["training_seeds"]:
                run = args.root / "training" / str(seed)
                run.parent.mkdir(exist_ok=True)
                launch("training", str(seed), [sys.executable, "-m", "scripts.train_hu20",
                    "--plan", str(args.plan), "--seed", str(seed),
                    "--out", str(run), "--deadline", str(record["deadline_unix_seconds"])])
            for opponent in plan["opponents"]:
                arms = ["uniform"]
                for seed in plan["training_seeds"]:
                    arms.extend([f"E{seed}-{i}" for i in range(3)])
                    arms.extend([f"C{seed}", f"A{seed}"])
                for arm in arms:
                    name = f"{opponent}-{arm}"
                    run = args.root / "development" / name
                    run.parent.mkdir(exist_ok=True)
                    launch("development", name, [sys.executable, "-m", "scripts.evaluate_hu20",
                        "--plan", str(args.plan), "--training-root", str(args.root / "training"),
                        "--arm", arm, "--opponent", opponent, "--phase", "development",
                        "--out", str(run), "--deadline", str(record["deadline_unix_seconds"])])
            record["status"] = "awaiting_decision"
        else:
            if args.decision is None:
                raise ValueError("Confirmation requires a committed extraction decision")
            decision = json.loads(args.decision.read_text())
            if (decision.get("plan_sha256") != digest(plan)
                    or decision.get("primary_extraction") not in ("C", "A")
                    or decision.get("development_digest") !=
                        _hash(args.root / "development-summary.json")):
                raise ValueError("Decision does not pin the completed development result")
            record["decision_sha256"] = _hash(args.decision)
            selected = decision["primary_extraction"]
            for opponent in plan["opponents"]:
                for arm in ["uniform", *(f"{selected}{seed}" for seed in plan["training_seeds"])]:
                    name = f"{opponent}-{arm}"
                    run = args.root / "confirmation" / name
                    run.parent.mkdir(exist_ok=True)
                    launch("confirmation", name, [sys.executable, "-m", "scripts.evaluate_hu20",
                        "--plan", str(args.plan), "--training-root", str(args.root / "training"),
                        "--arm", arm, "--opponent", opponent, "--phase", "confirmation",
                        "--out", str(run), "--deadline", str(record["deadline_unix_seconds"])])
            seeds = plan["training_seeds"]
            for i, hero_seed in enumerate(seeds):
                opponent_seed = seeds[(i+1) % len(seeds)]
                for arm in (f"E{hero_seed}-0", f"{selected}{hero_seed}"):
                    opponent = f"{selected}{opponent_seed}"
                    name = f"{arm}-versus-{opponent}"
                    run = args.root / "crossplay" / name
                    run.parent.mkdir(exist_ok=True)
                    launch("crossplay", name, [sys.executable, "-m", "scripts.evaluate_hu20",
                        "--plan", str(args.plan), "--training-root", str(args.root / "training"),
                        "--arm", arm, "--opponent", opponent, "--phase", "crossplay",
                        "--out", str(run), "--deadline", str(record["deadline_unix_seconds"])])
            record["status"] = "complete"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["updated_unix_seconds"] = time()
    save(record_path, record)
    print(json.dumps({"status": record["status"], "error": record.get("error")}), flush=True)
    return 2 if record["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
