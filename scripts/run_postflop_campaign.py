"""Run the frozen six-continuation M4 campaign, one checkpoint per process."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import time

from src.arena.artifacts import git, write_json
from src.arena.schedule import digest


def launch(command, log, hard_deadline):
    with log.open("w", encoding="utf-8") as output:
        try:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                    timeout=max(1, hard_deadline-time()), check=False)
            return result.returncode
        except subprocess.TimeoutExpired:
            output.write("Hard ten-hour campaign deadline expired\n")
            return 124


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or git("status", "--porcelain"):
        parser.error("A new output directory and committed clean source are required")
    plan = json.loads(args.plan.read_text())
    start = time()
    hard_deadline = start + plan["limits"]["max_campaign_seconds"]
    work_deadline = hard_deadline - 900
    args.out.mkdir(parents=True)
    write_json(args.out / "campaign.json", {
        "schema": "postflop-replication-campaign-v1",
        "plan_sha256": digest(plan), "source_revision": git("rev-parse", "HEAD"),
        "started_unix_seconds": start, "hard_deadline_unix_seconds": hard_deadline,
        "work_deadline_unix_seconds": work_deadline,
        "parent_checkpoint": str(args.parent),
    })
    python = sys.executable
    base = [python, "-m"]
    executed = []

    def phase(name, command):
        code = launch(command, args.out / f"{name}.log", hard_deadline)
        executed.append({"phase": name, "exit_code": code, "ended_unix_seconds": time()})
        write_json(args.out / "progress.json", executed)
        print(json.dumps(executed[-1]), flush=True)
        return code == 0

    # The independent observation schedule is distinct from playing evaluation.
    observation_plan = {**plan, "arms": ["U_safe"],
                        "suites": plan["observation_suites"]}
    write_json(args.out / "observation-plan.json", observation_plan)
    if not phase("independent-observations", base + [
        "scripts.evaluate_postflop_replication", "--plan", str(args.out / "observation-plan.json"),
        "--arm", "U_safe", "--checkpoint", str(args.parent),
        "--out", str(args.out / "independent-observations"),
        "--campaign-deadline", str(work_deadline)]):
        return 2
    coverage = json.loads((args.out / "independent-observations" / "coverage.json").read_text())
    observed = {}
    for suite in coverage.values():
        for row in suite["distinct"]:
            observed.setdefault(row["street"], set()).add(row["key"])
    write_json(args.out / "observation-set.json",
               {street: sorted(keys) for street, keys in sorted(observed.items())})
    training = args.out / "training"
    training.mkdir()
    for seed in plan["continuation_seeds"]:
        for mode in (1, 4):
            arm = f"k{mode}-{seed}"
            if not phase(f"train-{arm}", base + [
                "scripts.run_postflop_replication", "--plan", str(args.plan),
                "--parent", str(args.parent), "--out", str(training / arm),
                "--replicates", str(mode), "--seed", str(seed),
                "--campaign-deadline", str(work_deadline)]):
                return 2
    evaluation = args.out / "evaluation"
    evaluation.mkdir()
    for arm in plan["arms"]:
        command = base + ["scripts.evaluate_postflop_replication",
                          "--plan", str(args.plan), "--arm", arm,
                          "--out", str(evaluation / arm),
                          "--campaign-deadline", str(work_deadline)]
        if arm != "TAG":
            checkpoint = args.parent if arm in ("U_safe", "parent") else training / arm / "checkpoint.json.gz"
            command += ["--checkpoint", str(checkpoint)]
        if arm not in ("TAG", "U_safe", "parent"):
            command += ["--lineage", str(training / arm / "lineage.json")]
        if not phase(f"evaluate-{arm}", command):
            return 2
    if not phase("report", base + [
        "scripts.report_postflop_replication", "--plan", str(args.plan),
        "--run", str(args.out), "--parent", str(args.parent),
        "--observation-set", str(args.out / "observation-set.json"),
        "--out", str(args.out / "analysis.json")]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
