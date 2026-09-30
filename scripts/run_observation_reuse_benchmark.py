"""One guarded M4 benchmark sequence; retain every attempt and stop on mismatch."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from scripts.hu20_platform_pilot import write
from scripts.hu20_scaling_common import inventory
from scripts.hu20_scaling_supervise import run

PARENT = Path("/Users/dberweger/Local/hu20-training-scaling-pr116/results/"
              "hu20-scaling-m4-recovery-20260929-1008/training/B-2026093001/"
              "checkpoint-100000000.json.gz")
PARENT_HASH = "b560669df702057b9df72c90495195a60741e1c6c4b42603cc7c330e2615f64a"


def verify(left, right, out, *, resumed=False):
    a = json.loads((left / "result.json").read_text())
    b = json.loads((right / "result.json").read_text())
    fields = ["added_nodes", "overshoot_nodes", "iteration", "entries", "new_entries",
              "config", "origin_entries", "origin_iteration", "next_nodes", "next_streams",
              "final", "current", "next"]
    if not resumed:
        fields.append("work_sha256")
    checks = {key: a.get(key) == b.get(key) for key in fields}
    checks["complete"] = a.get("status") == b.get("status") == "complete"
    if a.get("trace_result") or b.get("trace_result"):
        checks["trace_sha256"] = a["trace_result"]["sha256"] == b["trace_result"]["sha256"]
        checks["trace_counts"] = all(a["trace_result"]["counts"][key] == b["trace_result"]["counts"][key]
                                     for key in ("observations", "keys_menus", "rng"))
    if resumed:
        original_rows = [json.loads(line) for line in (left / "iterations.jsonl").read_text().splitlines()]
        suffix = [row for row in original_rows if row["added_nodes"] > b["resume_added_nodes"]]
        resume_rows = [json.loads(line) for line in (right / "iterations.jsonl").read_text().splitlines()]
        checks["every_resumed_iteration"] = suffix == resume_rows
    result = {"left": str(left), "right": str(right), "checks": checks,
              "passed": all(checks.values())}
    write(out, result)
    if not result["passed"]:
        raise ValueError("Exact trainer/recovery equivalence failed: " + str(checks))
    return result


def execute(root):
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + 4 * 3600
    write(root / "clock.json", {"started": started, "deadline": deadline})
    note = Path("/tmp/DR_RESEARCH_M4_COORDINATION.txt")
    with note.open("a") as stream:
        stream.write(f"\nDoctor Research observation reuse CLAIM: coordinator {os.getpid()}, "
                     f"one heavy child, root {root}, deadline {deadline}. New Guy uses M1; "
                     "please request any M4 work here. No poker outcomes or main campaign.\n")
    jobs = [{"name": "focused-tests", "command": [sys.executable, "-m", "pytest", "-q",
             "tests/test_observation_reuse.py", "tests/test_hand_observations.py",
             "tests/test_blueprint.py", "tests/test_blueprint_replication.py",
             "tests/test_blueprint_native_reopening.py"]}]

    def train(name, variant, case, nodes, *, trace=False, resume=None):
        command = [sys.executable, "-m", "scripts.benchmark_observation_reuse", "--variant", variant,
                   "--out", str(root / name), "--nodes", str(nodes), "--deadline", str(deadline)]
        if resume:
            command += ["--resume", str(root / resume)]
        elif case == "mature":
            command += ["--parent", str(PARENT), "--parent-sha256", PARENT_HASH]
        if trace:
            command += ["--trace"]
        jobs.append({"name": name, "command": command})

    def check(name, left, right, resumed=False):
        command = [sys.executable, "-m", "scripts.run_observation_reuse_benchmark", "verify",
                   "--left", str(root / left), "--right", str(root / right),
                   "--out", str(root / (name + ".json"))]
        if resumed:
            command += ["--resumed"]
        jobs.append({"name": name, "command": command})

    for case in ("small", "mature"):
        for variant in ("original", "candidate"):
            train(f"trace-{case}-{variant}", variant, case, 100000, trace=True)
        check(f"trace-{case}-equivalence", f"trace-{case}-original", f"trace-{case}-candidate")
        for pair, order in enumerate((("original", "candidate"), ("candidate", "original"),
                                      ("original", "candidate")), 1):
            for variant in order:
                train(f"perf-{case}-{pair}-{variant}", variant, case, 1000000)
            check(f"perf-{case}-{pair}-equivalence", f"perf-{case}-{pair}-original",
                  f"perf-{case}-{pair}-candidate")
        for variant in ("original", "candidate"):
            name = f"resume-{case}-{variant}"
            direct = f"perf-{case}-1-{variant}"
            train(name, variant, case, 1000000, resume=direct)
            check(name + "-equivalence", direct, name, resumed=True)
        check(f"resume-{case}-cross-equivalence", f"resume-{case}-original", f"resume-{case}-candidate")
    write(root / "jobs.json", jobs)
    try:
        record = run(jobs, root / "supervisor", deadline, require_ac=True)
        write(root / "result.json", {"status": record["status"], "started": started,
              "finished": time.time(), "deadline": deadline,
              "attempt_count": len(record["attempts"])})
    finally:
        with note.open("a") as stream:
            stream.write(f"\nDoctor Research observation reuse RELEASE: coordinator {os.getpid()} "
                         f"closed, retained root {root}; see result/supervisor for status.\n")
    # Supervisor logs are closed before this global inventory is created.
    write(root.with_name(root.name + "-manifest.json"), inventory(root))
    return record["status"] == "complete"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--root", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--left", type=Path, required=True)
    check.add_argument("--right", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--resumed", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        return 0 if execute(args.root) else 1
    verify(args.left, args.right, args.out, resumed=args.resumed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
