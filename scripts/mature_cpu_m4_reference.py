"""Guarded M4 reference using the separately validated, immutable trainer source."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def execute(plan_path, runtime, root):
    plan = json.loads(plan_path.read_text())
    source = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    if source != plan["validated_trainer_source"]:
        raise ValueError("Reference source differs from validated trainer")
    if sys.version_info[:3] != (3, 11, 14) or sys.platform != "darwin":
        raise ValueError("Use pinned M4 Python 3.11.14")
    runtime = runtime.resolve()
    root = root.resolve()
    # The guarded supervisor lives in the validated runtime. Keep this PR's
    # engineering/reporting files separate from the trainer checkout.
    sys.path.insert(0, str(runtime))
    from scripts.hu20_scaling_supervise import run
    from scripts.hu20_scaling_common import inventory
    from scripts.hu20_platform_pilot import write

    os.chdir(runtime)
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + 3600
    write(root / "clock.json", {"started": started, "deadline": deadline,
          "runtime_source": source, "plan_sha256": sha256(plan_path.read_bytes()).hexdigest(),
          "driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest()})
    direct = [sys.executable, "-m", "scripts.benchmark_observation_reuse", "--variant", plan["variant"],
              "--out", str(root / "direct"), "--nodes", str(plan["added_complete_nodes"]),
              "--deadline", str(deadline), "--parent", plan["parent"]["checkpoint_path"],
              "--parent-sha256", plan["parent"]["checkpoint_sha256"]]
    resumed = [sys.executable, "-m", "scripts.benchmark_observation_reuse", "--variant", plan["variant"],
               "--out", str(root / "resumed"), "--nodes", str(plan["added_complete_nodes"]),
               "--deadline", str(deadline), "--resume", str(root / "direct")]
    check = [sys.executable, "-m", "scripts.run_observation_reuse_benchmark", "verify",
             "--left", str(root / "direct"), "--right", str(root / "resumed"),
             "--out", str(root / "resume-verification.json"), "--resumed"]
    jobs = [{"name": name, "command": cmd} for name, cmd in
            (("direct", direct), ("resumed", resumed), ("verify-resume", check))]
    write(root / "jobs.json", jobs)
    note = Path("/tmp/DR_RESEARCH_M4_COORDINATION.txt")
    with note.open("a") as stream:
        stream.write(f"\nDoctor Research mature CPU pilot M4 reference CLAIM: {os.getpid()}, "
                     f"source {source}, one child, root {root}, absolute deadline {deadline}. "
                     "5M fixed engineering nodes plus recovery, no rental or strength evaluation.\n")
    try:
        result = run(jobs, root / "supervisor", deadline, require_ac=True)
        write(root / "result.json", {"status": result["status"], "started": started,
                                     "finished": time.time(), "deadline": deadline})
    finally:
        with note.open("a") as stream:
            stream.write(f"\nDoctor Research mature CPU M4 reference RELEASE: {os.getpid()} closed; "
                         f"retained root {root}; no heavy child remains.\n")
    write(root.with_name(root.name + "-manifest.json"), inventory(root))
    return result["status"] == "complete"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if execute(args.plan.resolve(), args.runtime, args.root) else 1)
