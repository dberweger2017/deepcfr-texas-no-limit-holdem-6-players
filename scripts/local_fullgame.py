"""Run the declared local batch only when explicitly invoked; never auto-resume."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def used_bytes(directory):
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def resource_failure(elapsed, limit, rss, free, stored):
    if elapsed >= limit:
        return "wall_time_limit"
    if rss > 7 * GIB:
        return "process_memory_limit"
    if free < 12 * GIB:
        return "free_disk_limit"
    if stored > 8 * GIB:
        return "output_size_limit"
    return None


def guarded_run(command, output, *, seconds, poll_seconds=5):
    """Own a process group so a timed-out worker cannot continue training."""
    if seconds <= 0 or poll_seconds <= 0:
        raise ValueError("Worker requires a positive remaining time budget")
    started = time.monotonic()
    record = {"command": command, "limit_seconds": seconds, "status": "running"}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_suffix(".json"), record)
    with output.open("wb") as log:
        child = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True
        )
        record["pid"] = child.pid
        write_json(output.with_suffix(".json"), record)
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - started
                ps = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(child.pid)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if ps.returncode and child.poll() is None:
                    raise RuntimeError("Cannot measure worker memory")
                rss = int(ps.stdout.strip() or 0) * 1024
                reason = resource_failure(
                    elapsed,
                    seconds,
                    rss,
                    shutil.disk_usage(output.parent).free,
                    used_bytes(output.parent),
                )
                record.update(seconds=elapsed, rss_bytes=rss)
                if reason:
                    raise RuntimeError(reason)
                write_json(output.with_suffix(".json"), record)
                time.sleep(min(poll_seconds, max(0.01, seconds - elapsed)))
            record.update(returncode=child.returncode)
            if child.returncode:
                raise RuntimeError(f"worker_exit_{child.returncode}")
            record["status"] = "complete"
        except BaseException as exc:
            record.update(status="failed", error=str(exc))
            raise
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            record["seconds"] = time.monotonic() - started
            write_json(output.with_suffix(".json"), record)
    return record


def final_plan(training):
    from dataclasses import replace

    from src.holdem.experiment import Experiment

    class FinalExperiment(Experiment):
        def arena(self, scenario):
            return replace(super().arena(scenario), split="test")

    return FinalExperiment(
        **{
            **training.__dict__,
            "blocks": 4096,
            "evaluation_seed": 2026091811,
            "opponents": ("random",),
            "benchmarks": (),
            "reference": None,
        }
    )


def adjusted_interval(estimate):
    """Four two-sided intervals cover two seeds times two declared endpoints."""
    from scipy.stats import t

    interval = estimate["ci95"]
    if interval is None:
        return None
    center = estimate["bb_per_100"]
    degrees = estimate["blocks"] - 1
    factor = float(t.ppf(1 - 0.05 / 8, degrees) / t.ppf(0.975, degrees))
    half_width = (interval[1] - interval[0]) / 2 * factor
    return [center - half_width, center + half_width]


def verify_and_test(job, out):
    from dataclasses import asdict

    from src.holdem.checkpoint import load_training, save_policy, save_training
    from src.holdem.experiment import Experiment, evaluate
    from src.solver.neural.network import deterministic_cpu

    out.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((job / "manifest.json").read_text())
    training = Experiment.from_dict(provenance["plan"])
    if len(training.seeds) != 1 or len(training.scenarios) != 1:
        raise ValueError("Expected one completed training seed and scenario")
    result = json.loads((job / "result.json").read_text())
    if not result["complete"]:
        raise ValueError("Training did not complete")
    seed = training.seeds[0]
    directory = job / f"scenario-0-seed-{seed}"
    iteration = training.iterations
    marker = json.loads((directory / f"training-{iteration}.json").read_text())
    artifacts = [
        json.loads(line)
        for line in (directory / "artifacts.jsonl").read_text().splitlines()
    ]
    export = next(
        a
        for a in artifacts
        if a["iteration"] == iteration and a["kind"] == "holdem-average-v1"
    )
    with deterministic_cpu():
        trainer = load_training(
            directory / f"training-{iteration}.pt",
            marker["sha256"],
            manifest=provenance,
        )
        recovered = save_training(
            trainer, out / "recovered-training.pt", manifest=provenance
        )
        exported = save_policy(
            trainer, out / "recovered-average.pt", manifest=provenance
        )
    if recovered != marker["sha256"] or exported != export["sha256"]:
        raise ValueError("Fresh-process recovery bytes differ")
    write_json(
        out / "recovery.json",
        {"training_sha256": recovered, "policy_sha256": exported, "matching": True},
    )
    # Original models stay intact; remove only verified duplicate recovery bytes.
    (out / "recovered-training.pt").unlink()
    (out / "recovered-average.pt").unlink()
    final = final_plan(training)
    write_json(out / "test-plan.json", asdict(final.arena(final.scenarios[0])))
    report = evaluate(
        trainer, final.scenarios[0], final, out, provenance, time.perf_counter() + 1800
    )
    if report["status"] != "valid":
        raise ValueError("Invalid final-test arena")
    comparison = report["scenarios"][training.scenarios[0].name]["comparison"]
    endpoints = {
        name: {
            **comparison[name],
            "familywise_interval": adjusted_interval(comparison[name]),
        }
        for name in ("candidate", "paired_difference")
    }
    passed = all(
        row["familywise_interval"] is not None and row["familywise_interval"][0] > 0
        for row in endpoints.values()
    )
    write_json(
        out / "final-summary.json",
        {
            "seed": seed,
            "endpoints": endpoints,
            "competence_check_passed": passed,
            "promoted": False,
        },
    )


def campaign(plan_path, out):
    from src.arena.artifacts import environment, git, source_fingerprint
    from src.holdem.experiment import Experiment

    data = json.loads(plan_path.read_text())
    training = Experiment.from_dict(data)
    if training.seeds != (2026091802, 2026091803) or len(training.scenarios) != 1:
        raise ValueError("Use the declared two-seed local plan")
    if git("status", "--porcelain"):
        raise ValueError("Commit the campaign source before launch")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 20 * GIB:
        raise RuntimeError("Need 20 GiB free before launching")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", data)
    write_json(
        out / "manifest.json",
        {
            "revision": git("rev-parse", "HEAD"),
            "source_sha256": source_fingerprint(),
            "environment": environment(),
        },
    )
    status = {"state": "running", "seeds": [], "promoted": False}
    write_json(out / "status.json", status)
    try:
        for seed in training.seeds:
            status["active_seed"] = seed
            write_json(out / "status.json", status)
            guarded_run(
                [
                    sys.executable,
                    "-m",
                    "scripts.train_holdem",
                    "--plan",
                    str(out / "plan.json"),
                    "--seed",
                    str(seed),
                    "--out",
                    str(out / f"seed-{seed}"),
                ],
                out / f"seed-{seed}.log",
                seconds=21600,
            )
            status["seeds"].append(seed)
        remaining = 1800.0
        for seed in training.seeds:
            status.update(phase="verification_and_final_test", active_seed=seed)
            write_json(out / "status.json", status)
            record = guarded_run(
                [
                    sys.executable,
                    "-m",
                    "scripts.local_fullgame",
                    "--verify-job",
                    str(out / f"seed-{seed}"),
                    "--out",
                    str(out / f"final-{seed}"),
                ],
                out / f"final-{seed}.log",
                seconds=remaining,
            )
            remaining -= record["seconds"]
        summaries = [
            json.loads((out / f"final-{seed}" / "final-summary.json").read_text())
            for seed in training.seeds
        ]
        write_json(
            out / "summary.json",
            {
                "seeds": summaries,
                "competence_check_passed": all(
                    s["competence_check_passed"] for s in summaries
                ),
                "promoted": False,
            },
        )
        status.update(state="complete", active_seed=None)
    except BaseException as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        write_json(out / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", type=Path)
    mode.add_argument("--verify-job", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    output = args.out.resolve()
    if args.verify_job:
        verify_and_test(args.verify_job.resolve(), output)
    else:
        campaign(args.plan, output)


if __name__ == "__main__":
    main()
