"""Measure frozen refit policies and original current-versus-average play."""

import argparse
import gzip
import json
import subprocess
import sys
from dataclasses import asdict
from hashlib import file_digest
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import torch
from scipy.stats import t

from src.arena.artifacts import environment, git, source_fingerprint, write_json
from src.arena.policies import make_policy
from src.arena.report import performance, summarize
from src.arena.runner import run_schedule
from src.arena.schedule import Plan, Scenario, build_schedule, canonical, digest
from src.holdem.betting import BettingNetwork
from src.holdem.checkpoint import load_training
from src.holdem.policy import FrozenProfile
from src.holdem.policy_diagnostics import replay_policy_changes
from src.holdem.timing import peak_rss_bytes
from src.solver.neural.network import deterministic_cpu

ARMS = {
    "clip64": (1.0, 64),
    "clip256": (1.0, 256),
    "unclipped64": (None, 64),
    "unclipped256": (None, 256),
}
COMPARISONS = (
    ("clip256", "clip64"),
    ("unclipped64", "clip64"),
    ("unclipped256", "clip64"),
    ("current", "average"),
)


def checksum(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def simultaneous_interval(plan, rows, *, comparisons, material_bb100):
    block_values = []
    for index in range(plan.blocks):
        arms = {}
        for arm in ("candidate", "baseline"):
            selected = [r for r in rows if r["block"] == index and r["arm"] == arm]
            arms[arm] = mean(
                100 * r["candidate_chips"] / r["big_blind"] for r in selected
            )
        block_values.append(arms["candidate"] - arms["baseline"])
    center = mean(block_values)
    if len(block_values) < 30 or stdev(block_values) == 0:
        return {
            "ci_family95": None,
            "interpretation": "insufficient_variation_or_blocks",
        }
    radius = float(t.ppf(1 - 0.05 / (2 * comparisons), len(block_values) - 1)) * (
        stdev(block_values) / sqrt(len(block_values))
    )
    lower, upper = center - radius, center + radius
    return {
        "ci_family95": [lower, upper],
        "candidate_better": lower > 0,
        "baseline_better": upper < 0,
        "within_material_margin": lower > -material_bb100 and upper < material_bb100,
        "candidate_materially_better": lower > material_bb100,
        "baseline_materially_better": upper < -material_bb100,
    }


def evaluate_pair(plan, policies, output, *, deadline, family_size, material_bb100):
    output.mkdir()
    rows, timings = [], []
    started = perf_counter()

    class BoundedPlayer:
        def __init__(self, player):
            self.player = player

        def choose_action(self, view):
            if perf_counter() >= deadline:
                raise TimeoutError("Policy comparison exceeded its deadline")
            return self.player.choose_action(view)

    def factory(name, seed):
        policy = (
            policies[name].player(seed) if name in policies else make_policy(name, seed)
        )
        return BoundedPlayer(policy)

    blocks = build_schedule(plan)
    write_json(
        output / "schedule.json",
        {"plan": asdict(plan), "blocks": [asdict(b) for b in blocks]},
    )
    path = output / "outcomes.jsonl.gz"
    with (
        path.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as stream,
    ):

        def emit(row, timing):
            stream.write((canonical(row) + "\n").encode())
            rows.append(row)
            timings.append(timing)

        valid = run_schedule(plan, blocks, emit, factory=factory)
    report = summarize(plan, rows)
    report["performance"] = performance(timings, perf_counter() - started)
    report["outcome_file"] = {"path": path.name, "sha256": checksum(path)}
    report["blocks_sha256"] = digest([asdict(b) for b in blocks])
    if valid and report["status"] == "valid":
        report["multiplicity"] = simultaneous_interval(
            plan, rows, comparisons=family_size, material_bb100=material_bb100
        )
    write_json(output / "report.json", report)
    if not valid or report["status"] != "valid":
        raise RuntimeError("Invalid comparison; partial outcomes retained")
    return report


def load_refits(job, directory, width):
    profiles = {}
    for name, (clip, steps) in ARMS.items():
        selected = [
            f for f in job["fits"] if f["gradient_clip"] == clip and f["steps"] == steps
        ]
        if sorted(f["role"] for f in selected) != list(range(6)):
            raise ValueError("Each refit profile needs exactly six roles")
        models = []
        for fit in sorted(selected, key=lambda f: f["role"]):
            record = fit["weights"]
            if Path(record["path"]).name != record["path"]:
                raise ValueError("Expected a model filename, not a path")
            path = directory / record["path"]
            if checksum(path) != record["sha256"]:
                raise ValueError("Refit weight checksum mismatch")
            model = BettingNetwork(width)
            model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True), strict=True
            )
            models.append(model)
        profiles[name] = FrozenProfile(models)
    return profiles


def worker(settings, refits, output, seed):
    started = perf_counter()
    deadline = started + settings["maximum_seconds_per_seed"]
    output.mkdir(parents=True, exist_ok=False)
    result = {"seed": seed, "status": "running", "comparisons": [], "diagnostics": []}
    try:
        evidence_path = Path(settings["fitting_report"])
        if checksum(evidence_path) != settings["fitting_report_sha256"]:
            raise ValueError("The frozen fitting report changed")
        evidence = json.loads(evidence_path.read_text())
        job = next(j for j in evidence["jobs"] if j["seed"] == seed)
        spec = next(
            s
            for s in evidence["manifest"]["settings"]["checkpoints"]
            if s["seed"] == seed
        )
        original_manifest = json.loads(
            (refits / "inputs" / f"manifest-{seed}.json").read_text()
        )
        with deterministic_cpu():
            trainer = load_training(
                refits / "inputs" / f"training-{seed}.pt",
                spec["sha256"],
                manifest=original_manifest,
            )
            if (
                trainer.config.seed != seed
                or trainer.iteration != 256
                or trainer.table.capacity != 6
            ):
                raise ValueError(
                    "Expected the declared six-player iteration-256 checkpoint"
                )
            result["loading_seconds"] = perf_counter() - started
            profiles = load_refits(
                job, refits / f"seed-{seed}", trainer.config.fit.width
            )
            profiles["current"] = trainer.current_profile()
            historical = trainer.average_policy()
            result["provenance"] = {
                "checkpoint_sha256": spec["sha256"],
                "profiles": {name: p.fingerprint for name, p in profiles.items()},
                "average_fingerprints_sha256": digest(historical.fingerprints),
                "average_profile_count": len(historical.fingerprints),
            }
            for memory in trainer.memories:
                result["diagnostics"].append(
                    {
                        "role": memory.role,
                        "policies": replay_policy_changes(
                            memory,
                            {n: p._models[memory.role] for n, p in profiles.items()},
                            control="clip64",
                            deadline=deadline,
                        ),
                    }
                )
                write_json(output / "report.json", result)
            policies = {**profiles, "average": historical}
            for candidate, baseline in COMPARISONS:
                plan = Plan(
                    (
                        Scenario(
                            "six-100bb",
                            (200,) * 6,
                            small_blind=1,
                            big_blind=2,
                            chip_unit="1",
                        ),
                    ),
                    candidate=candidate,
                    baseline=baseline,
                    opponents=tuple(settings["opponents"]),
                    blocks=settings["blocks"],
                    root_seed=settings["evaluation_seed"],
                    split="validation",
                )
                name = f"{candidate}-vs-{baseline}"
                report = evaluate_pair(
                    plan,
                    policies,
                    output / name,
                    deadline=deadline,
                    family_size=12,
                    material_bb100=settings["material_bb100"],
                )
                result["comparisons"].append({"name": name, "report": report})
                write_json(output / "report.json", result)
                print(f"{seed}: {name} completed", flush=True)
            for profile in profiles.values():
                profile.assert_unchanged()
            if (
                digest(trainer.average_policy().fingerprints)
                != result["provenance"]["average_fingerprints_sha256"]
            ):
                raise ArithmeticError("Historical average changed")
            result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["wall_seconds"] = perf_counter() - started
        result["peak_process_rss_bytes"] = peak_rss_bytes()
        write_json(output / "report.json", result)


def run(plan, refits, output):
    settings = json.loads(plan.read_text())
    if (
        settings["version"] != 1
        or settings["seeds"] != [307, 311, 313]
        or settings["blocks"] != 1024
        or settings["maximum_seconds_per_seed"] != 900
        or settings["maximum_total_seconds"] != 2700
    ):
        raise ValueError("Expected the bounded three-seed policy comparison protocol")
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    result = {
        "status": "running",
        "jobs": [{"seed": s, "status": "unattempted"} for s in settings["seeds"]],
    }
    write_json(
        output / "manifest.json",
        {
            "plan": settings,
            "plan_sha256": checksum(plan),
            "revision": git("rev-parse", "HEAD"),
            "source_sha256": source_fingerprint(),
            "environment": environment(),
            "fitting_report_sha256": settings["fitting_report_sha256"],
        },
    )
    try:
        for job in result["jobs"]:
            remaining = settings["maximum_total_seconds"] - (perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("Campaign deadline expired")
            job["status"] = "running"
            write_json(output / "result.json", result)
            with (output / f"seed-{job['seed']}.log").open("x") as log:
                try:
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "scripts.compare_holdem_policies",
                            "--plan",
                            str(plan),
                            "--refits",
                            str(refits),
                            "--out",
                            str(output),
                            "--worker",
                            str(job["seed"]),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=min(remaining, settings["maximum_seconds_per_seed"]),
                    )
                except subprocess.TimeoutExpired:
                    job["status"] = "timeout"
                    raise
            job.update(
                exit_code=process.returncode,
                status="completed" if process.returncode == 0 else "failed",
            )
            if process.returncode:
                raise RuntimeError("A seed failed; remaining jobs are unattempted")
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["wall_seconds"] = perf_counter() - started
        write_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--refits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--worker", type=int, choices=(307, 311, 313))
    args = parser.parse_args()
    if args.worker is not None:
        worker(
            json.loads(args.plan.read_text()),
            args.refits,
            args.out / f"seed-{args.worker}",
            args.worker,
        )
    else:
        run(args.plan, args.refits, args.out)


if __name__ == "__main__":
    main()
