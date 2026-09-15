"""Run bounded CPU comparisons and a predeclared frozen-replay fitting sweep."""

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import monotonic, sleep

from scripts.pilot_replay import export_replay, load_replay
from src.solver.experiment import ROOT, canonical, write_json
from src.solver.neural.experiment import Plan, provenance
from src.solver.neural.solver import Config


def validate(settings: dict) -> None:
    if settings["version"] != 1:
        raise ValueError("Unsupported CPU pilot plan")
    for field, maximum in (
        ("workers", 8),
        ("maximum_seconds", 14400),
        ("job_seconds", 840),
    ):
        if type(settings[field]) is not int or not 1 <= settings[field] <= maximum:
            raise ValueError(f"Invalid {field}")
    seeds = settings["benchmark_seeds"]
    if not seeds or len(seeds) > 8 or len(set(seeds)) != len(seeds):
        raise ValueError("Expected unique benchmark seeds")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Invalid benchmark seed")
    replay_seeds = settings["replay_seeds"]
    if replay_seeds != [11, 29, 47]:
        raise ValueError("The fitting pilot requires all three retained Leduc seeds")
    plan = Plan.from_dict(settings["benchmark"])
    if plan.maximum_seconds > settings["job_seconds"]:
        raise ValueError("Training budget exceeds the worker deadline")
    variants = settings["variants"]
    if (
        not variants
        or len(variants) > 8
        or len({v["name"] for v in variants}) != len(variants)
    ):
        raise ValueError("Expected unique fitting variants")
    for variant in variants:
        name, fit = variant["name"], variant["fit"]
        if not name or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name
        ):
            raise ValueError("Invalid variant name")
        if set(fit) != {"hidden", "steps", "batch_size", "learning_rate"}:
            raise ValueError("Unexpected fitting parameters")
        Config(**{k: v for k, v in fit.items() if k != "steps"})
        if type(fit["steps"]) is not int or not 1 <= fit["steps"] <= 24000:
            raise ValueError("Invalid fitting steps")


def run_jobs(
    jobs: list[dict],
    output: Path,
    *,
    workers: int,
    deadline: float,
    job_seconds: float,
    module: str = "scripts.pilot_worker",
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    active = {}
    records = []
    next_job = 0
    environment = {
        **os.environ,
        **{
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }

    def stop(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    try:
        while next_job < len(jobs) or active:
            if monotonic() >= deadline:
                raise TimeoutError("CPU pilot reached its total runtime limit")
            while next_job < len(jobs) and len(active) < workers:
                index = next_job
                job_path = output / f"job-{index:03d}.json"
                write_json(job_path, jobs[index])
                result_path = output / f"job-{index:03d}"
                log = (output / f"job-{index:03d}.log").open("w")
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            module,
                            "--job",
                            str(job_path.resolve()),
                            "--out",
                            str(result_path.resolve()),
                        ],
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    log.close()
                active[index] = (process, monotonic(), result_path)
                next_job += 1
            for index, (process, began, path) in list(active.items()):
                if process.poll() is None:
                    if monotonic() - began >= job_seconds:
                        raise TimeoutError(f"Worker {index} reached its runtime limit")
                    continue
                elapsed = monotonic() - began
                if process.returncode:
                    raise RuntimeError(f"Worker {index} failed; see {path.name}.log")
                result = json.loads((path / "report.json").read_text())
                if result["status"] != "completed":
                    raise RuntimeError(f"Worker {index} did not complete")
                records.append(
                    {"index": index, "elapsed_seconds": elapsed, "report": result}
                )
                del active[index]
                write_json(
                    output / "progress.json", sorted(records, key=lambda r: r["index"])
                )
            if active:
                sleep(0.05)
    finally:
        for process, _, _ in active.values():
            stop(process)
    return {
        "workers": workers,
        "wall_seconds": monotonic() - started,
        "jobs": sorted(records, key=lambda r: r["index"]),
    }


def same_training(serial: dict, parallel: dict) -> bool:
    return canonical([row["report"]["result"] for row in serial["jobs"]]) == canonical(
        [row["report"]["result"] for row in parallel["jobs"]]
    )


def host_details() -> dict:
    command = (
        ["sysctl", "-n", "machdep.cpu.brand_string"]
        if sys.platform == "darwin"
        else ["lscpu"]
    )
    try:
        cpu = subprocess.check_output(command, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        cpu = platform.processor()
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "platform": platform.platform()}


def run_pilot(settings: dict, output: Path, replays: Path | None = None) -> dict:
    validate(settings)
    started = monotonic()
    output.mkdir(parents=True, exist_ok=False)
    scripts = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
        for p in (
            ROOT / "scripts/cpu_pilot.py",
            ROOT / "scripts/pilot_replay.py",
            ROOT / "scripts/pilot_worker.py",
        )
    }
    manifest = {
        **provenance(settings),
        "pilot_sources": scripts,
        "host": host_details(),
        "mode": "local-serial" if replays is None else "remote-full",
        "replays": {},
    }
    report = {"status": "running", "phases": {}}
    write_json(output / "manifest.json", manifest)
    try:
        if replays is not None:
            for seed in settings["replay_seeds"]:
                path = replays / str(seed)
                _, _, metadata = load_replay(path)
                if (metadata["game"], metadata["seed"], metadata["iteration"]) != (
                    "leduc",
                    seed,
                    120,
                ):
                    raise ValueError("Expected the original completed Leduc replay")
                manifest["replays"][str(seed)] = {
                    "manifest_sha256": sha256(
                        (path / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "replay_sha256": metadata["replay_sha256"],
                }
            write_json(output / "manifest.json", manifest)
        plan = Plan.from_dict(settings["benchmark"])
        jobs = [
            {
                "kind": "training",
                "plan": asdict(
                    replace(plan, training=replace(plan.training, seed=seed))
                ),
            }
            for seed in settings["benchmark_seeds"]
        ]
        deadline = started + settings["maximum_seconds"]
        for workers in [1] if replays is None else [1, settings["workers"]]:
            name = f"training-{workers}"
            report["phases"][name] = run_jobs(
                jobs,
                output / name,
                workers=workers,
                deadline=deadline,
                job_seconds=settings["job_seconds"],
            )
            write_json(output / "report.json", report)
        if replays is not None:
            serial = report["phases"]["training-1"]
            parallel = report["phases"][f"training-{settings['workers']}"]
            if not same_training(serial, parallel):
                raise ArithmeticError("Serial and parallel training results differ")
            report["parallel_equivalent"] = True
            report["throughput_speedup"] = (
                serial["wall_seconds"] / parallel["wall_seconds"]
            )
            # Require a meaningful gain before paying for concurrent fitting workers.
            chosen = settings["workers"] if report["throughput_speedup"] >= 1.2 else 1
            report["fitting_workers"] = chosen
            jobs = [
                {
                    "kind": "refit",
                    "replay": str((replays / str(seed)).resolve()),
                    "variant": variant["name"],
                    "seed": seed,
                    "fit": variant["fit"],
                    "maximum_seconds": settings["job_seconds"] - 20,
                }
                for variant in settings["variants"]
                for seed in settings["replay_seeds"]
            ]
            report["phases"]["fitting"] = run_jobs(
                jobs,
                output / "fitting",
                workers=chosen,
                deadline=deadline,
                job_seconds=settings["job_seconds"],
            )
            for seed, pin in manifest["replays"].items():
                path = replays / seed
                if (
                    sha256((path / "manifest.json").read_bytes()).hexdigest()
                    != pin["manifest_sha256"]
                    or sha256((path / "replay.npz").read_bytes()).hexdigest()
                    != pin["replay_sha256"]
                ):
                    raise ArithmeticError("Pilot input replay changed during execution")
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "timed_out" if isinstance(exc, TimeoutError) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = monotonic() - started
        report["model_promoted"] = False
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--training", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument(
        "--replays", type=Path, help="Enables the remote parallel and fitting phases"
    )
    args = parser.parse_args()
    if args.command == "export":
        export_replay(args.training, args.out)
    else:
        run_pilot(json.loads(args.plan.read_text()), args.out, args.replays)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
