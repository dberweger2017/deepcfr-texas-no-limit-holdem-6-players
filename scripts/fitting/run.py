"""Import, smoke-check, or execute the predeclared strategy-fitting experiment."""

import argparse
import json
import platform
import resource
import sys
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from time import monotonic, perf_counter

import torch

from scripts.cpu_pilot import host_details, run_jobs
from scripts.fitting.data import (
    PLAN,
    digest,
    export_inputs,
    load_input,
    load_plan,
    manifest,
)
from scripts.fitting.optimizer import fitting_seed, optimize
from scripts.fitting.report import diagnose, screen
from src.solver.evaluate import evaluate
from src.solver.experiment import canonical, write_json
from src.solver.neural.checkpoint import atomic_write
from src.solver.neural.network import deterministic_cpu
from src.solver.sequence_form import solve


def memory_hash(memory):
    value = sha256()
    for name in ("infos", "iterations", "targets"):
        value.update(getattr(memory, name)[: memory.size].tobytes())
    return value.hexdigest()


def worker(plan, job, output):
    recipe = next(r for r in plan["recipes"] if r["name"] == job["recipe"])
    if (
        type(job["replicate"]) is not int
        or job["replicate"] not in plan["fit_replicates"]
        or job["mode"] not in ("scored", "smoke")
    ):
        raise ValueError("Unexpected fit replicate or mode")
    output.mkdir(parents=True, exist_ok=False)
    began = perf_counter()
    scored = job["mode"] == "scored"
    report = {
        "status": "running",
        "scored": scored,
        "seed": job["seed"],
        "replicate": job["replicate"],
        "recipe": recipe["name"],
        "evaluations": [],
    }
    write_json(output / "manifest.json", {**manifest(plan), "job": job})
    try:
        if scored:
            require_runtime()
        solver, memory, origin = load_input(
            plan, Path(job["inputs"]), job["seed"], job["input_index_sha256"]
        )
        fixed = deepcopy(plan["fixed"])
        if not scored:
            fixed["steps"] = plan["resources"]["local_smoke_steps_per_recipe"]
            fixed["evaluation_steps"] = [fixed["steps"]]
        deadline = began + (
            plan["resources"]["per_fit_seconds"] - 5
            if scored
            else plan["resources"]["local_smoke_maximum_seconds"]
        )
        seed = fitting_seed(job["seed"], job["replicate"], origin["iteration"])
        report.update(
            fit_seed=seed,
            replay_sha256=origin["replay_sha256"],
            input_memory_sha256=memory_hash(memory),
        )
        with deterministic_cpu():
            oracle = solve(solver.tree)
            oracle_evaluation = evaluate(solver.tree, oracle.policy).to_dict()
            if (
                max(
                    oracle.upper_value - oracle.lower_value,
                    oracle.maximum_residual,
                    oracle_evaluation["exploitability"],
                )
                > 1e-8
            ):
                raise ArithmeticError("Independent equilibrium check failed")
            report["oracle"] = {
                "value": oracle.lower_value,
                "evaluation": oracle_evaluation,
            }

            def observe(step, model, optimizer):
                row, infos = diagnose(
                    solver, memory, model, origin["iteration"], oracle.lower_value
                )
                info_path = output / f"information-sets-{step:06d}.json"
                write_json(info_path, infos)
                row.update(
                    step=step,
                    optimizer=optimizer,
                    information_sets_sha256=digest(info_path),
                    elapsed_seconds=perf_counter() - began,
                    cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime
                    + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                )
                report["evaluations"].append(row)
                write_json(output / "report.json", report)

            model = optimize(
                solver,
                memory,
                recipe,
                fixed,
                seed,
                origin["iteration"],
                deadline,
                observe,
            )
        if memory_hash(memory) != report["input_memory_sha256"]:
            raise ArithmeticError("Frozen replay changed during fitting")
        if scored and recipe["name"] == "minibatch-fixed" and job["replicate"] == 0:
            spec = next(s for s in plan["source"]["inputs"] if s["seed"] == job["seed"])
            if (
                report["evaluations"][-1]["policy_sha256"]
                != spec["original_control_policy_sha256"]
            ):
                raise ArithmeticError("Original control policy hash did not reproduce")
            report["original_control_reproduced"] = True
        data = BytesIO()
        torch.save(
            {
                "weights": model.state_dict(),
                "fixed": fixed,
                "recipe": recipe,
                "fit_seed": seed,
                "replay_sha256": origin["replay_sha256"],
            },
            data,
        )
        report["model_file_sha256"] = atomic_write(output / "model.pt", data.getvalue())
        report["status"] = "completed"
    except BaseException as exc:
        report["status"], report["error"] = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = perf_counter() - began
        report["peak_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        write_json(output / "report.json", report)
    return report


def verify_artifacts(path, report):
    if digest(path / "model.pt") != report["model_file_sha256"]:
        raise ValueError("Final model hash differs")
    for row in report["evaluations"]:
        if (
            digest(path / f"information-sets-{row['step']:06d}.json")
            != row["information_sets_sha256"]
        ):
            raise ValueError("Information-set report hash differs")
    saved = json.loads((path / "manifest.json").read_text())
    current = manifest(saved["plan"])
    for field in ("source_sha256", "diagnostic_source_sha256"):
        if saved[field] != current[field]:
            raise ValueError("Study source changed during execution")


def require_runtime():
    if (
        sys.platform != "linux"
        or platform.python_version_tuple()[:2] != ("3", "11")
        or any(
            version(k) != v
            for k, v in {
                "torch": "2.5.1+cpu",
                "numpy": "1.26.4",
                "scipy": "1.17.1",
            }.items()
        )
    ):
        raise ValueError(
            "Scored fits require the declared Linux Python 3.11 CPU runtime"
        )


def execute(plan, inputs, output, workers, mode, seconds):
    if (
        type(workers) is not int
        or not 1 <= workers <= plan["resources"]["maximum_workers"]
    ):
        raise ValueError("Invalid worker count")
    if not 0 < seconds <= 3300:
        raise ValueError("Reserve rental time for retrieval and shutdown")
    if mode not in ("smoke", "scored"):
        raise ValueError("Unknown run mode")
    if mode == "smoke" and workers != 1:
        raise ValueError("Local smoke must be serial")
    if mode == "scored":
        require_runtime()
    began = monotonic()
    output.mkdir(parents=True, exist_ok=False)
    index_hash = digest(inputs / "index.json")
    write_json(
        output / "manifest.json",
        {
            **manifest(plan),
            "input_index_sha256": index_hash,
            "host": host_details(),
            "mode": mode,
            "workers": workers,
            "maximum_seconds": seconds,
        },
    )
    jobs = [
        {
            "inputs": str(inputs.resolve()),
            "input_index_sha256": index_hash,
            "seed": s["seed"],
            "replicate": r,
            "recipe": recipe["name"],
            "mode": mode,
        }
        for recipe in plan["recipes"]
        for r in plan["fit_replicates"]
        for s in plan["source"]["inputs"]
    ]
    report = {"status": "running", "mode": mode, "model_promoted": False}
    try:
        if mode == "smoke":
            deadline = began + min(
                seconds, plan["resources"]["local_smoke_maximum_seconds"]
            )
            selected = [
                j for j in jobs if j["seed"] == jobs[0]["seed"] and j["replicate"] == 0
            ]
            result = run_jobs(
                selected,
                output / "smoke",
                workers=1,
                deadline=deadline,
                job_seconds=120,
                module="scripts.fitting.run",
            )
            report["jobs"] = result["jobs"]
        else:
            deadline = began + seconds
            control = run_jobs(
                jobs[:1],
                output / "control",
                workers=1,
                deadline=deadline,
                job_seconds=180,
                module="scripts.fitting.run",
            )
            report["control"] = control
            write_json(output / "report.json", report)
            remaining = jobs[1:]
            # Allow a factor of two for contention plus final reporting/retrieval.
            projected = (
                control["wall_seconds"]
                * ((len(remaining) + workers - 1) // workers)
                * 2
                + 120
            )
            if monotonic() + projected >= deadline:
                raise TimeoutError(
                    "Control timing leaves insufficient budget for the complete matrix"
                )
            batch = run_jobs(
                remaining[:workers],
                output / "first-batch",
                workers=workers,
                deadline=deadline,
                job_seconds=180,
                module="scripts.fitting.run",
            )
            report["first_batch"] = batch
            write_json(output / "report.json", report)
            remaining = remaining[workers:]
            projected = (
                batch["wall_seconds"]
                * ((len(remaining) + workers - 1) // workers)
                * 1.25
                + 120
            )
            if monotonic() + projected >= deadline:
                raise TimeoutError(
                    "Concurrent throughput leaves insufficient budget for the complete matrix"
                )
            rest = run_jobs(
                remaining,
                output / "remaining",
                workers=workers,
                deadline=deadline,
                job_seconds=180,
                module="scripts.fitting.run",
            )
            report["remaining"] = rest
            for directory, batch_result in (
                ("control", control),
                ("first-batch", batch),
                ("remaining", rest),
            ):
                for job in batch_result["jobs"]:
                    verify_artifacts(
                        output / directory / f"job-{job['index']:03d}", job["report"]
                    )
            reports = [j["report"] for b in (control, batch, rest) for j in b["jobs"]]
            report["screen"] = screen(plan, reports)
        report["status"] = "completed"
    except BaseException as exc:
        report["status"], report["error"] = (
            "inconclusive",
            f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        report["wall_seconds"] = monotonic() - began
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--phase", choices=("import", "smoke", "scored"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=2400)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.job:
        worker(plan, json.loads(args.job.read_text()), args.out)
    elif args.phase == "import":
        if args.archive is None:
            parser.error("Import requires --archive")
        export_inputs(plan, args.archive, args.out)
    elif args.phase in ("smoke", "scored") and args.inputs is not None:
        execute(plan, args.inputs, args.out, args.workers, args.phase, args.seconds)
    else:
        parser.error("Choose a phase and its inputs")
    print(canonical({"status": "completed", "output": str(args.out)}))


if __name__ == "__main__":
    main()
