"""Calibrate CPU throughput, explore strategy fits, then confirm a frozen selection."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic

from scripts.cpu_pilot import host_details, run_jobs, same_training
from src.solver.experiment import canonical, write_json
from src.solver.neural.campaign import summarize
from src.solver.neural.experiment import provenance
from src.solver.neural.study import Study, load_exploration, select


def calibrate(study, output, deadline):
    base = study.exploration.training
    plan = replace(
        base,
        iterations=24,
        evaluation_interval=24,
        maximum_seconds=800,
        execution="local",
        training=replace(base.training, seed=887),
    )
    sample = run_jobs(
        [{"kind": "training", "plan": asdict(plan)}],
        output / "representative",
        workers=1,
        deadline=deadline,
        job_seconds=840,
    )
    plan = replace(plan, iterations=4, evaluation_interval=4)
    jobs = [
        {
            "kind": "training",
            "plan": asdict(
                replace(plan, training=replace(plan.training, seed=907 + i))
            ),
        }
        for i in range(16)
    ]
    scales = {}
    for workers in (4, 8, 16):
        scales[workers] = run_jobs(
            jobs,
            output / f"workers-{workers}",
            workers=workers,
            deadline=deadline,
            job_seconds=840,
        )
        if not same_training(scales[4], scales[workers]):
            raise ArithmeticError("Worker counts changed the training results")
        write_json(
            output / "calibration-progress.json",
            {"representative": sample, "scales": scales},
        )
    fastest = min(scales, key=lambda n: scales[n]["wall_seconds"])
    return {
        "representative": sample,
        "scales": scales,
        "suggested_workers": fastest,
        "parallel_equivalent": True,
        "projected_collection_seconds_per_seed": sample["wall_seconds"]
        * base.iterations
        / 24,
    }


def execute(study, phase, output, workers, exploration=None):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Use between one and sixteen independent workers")
    if phase == "confirm" and exploration is None:
        raise ValueError("Confirmation requires the complete exploration bundle")
    started = monotonic()
    deadline = started + study.maximum_seconds
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "manifest.json",
        {
            **provenance(asdict(study)),
            "phase": phase,
            "workers": workers,
            "host": host_details(),
        },
    )
    report = {"status": "running", "phase": phase}
    try:
        if phase == "calibrate":
            report.update(calibrate(study, output, deadline))
        elif phase == "explore":
            jobs = [
                {"kind": "explore", "study": asdict(study), "seed": seed}
                for seed in study.exploration.seeds
            ]
            batch = run_jobs(
                jobs,
                output / "seeds",
                workers=workers,
                deadline=deadline,
                job_seconds=study.worker_seconds + 60,
                module="scripts.study_worker",
            )
            report["batch"] = batch
            report["selection"] = select(
                study, [row["report"]["result"] for row in batch["jobs"]]
            )
            write_json(output / "selection.json", report["selection"])
        elif phase == "confirm":
            paths = [
                exploration / "seeds" / f"job-{i:03d}" / "study"
                for i in range(len(study.exploration.seeds))
            ]
            reports, hashes = load_exploration(study, paths)
            selection = select(study, reports)
            stored = json.loads((exploration / "selection.json").read_text())
            if (
                canonical(selection) != canonical(stored)
                or selection["selected"] is None
            ):
                raise ValueError("No eligible frozen selection to confirm")
            campaign = study.confirmation(selection["selected"])
            write_json(
                output / "selection.json",
                {
                    **selection,
                    "exploration_report_hashes": hashes,
                    "campaign": asdict(campaign),
                },
            )
            report["gates"] = {}
            for name, spec in (("leduc", campaign), ("kuhn", study.kuhn)):
                jobs = [
                    {"kind": "confirm", "campaign": asdict(spec), "seed": seed}
                    for seed in spec.seeds
                ]
                run_jobs(
                    jobs,
                    output / name,
                    workers=workers,
                    deadline=deadline,
                    job_seconds=study.worker_seconds + 60,
                    module="scripts.study_worker",
                )
                paths = [
                    output / name / f"job-{i:03d}" / "campaign"
                    for i in range(len(spec.seeds))
                ]
                report["gates"][name] = summarize(
                    spec, paths, output / f"{name}-summary"
                )
                write_json(output / "report.json", report)
            report["gate_passed"] = all(
                gate["status"] == "passed" for gate in report["gates"].values()
            )
        else:
            raise ValueError("Unknown study phase")
        report["status"] = "completed"
    except BaseException as exc:
        report["status"], report["error"] = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = monotonic() - started
        report["model_promoted"] = False
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("calibrate", "explore", "confirm"), required=True
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--exploration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = execute(
        Study.from_dict(json.loads(args.study.read_text())),
        args.phase,
        args.out,
        args.workers,
        args.exploration,
    )
    print(
        json.dumps(
            {"status": result["status"], "report": str(args.out / "report.json")}
        )
    )


if __name__ == "__main__":
    main()
