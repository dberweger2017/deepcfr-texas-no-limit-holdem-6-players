"""Fresh-seed readiness for the exported snapshot average; no fitting sweep."""

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import monotonic

from scripts.cpu_pilot import host_details, run_jobs
from scripts.run_neural_readiness import rental_time_remaining
from src.solver.experiment import ROOT, write_json
from src.solver.neural.campaign import Campaign, run_seed, summarize
from src.solver.neural.experiment import provenance

PLAN = ROOT / "configs/solver/snapshot-readiness-v1.json"
PLAN_SHA256 = "bff6ea75dff3b26677dbd327ec35fd984b8d3f344906a9176d36b977f1c1814a"


def load_plan():
    data = PLAN.read_bytes()
    if sha256(data).hexdigest() != PLAN_SHA256:
        raise ValueError("Expected the frozen snapshot readiness plan")
    return json.loads(data)


def execute(settings, output, *, seconds, smoke=False, rental=None):
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "manifest.json",
        {
            **provenance(settings),
            "mode": "smoke" if smoke else "confirmation",
            "host": host_details(),
            "rental": rental,
            "effective_seconds": seconds,
        },
    )
    jobs = [
        {"campaign": value, "seed": seed}
        for value in settings["campaigns"].values()
        for seed in value["seeds"]
    ]
    report = {"status": "inconclusive", "scored": not smoke}
    try:
        resources = settings["resources"]
        report["execution"] = run_jobs(
            jobs,
            output / "jobs",
            workers=resources["workers"],
            deadline=monotonic() + seconds,
            job_seconds=resources["worker_seconds"],
            module="scripts.run_snapshot_readiness",
        )
        report["games"] = {
            game: summarize(
                Campaign.from_dict(value),
                [
                    output / "jobs" / f"job-{index:03d}" / "campaign"
                    for index, job in enumerate(jobs)
                    if job["campaign"]["training"]["game"] == game
                ],
                output / game,
            )
            for game, value in settings["campaigns"].items()
        }
        report["status"] = (
            "smoke_completed"
            if smoke
            else "passed"
            if all(g["status"] == "passed" for g in report["games"].values())
            else "failed"
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--job", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hourly-usd", type=float)
    parser.add_argument("--rental-started-at", type=datetime.fromisoformat)
    args = parser.parse_args()
    if args.job:
        job = json.loads(args.job.read_text())
        args.out.mkdir(parents=True, exist_ok=False)
        result = run_seed(
            Campaign.from_dict(job["campaign"]), job["seed"], args.out / "campaign"
        )
        report = {
            "status": "completed"
            if result["training_status"] == "completed"
            else "failed",
            "campaign": result,
        }
        write_json(args.out / "report.json", report)
        return
    settings = load_plan()
    rental = None
    if args.smoke:
        for game, value in settings["campaigns"].items():
            campaign = Campaign.from_dict(value)
            training = replace(
                campaign.training,
                iterations=2,
                evaluation_interval=1,
                maximum_seconds=60,
                execution="local",
                training=replace(
                    campaign.training.training,
                    hidden=8,
                    traversals=8,
                    advantage_steps=2,
                    batch_size=8,
                    capacity=32,
                ),
            )
            settings["campaigns"][game] = asdict(
                replace(campaign, training=training, seeds=(101, 103, 107))
            )
        settings["resources"].update(workers=1, worker_seconds=90)
        seconds = 300
    else:
        if args.hourly_usd is None or args.rental_started_at is None:
            parser.error("Confirmation requires the actual rental quote and start time")
        seconds = rental_time_remaining(
            settings["resources"],
            args.hourly_usd,
            args.rental_started_at,
            datetime.now(timezone.utc),
        )
        if provenance(settings)["dirty"]:
            parser.error("Confirmation requires a clean, committed checkout")
        rental = {
            "hourly_usd": args.hourly_usd,
            "started_at": args.rental_started_at.isoformat(),
        }
    result = execute(
        settings, args.out, seconds=seconds, smoke=args.smoke, rental=rental
    )
    print(
        json.dumps(
            {"status": result["status"], "report": str(args.out / "report.json")}
        )
    )
    return 0 if result["status"] in {"passed", "smoke_completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
