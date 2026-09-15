"""Run the frozen readiness campaign or a short check on non-confirmation seeds."""

import argparse
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import monotonic

from scripts.cpu_pilot import host_details, run_jobs
from src.solver.experiment import ROOT, write_json
from src.solver.neural.campaign import Campaign, summarize
from src.solver.neural.experiment import provenance
from src.solver.neural.readiness import run_job

PLAN = ROOT / "configs/solver/neural-readiness-v1.json"
PLAN_SHA256 = "c856f6430128896f65c43f1fb05a87a5f4cdc3c5053dfd25ed2a5767098a6de8"


def load_plan():
    raw = PLAN.read_bytes()
    if sha256(raw).hexdigest() != PLAN_SHA256:
        raise ValueError("Expected the frozen neural-readiness-v1 plan")
    return json.loads(raw)


def smoke_plan(settings):
    settings = json.loads(json.dumps(settings))
    for game, value in settings["campaigns"].items():
        campaign = Campaign.from_dict(value)
        plan = replace(
            campaign.training,
            iterations=2,
            evaluation_interval=1,
            maximum_seconds=60,
            execution="local",
            training=replace(
                campaign.training.training,
                hidden=8,
                strategy_hidden=8,
                traversals=8,
                advantage_steps=2,
                strategy_steps=3,
                batch_size=8,
                capacity=32,
            ),
        )
        settings["campaigns"][game] = asdict(
            replace(campaign, training=plan, seeds=(101, 103, 107))
        )
    settings["comparison"]["maximum_seconds"] = 30
    settings["resources"].update(workers=1, worker_seconds=90, maximum_seconds=300)
    return settings


def rental_time_remaining(resources, hourly_usd, started_at, now):
    if (
        not math.isfinite(hourly_usd)
        or not 0 < hourly_usd <= resources["maximum_hourly_usd"]
    ):
        raise ValueError("Quote exceeds the frozen all-in hourly limit")
    if started_at.tzinfo is None:
        raise ValueError("Rental start needs an explicit timezone")
    elapsed = (now - started_at).total_seconds()
    if elapsed < 0:
        raise ValueError("Rental start cannot be in the future")
    cap = min(resources["rental_seconds"], resources["maximum_usd"] / hourly_usd * 3600)
    remaining = min(
        resources["maximum_seconds"],
        cap - elapsed - resources["retrieval_reserve_seconds"],
    )
    if remaining <= 0:
        raise ValueError("Rental has no training time left after the retrieval reserve")
    return remaining


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
        {
            "campaign": value,
            "seed": seed,
            "comparison_seconds": settings["comparison"]["maximum_seconds"]
            if game == settings["comparison"]["game"]
            else None,
        }
        for game, value in settings["campaigns"].items()
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
            module="scripts.run_neural_readiness",
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
        comparisons = [
            row["report"]["comparison"]
            for row in report["execution"]["jobs"]
            if "comparison" in row["report"]
        ]
        expected = settings["campaigns"][settings["comparison"]["game"]]["seeds"]
        if [row["seed"] for row in comparisons] != list(expected):
            raise ValueError("Every declared paired comparison must be retained")
        report["paired_comparisons"] = comparisons
        report["regressions"] = [
            r["seed"] for r in comparisons if r["exploitability_delta"] > 0
        ]
        report["status"] = (
            "smoke_completed"
            if smoke
            else "passed"
            if all(r["status"] == "passed" for r in report["games"].values())
            else "failed"
        )
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
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
        run_job(json.loads(args.job.read_text()), args.out)
        return 0
    settings = load_plan()
    rental = None
    if args.smoke:
        settings = smoke_plan(settings)
        seconds = settings["resources"]["maximum_seconds"]
    else:
        if args.hourly_usd is None or args.rental_started_at is None:
            parser.error("--run requires --hourly-usd and --rental-started-at")
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
