"""Run a bounded tabular blueprint pilot and evaluate its frozen export."""

import argparse
import json
import resource
import shutil
import signal
import sys
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path
from time import monotonic

from src.arena.artifacts import environment, git, write_json
from src.arena.catalog import Checkpoint
from src.arena.registry import PolicyRegistry
from src.arena.run import run
from src.arena.schedule import Plan
from src.blueprint.artifact import (
    FrozenBlueprint,
    export_policy,
    load_training,
    save_training,
)
from src.blueprint.diagnostics import preflop_first_action
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Table


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--checkpoint-seconds", type=float, default=1800)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-rss-gib", type=float)
    parser.add_argument("--min-free-gib", type=float)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if (
        args.workers < 1
        or not isfinite(args.checkpoint_seconds)
        or not 0 < args.checkpoint_seconds
        or any(
            value is not None and (not isfinite(value) or value <= 0)
            for value in (args.max_wall_seconds, args.max_rss_gib, args.min_free_gib)
        )
    ):
        parser.error("Execution limits must be positive")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    target = plan["iterations"] if args.stop_after is None else args.stop_after
    if type(target) is not int or not 0 <= target <= plan["iterations"]:
        parser.error("Stop after must lie between zero and planned iterations")
    table_data = plan["table"]
    table = Table(
        tuple(table_data["player_ids"]),
        tuple(table_data["stacks"]),
        table_data["button"],
        table_data["small_blind"],
        table_data["big_blind"],
        table_data["chip_unit"],
    )
    config = PilotConfig(**plan["trainer"])
    trainer = (
        load_training(args.resume) if args.resume else BlueprintTrainer(table, config)
    )
    if trainer.table != table or (
        trainer.config != config
        and (
            config.max_entries < trainer.config.max_entries
            or replace(trainer.config, max_entries=config.max_entries) != config
        )
    ):
        parser.error("Resume checkpoint differs from the frozen pilot plan")
    # The entry ceiling is an operational stop bound, not a learning parameter.
    # A measured increase can resume the same seed without changing its policy.
    trainer.config = config
    if trainer.iteration > target:
        parser.error("Resume checkpoint is beyond the requested stopping point")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "manifest.json",
        {
            "plan": plan,
            "revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "environment": environment(),
            "resumed_from": str(args.resume) if args.resume else None,
            "execution": {
                "checkpoint_seconds": args.checkpoint_seconds,
                "max_wall_seconds": args.max_wall_seconds,
                "max_rss_gib": args.max_rss_gib,
                "min_free_gib": args.min_free_gib,
                "workers": args.workers,
            },
        },
    )
    started = monotonic()
    last_checkpoint = started
    reports = []
    checkpoints = []
    checkpoint_hash = None
    stop_reason = None
    interrupted = False

    def request_stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    def save_boundary():
        nonlocal checkpoint_hash, last_checkpoint
        save_started = monotonic()
        checkpoint_hash = save_training(trainer, args.out / "checkpoint.json.gz")
        checkpoints.append(
            {
                "iteration": trainer.iteration,
                "seconds": monotonic() - save_started,
                "bytes": (args.out / "checkpoint.json.gz").stat().st_size,
                "sha256": checkpoint_hash,
            }
        )
        write_json(args.out / "iterations.json", reports)
        write_json(args.out / "checkpoints.json", checkpoints)
        last_checkpoint = monotonic()

    old_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for _ in range(trainer.iteration, target):
            report = trainer.step(workers=args.workers)
            aggregate_rss = _peak_rss_bytes() + report.worker_rss_sum_bytes
            reports.append(
                {
                    **asdict(report),
                    "nodes_per_second": report.nodes / report.elapsed_seconds,
                    "peak_rss_bytes": _peak_rss_bytes(),
                    "conservative_worker_plus_parent_rss_bytes": aggregate_rss,
                }
            )
            elapsed = monotonic() - started
            if interrupted:
                stop_reason = "signal"
            elif args.max_wall_seconds is not None and elapsed >= args.max_wall_seconds:
                stop_reason = "wall_time"
            elif (
                args.max_rss_gib is not None
                and aggregate_rss >= args.max_rss_gib * 1024**3
            ):
                stop_reason = "rss"
            elif (
                args.min_free_gib is not None
                and shutil.disk_usage(args.out).free <= args.min_free_gib * 1024**3
            ):
                stop_reason = "disk"
            if (
                checkpoint_hash is None
                or monotonic() - last_checkpoint >= args.checkpoint_seconds
                or stop_reason is not None
                or trainer.iteration == target
            ):
                save_boundary()
            if stop_reason is not None:
                break
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    if checkpoint_hash is None:
        save_boundary()
    if trainer.iteration < target:
        write_json(
            args.out / "result.json",
            {
                "iteration": trainer.iteration,
                "entries": len(trainer.nodes),
                "checkpoint_sha256": checkpoint_hash,
                "status": "stopped",
                "stop_reason": stop_reason,
                "peak_rss_bytes": _peak_rss_bytes(),
            },
        )
        return 2

    policy_path = args.out / "policy.json.gz"
    policy_hash = export_policy(trainer, policy_path)
    models_dir = args.out / "models"
    models_dir.mkdir()
    (models_dir / f"{policy_hash}.json.gz").write_bytes(policy_path.read_bytes())
    eval_plan = Plan.from_dict(
        {
            **plan["evaluation"],
            "models": [
                asdict(
                    Checkpoint(
                        "blueprint",
                        str(policy_path),
                        policy_hash,
                        "holdem-blueprint-v1",
                    )
                )
            ],
        }
    )
    frozen = FrozenBlueprint(eval_plan.models[0], models_dir / f"{policy_hash}.json.gz")
    diagnostic = preflop_first_action(frozen, trainer.table)
    write_json(args.out / "card_probe.json", diagnostic)
    evaluation = run(
        eval_plan,
        args.out / "evaluation",
        registry=PolicyRegistry(eval_plan, artifact_dir=models_dir),
    )
    result = {
        "iteration": trainer.iteration,
        "entries": len(trainer.nodes),
        "checkpoint_sha256": checkpoint_hash,
        "policy_sha256": policy_hash,
        "evaluation_status": evaluation["status"],
        "completed_hands": evaluation["completed_hands"],
        "preflop_trained_classes": diagnostic["trained_infosets"],
        "preflop_distinct_distributions": diagnostic["distinct_distributions"],
        "strength_claim": False,
        "peak_rss_bytes": _peak_rss_bytes(),
        "checkpoints": len(checkpoints),
    }
    write_json(args.out / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if evaluation["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
