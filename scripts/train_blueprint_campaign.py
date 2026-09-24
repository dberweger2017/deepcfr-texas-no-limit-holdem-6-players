"""Resume one blueprint seed with retained checkpoints and live poker measurements."""

import argparse
import json
import os
import resource
import shutil
import signal
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from math import isfinite
from pathlib import Path
from time import monotonic

from src.arena.artifacts import environment, git, write_json
from src.arena.schedule import Plan
from src.blueprint.artifact import load_training, save_training
from src.blueprint.evaluation import evaluate
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Table


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _append(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _hash(path: Path):
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--max-rss-gib", type=float, required=True)
    parser.add_argument("--min-free-gib", type=float, required=True)
    parser.add_argument("--checkpoint-seconds", type=float, default=3600)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--starting-nodes", type=int)
    parser.add_argument("--confirm-final", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or any(
        not isfinite(value) or value <= 0
        for value in (args.max_wall_seconds, args.max_rss_gib, args.min_free_gib, args.checkpoint_seconds)
    ):
        parser.error("Execution limits must be positive")
    plan = json.loads(args.plan.read_text())
    campaign = json.loads(args.campaign.read_text())
    starting_nodes = campaign["source_nodes"] if args.starting_nodes is None else args.starting_nodes
    if type(starting_nodes) is not int or starting_nodes < 0:
        parser.error("Starting nodes must be a nonnegative integer")
    milestones = campaign["entry_milestones"]
    if (
        not milestones
        or any(type(v) is not int or v <= 0 for v in milestones)
        or sorted(set(milestones)) != milestones
    ):
        parser.error("Entry milestones must be distinct ascending positive integers")
    evaluations = {name: Plan.from_dict(value) for name, value in campaign["evaluations"].items()}
    if set(evaluations) != {"random", "styles"}:
        parser.error("Campaign requires random and styles evaluations")
    confirmation = Plan.from_dict(campaign["confirmation"])
    if confirmation.split != "test" or confirmation.opponents != ("random",):
        parser.error("Confirmation must use fresh test-split random deals")
    table_data = plan["table"]
    table = Table(
        tuple(table_data["player_ids"]), tuple(table_data["stacks"]),
        table_data["button"], table_data["small_blind"],
        table_data["big_blind"], table_data["chip_unit"],
    )
    config = PilotConfig(**plan["trainer"])
    trainer = load_training(args.resume)
    source_hash = _hash(args.resume)
    if args.starting_nodes is None and (
        trainer.iteration != campaign["source_iteration"]
        or source_hash != campaign["source_sha256"]
    ):
        parser.error("Initial source checkpoint does not match the frozen campaign")
    if trainer.table != table or (
        trainer.config != config
        and (config.max_entries < trainer.config.max_entries
             or replace(trainer.config, max_entries=config.max_entries) != config)
    ):
        parser.error("Resume checkpoint differs from the frozen campaign plan")
    trainer.config = config
    if trainer.iteration >= plan["iterations"]:
        parser.error("Resume checkpoint has reached target iterations")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "snapshots").mkdir()
    (args.out / "evaluations").mkdir()
    write_json(args.out / "manifest.json", {
        "plan": plan, "campaign": campaign, "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")), "environment": environment(),
        "resumed_from": str(args.resume), "source_sha256": source_hash,
        "execution": {
            "max_wall_seconds": args.max_wall_seconds, "max_rss_gib": args.max_rss_gib,
            "min_free_gib": args.min_free_gib, "checkpoint_seconds": args.checkpoint_seconds,
            "workers": args.workers, "confirm_final": args.confirm_final,
            "starting_nodes": starting_nodes,
        },
    })
    started = monotonic()
    last_checkpoint = started
    total_nodes = starting_nodes
    phase_nodes = 0
    total_step_seconds = 0.0
    last_worker_rss = 0
    last_evaluated_iteration = -1
    next_milestone = next((v for v in milestones if v > len(trainer.nodes)), None)
    interrupted = False
    stop_reason = None

    def request_stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    def save_boundary(*, measure=False):
        nonlocal last_checkpoint, last_evaluated_iteration
        saved_at = monotonic()
        path = args.out / "checkpoint.json.gz"
        checkpoint_hash = save_training(trainer, path)
        snapshot = None
        if measure and trainer.iteration != last_evaluated_iteration:
            snapshot = args.out / "snapshots" / f"iteration-{trainer.iteration}.json.gz"
            os.link(path, snapshot)
        row = {
            "iteration": trainer.iteration, "entries": len(trainer.nodes),
            "nodes": total_nodes, "step_seconds": total_step_seconds,
            "wall_seconds": monotonic() - started,
            "nodes_per_step_second": phase_nodes / total_step_seconds if total_step_seconds else None,
            "peak_rss_bytes": _rss_bytes(),
            "conservative_rss_bytes": _rss_bytes() + last_worker_rss,
            "checkpoint_bytes": path.stat().st_size,
            "checkpoint_seconds": monotonic() - saved_at,
            "checkpoint_sha256": checkpoint_hash,
            "snapshot": str(snapshot) if snapshot else None,
        }
        _append(args.out / "progress.jsonl", row)
        last_checkpoint = monotonic()
        if snapshot is not None:
            for name, eval_plan in evaluations.items():
                measured = evaluate(trainer, eval_plan)
                output = args.out / "evaluations" / f"iteration-{trainer.iteration}-{name}.json"
                write_json(output, measured)
                scenario = measured["report"]["scenarios"][eval_plan.scenarios[0].name]
                _append(args.out / "evaluation.jsonl", {
                    "iteration": trainer.iteration, "entries": len(trainer.nodes),
                    "checkpoint_sha256": checkpoint_hash, "benchmark": name,
                    "status": measured["report"]["status"],
                    "completed_hands": measured["report"]["completed_hands"],
                    "comparison": scenario["comparison"],
                    "coverage": measured["coverage"],
                    "schedule_sha256": measured["report"]["schedule_sha256"],
                })
                if measured["report"]["status"] != "valid":
                    raise RuntimeError(f"Invalid {name} arena evaluation at iteration {trainer.iteration}")
            last_evaluated_iteration = trainer.iteration
        return checkpoint_hash

    old_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    checkpoint_hash = None
    try:
        checkpoint_hash = save_boundary(measure=True)
        for _ in range(trainer.iteration, plan["iterations"]):
            report = trainer.step(workers=args.workers)
            total_nodes += report.nodes
            phase_nodes += report.nodes
            total_step_seconds += report.elapsed_seconds
            last_worker_rss = max(last_worker_rss, report.worker_rss_sum_bytes)
            if interrupted:
                stop_reason = "signal"
            elif monotonic() - started >= args.max_wall_seconds:
                stop_reason = "wall_time"
            elif _rss_bytes() + last_worker_rss >= args.max_rss_gib * 1024**3:
                stop_reason = "rss"
            elif shutil.disk_usage(args.out).free <= args.min_free_gib * 1024**3:
                stop_reason = "disk"
            milestone = next_milestone is not None and len(trainer.nodes) >= next_milestone
            if milestone:
                next_milestone = next((v for v in milestones if v > len(trainer.nodes)), None)
            if milestone or stop_reason or monotonic() - last_checkpoint >= args.checkpoint_seconds or trainer.iteration == plan["iterations"]:
                checkpoint_hash = save_boundary(measure=bool(milestone or stop_reason or trainer.iteration == plan["iterations"]))
            if stop_reason:
                break
    except Exception as exc:
        write_json(args.out / "failure.json", {
            "iteration": trainer.iteration, "entries": len(trainer.nodes),
            "checkpoint_sha256": checkpoint_hash,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    result = {
        "status": "complete" if trainer.iteration == plan["iterations"] else "stopped",
        "stop_reason": stop_reason, "iteration": trainer.iteration,
        "entries": len(trainer.nodes), "nodes": total_nodes,
        "checkpoint_sha256": checkpoint_hash,
        "peak_rss_bytes": _rss_bytes(),
    }
    if args.confirm_final:
        if last_evaluated_iteration != trainer.iteration:
            checkpoint_hash = save_boundary(measure=True)
            result["checkpoint_sha256"] = checkpoint_hash
        try:
            final = evaluate(trainer, confirmation)
        except Exception as exc:
            write_json(args.out / "failure.json", {
                "iteration": trainer.iteration, "entries": len(trainer.nodes),
                "checkpoint_sha256": checkpoint_hash,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        write_json(args.out / "evaluations" / f"iteration-{trainer.iteration}-random-final.json", final)
        scenario = final["report"]["scenarios"][confirmation.scenarios[0].name]
        _append(args.out / "evaluation.jsonl", {
            "iteration": trainer.iteration, "entries": len(trainer.nodes),
            "checkpoint_sha256": checkpoint_hash, "benchmark": "random_final",
            "status": final["report"]["status"],
            "completed_hands": final["report"]["completed_hands"],
            "comparison": scenario["comparison"],
            "coverage": final["coverage"],
            "schedule_sha256": final["report"]["schedule_sha256"],
        })
        if final["report"]["status"] != "valid":
            write_json(args.out / "failure.json", {
                "iteration": trainer.iteration, "entries": len(trainer.nodes),
                "checkpoint_sha256": checkpoint_hash,
                "error": "Invalid final random arena evaluation",
            })
            raise RuntimeError("Invalid final random arena evaluation")
    write_json(args.out / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
