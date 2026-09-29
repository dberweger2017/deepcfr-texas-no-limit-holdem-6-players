"""Measure a saved blueprint on one host without changing its learning recipe."""

import argparse
import gc
import json
import resource
import sys
from dataclasses import asdict
from hashlib import sha256
from math import isfinite
from pathlib import Path
from time import monotonic

from src.arena.artifacts import environment, git, write_json
from src.arena.catalog import Checkpoint
from src.arena.registry import PolicyRegistry
from src.arena.run import run
from src.arena.schedule import Plan
from src.blueprint.artifact import export_policy, load_training, save_training
from src.blueprint.diagnostics import preflop_first_action
from src.blueprint.solver import BlueprintTrainer, FORMAT


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cgroup_bytes(name: str) -> int | None:
    path = Path("/sys/fs/cgroup") / name
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _manifest(args, source_hash: str) -> dict:
    return {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": source_hash,
        "revision": git("rev-parse", "HEAD"),
        "dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no")),
        "environment": environment(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def _count_held_out_lookups(blueprint) -> dict:
    """Count real arena queries without changing their returned decisions."""
    counts: dict[str, dict[str, int]] = {}
    distribution = blueprint.distribution

    def measured(view):
        menu, probabilities, trained = distribution(view)
        street = counts.setdefault(view.street.value, {"trained": 0, "fallback": 0})
        street["trained" if trained else "fallback"] += 1
        return menu, probabilities, trained

    blueprint.distribution = measured
    return counts


def _worker(args) -> int:
    if args.workers < 1 or args.steps < 1:
        raise ValueError("Workers and steps must be positive")
    if not isfinite(args.max_rss_gib) or args.max_rss_gib <= 0:
        raise ValueError("RSS limit must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    source_hash = _digest(args.checkpoint)
    write_json(args.out / "manifest.json", _manifest(args, source_hash))
    if source_hash != args.expected_sha256:
        raise ValueError("Source checkpoint hash differs from the frozen protocol")
    started = monotonic()
    trainer = load_training(args.checkpoint)
    loaded_seconds = monotonic() - started
    initial_iteration = trainer.iteration
    initial_entries = len(trainer.nodes)
    if _rss_bytes() >= args.max_rss_gib * 1024**3:
        raise MemoryError("Source checkpoint exceeds the benchmark RSS limit")
    reports = []
    for _ in range(args.steps):
        report = trainer.step(workers=args.workers)
        reports.append(report)
        if _rss_bytes() + report.worker_rss_sum_bytes >= args.max_rss_gib * 1024**3:
            write_json(
                args.out / "result.json",
                {
                    "status": "rss_stop",
                    "iteration": trainer.iteration,
                    "entries": len(trainer.nodes),
                    "peak_parent_rss_bytes": _rss_bytes(),
                    "worker_rss_sum_bytes": report.worker_rss_sum_bytes,
                },
            )
            return 2
    work_seconds = sum(report.elapsed_seconds for report in reports)
    save_started = monotonic()
    checkpoint = args.out / "checkpoint.json.gz"
    checkpoint_hash = save_training(trainer, checkpoint)
    save_seconds = monotonic() - save_started
    write_json(
        args.out / "result.json",
        {
            "status": "complete",
            "source_sha256": source_hash,
            "workers": args.workers,
            "steps": args.steps,
            "initial_iteration": initial_iteration,
            "iteration": trainer.iteration,
            "initial_entries": initial_entries,
            "entries": len(trainer.nodes),
            "nodes": sum(report.nodes for report in reports),
            "load_seconds": loaded_seconds,
            "work_seconds": work_seconds,
            "nodes_per_work_second": sum(report.nodes for report in reports) / work_seconds,
            "save_seconds": save_seconds,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_hash,
            "peak_parent_rss_bytes": _rss_bytes(),
            "max_worker_rss_sum_bytes": max(report.worker_rss_sum_bytes for report in reports),
            "cgroup_memory_peak_bytes": _cgroup_bytes("memory.peak"),
        },
    )
    return 0


def _learning(args) -> int:
    plan_data = json.loads(args.plan.read_text(encoding="utf-8"))
    if (
        plan_data.get("candidate") != "blueprint"
        or plan_data.get("baseline") != "blueprint_uniform"
    ):
        raise ValueError("Learning plan must compare blueprint with blueprint_uniform")
    if len(plan_data.get("scenarios", ())) != 1:
        raise ValueError("Learning check requires one frozen scenario")
    if not isfinite(args.max_rss_gib) or args.max_rss_gib <= 0:
        raise ValueError("RSS limit must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    source_hash = _digest(args.checkpoint)
    write_json(args.out / "manifest.json", {**_manifest(args, source_hash), "plan": plan_data})
    if source_hash != args.expected_sha256:
        raise ValueError("Source checkpoint hash differs from the frozen protocol")
    models_dir = args.out / "models"
    models_dir.mkdir()
    started = monotonic()
    trainer = load_training(args.checkpoint)
    table, config, iteration = trainer.table, trainer.config, trainer.iteration
    load_seconds = monotonic() - started
    if _rss_bytes() >= args.max_rss_gib * 1024**3:
        raise MemoryError("Source checkpoint exceeds the learning-check RSS limit")
    exported = {}
    for strategy in ("current", "average"):
        export_started = monotonic()
        temporary = models_dir / f"{strategy}.json.gz"
        digest = export_policy(trainer, temporary, strategy=strategy)
        final = models_dir / f"{digest}.json.gz"
        temporary.rename(final)
        exported[strategy] = {
            "sha256": digest,
            "bytes": final.stat().st_size,
            "seconds": monotonic() - export_started,
            "peak_rss_bytes": _rss_bytes(),
        }
        write_json(args.out / "exports.json", exported)
        if _rss_bytes() >= args.max_rss_gib * 1024**3:
            raise MemoryError("Policy export exceeded the learning-check RSS limit")
    uniform = BlueprintTrainer(table, config)
    temporary = models_dir / "uniform.json.gz"
    uniform_hash = export_policy(uniform, temporary)
    temporary.rename(models_dir / f"{uniform_hash}.json.gz")
    del trainer, uniform
    gc.collect()

    results = {}
    for strategy in ("current", "average"):
        candidate_hash = exported[strategy]["sha256"]
        specs = (
            Checkpoint(
                "blueprint",
                str(models_dir / f"{candidate_hash}.json.gz"),
                candidate_hash,
                FORMAT,
            ),
            Checkpoint(
                "blueprint_uniform",
                str(models_dir / f"{uniform_hash}.json.gz"),
                uniform_hash,
                FORMAT,
            ),
        )
        plan = Plan.from_dict({**plan_data, "models": [asdict(spec) for spec in specs]})
        registry = PolicyRegistry(plan, artifact_dir=models_dir)
        probe = preflop_first_action(registry.models["blueprint"], table)
        held_out_lookups = _count_held_out_lookups(registry.models["blueprint"])
        report = run(plan, args.out / strategy, registry=registry)
        results[strategy] = {
            "status": report["status"],
            "completed_hands": report["completed_hands"],
            "invalid_actions": report["invalid_actions"],
            "comparison": report["scenarios"][plan.scenarios[0].name]["comparison"],
            "preflop_probe": probe,
            "held_out_lookups": held_out_lookups,
            "arena_wall_seconds": report["performance"]["wall_seconds"],
            "peak_rss_bytes": _rss_bytes(),
        }
        write_json(
            args.out / "result.json",
            {
                "source_sha256": source_hash,
                "iteration": iteration,
                "load_seconds": load_seconds,
                "exports": exported,
                "uniform_sha256": uniform_hash,
                "strategies": results,
                "cgroup_memory_peak_bytes": _cgroup_bytes("memory.peak"),
            },
        )
        del registry
        gc.collect()
        if _rss_bytes() >= args.max_rss_gib * 1024**3:
            raise MemoryError("Arena exceeded the learning-check RSS limit")
    return 0 if all(result["status"] == "valid" for result in results.values()) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--checkpoint", required=True, type=Path)
    worker.add_argument("--expected-sha256", required=True)
    worker.add_argument("--out", required=True, type=Path)
    worker.add_argument("--workers", required=True, type=int)
    worker.add_argument("--steps", default=64, type=int)
    worker.add_argument("--max-rss-gib", default=48.0, type=float)
    learning = sub.add_parser("learning")
    learning.add_argument("--checkpoint", required=True, type=Path)
    learning.add_argument("--expected-sha256", required=True)
    learning.add_argument("--plan", required=True, type=Path)
    learning.add_argument("--out", required=True, type=Path)
    learning.add_argument("--max-rss-gib", default=48.0, type=float)
    args = parser.parse_args(argv)
    return _worker(args) if args.command == "worker" else _learning(args)


if __name__ == "__main__":
    raise SystemExit(main())
