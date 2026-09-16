"""Replay a saved collection root with explicit limits, without fitting a model."""

import argparse
import cProfile
import json
import platform
import resource
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from src.arena.artifacts import environment, git, source_fingerprint
from src.game.hand import Hand
from src.holdem.checkpoint import load_training
from src.holdem.collection import (
    CollectionLimitExceeded,
    collect_traversal,
    collection_seed,
)


def traversal_digest(result):
    digest = sha256()
    # Stream records to avoid materializing a second copy of a large traversal.
    for record in (result.root, *result.targets, *result.executions):
        digest.update(
            json.dumps(
                asdict(record), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        digest.update(b"\n")
    digest.update(repr((result.value_bb, result.nodes, result.terminals)).encode())
    return digest.hexdigest()


def load_root(run, job, iteration, traverser, sample):
    manifest = json.loads((run / "manifest.json").read_text())
    marker = json.loads((run / job / f"training-{iteration}.json").read_text())
    checkpoint = run / job / f"training-{iteration}.pt"
    trainer = load_training(checkpoint, marker["sha256"], manifest=manifest)
    if trainer.iteration != iteration:
        raise ValueError("Checkpoint iteration differs from its marker")
    config = trainer.config
    if not 0 <= traverser < len(trainer.table.stacks) or sample < 0:
        raise ValueError("Invalid traversal coordinates")
    next_iteration = iteration + 1
    table = (
        replace(
            trainer.table,
            button=(trainer.table.button + iteration) % len(trainer.table.stacks),
        )
        if config.rotate_button
        else trainer.table
    )
    seat = table.seat_numbers[traverser]
    hand = Hand.start(
        table,
        hand_id=f"collection-{next_iteration}-{seat}-{sample}",
        seed=collection_seed(config.seed, next_iteration, seat, sample, "deal"),
    )
    return trainer, hand, marker, manifest


def profile_root(
    run,
    job,
    iteration,
    traverser,
    sample,
    max_nodes,
    max_seconds,
    out,
    *,
    instrument=False,
):
    out.mkdir(parents=True, exist_ok=False)
    trainer, hand, marker, manifest = load_root(run, job, iteration, traverser, sample)
    config = trainer.config
    next_iteration = iteration + 1
    table = hand.table
    seat = table.seat_numbers[traverser]
    profile = trainer.current_profile()
    profiler = cProfile.Profile() if instrument else None
    report = {
        "kind": "holdem-collection-profile-v1",
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": source_fingerprint(),
        "environment": environment(),
        "checkpoint_sha256": marker["sha256"],
        "checkpoint_manifest": manifest,
        "profile_sha256": profile.fingerprint,
        "table": asdict(table),
        "iteration": next_iteration,
        "seed": config.seed,
        "traverser": traverser,
        "sample": sample,
        "max_nodes": max_nodes,
        "max_seconds": max_seconds,
        "instrumented": instrument,
    }
    started = perf_counter()
    if profiler:
        profiler.enable()
    try:
        result = collect_traversal(
            hand,
            profile,
            traverser,
            iteration=next_iteration,
            action_seed=collection_seed(
                config.seed, next_iteration, seat, sample, "opponents"
            ),
            max_nodes=max_nodes,
            deadline=started + max_seconds,
        )
    except CollectionLimitExceeded as error:
        report.update(status="collection_limit", error=str(error))
        result = None
    finally:
        if profiler:
            profiler.disable()
        report["collection_seconds"] = perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"] = (
            rss if platform.system() == "Darwin" else rss * 1024
        )
    if profiler:
        profiler.dump_stats(out / "collection.prof")
    if result is not None:
        report.update(
            status="completed",
            nodes=result.nodes,
            terminals=result.terminals,
            targets=len(result.targets),
            executions=len(result.executions),
            value_bb=result.value_bb,
            traversal_sha256=traversal_digest(result),
        )
    report["total_seconds"] = perf_counter() - started
    (out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument(
        "--iteration",
        type=int,
        required=True,
        help="Completed checkpoint iteration; collect the following one",
    )
    parser.add_argument("--traverser", type=int, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=50_000)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_nodes < 1 or not 0 < args.max_seconds <= 300 or args.iteration < 1:
        parser.error("Use positive node/iteration limits and at most 300 seconds")
    report = profile_root(
        args.run,
        args.job,
        args.iteration,
        args.traverser,
        args.sample,
        args.max_nodes,
        args.max_seconds,
        args.out,
        instrument=args.instrument,
    )
    print(
        json.dumps(
            {k: report[k] for k in ("status", "collection_seconds", "total_seconds")}
        )
    )


if __name__ == "__main__":
    main()
