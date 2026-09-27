"""Outcome-free, separate-process M4 throughput and memory preflight."""

import argparse
import json
import resource
import subprocess
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import monotonic

from src.arena.artifacts import environment, git, write_json
from src.blueprint.artifact import load_training
from src.blueprint.lookup import REPLICATION_PARENT


def _hash(path):
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _system(command):
    if sys.platform != "darwin":
        return None
    result = subprocess.run(command, text=True, capture_output=True,
                            check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replicates", type=int, choices=(1, 4), required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026092701)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.out.exists():
        parser.error("A positive step count and new output path are required")
    if _hash(args.checkpoint) != REPLICATION_PARENT:
        parser.error("The immutable parent checkpoint SHA-256 differs")
    args.out.mkdir(parents=True)
    started = monotonic()
    manifest = {
        "schema": "postflop-replication-resource-preflight-v1",
        "parent_checkpoint_sha256": REPLICATION_PARENT,
        "source_revision": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain")),
        "replicates": args.replicates, "steps": args.steps, "seed": args.seed,
        "environment": environment(),
        "swap_before": _system(["sysctl", "vm.swapusage"]),
        "memory_pressure_before": _system(["memory_pressure", "-Q"]),
    }
    write_json(args.out / "manifest.json", manifest)
    rows = []
    status, reason = "complete", None
    try:
        trainer = load_training(args.checkpoint)
        load_seconds = monotonic() - started
        trainer.config = replace(trainer.config, seed=args.seed,
                                 postflop_replicates=args.replicates,
                                 max_nodes=250_000, max_seconds=300)
        if _rss() >= 10.5 * 1024**3:
            raise MemoryError("Parent load exceeds the M4 RSS cap")
        for _ in range(args.steps):
            report = trainer.step()
            rows.append({**asdict(report), "updated_keys": len(report.updated_keys),
                         "rss_bytes": _rss()})
            write_json(args.out / "steps.json", rows)
            if _rss() >= 10.5 * 1024**3:
                raise MemoryError("Preflight exceeds the M4 RSS cap")
    except Exception as exc:
        status, reason = "failed", f"{type(exc).__name__}: {exc}"
        load_seconds = locals().get("load_seconds")
    result = {
        "status": status, "reason": reason, "completed_steps": len(rows),
        "load_seconds": load_seconds,
        "elapsed_seconds": monotonic() - started,
        "completed_nodes": sum(row["nodes"] for row in rows),
        "peak_process_rss_bytes": _rss(),
        "swap_after": _system(["sysctl", "vm.swapusage"]),
        "memory_pressure_after": _system(["memory_pressure", "-Q"]),
    }
    write_json(args.out / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
