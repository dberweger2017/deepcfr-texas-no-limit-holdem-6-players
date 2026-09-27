"""Replay a fixed K1 continuation with read-only policy-window captures."""

import argparse
import gzip
import json
import os
import resource
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from time import monotonic, time

from src.blueprint.artifact import load_training, save_training
from src.blueprint.lookup import REPLICATION_PARENT
from src.blueprint.windowed import (EXTRACTION, _hash, build_index,
                                    collect_preflop, write_snapshot)


def rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def system(command):
    if sys.platform != "darwin":
        return None
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with path.open("a") as out:
        out.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


def guard(out, deadline, limits):
    if time() > deadline:
        raise TimeoutError("Ten-hour campaign deadline")
    if rss() > limits["max_rss_gib"] * 1024**3:
        raise MemoryError("Process RSS ceiling")
    if shutil.disk_usage(out).free < limits["min_free_gib"] * 1024**3:
        raise RuntimeError("Free-disk ceiling")


def run(plan, parent, original, lineage_file, out, seed, deadline, *, preflight=False):
    if out.exists():
        raise FileExistsError(out)
    if _hash(parent) != REPLICATION_PARENT:
        raise ValueError("Parent hash mismatch")
    lineage = json.loads(lineage_file.read_text())
    if seed not in plan["continuation_seeds"] or lineage["continuation_seed"] != seed:
        raise ValueError("Continuation seed mismatch")
    if _hash(original) != lineage["output_checkpoint_sha256"]:
        raise ValueError("Original K1 hash mismatch")
    out.mkdir(parents=True)
    started = monotonic()
    manifest = {"schema": EXTRACTION, "seed": seed, "parent_sha256": REPLICATION_PARENT,
                "original_checkpoint_sha256": _hash(original), "lineage_sha256": _hash(lineage_file),
                "plan": plan, "preflight": preflight, "swap_before": system(["sysctl", "vm.swapusage"]),
                "memory_pressure_before": system(["memory_pressure", "-Q"])}
    write_json(out / "run.json", manifest)
    result = {"status": "incomplete", "stage": "load", "stop_reason": None}
    snapshots, counters = [], {}
    try:
        guard(out, deadline, plan["limits"])
        trainer = load_training(original if preflight else parent)
        result["load_seconds"] = monotonic() - started
        if preflight:
            # One full-sized profile and one root per seat measure read-only
            # capture/collection cost without opening any poker outcomes.
            capture_start = monotonic()
            snap = out / "resource-snapshot.jsonl.gz"
            result["snapshot_sha256"] = write_snapshot(trainer, snap)
            result["snapshot_seconds"] = monotonic() - capture_start
            result["snapshot_bytes"] = snap.stat().st_size
            collect_start = monotonic()
            result["collector"] = collect_preflop(trainer, 0, plan["collector_roots_per_seat"], plan["collector_seed"],
                                                   counters, max_visited=plan["limits"]["max_collector_states"])
            result["collector_seconds"] = monotonic() - collect_start
            result["status"] = "complete"
            return result
        trainer.config = replace(trainer.config, seed=seed, postflop_replicates=1,
                                 max_nodes=plan["limits"]["max_nodes_per_iteration"],
                                 max_entries=plan["limits"]["max_entries"],
                                 max_seconds=plan["limits"]["max_seconds_per_iteration"])
        total_nodes = 0
        next_capture = 0
        while total_nodes < plan["additional_nodes_per_run"]:
            guard(out, deadline, plan["limits"])
            report = trainer.step()
            total_nodes += report.nodes
            append(out / "iterations.jsonl", {"iteration": trainer.iteration,
                   "additional_nodes": total_nodes, "nodes": report.nodes,
                   "elapsed_seconds": monotonic()-started, "rss_bytes": rss()})
            if next_capture < 8 and total_nodes >= plan["capture_nodes"][next_capture]:
                capture_start = monotonic()
                path = out / f"snapshot-{next_capture}.jsonl.gz"
                digest = write_snapshot(trainer, path)
                guard(out, deadline, plan["limits"])
                collected = collect_preflop(trainer, next_capture,
                                            plan["collector_roots_per_seat"],
                                            plan["collector_seed"], counters,
                                            max_visited=plan["limits"]["max_collector_states"])
                guard(out, deadline, plan["limits"])
                row = {"index": next_capture, "requested_nodes": plan["capture_nodes"][next_capture],
                       "completed_nodes": total_nodes, "iteration": trainer.iteration,
                       "snapshot_sha256": digest, "snapshot_bytes": path.stat().st_size,
                       "collector": collected, "capture_seconds": monotonic()-capture_start,
                       "rss_bytes": rss()}
                append(out / "captures.jsonl", row)
                snapshots.append(path)
                next_capture += 1
        if next_capture != 8 or total_nodes > 20_250_000:
            raise ValueError("Incomplete or overshot published capture window")
        result["stage"] = "checkpoint"
        checkpoint = out / "replay-checkpoint.json.gz"
        result["replay_checkpoint_sha256"] = save_training(trainer, checkpoint)
        result["checkpoint_byte_identical"] = _hash(checkpoint) == _hash(original)
        if not result["checkpoint_byte_identical"]:
            raise ValueError("Replay checkpoint differs from original K1")
        result["stage"] = "index"
        with gzip.open(out / "preflop-counters.json.gz", "wt", encoding="utf-8") as saved:
            json.dump(counters, saved, sort_keys=True, separators=(",", ":"))
        index = out / "policy-index.sqlite"
        index_stats = build_index(snapshots, counters, index)
        capture_rows = [json.loads(line) for line in (out / "captures.jsonl").read_text().splitlines()]
        identity = {"schema": EXTRACTION, "source_checkpoint_sha256": _hash(original),
                    "source_parent_sha256": REPLICATION_PARENT,
                    "source_lineage_sha256": _hash(lineage_file),
                    "snapshot_sha256": [row["snapshot_sha256"] for row in capture_rows],
                    "requested_nodes": plan["capture_nodes"],
                    "completed_nodes": [row["completed_nodes"] for row in capture_rows],
                    "iterations": [row["iteration"] for row in capture_rows],
                    "snapshot_weights": [0.125]*8,
                    "collector_seed": plan["collector_seed"],
                    "collector_roots_per_seat": plan["collector_roots_per_seat"],
                    "preflop_fallback": "final-current when no collected action mass",
                    "postflop_fallback": "uniform per absent profile",
                    "abstraction": trainer.config.abstraction, "raise_cap": trainer.config.raise_cap,
                    "lookup_mode": "button-zero-compatible-v1",
                    "artifact_sha256": index_stats["artifact_sha256"],
                    "index_stats": index_stats}
        write_json(out / "policy-manifest.json", identity)
        result.update(status="complete", stage="done", additional_nodes=total_nodes,
                      final_iteration=trainer.iteration, index_stats=index_stats)
    except Exception as exc:
        result["stop_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        result.update(elapsed_seconds=monotonic()-started,
                      peak_process_rss_bytes=rss(),
                      swap_after=system(["sysctl", "vm.swapusage"]),
                      memory_pressure_after=system(["memory_pressure", "-Q"]))
        write_json(out / "result.json", result)
        write_json(out / "checksums.json", {str(path.relative_to(out)): _hash(path)
                   for path in out.rglob("*") if path.is_file() and path.name != "checksums.json"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run(json.loads(args.plan.read_text()), args.parent, args.original,
                 args.lineage, args.out, args.seed, args.deadline, preflight=args.preflight)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
