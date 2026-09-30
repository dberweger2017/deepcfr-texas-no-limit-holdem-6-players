"""One frozen HU20 lineage; exact state checks, with no poker evaluation."""

import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
import platform
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import zlib


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def fingerprint(path):
    """Keep transport bytes and the entire uncompressed state separate."""
    data = path.read_bytes()
    payload = gzip.decompress(data)
    return {"sha256": sha256(data).hexdigest(), "bytes": len(data),
            "uncompressed_sha256": sha256(payload).hexdigest(),
            "uncompressed_bytes": len(payload), "gzip_header_hex": data[:10].hex()}


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def environment():
    import pokers
    import importlib.metadata
    distribution = importlib.metadata.distribution("pokers")
    binaries = {}
    for item in distribution.files or ():
        if str(item).endswith((".so", ".pyd")):
            path = distribution.locate_file(item)
            binaries[str(item)] = sha256(path.read_bytes()).hexdigest()
    return {"python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "zlib_build": zlib.ZLIB_VERSION, "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
            "engine_origin": distribution.read_text("direct_url.json"),
            "engine_binaries": binaries,
            "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}


def run(plan, out, resume=None):
    from src.blueprint.artifact import export_policy, load_training, save_training
    from src.blueprint.solver import BlueprintTrainer, PilotConfig, _seed
    from src.game.hand import Table
    from random import Random

    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + plan["max_seconds"]
    config = PilotConfig(**plan["config"])
    completed = 0
    if resume is None:
        trainer = BlueprintTrainer(Table(("player-0", "player-1"), (2000, 2000)), config)
        origin = {"from_zero": True, "entries": 0, "iteration": 0}
    else:
        metadata = json.loads((resume / "midpoint.json").read_text())
        checkpoint = resume / "midpoint.json.gz"
        if fingerprint(checkpoint) != metadata["checkpoint"]:
            raise ValueError("Midpoint checkpoint changed")
        trainer = load_training(checkpoint)
        if trainer.config != config or trainer.iteration != metadata["iteration"]:
            raise ValueError("Resume configuration/iteration changed")
        completed = metadata["completed_nodes"]
        origin = {"from_zero": False, "checkpoint": metadata["checkpoint"],
                  "completed_nodes": completed, "iteration": trainer.iteration}
        save_training(trainer, out / "reload.json.gz")
        if fingerprint(out / "reload.json.gz") != metadata["checkpoint"]:
            raise ValueError("Save/load did not reproduce the midpoint")
    result = {"status": "running", "plan_sha256": sha256(canonical(plan)).hexdigest(),
              "environment": environment(), "origin": origin, "config": asdict(config)}
    write(out / "attempt.json", result)
    save_training(trainer, out / "initial.json.gz")
    work_digest = sha256()
    training_seconds = 0.0
    try:
        with (out / "iterations.jsonl").open("w") as log:
            while completed < plan["completed_node_target"]:
                if (time.monotonic() >= deadline or peak_rss() > plan["max_rss_gib"] * 2**30
                        or shutil.disk_usage(out).free < plan["min_free_gib"] * 2**30):
                    raise RuntimeError("Time, RSS or disk pilot guard reached")
                before = time.monotonic()
                report = trainer.step(workers=1, cancelled=lambda: time.monotonic() >= deadline)
                training_seconds += time.monotonic() - before
                completed += report.nodes
                row = asdict(report)
                row.pop("updated_keys")
                for field in ("elapsed_seconds", "replay_seconds", "worker_rss_sum_bytes"):
                    row.pop(field)
                row["completed_nodes"] = completed
                work_digest.update(canonical(row) + b"\n")
                log.write(json.dumps(row, sort_keys=True) + "\n")
                if (resume is None and completed >= plan["resume_checkpoint_target"]
                        and not (out / "midpoint.json").exists()):
                    save_training(trainer, out / "midpoint.json.gz")
                    write(out / "midpoint.json", {"completed_nodes": completed,
                          "iteration": trainer.iteration,
                          "checkpoint": fingerprint(out / "midpoint.json.gz")})
        before = time.monotonic()
        save_training(trainer, out / "final.json.gz")
        checkpoint_seconds = time.monotonic() - before
        before = time.monotonic()
        export_policy(trainer, out / "current.json.gz")
        export_seconds = time.monotonic() - before
        # No persistent RNG object crosses an iteration boundary. Every root is
        # derived from seed/iteration/seat/sample/stream; record the next roots
        # and initial Python action-RNG state explicitly, without consuming it.
        streams = [{"seat": seat, "deal": _seed(config.seed, trainer.iteration + 1, seat, 0, "deal"),
                    "actions": _seed(config.seed, trainer.iteration + 1, seat, 0, "actions"),
                    "action_rng_state_sha256": sha256(canonical(Random(
                        _seed(config.seed, trainer.iteration + 1, seat, 0, "actions")).getstate())).hexdigest()}
                   for seat in range(2)]
        result.update(status="complete", completed_nodes=completed,
                      overshoot_nodes=completed-plan["completed_node_target"],
                      iteration=trainer.iteration, entries=len(trainer.nodes),
                      non_timing_work_sha256=work_digest.hexdigest(),
                      training_seconds=training_seconds,
                      training_nodes_per_second=(completed-origin.get("completed_nodes", 0))/training_seconds,
                      checkpoint_seconds=checkpoint_seconds, export_seconds=export_seconds,
                      final=fingerprint(out / "final.json.gz"),
                      current=fingerprint(out / "current.json.gz"), next_streams=streams)
        # Match the historical M4 next-iteration resume reference as well.
        next_report = trainer.step(workers=1, cancelled=lambda: time.monotonic() >= deadline)
        save_training(trainer, out / "next.json.gz")
        result.update(next=fingerprint(out / "next.json.gz"), next_nodes=next_report.nodes,
                      next_iteration=trainer.iteration)
    except Exception as exc:
        result.update(status="failed", failure=f"{type(exc).__name__}: {exc}",
                      completed_nodes=completed, iteration=trainer.iteration,
                      discarded_nodes=trainer.last_attempt_nodes,
                      discarded_work=trainer.last_attempt_work)
        save_training(trainer, out / "partial.json.gz")
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=peak_rss())
        write(out / "result.json", result)
    return result


def compare(left, right, out):
    """Run on Linux/M4 only; compare all trainer fields without tolerances."""
    result = {"left": str(left), "right": str(right), "files": {}, "equal": True}
    for name in ("final.json.gz", "current.json.gz", "next.json.gz"):
        a, b = left / name, right / name
        af, bf = fingerprint(a), fingerprint(b)
        # JSONL checkpoint records are sorted by key by the unchanged writer;
        # current exports are sorted JSON. Comparing every decompressed byte is
        # stronger than comparing selected nodes or a floating point tolerance.
        semantic_equal = af["uncompressed_sha256"] == bf["uncompressed_sha256"]
        row = {"left": af, "right": bf, "semantic_equal": semantic_equal,
               "byte_equal": af["sha256"] == bf["sha256"]}
        if not semantic_equal:
            aa = gzip.decompress(a.read_bytes()).decode().splitlines()
            bb = gzip.decompress(b.read_bytes()).decode().splitlines()
            first = next((i for i, (x, y) in enumerate(zip(aa, bb)) if x != y), min(len(aa), len(bb)))
            row["first_different_record"] = first
            row["left_record"] = json.loads(aa[first]) if first < len(aa) else None
            row["right_record"] = json.loads(bb[first]) if first < len(bb) else None
        result["files"][name] = row
        result["equal"] &= semantic_equal
    write(out, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("run")
    train.add_argument("--plan", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    check = sub.add_parser("compare")
    check.add_argument("--left", type=Path, required=True)
    check.add_argument("--right", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run(json.loads(args.plan.read_text()), args.out, args.resume)
    else:
        result = compare(args.left, args.right, args.out)
    print(json.dumps({k: result[k] for k in ("status", "equal") if k in result}))
    return result.get("status") == "failed" or result.get("equal") is False


if __name__ == "__main__":
    raise SystemExit(main())
