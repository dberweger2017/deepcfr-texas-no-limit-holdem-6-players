"""Fixed-work trainer engineering only; execute on M4, never travelling M1."""

import argparse
import ast
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import subprocess
import time

from scripts.hu20_platform_pilot import canonical, environment, peak_rss, write

BASE = "7d74b6c"


def original_observe():
    """Execute the literal merged-base method, not a reimplemented control."""
    import src.game.hand as module
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:src/game/hand.py"], text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Hand")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "observe")
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[method], type_ignores=[]), "merged-base-observe", "exec"), namespace)
    return namespace["observe"]


def fingerprint(path):
    digest = sha256()
    with path.open("rb") as source:
        header = source.read(10)
        digest.update(header)
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    payload = sha256()
    count = 0
    with gzip.open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            payload.update(chunk)
            count += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size,
            "uncompressed_sha256": payload.hexdigest(), "uncompressed_bytes": count,
            "gzip_header_hex": header.hex()}


def install_trace():
    import src.blueprint.solver as solver
    import src.game.hand as hand_module
    from src.game.hand import Hand
    digests = {key: sha256() for key in ("observations", "keys_menus", "rng")}
    counts = {key: 0 for key in (*digests, "replays")}

    def record(kind, value):
        digests[kind].update(canonical(value) + b"\n")
        counts[kind] += 1

    observe = Hand.observe
    replay = hand_module.replay
    key_fn = solver.information_key

    def observed(self, seat, previous_hands=()):
        view = observe(self, seat, previous_hands)
        record("observations", asdict(view))
        return view

    def replayed(*args, **kwargs):
        counts["replays"] += 1
        return replay(*args, **kwargs)

    def keyed(view, menu, **kwargs):
        key = key_fn(view, menu, **kwargs)
        record("keys_menus", [key, [asdict(item) for item in menu]])
        return key

    class TracedRandom(Random):
        def __init__(self, seed):
            super().__init__(seed)
            record("rng", ["initial", seed, self.getstate()])

        def choices(self, *args, **kwargs):
            before = self.getstate()
            chosen = super().choices(*args, **kwargs)
            record("rng", ["choice", before, chosen, self.getstate()])
            return chosen

    Hand.observe = observed
    hand_module.replay = replayed
    solver.information_key = keyed
    solver.Random = TracedRandom
    return lambda: {"sha256": {key: value.hexdigest() for key, value in digests.items()}, "counts": counts}


def run(args):
    from src.blueprint.artifact import export_policy, load_training, save_training
    from src.blueprint.solver import BlueprintTrainer, PilotConfig, _seed
    from src.game.hand import Hand, Table
    import shutil

    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if args.variant == "original":
        Hand.observe = original_observe()
    trace = install_trace() if args.trace else None
    result = {"status": "running", "variant": args.variant, "trace": args.trace,
              "environment": environment(), "target_added_nodes": args.nodes,
              "parent": str(args.parent) if args.parent else None}
    write(args.out / "attempt.json", result)
    completed = 0
    durations = {}

    def timed(name, fn):
        before = time.monotonic()
        value = fn()
        durations[name] = time.monotonic() - before
        return value

    if args.resume:
        midpoint = json.loads((args.resume / "midpoint.json").read_text())
        checkpoint = args.resume / "midpoint.json.gz"
        if fingerprint(checkpoint) != midpoint["fingerprint"]:
            raise ValueError("Midpoint changed before resume")
        trainer = timed("load_seconds", lambda: load_training(checkpoint))
        completed = midpoint["added_nodes"]
        result["origin_entries"] = midpoint["origin_entries"]
        result["origin_iteration"] = midpoint["origin_iteration"]
        result["resume_added_nodes"] = completed
        reloaded = args.out / "reload.json.gz"
        timed("reload_save_seconds", lambda: save_training(trainer, reloaded))
        if fingerprint(reloaded) != midpoint["fingerprint"]:
            raise ValueError("Fresh-process reload bytes differ")
    elif args.parent:
        before = fingerprint(args.parent)
        if before["sha256"] != args.parent_sha256:
            raise ValueError("Mature input checkpoint hash mismatch")
        result["parent_fingerprint"] = before
        trainer = timed("load_seconds", lambda: load_training(args.parent))
    else:
        trainer = timed("load_seconds", lambda: BlueprintTrainer(
            Table(("player-0", "player-1"), (2000, 2000)),
            PilotConfig(seed=2026093011, raise_cap=None, roots_per_seat=1,
                        max_nodes=250000, max_entries=4000000, max_seconds=300,
                        abstraction="hu20-native-reopening-ordered-history-card-v1",
                        game="hu20-native-reopening-20bb-52card-no-ante-rake-v1")))
    result.setdefault("origin_entries", len(trainer.nodes))
    result.setdefault("origin_iteration", trainer.iteration)
    result["config"] = asdict(trainer.config)
    work_digest = sha256()
    training_seconds = 0.0
    try:
        with (args.out / "iterations.jsonl").open("w") as log:
            while completed < args.nodes:
                if (time.time() >= args.deadline or peak_rss() >= 10.5 * 2**30
                        or shutil.disk_usage(args.out).free < 8 * 2**30):
                    raise RuntimeError("Engineering deadline/RSS/disk guard")
                before = time.monotonic()
                report = trainer.step(workers=1, cancelled=lambda: time.time() >= args.deadline)
                training_seconds += time.monotonic() - before
                completed += report.nodes
                row = asdict(report)
                for key in ("elapsed_seconds", "replay_seconds", "worker_rss_sum_bytes"):
                    row.pop(key)
                row["added_nodes"] = completed
                work_digest.update(canonical(row) + b"\n")
                log.write(canonical(row).decode() + "\n")
                if (not args.resume and completed >= args.nodes // 2
                        and not (args.out / "midpoint.json").exists()):
                    checkpoint = args.out / "midpoint.json.gz"
                    timed("midpoint_save_seconds", lambda: save_training(trainer, checkpoint))
                    write(args.out / "midpoint.json", {
                        "added_nodes": completed, "iteration": trainer.iteration,
                        "origin_entries": result["origin_entries"],
                        "origin_iteration": result["origin_iteration"],
                        "fingerprint": fingerprint(checkpoint)})
        final = args.out / "final.json.gz"
        current = args.out / "current.json.gz"
        timed("final_save_seconds", lambda: save_training(trainer, final))
        timed("export_seconds", lambda: export_policy(trainer, current))
        result.update(added_nodes=completed, overshoot_nodes=completed-args.nodes,
                      iteration=trainer.iteration, entries=len(trainer.nodes),
                      new_entries=len(trainer.nodes)-result["origin_entries"],
                      work_sha256=work_digest.hexdigest(), training_seconds=training_seconds,
                      nodes_per_second=(completed-result.get("resume_added_nodes", 0))/training_seconds,
                      final=fingerprint(final), current=fingerprint(current))
        result["next_streams"] = [{"seat": seat,
            "deal": _seed(trainer.config.seed, trainer.iteration+1, seat, 0, "deal"),
            "actions": _seed(trainer.config.seed, trainer.iteration+1, seat, 0, "actions"),
            "rng_sha256": sha256(canonical(Random(_seed(trainer.config.seed,
                              trainer.iteration+1, seat, 0, "actions")).getstate())).hexdigest()}
            for seat in range(2)]
        next_report = trainer.step(workers=1, cancelled=lambda: time.time() >= args.deadline)
        next_path = args.out / "next.json.gz"
        timed("next_save_seconds", lambda: save_training(trainer, next_path))
        result.update(next=fingerprint(next_path), next_nodes=next_report.nodes, status="complete")
        if trace:
            result["trace_result"] = trace()
        if args.parent and fingerprint(args.parent) != result["parent_fingerprint"]:
            raise ValueError("Original input changed during benchmark")
    except Exception as exc:
        result.update(status="failed", failure=f"{type(exc).__name__}: {exc}",
                      added_nodes=completed, discarded_nodes=trainer.last_attempt_nodes,
                      discarded_work=trainer.last_attempt_work)
        save_training(trainer, args.out / "partial.json.gz")
        raise
    finally:
        result.update(durations=durations, elapsed_seconds=time.monotonic()-started,
                      peak_rss_bytes=peak_rss())
        write(args.out / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("original", "candidate"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-sha256")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    if args.nodes <= 0 or bool(args.parent) != bool(args.parent_sha256):
        parser.error("Positive fixed work and a hash for every parent required")
    run(args)


if __name__ == "__main__":
    main()
