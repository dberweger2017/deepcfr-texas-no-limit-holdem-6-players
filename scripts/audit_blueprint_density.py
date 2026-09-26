"""Stream one hash-pinned blueprint JSONL checkpoint into bounded histograms."""

import argparse
import gzip
import json
import resource
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import monotonic, time

from src.arena.artifacts import environment, git, write_json
from src.blueprint.abstraction import SCHEMA
from src.blueprint.lookup import BUTTON_ZERO_CHECKPOINTS


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _file_hash(path):
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(hist, count, fraction):
    target = max(1, int((count - 1) * fraction) + 1)
    reached = 0
    for value, frequency in sorted(hist.items()):
        reached += frequency
        if reached >= target:
            return value
    raise AssertionError("Empty visit histogram")


def audit(path: Path, expected_sha256: str, *, max_wall_seconds: int = 7200,
          max_rss_gib: float = 10.5):
    started = monotonic()
    if expected_sha256 not in BUTTON_ZERO_CHECKPOINTS:
        raise ValueError("Checkpoint is outside the verified button-zero lineage")
    digest = _file_hash(path)
    if digest != expected_sha256:
        raise ValueError("Checkpoint SHA-256 mismatch")
    if monotonic() - started >= max_wall_seconds:
        raise TimeoutError("Checkpoint hash verification reached the wall limit")
    with gzip.open(path, "rt", encoding="utf-8") as source:
        header = json.loads(source.readline())
        if (header.get("kind") != "training" or
                header.get("checkpoint_format") != "jsonl-v2" or
                header.get("abstraction") != SCHEMA or
                header.get("iteration") != BUTTON_ZERO_CHECKPOINTS[digest] or
                header["table"]["button"] != 0 or
                tuple(header["table"]["stacks"]) != (10_000,) * 6):
            raise ValueError("Checkpoint header differs from declared provenance")
        hist = Counter()
        entries = visits_total = both_fold_check = near_pure = 0
        for line in source:
            key, names, regrets, averages, visits = json.loads(line)
            if (not isinstance(key, str) or len(key) != 32 or
                    type(visits) is not int or visits < 0 or
                    not names or len(names) != len(regrets) or len(names) != len(averages)):
                raise ValueError(f"Invalid checkpoint row {entries + 1}")
            entries += 1
            visits_total += visits
            hist[visits] += 1
            both_fold_check += "fold" in names and "check" in names
            positive = [max(0.0, value) for value in regrets]
            total = sum(positive)
            maximum = max(positive) / total if total > 0 else 1 / len(names)
            near_pure += maximum >= 0.99
            if entries % 100_000 == 0:
                if monotonic() - started >= max_wall_seconds:
                    raise TimeoutError("Checkpoint density audit reached its wall limit")
                if _rss_bytes() >= max_rss_gib * 1024**3:
                    raise MemoryError("Checkpoint density audit reached its RSS limit")
    if not entries:
        raise ValueError("Checkpoint has no entries")
    return {
        "checkpoint_sha256": digest,
        "file_bytes": path.stat().st_size,
        "iteration": header["iteration"],
        "abstraction": header["abstraction"],
        "training_table": header["table"],
        "training_config": header["config"],
        "entries": entries,
        "visits_total": visits_total,
        "visits_mean": visits_total / entries,
        "visit_histogram": {str(value): frequency for value, frequency in sorted(hist.items())},
        "visits_median": _quantile(hist, entries, 0.5),
        "visits_p90": _quantile(hist, entries, 0.9),
        "visits_p95": _quantile(hist, entries, 0.95),
        "visits_p99": _quantile(hist, entries, 0.99),
        "visits_p999": _quantile(hist, entries, 0.999),
        "fraction_one_visit": hist[1] / entries,
        "fraction_at_most_two": sum(n for v, n in hist.items() if v <= 2) / entries,
        "fraction_at_most_five": sum(n for v, n in hist.items() if v <= 5) / entries,
        "fraction_at_least_twenty": sum(n for v, n in hist.items() if v >= 20) / entries,
        "fold_and_check_entries": both_fold_check,
        "fraction_fold_and_check": both_fold_check / entries,
        "near_pure_current_threshold": 0.99,
        "near_pure_current_entries": near_pure,
        "near_pure_current_fraction": near_pure / entries,
        "near_pure_note": "descriptive current regret-matched policies, not a quality test",
        "elapsed_seconds": monotonic() - started,
        "peak_process_rss_bytes": _rss_bytes(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    parser.add_argument("--max-rss-gib", type=float, default=10.5)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output already exists")
    result = audit(args.checkpoint, args.expected_sha256,
                   max_wall_seconds=args.max_wall_seconds,
                   max_rss_gib=args.max_rss_gib)
    write_json(args.out, {
        "schema": "blueprint-checkpoint-density-v1",
        "source_revision": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain")),
        "environment": environment(),
        "started_unix_seconds": time() - result["elapsed_seconds"],
        "result": result,
    })
    print(json.dumps({key: result[key] for key in
                      ("checkpoint_sha256", "entries", "visits_total",
                       "elapsed_seconds", "peak_process_rss_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
