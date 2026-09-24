"""Compare one frozen blueprint with and without bounded postflop search."""

import argparse
import json
import os
import resource
import sys
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from math import isfinite
from pathlib import Path
from time import monotonic

from src.arena.artifacts import environment, git, write_json
from src.arena.policies import make_policy
from src.arena.report import summarize
from src.arena.runner import run_schedule
from src.arena.schedule import Plan, build_schedule, digest
from src.blueprint.artifact import load_training
from src.blueprint.search import LiveBlueprint, SearchConfig, SearchPlayer


class CampaignLimitExceeded(RuntimeError):
    """The comparison stopped at a declared wall or memory boundary."""


class _Baseline:
    def __init__(self, blueprint, seed):
        from random import Random

        self.blueprint = blueprint
        self.random = Random(seed)

    def choose_action(self, view):
        menu, probabilities, _ = self.blueprint.distribution(view)
        return self.random.choices(menu, weights=probabilities, k=1)[0].action


def _sha256(path: Path) -> str:
    result = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _append(path: Path, value: dict):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def run(checkpoint: Path, expected_sha256: str, config: dict, out: Path) -> dict:
    started = monotonic()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("A full expected checkpoint SHA-256 is required")
    actual = _sha256(checkpoint)
    if actual != expected_sha256:
        raise ValueError("Blueprint checkpoint hash mismatch")
    search_config = SearchConfig(**config["search"])
    limits = config["execution"]
    if any(
        not isinstance(limits.get(key), (int, float))
        or not isfinite(limits[key]) or limits[key] <= 0
        for key in ("max_wall_seconds", "max_rss_gib")
    ):
        raise ValueError("Comparison needs positive wall and RSS limits")
    plans = {name: Plan.from_dict(value) for name, value in config["comparisons"].items()}
    if not plans or any(
        plan.candidate != "blueprint_search"
        or plan.baseline != "blueprint_live"
        or plan.split != "validation"
        or plan.models
        or any(s.mode != "fixed" for s in plan.scenarios)
        for plan in plans.values()
    ):
        raise ValueError("Search comparisons need fixed, paired validation plans")
    trainer = load_training(checkpoint)
    blueprint = LiveBlueprint(trainer)
    if monotonic() - started >= limits["max_wall_seconds"]:
        raise CampaignLimitExceeded("Checkpoint load reached the wall limit")
    if _rss_bytes() >= limits["max_rss_gib"] * 1024**3:
        raise CampaignLimitExceeded("Checkpoint load reached the RSS limit")
    if any(
        len(s.stacks) != trainer.table.capacity
        for plan in plans.values() for s in plan.scenarios
    ):
        raise ValueError("Comparison table size differs from the checkpoint")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual,
        "checkpoint_iteration": trainer.iteration,
        "checkpoint_entries": len(trainer.nodes),
        "comparison": config,
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "environment": environment(),
    })
    reports = {}
    stop_reason = None
    for name, plan in plans.items():
        rows = []
        candidate_latencies = []
        players = []

        def factory(policy_name, seed):
            if policy_name == "blueprint_search":
                player = SearchPlayer(blueprint, seed, search_config)
                players.append(player)
                return player
            if policy_name == "blueprint_live":
                return _Baseline(blueprint, seed)
            return make_policy(policy_name, seed)

        def emit(row, timing):
            compact = {
                key: value for key, value in row.items()
                if key not in {"events", "reloads", "outcome_sha256"}
            }
            compact["outcome_sha256"] = digest(compact)
            _append(out / f"{name}-hands.jsonl", compact)
            rows.append(compact)
            if row["arm"] == "candidate":
                candidate_latencies.extend(
                    item["seconds"] for item in timing["decisions"]
                    if item["player_id"] == "player-0"
                )
            if monotonic() - started >= limits["max_wall_seconds"]:
                raise CampaignLimitExceeded("Search comparison reached its wall limit")
            if _rss_bytes() >= limits["max_rss_gib"] * 1024**3:
                raise CampaignLimitExceeded("Search comparison reached its RSS limit")

        try:
            valid = run_schedule(plan, build_schedule(plan), emit, factory=factory)
        except CampaignLimitExceeded as exc:
            stop_reason = f"{type(exc).__name__}: {exc}"
            write_json(out / "stopped.json", {
                "benchmark": name, "reason": stop_reason,
                "retained_hands": len(rows), "peak_process_rss_bytes": _rss_bytes(),
            })
            break
        report = summarize(plan, rows)
        street_counts = sum((p.by_street for p in players), Counter())
        search_latencies = [value for p in players for value in p.search_seconds]
        telemetry = {
            "search_attempts": sum(p.attempts for p in players),
            "search_completed": sum(p.completed for p in players),
            "search_fallbacks": sum(p.fallbacks for p in players),
            "search_by_street": {
                street: {
                    metric: street_counts[(street, metric)]
                    for metric in ("attempts", "completed", "fallbacks")
                }
                for street in ("flop", "turn", "river")
            },
            "candidate_decisions": len(candidate_latencies),
            "decision_seconds_p50": _percentile(candidate_latencies, 0.5),
            "decision_seconds_p95": _percentile(candidate_latencies, 0.95),
            "decision_seconds_max": max(candidate_latencies, default=None),
            "search_seconds_p50": _percentile(search_latencies, 0.5),
            "search_seconds_p95": _percentile(search_latencies, 0.95),
            "search_seconds_max": max(search_latencies, default=None),
            "peak_process_rss_bytes": _rss_bytes(),
        }
        reports[name] = {"report": report, "telemetry": telemetry, "plan": asdict(plan)}
        write_json(out / f"{name}-report.json", reports[name])
        if not valid or report["status"] != "valid":
            break
    result = {
        "checkpoint_sha256": actual,
        "search": asdict(search_config),
        "comparisons": reports,
        "status": "stopped" if stop_reason else "valid" if len(reports) == len(plans) and all(
            item["report"]["status"] == "valid" for item in reports.values()
        ) else "failed",
    }
    write_json(out / "result.json", result)
    write_json(out / "checksums.json", {
        path.name: _sha256(path)
        for path in sorted(out.iterdir()) if path.is_file()
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        args.checkpoint, args.expected_sha256,
        json.loads(args.plan.read_text()), args.out,
    )
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
