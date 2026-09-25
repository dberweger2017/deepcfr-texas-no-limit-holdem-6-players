"""Compare a frozen blueprint's rollout or local-CFR player on paired deals."""

import argparse
import json
import os
import resource
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from math import isfinite
from pathlib import Path
from time import monotonic

from src.arena.artifacts import environment, git, write_json
from src.arena.policies import make_policy
from src.arena.report import estimate, summarize
from src.arena.runner import run_schedule
from src.arena.schedule import Plan, build_schedule, digest
from src.blueprint.artifact import load_training
from src.blueprint.local_cfr import LocalCFRConfig, LocalCFRPlayer
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


def _log_progress(writer, name, scenario, rates, players, step):
    prefix = f"{name}/{scenario}"
    for arm in ("candidate", "baseline"):
        value = estimate(rates[arm])
        writer.add_scalar(f"{prefix}/{arm}_bb_per_100", value["bb_per_100"], step)
        if value["ci95"] is not None:
            writer.add_scalar(f"{prefix}/{arm}_ci95_lower", value["ci95"][0], step)
            writer.add_scalar(f"{prefix}/{arm}_ci95_upper", value["ci95"][1], step)
        group = players[arm]
        writer.add_scalar(f"{prefix}/{arm}_search_attempts", sum(p.attempts for p in group), step)
        writer.add_scalar(f"{prefix}/{arm}_search_fallbacks", sum(p.fallbacks for p in group), step)
        latencies = [value for p in group for value in p.search_seconds]
        if latencies:
            writer.add_scalar(
                f"{prefix}/{arm}_search_p95_seconds", _percentile(latencies, 0.95), step,
            )
        cycles = [cycle for player in group for cycle in getattr(player, "cycles", ())]
        if any(hasattr(player, "cycles") for player in group):
            attempts = sum(player.attempts for player in group)
            completed = sum(player.completed for player in group)
            writer.add_scalar(
                f"{prefix}/{arm}_solver_completion_rate",
                completed / attempts if attempts else 0.0, step,
            )
            writer.add_scalar(
                f"{prefix}/{arm}_solver_range_updates",
                sum(player.range_updates for player in group), step,
            )
        if cycles:
            writer.add_scalar(f"{prefix}/{arm}_solver_cycles_min", min(cycles), step)
            writer.add_scalar(f"{prefix}/{arm}_solver_cycles_median", _percentile(cycles, 0.5), step)
            writer.add_scalar(f"{prefix}/{arm}_solver_nodes", sum(p.nodes for p in group), step)
            writer.add_scalar(f"{prefix}/{arm}_solver_leaf_choices", sum(p.leaf_choices for p in group), step)
            for cycle in (16, 32, 64, 128):
                snapshots = [snapshot for player in group
                             for attempt in getattr(player, "attempt_records", ())
                             for snapshot in attempt["diagnostics"]
                             if snapshot["cycle"] == cycle]
                if not snapshots:
                    continue
                for key in ("target_l1_from_previous", "mean_positive_regret",
                            "max_positive_regret", "action_infosets", "leaf_infosets"):
                    values = [snapshot[key] for snapshot in snapshots
                              if snapshot[key] is not None]
                    if values:
                        writer.add_scalar(f"{prefix}/{arm}/cycle_{cycle}/{key}",
                                          sum(values) / len(values), step)
                for index, style in enumerate(("blueprint", "fold", "call", "raise")):
                    writer.add_scalar(f"{prefix}/{arm}/cycle_{cycle}/leaf_{style}",
                                      sum(snapshot["leaf_style_mean_policy"][index]
                                          for snapshot in snapshots) / len(snapshots), step)
    paired = estimate([a - b for a, b in zip(rates["candidate"], rates["baseline"], strict=True)])
    writer.add_scalar(f"{prefix}/paired_bb_per_100", paired["bb_per_100"], step)
    if paired["ci95"] is not None:
        writer.add_scalar(f"{prefix}/paired_ci95_lower", paired["ci95"][0], step)
        writer.add_scalar(f"{prefix}/paired_ci95_upper", paired["ci95"][1], step)
    writer.add_scalar(f"{prefix}/peak_rss_gib", _rss_bytes() / 1024**3, step)
    writer.flush()


def run(
    checkpoint: Path, expected_sha256: str, config: dict, out: Path,
    *, tensorboard: bool = False,
) -> dict:
    started = monotonic()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("A full expected checkpoint SHA-256 is required")
    actual = _sha256(checkpoint)
    if actual != expected_sha256:
        raise ValueError("Blueprint checkpoint hash mismatch")
    search_config = SearchConfig(**config["search"])
    local_config = LocalCFRConfig(**config["local_cfr"]) if "local_cfr" in config else None
    limits = config["execution"]
    if any(
        not isinstance(limits.get(key), (int, float))
        or not isfinite(limits[key]) or limits[key] <= 0
        for key in ("max_wall_seconds", "max_rss_gib")
    ):
        raise ValueError("Comparison needs positive wall and RSS limits")
    min_free_gib = limits.get("min_free_gib", 0)
    if not isinstance(min_free_gib, (int, float)) or not isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("Comparison needs a nonnegative free-disk guard")
    plans = {name: Plan.from_dict(value) for name, value in config["comparisons"].items()}
    if not plans or any(
        plan.candidate != ("blueprint_local_cfr" if local_config else "blueprint_search")
        or plan.baseline not in ({"blueprint_search"} if local_config else {"blueprint_live", "blueprint_search_original"})
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
    out.parent.mkdir(parents=True, exist_ok=True)
    if min_free_gib and shutil.disk_usage(out.parent).free <= min_free_gib * 1024**3:
        raise CampaignLimitExceeded("Checkpoint load reached the free-disk guard")
    if any(
        len(s.stacks) != trainer.table.capacity
        for plan in plans.values() for s in plan.scenarios
    ):
        raise ValueError("Comparison table size differs from the checkpoint")
    out.mkdir(parents=True, exist_ok=False)
    writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(out / "tensorboard"))
    write_json(out / "manifest.json", {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual,
        "checkpoint_iteration": trainer.iteration,
        "checkpoint_entries": len(trainer.nodes),
        "comparison": config,
        "tensorboard_every_blocks": (8 if local_config else 64) if tensorboard else None,
        "tensorboard_heartbeat_seconds": 300 if local_config and tensorboard else None,
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "environment": environment(),
    })
    reports = {}
    stop_reason = None
    for name, plan in plans.items():
        rows = []
        candidate_latencies = []
        players = {"candidate": [], "baseline": []}
        block_rows = {}
        rates = {
            scenario.name: {"candidate": [], "baseline": []}
            for scenario in plan.scenarios
        }
        scenarios = {scenario.name: scenario for scenario in plan.scenarios}
        last_progress = monotonic()

        def factory(policy_name, seed):
            if policy_name == "blueprint_local_cfr":
                player = LocalCFRPlayer(blueprint, seed, local_config, search_config)
                players["candidate"].append(player)
                return player
            if policy_name == "blueprint_search":
                player = SearchPlayer(blueprint, seed, search_config)
                players["baseline" if local_config else "candidate"].append(player)
                return player
            if policy_name == "blueprint_search_original":
                player = SearchPlayer(
                    blueprint, seed,
                    SearchConfig(**{**asdict(search_config), "variant": "original"}),
                )
                players["baseline"].append(player)
                return player
            if policy_name == "blueprint_live":
                return _Baseline(blueprint, seed)
            return make_policy(policy_name, seed)

        def emit(row, timing):
            nonlocal last_progress
            compact = {
                key: value for key, value in row.items()
                if key not in {"events", "reloads", "outcome_sha256"}
            }
            compact["outcome_sha256"] = digest(compact)
            _append(out / f"{name}-hands.jsonl", compact)
            rows.append(compact)
            if writer is not None:
                scenario = scenarios[row["scenario"]]
                key = (row["scenario"], row["block"])
                group = block_rows.setdefault(key, [])
                group.append(compact)
                expected = 2 * len(scenario.stacks) * scenario.hands_per_rotation
                if len(group) == expected:
                    if all(item["status"] == "completed" for item in group):
                        for arm in ("candidate", "baseline"):
                            outcomes = [
                                item["candidate_chips"] for item in group if item["arm"] == arm
                            ]
                            rates[scenario.name][arm].append(
                                100 * sum(outcomes) / (len(outcomes) * scenario.big_blind)
                            )
                        step = len(rates[scenario.name]["candidate"])
                        cadence = 8 if local_config else 64
                        if step % cadence == 0 or step == plan.blocks \
                                or local_config and monotonic() - last_progress >= 300:
                            _log_progress(
                                writer, name, scenario.name, rates[scenario.name], players, step,
                            )
                            last_progress = monotonic()
                    del block_rows[key]
            if row["arm"] == "candidate":
                candidate_latencies.extend(
                    item["seconds"] for item in timing["decisions"]
                    if item["player_id"] == "player-0"
                )
            if monotonic() - started >= limits["max_wall_seconds"]:
                raise CampaignLimitExceeded("Search comparison reached its wall limit")
            if _rss_bytes() >= limits["max_rss_gib"] * 1024**3:
                raise CampaignLimitExceeded("Search comparison reached its RSS limit")
            if min_free_gib and shutil.disk_usage(out).free <= min_free_gib * 1024**3:
                raise CampaignLimitExceeded("Search comparison reached its free-disk guard")

        try:
            valid = run_schedule(plan, build_schedule(plan), emit, factory=factory)
        except CampaignLimitExceeded as exc:
            stop_reason = f"{type(exc).__name__}: {exc}"
            write_json(out / "stopped.json", {
                "benchmark": name, "reason": stop_reason,
                "retained_hands": len(rows), "peak_process_rss_bytes": _rss_bytes(),
            })
            break
        finally:
            if local_config is not None:
                for player in players["candidate"]:
                    for record in player.attempt_records:
                        _append(out / f"{name}-solver-attempts.jsonl", {
                            "policy_seed": player.seed, **record,
                        })
        report = summarize(plan, rows)
        telemetry = {
            "candidate_decisions": len(candidate_latencies),
            "decision_seconds_p50": _percentile(candidate_latencies, 0.5),
            "decision_seconds_p95": _percentile(candidate_latencies, 0.95),
            "decision_seconds_max": max(candidate_latencies, default=None),
            "peak_process_rss_bytes": _rss_bytes(),
        }
        for arm in ("candidate", "baseline"):
            group = players[arm]
            counts = sum((p.by_street for p in group), Counter())
            coverage = sum((p.coverage for p in group), Counter())
            latencies = [value for p in group for value in p.search_seconds]
            telemetry[arm] = {
                "search_attempts": sum(p.attempts for p in group),
                "search_completed": sum(p.completed for p in group),
                "search_fallbacks": sum(p.fallbacks for p in group),
                "search_by_street": {
                    street: {
                        metric: counts[(street, metric)]
                        for metric in ("attempts", "completed", "fallbacks")
                    }
                    for street in ("flop", "turn", "river")
                },
                "lookup_coverage": {
                    phase: {
                        label: coverage[(phase, label)]
                        for label in ("trained", "untrained", "off_tree", "off_menu_action")
                    }
                    for phase in ("range", "continuation")
                },
                "search_seconds_p50": _percentile(latencies, 0.5),
                "search_seconds_p95": _percentile(latencies, 0.95),
                "search_seconds_max": max(latencies, default=None),
                "solver_cycles_min": min((cycle for p in group for cycle in getattr(p, "cycles", ())), default=None),
                "solver_cycles_median": _percentile(
                    [cycle for p in group for cycle in getattr(p, "cycles", ())], 0.5,
                ),
                "solver_nodes": sum(getattr(p, "nodes", 0) for p in group),
                "solver_leaf_choices": sum(getattr(p, "leaf_choices", 0) for p in group),
                "solver_range_updates": sum(getattr(p, "range_updates", 0) for p in group),
                "solver_diagnostics": {
                    str(cycle): {
                        key: (
                            sum(values) / len(values) if values else None
                        )
                        for key in ("target_l1_from_previous", "mean_positive_regret",
                                    "max_positive_regret", "action_infosets", "leaf_infosets")
                        for values in [[snapshot[key] for p in group
                                        for record in getattr(p, "attempt_records", ())
                                        for snapshot in record["diagnostics"]
                                        if snapshot["cycle"] == cycle and snapshot[key] is not None]]
                    }
                    for cycle in (16, 32, 64, 128)
                } if any(hasattr(p, "attempt_records") for p in group) else None,
                "delegated_search_attempts": sum(getattr(getattr(p, "other", None), "attempts", 0) for p in group),
                "delegated_search_fallbacks": sum(getattr(getattr(p, "other", None), "fallbacks", 0) for p in group),
            }
        telemetry.update({
            key: telemetry["candidate"][key]
            for key in ("search_attempts", "search_completed", "search_fallbacks", "search_by_street",
                        "search_seconds_p50", "search_seconds_p95", "search_seconds_max")
        })
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
    if local_config is not None:
        candidate = [item["telemetry"]["candidate"] for item in reports.values()]
        attempts = sum(item["search_attempts"] for item in candidate)
        completed = sum(item["search_completed"] for item in candidate)
        minimum = min(
            (item["solver_cycles_min"] for item in candidate
             if item["solver_cycles_min"] is not None), default=None,
        )
        slowest = max(
            (item["search_seconds_max"] for item in candidate
             if item["search_seconds_max"] is not None), default=None,
        )
        peak = max((item["telemetry"]["peak_process_rss_bytes"]
                    for item in reports.values()), default=_rss_bytes())
        result["feasibility"] = {
            "eligible_attempts": attempts,
            "completed": completed,
            "completion_rate": completed / attempts if attempts else None,
            "minimum_completed_cycles": minimum,
            "maximum_solver_seconds": slowest,
            "peak_process_rss_bytes": peak,
            "passed": result["status"] == "valid" and attempts > 0
            and completed / attempts >= 0.95 and minimum is not None
            and minimum >= local_config.min_cycles
            and slowest is not None and slowest <= local_config.max_seconds + 0.05
            and peak < limits["max_rss_gib"] * 1024**3,
        }
    write_json(out / "result.json", result)
    if writer is not None:
        writer.close()
    write_json(out / "checksums.json", {
        str(path.relative_to(out)): _sha256(path)
        for path in sorted(out.rglob("*")) if path.is_file()
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        args.checkpoint, args.expected_sha256,
        json.loads(args.plan.read_text()), args.out, tensorboard=args.tensorboard,
    )
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
