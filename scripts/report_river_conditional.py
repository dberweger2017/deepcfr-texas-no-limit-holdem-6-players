"""Analyze a complete frozen conditional river run using root-cluster inference."""

import argparse
import json
import re
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from scripts.evaluate_river_quality import _hash


ARMS = ("river_cfr", "rollout_normal", "rollout_matched")
CONTROLS = ARMS[1:]


def _interval(values, *, seed=2026092704, draws=20_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(values), size=(draws, len(values)))
    means = np.mean(values[indices], axis=1)
    return [float(np.quantile(means, q)) for q in (0.0125, 0.9875)]


def analyze(plan, cases_source, manifest, result, rows):
    case_map = {case["id"]: case for case in cases_source["cases"]}
    roots = {row["case_id"]: row for row in rows if row["phase"] == "root"}
    plays = {(row["case_id"], row["repetition"], row["arm"]): row
             for row in rows if row["phase"] == "play"}
    expected = len(plan["case_ids"]) * plan["deals_per_root"] * len(ARMS)
    raw_root_count = sum(row["phase"] == "root" for row in rows)
    raw_play_count = sum(row["phase"] == "play" for row in rows)
    if (manifest["dirty"] or manifest["plan"] != plan
            or result["status"] != "valid" or len(roots) != len(plan["case_ids"])
            or raw_root_count != len(roots) or raw_play_count != len(plays)
            or len(plays) != expected or result["retained_play_attempts"] != expected
            or any(row["status"] != "completed" for row in roots.values())
            or any("profile_path" not in row for row in roots.values())
            or any(row["status"] != "completed" for row in plays.values())):
        return {"status": "incomplete", "retained_play_attempts": len(plays),
                "expected_play_attempts": expected, "run_status": result["status"],
                "run_error": result["error"]}
    root_metrics = []
    for case_id in plan["case_ids"]:
        case = case_map[case_id]
        root = roots[case_id]
        arm_returns = defaultdict(list)
        differences = {control: [] for control in CONTROLS}
        for repetition in range(plan["deals_per_root"]):
            triplet = [plays[(case_id, repetition, arm)] for arm in ARMS]
            if (triplet[0]["holes"] != triplet[1]["holes"]
                    or triplet[0]["holes"] != triplet[2]["holes"]
                    or len({row["opponent_seed"] for row in triplet}) != 1):
                raise ValueError("A paired deal or opponent seed differs across arms")
            for row in triplet:
                arm_returns[row["arm"]].append(row["payoff_bb"])
            for control in CONTROLS:
                differences[control].append(
                    triplet[0]["payoff_bb"] -
                    next(row["payoff_bb"] for row in triplet if row["arm"] == control)
                )
        root_metrics.append({
            "case_id": case_id, "hero_position": case["hero_position"],
            "opponent_style": case["opponent_style"],
            "root_pot_bb": root["root_pot_bb"],
            "completed_sweeps": root["completed_sweeps"],
            "solver_seconds": root.get("solver_seconds", root["seconds"]),
            "arm_mean_return_bb": {arm: float(np.mean(arm_returns[arm]))
                                   for arm in ARMS},
            "paired_mean_difference_bb": {control: float(np.mean(differences[control]))
                                          for control in CONTROLS},
        })
    comparisons = {}
    for control in CONTROLS:
        values = [root["paired_mean_difference_bb"][control]
                  for root in root_metrics]
        scaled = [value / root["root_pot_bb"]
                  for value, root in zip(values, root_metrics)]
        comparisons[control] = {
            "mean_difference_bb": float(np.mean(values)),
            "bonferroni_97_5_percent_interval_bb": _interval(values),
            "mean_difference_per_root_pot": float(np.mean(scaled)),
            "bonferroni_97_5_percent_interval_per_root_pot": _interval(scaled),
            "roots_positive": sum(value > 0 for value in values),
        }
    arm_means = {arm: float(np.mean([root["arm_mean_return_bb"][arm]
                                      for root in root_metrics])) for arm in ARMS}
    fallbacks = Counter()
    delegations = Counter()
    coverage = Counter()
    durations = defaultdict(list)
    worlds = defaultdict(list)
    for row in plays.values():
        arm = row["arm"]
        durations[arm].append(row["seconds"])
        fallbacks[arm] += row.get("rollout_fallbacks", 0)
        delegations[arm] += row.get("delegations", 0)
        coverage.update({f"{arm}:{key}": value for key, value in
                         row.get("continuation_coverage", {}).items()})
        worlds[arm].extend(row.get("rollout_worlds_completed", []))
    telemetry = {arm: {
        "play_attempts": sum(row["arm"] == arm for row in plays.values()),
        "rollout_fallbacks": fallbacks[arm],
        "off_tree_delegations": delegations[arm],
        "mean_play_seconds": float(np.mean(durations[arm])),
        "max_play_seconds": max(durations[arm]),
        "rollout_worlds_completed_mean": (
            float(np.mean(worlds[arm])) if worlds[arm] else None),
        "rollout_worlds_completed_min": min(worlds[arm]) if worlds[arm] else None,
    } for arm in ARMS}
    group_effects = {}
    for field in ("hero_position", "opponent_style"):
        group_effects[field] = {}
        for label in sorted({root[field] for root in root_metrics}):
            subset = [root for root in root_metrics if root[field] == label]
            group_effects[field][label] = {
                "roots": len(subset),
                "mean_difference_bb": {control: float(np.mean([
                    root["paired_mean_difference_bb"][control] for root in subset
                ])) for control in CONTROLS},
            }
    def swap_used(value):
        match = re.search(r"used = ([0-9.]+)M", value or "")
        return float(match.group(1)) if match else None

    before_swap = swap_used(manifest["swap_before"])
    after_swap = swap_used(result["swap_after"])
    resource_pass = (
        result["elapsed_seconds"] <= plan["max_wall_seconds"]
        and result["peak_process_rss_bytes"] < plan["max_rss_gib"] * 1024**3
        and (before_swap is None or after_swap is None or after_swap <= before_swap)
    )
    return {
        "status": "complete" if resource_pass else "resource_violation",
        "resource_pass": resource_pass, "scope": "conditional river only",
        "roots": len(root_metrics), "deals_per_root": plan["deals_per_root"],
        "paired_deals": len(root_metrics) * plan["deals_per_root"],
        "comparisons": comparisons, "arm_mean_return_bb": arm_means,
        "telemetry": telemetry, "root_metrics": root_metrics,
        "group_effects": group_effects,
        "continuation_coverage": dict(coverage),
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_process_rss_bytes": result["peak_process_rss_bytes"],
        "swap_before": manifest["swap_before"], "swap_after": result["swap_after"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    checksums = json.loads((args.run / "checksums.json").read_text())
    for name, digest in checksums.items():
        if _hash(args.run / name) != digest:
            raise ValueError(f"Run artifact hash mismatch: {name}")
    plan = json.loads(args.plan.read_text())
    cases = json.loads(args.cases.read_text())
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["cases_sha256"] != _hash(args.cases):
        raise ValueError("Frozen cases hash differs from run manifest")
    rows = [json.loads(line) for line in (args.run / "rows.jsonl").read_text().splitlines()]
    for row in rows:
        if row["phase"] == "root" and row["status"] == "completed" and "profile_path" in row:
            if _hash(args.run / row["profile_path"]) != row["profile_sha256"]:
                raise ValueError(f"Saved profile hash differs for {row['case_id']}")
    result = json.loads((args.run / "result.json").read_text())
    analysis = analyze(plan, cases, manifest, result, rows)
    analysis["artifacts_sha256"] = sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()
    args.out.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: analysis[key] for key in ("status", "artifacts_sha256")}))
    return 0 if analysis["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
