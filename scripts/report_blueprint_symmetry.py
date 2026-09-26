"""Validate seven-arm artifacts and estimate effects over independent blocks."""

import argparse
import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

from scripts.evaluate_blueprint_symmetry import ARMS, _hash
from src.arena.schedule import digest

PRIMARY = ("B_canonical_safe", "U_safe")
CONTRASTS = (
    ("B_canonical_safe", "U_safe"),
    ("B_canonical", "B_legacy"),
    ("B_canonical_safe", "B_legacy_safe"),
    ("U_safe", "U"),
    ("B_canonical_safe", "TAG"),
)


def _estimate(values, confidence):
    if not values:
        return {"blocks": 0, "bb_per_100": None, "interval": None}
    point = mean(values)
    interval = None
    if len(values) >= 30:
        margin = float(t.ppf((1 + confidence) / 2, len(values) - 1)) * stdev(values) / sqrt(len(values))
        interval = [point - margin, point + margin]
    return {"blocks": len(values), "bb_per_100": point,
            "confidence": confidence, "interval": interval}


def analyze(plan, manifest, result, rows, coverage):
    expected = sum(suite["blocks"] for suite in plan["suites"].values()) * 6 * len(ARMS)
    expected_keys = {
        (suite_name, block, rotation, arm)
        for suite_name, suite in plan["suites"].items()
        for block in range(suite["blocks"])
        for rotation in range(6)
        for arm in ARMS
    }
    keys = [(row["suite"], row["block"], row["rotation"], row["arm"])
            for row in rows]
    if (manifest["source_dirty"] or manifest["plan"] != plan
            or manifest["checkpoint_sha256"] != plan["checkpoint_sha256"]
            or manifest["plan_sha256"] != digest(plan)
            or result["status"] != "complete" or plan["resource_only"]
            or len(rows) != expected or set(keys) != expected_keys
            or result["attempts"] != expected
            or any(row["status"] != "completed" for row in rows)
            or any(row["candidate_chips"] is None for row in rows)
            or any(sum(row["net_chips"]) != 0 or
                   row["candidate_chips"] != row["net_chips"][row["rotation"]]
                   for row in rows)
            or any(digest({k: v for k, v in row.items() if k != "outcome_sha256"})
                   != row["outcome_sha256"] for row in rows)):
        return {"status": "incomplete", "expected_attempts": expected,
                "retained_attempts": len(rows), "run_status": result["status"],
                "stop_reason": result["stop_reason"]}
    if result["peak_process_rss_bytes"] >= plan["limits"]["max_rss_gib"] * 1024**3:
        return {"status": "resource_violation", "reason": "RSS cap"}
    if result["elapsed_seconds"] >= plan["limits"]["max_wall_seconds"]:
        return {"status": "resource_violation", "reason": "wall cap"}
    if any(sum(row["decision_weighted_visit_histogram"].values()) !=
           row["decision_weighted_found"] or
           sum(row["distinct_found_visit_histogram"].values()) !=
           row["distinct_found"] for row in coverage["coverage_rows"]):
        return {"status": "telemetry_invalid", "reason": "visit histogram total"}
    by_key = {key: row for key, row in zip(keys, rows, strict=True)}
    for suite_name, suite in plan["suites"].items():
        for block in range(suite["blocks"]):
            for rotation in range(6):
                group = [by_key[(suite_name, block, rotation, arm)] for arm in ARMS]
                if (len({row["deal_seed"] for row in group}) != 1 or
                        len({tuple(row["opponents"]) for row in group}) != 1):
                    raise ValueError("Paired arm schedules differ")
    suites = {}
    for suite_name, suite in plan["suites"].items():
        block_rates = {arm: [] for arm in ARMS}
        for block in range(suite["blocks"]):
            for arm in ARMS:
                chips = [by_key[(suite_name, block, rotation, arm)]["candidate_chips"]
                         for rotation in range(6)]
                block_rates[arm].append(100 * sum(chips) / (6 * 100))
        absolute = {arm: _estimate(values, 0.95)
                    for arm, values in block_rates.items()}
        contrasts = {}
        for candidate, baseline in CONTRASTS:
            values = [a - b for a, b in zip(block_rates[candidate], block_rates[baseline], strict=True)]
            confidence = 0.975 if suite_name == "styles" and (candidate, baseline) == PRIMARY else 0.95
            contrasts[f"{candidate} - {baseline}"] = _estimate(values, confidence)
        suites[suite_name] = {
            "blocks": suite["blocks"], "arms": absolute,
            "contrasts": contrasts,
            "block_rates_bb_per_100": block_rates,
        }
    weighted = defaultdict(Counter)
    distinct = defaultdict(Counter)
    for row in coverage["coverage_rows"]:
        key = (row["arm"], row["street"], row["button"], row["lookup_mode"])
        weighted[key]["found"] += row["decision_weighted_found"]
        weighted[key]["missing"] += row["decision_weighted_missing"]
        weighted[key]["single_visit"] += row["decision_weighted_visit_histogram"].get("1", 0)
        distinct[key]["found"] += row["distinct_found"]
        distinct[key]["missing"] += row["distinct_missing"]
        distinct[key]["single_visit"] += row["distinct_found_visit_histogram"].get("1", 0)
    coverage_rates = []
    for key in sorted(weighted):
        w, d = weighted[key], distinct[key]
        coverage_rates.append({
            "arm": key[0], "street": key[1], "button": key[2], "lookup_mode": key[3],
            "decision_weighted": dict(w), "distinct_keys_in_group": dict(d),
        })
    actions = Counter()
    for row in coverage["action_counts"]:
        actions[(row["arm"], row["action"])] += row["count"]
    return {
        "status": "complete", "checkpoint_sha256": plan["checkpoint_sha256"],
        "suites": suites, "coverage_by_street_button": coverage_rates,
        "same_decision_lookup_counts": coverage["same_decision_counts"],
        "action_counts": [{"arm": arm, "action": action, "count": count}
                          for (arm, action), count in sorted(actions.items())],
        "wrapper_interventions": result["wrapper_interventions"],
        "decisions": coverage["decisions"],
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_process_rss_bytes": result["peak_process_rss_bytes"],
        "swap_before": manifest["swap_before"], "swap_after": result["swap_after"],
        "memory_pressure_before": manifest["memory_pressure_before"],
        "memory_pressure_after": result["memory_pressure_after"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    checks = json.loads((args.run / "checksums.json").read_text())
    for name, expected in checks.items():
        if _hash(args.run / name) != expected:
            raise ValueError(f"Run artifact hash mismatch: {name}")
    plan = json.loads(args.plan.read_text())
    manifest = json.loads((args.run / "manifest.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    rows = [json.loads(line) for line in (args.run / "hands.jsonl").read_text().splitlines()]
    coverage = json.loads((args.run / "reached-decisions.json").read_text())
    summary = analyze(plan, manifest, result, rows, coverage)
    summary["artifact_checksums_sha256"] = digest(checks)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": summary["status"],
                      "artifact_checksums_sha256": summary["artifact_checksums_sha256"]}))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
