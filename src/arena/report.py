"""Count independent deal/session blocks, never rotations, as statistical samples."""

from collections import defaultdict
from math import sqrt
from statistics import mean, stdev

import numpy as np
from scipy.stats import t

from src.arena.schedule import Plan, digest, schedule_document

MINIMUM_BLOCKS = 30
CONFIDENCE = 0.95


def estimate(values: list[float]) -> dict:
    result = {
        "blocks": len(values),
        "bb_per_100": mean(values) if values else None,
        "ci95": None,
        "reason": "too_few_blocks",
    }
    if len(values) < MINIMUM_BLOCKS:
        return result
    deviation = stdev(values)
    if deviation == 0:
        result["reason"] = "no_observed_variation"
        return result
    margin = float(t.ppf(0.975, len(values) - 1)) * deviation / sqrt(len(values))
    result.update(
        ci95=[result["bb_per_100"] - margin, result["bb_per_100"] + margin], reason=None
    )
    return result


def comparison(candidate: list[float], baseline: list[float]) -> dict:
    difference = estimate([a - b for a, b in zip(candidate, baseline, strict=True)])
    interval = difference["ci95"]
    conclusion = "inconclusive"
    if interval is not None:
        if interval[0] > 0:
            conclusion = "candidate_better"
        elif interval[1] < 0:
            conclusion = "baseline_better"
    return {
        "candidate": estimate(candidate),
        "baseline": estimate(baseline),
        "paired_difference": difference,
        "conclusion": conclusion,
    }


def summarize(plan: Plan, rows: list[dict]) -> dict:
    expected = {
        (s.name, block, rotation, arm, hand)
        for s in plan.scenarios
        for block in range(plan.blocks)
        for rotation in range(len(s.stacks))
        for arm in ("candidate", "baseline")
        for hand in range(s.hands_per_rotation)
    }
    keys = [
        (r["scenario"], r["block"], r["rotation"], r["arm"], r["hand"]) for r in rows
    ]
    intact = all(
        digest({k: v for k, v in r.items() if k != "outcome_sha256"})
        == r["outcome_sha256"]
        for r in rows
    )
    completed = sum(r["status"] == "completed" for r in rows)
    valid = (
        intact
        and len(keys) == len(expected)
        and set(keys) == expected
        and completed == len(expected)
    )
    result = {
        "status": "valid" if valid else "invalid",
        "candidate_policy": plan.candidate,
        "baseline_policy": plan.baseline,
        "opponent_pool": list(plan.opponents),
        "split": plan.split,
        "schedule_sha256": digest(schedule_document(plan)),
        "requested_hands": len(expected),
        "attempted_hands": len(rows),
        "completed_hands": completed,
        "failed_hands": len(rows) - completed,
        "unattempted_hands": len(expected - set(keys)),
        "invalid_actions": sum(r["status"] == "invalid_action" for r in rows),
        "outcomes_sha256": digest(rows),
        "scenarios": {},
    }
    for scenario in plan.scenarios:
        records = [r for r in rows if r["scenario"] == scenario.name]
        counts = {
            arm: {
                "completed": sum(
                    r["status"] == "completed" and r["arm"] == arm for r in records
                ),
                "dealt_in": sum(
                    r["status"] == "completed"
                    and r["arm"] == arm
                    and "player-0" in r["participants"]
                    for r in records
                ),
            }
            for arm in ("candidate", "baseline")
        }
        item = {
            "mode": scenario.mode,
            "initial_players": len(scenario.stacks),
            "starting_stacks_bb": [v / scenario.big_blind for v in scenario.stacks],
            "counts": counts,
            "comparison": None,
            "lineups": [],
        }
        if valid:
            rates = defaultdict(dict)
            lineups = defaultdict(list)
            for index in range(plan.blocks):
                block = [r for r in records if r["block"] == index]
                lineups[tuple(block[0]["opponents"])].append(index)
                for arm in ("candidate", "baseline"):
                    outcomes = [r["candidate_chips"] for r in block if r["arm"] == arm]
                    rates[arm][index] = (
                        100 * sum(outcomes) / (len(outcomes) * scenario.big_blind)
                    )
            item["comparison"] = comparison(
                list(rates["candidate"].values()), list(rates["baseline"].values())
            )
            for lineup, indices in sorted(lineups.items()):
                item["lineups"].append(
                    {
                        "opponents": list(lineup),
                        **comparison(
                            [rates["candidate"][i] for i in indices],
                            [rates["baseline"][i] for i in indices],
                        ),
                    }
                )
        result["scenarios"][scenario.name] = item
    return result


def performance(timings: list[dict], wall_seconds: float) -> dict:
    result = {"wall_seconds": wall_seconds, "arms": {}}
    for arm in ("candidate", "baseline"):
        samples = [
            d["seconds"]
            for row in timings
            if row["arm"] == arm
            for d in row["decisions"]
            if d["player_id"] == "player-0"
        ]
        result["arms"][arm] = {
            "decisions": len(samples),
            "mean_action_ms": 1000 * mean(samples) if samples else None,
            "p95_action_ms": float(np.quantile(samples, 0.95)) * 1000
            if samples
            else None,
        }
    return result


def markdown(report: dict) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Run status: **{report['status']}**.",
        "",
        f"Candidate: `{report['candidate_policy']}`. Baseline: `{report['baseline_policy']}`. Split: `{report['split']}`.",
        "",
        f"Opponent pool: {report['opponent_pool']}. Full inputs and environment: [manifest.json](manifest.json).",
        "",
        (
            f"Completed {report['completed_hands']} of {report['requested_hands']} scheduled hands "
            f"across both arms; {report['failed_hands']} failed, {report['unattempted_hands']} unattempted, "
            f"{report['invalid_actions']} invalid actions."
        ),
        "",
        (
            "Rates use scheduled table hands, including dealt-out session hands. Confidence intervals use "
            "independent blocks; paired rotations and hands within a session are not independent samples."
        ),
        "",
    ]
    legacy = {
        name: details
        for name, details in report.get("policies", {}).items()
        if details["kind"] == "legacy-standard-v1"
    }
    if legacy:
        lines += [
            "## Frozen checkpoint adapters",
            "",
            (
                "These comparisons use legacy absolute-seat features and additional-raise sizing. "
                "The old training rules and learning algorithm are not validated by loading the weights."
            ),
            "",
        ]
        for name, details in legacy.items():
            training_seed = details["training_seed"]
            provenance = str(training_seed) if training_seed is not None else "unknown"
            lines.append(
                f"- `{name}`: {details['num_players']} players, training seed {provenance}, "
                f"SHA-256 `{details['weights_sha256']}`."
            )
        lines.append("")
    for name, item in report["scenarios"].items():
        lines += [
            f"## {name}",
            "",
            (
                f"{item['mode']}; {item['initial_players']} initial players; "
                f"starting stacks in BB: {item['starting_stacks_bb']}."
            ),
            "",
        ]
        summary = item["comparison"]
        if summary is None:
            lines += ["No strength estimate: the run is incomplete or invalid.", ""]
            continue
        lines += [
            "| Measure | BB/100 | 95% interval | Independent blocks |",
            "| --- | ---: | --- | ---: |",
        ]
        for key in ("candidate", "baseline", "paired_difference"):
            value = summary[key]
            ci = value["ci95"]
            interval = (
                f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci is not None else value["reason"]
            )
            lines.append(
                f"| {key} | {value['bb_per_100']:.2f} | {interval} | {value['blocks']} |"
            )
        lines += ["", f"Comparison: **{summary['conclusion']}**.", ""]
    perf = report["performance"]
    lines += ["## Timing", "", f"Wall time: {perf['wall_seconds']:.3f} seconds.", ""]
    for arm, values in perf["arms"].items():
        if values["decisions"]:
            lines.append(
                f"- {arm}: {values['decisions']} decisions, mean {values['mean_action_ms']:.3f} ms, "
                f"p95 {values['p95_action_ms']:.3f} ms."
            )
    lines += [
        "",
        (
            "This comparison applies only to the declared opponents and scenario. It does not certify "
            "general poker strength. Intervals are approximate, unadjusted for multiple comparisons, "
            "and assume the block budget was fixed before inspecting results."
        ),
        "",
    ]
    return "\n".join(lines)
