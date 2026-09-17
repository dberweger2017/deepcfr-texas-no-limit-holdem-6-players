"""Compare the frozen online branching campaign from retained arena outcomes."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

from src.arena.artifacts import write_json
from src.arena.report import estimate, summarize
from src.arena.schedule import digest
from src.holdem.experiment import Experiment

SEEDS = (719, 727, 733)
CHECKPOINTS = (128, 256, 384, 512)
ARMS = ("first", "second")


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def paired_rates(plan, first, second):
    """The six rotations of a deal are one observation, not six samples."""
    for rows in (first, second):
        if summarize(plan, rows)["status"] != "valid":
            raise ValueError("Incomplete or altered arena outcomes")
        blinds = {scenario.name: scenario.big_blind for scenario in plan.scenarios}
        if any(row["big_blind"] != blinds[row["scenario"]] for row in rows):
            raise ValueError("Outcome chip units differ from the declared game")
    controls = [
        sorted(
            (r for r in rows if r["arm"] == "baseline"),
            key=lambda r: (r["block"], r["rotation"]),
        )
        for rows in (first, second)
    ]
    if controls[0] != controls[1]:
        raise ValueError("Uniform controls do not reproduce on the shared schedule")
    values = []
    for rows in (first, second):
        values.append(
            [
                mean(
                    100 * r["candidate_chips"] / r["big_blind"]
                    for r in rows
                    if r["arm"] == "candidate" and r["block"] == block
                )
                for block in range(plan.blocks)
            ]
        )
    differences = [b - a for a, b in zip(*values, strict=True)]
    return {
        "first": estimate(values[0]),
        "second": estimate(values[1]),
        "difference": estimate(differences),
        "block_differences_bb100": differences,
    }


def family_interval(values):
    if len(values) < 30 or stdev(values) == 0:
        return None
    radius = (
        float(t.ppf(1 - 0.05 / (2 * len(SEEDS)), len(values) - 1))
        * stdev(values)
        / sqrt(len(values))
    )
    return [mean(values) - radius, mean(values) + radius]


def decision_counts(rows):
    streets, actions, all_ins = Counter(), Counter(), Counter()
    hands_with_postflop = 0
    for row in rows:
        if row["arm"] != "candidate":
            continue
        started = row["events"][0]
        hero = started["seat_numbers"][started["player_ids"].index("player-0")]
        remaining = dict(zip(started["seat_numbers"], started["stacks"], strict=True))
        postflop = False
        for event in row["events"]:
            if event["event"] == "BlindPosted":
                remaining[event["seat"]] -= event["amount"]
            elif event["event"] == "ActionTaken":
                seat = event["seat"]
                remaining[seat] -= event["paid"]
                if seat == hero:
                    street = event["street"]
                    streets[street] += 1
                    actions[event["action"]["kind"]] += 1
                    postflop |= street != "preflop"
                    if event["paid"] > 0 and remaining[seat] == 0:
                        all_ins[street] += 1
        hands_with_postflop += postflop
    return {
        "decisions_by_street": dict(streets),
        "actions": dict(actions),
        "all_ins_by_street": dict(all_ins),
        "hands_with_postflop_decision": hands_with_postflop,
    }


def training_summary(job):
    timing = lines(job / "training-timing.jsonl")
    reports = lines(job / "iteration-reports.jsonl")
    if [r["iteration"] for r in timing] != list(range(1, 513)) or any(
        r["status"] != "complete" for r in timing
    ):
        raise ValueError("Missing or failed training iterations")
    if [r["iteration"] for r in reports] != list(range(1, 513)):
        raise ValueError("Missing training diagnostics")
    cumulative, elapsed = {}, 0.0
    streets, positions = Counter(), {}
    roots, postflop_roots = 0, 0
    for row in timing:
        elapsed += row["total_seconds"]
        if row["iteration"] in CHECKPOINTS:
            cumulative[row["iteration"]] = elapsed
        for role in row["collection_coverage"]:
            roots += role["roots"]
            postflop_roots += role["roots_with_postflop"]
            streets.update(role["records_by_street"])
            counts = positions.setdefault(role["position_from_button"], Counter())
            counts.update(
                roots=role["roots"], roots_with_postflop=role["roots_with_postflop"]
            )
            counts.update(role["records_by_street"])
    fits = [role["fit"] for r in reports for role in r["roles"] if role["fit"]]
    return {
        "seconds": {
            k: sum(r[k] for r in timing)
            for k in (
                "total_seconds",
                "collection_seconds",
                "fitting_seconds",
                "replay_seconds",
            )
        },
        "cumulative_seconds": cumulative,
        "checkpoint_at_1800_seconds": max(
            (i for i, seconds in cumulative.items() if seconds <= 1800), default=None
        ),
        "roots": roots,
        "roots_with_postflop": postflop_roots,
        "records_by_street": dict(streets),
        "coverage_by_position": positions,
        "nodes": sum(r["nodes"] for r in timing),
        "peak_process_rss_bytes": max(r["peak_process_rss_bytes"] for r in timing),
        "replay_stored_final": timing[-1]["replay_stored"],
        "replay_seen_final": timing[-1]["replay_seen"],
        "max_inverse_reach": max(r["max_inverse_reach"] for r in reports),
        "max_regret_update_bb": max(r["max_regret_update_bb"] for r in reports),
        "clipped_steps": sum(f["clipped_steps"] for f in fits),
        "fit_steps": sum(f["steps"] for f in fits),
        "max_gradient_norm": max(f["max_gradient_norm"] for f in fits),
        "checkpoint_seconds": sum(
            r["seconds"] for r in lines(job / "checkpoint-timing.jsonl")
        ),
        "evaluation_seconds": sum(
            read(p)["wall_seconds"] for p in job.glob("timing-*.json")
        ),
    }


def evaluation_rows(job, arena, iteration, suite, seed):
    suffix = str(iteration) + ("-random" if suite == "random" else "")
    rows = read(job / f"outcomes-{suffix}.json")
    saved = read(job / f"evaluation-{suffix}.json")
    rebuilt = summarize(arena, rows)
    if any(saved[k] != v for k, v in rebuilt.items()):
        raise ValueError("Saved evaluation does not match its outcomes")
    exports = [
        entry
        for entry in lines(job / "artifacts.jsonl")
        if entry["iteration"] == iteration and entry["kind"] == "holdem-average-v1"
    ]
    if (
        len(exports) != 1
        or saved["policy_sha256"] != exports[0]["sha256"]
        or saved["training_seed"] != seed
        or saved["iteration"] != iteration
    ):
        raise ValueError("Evaluation does not identify the scheduled model")
    return rows


def report(root):
    plans = {
        arm: Experiment.from_dict(
            read(Path(f"configs/holdem/branching-online-{arm}.json"))
        )
        for arm in ARMS
    }
    jobs, training, manifests = {}, {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}-{seed}"
            directory = root / f"branching-online-{name}"
            manifest = read(directory / "manifest.json")
            expected = asdict(replace(plans[arm], seeds=(seed,)))
            if digest(manifest["plan"]) != digest(expected):
                raise ValueError(f"Plan differs from the frozen protocol: {name}")
            if (
                not read(directory / "result.json")["complete"]
                or (directory / "failure.json").exists()
            ):
                raise ValueError(f"Campaign job did not complete: {name}")
            job = directory / f"scenario-0-seed-{seed}"
            jobs[name] = job
            training[name] = training_summary(job)
            manifests[name] = manifest
    reference = manifests["first-719"]
    for manifest in manifests.values():
        for key in ("revision", "source_sha256", "environment", "contract"):
            if manifest[key] != reference[key]:
                raise ValueError(f"Campaign provenance differs: {key}")
    comparisons, primary = [], []
    for seed in SEEDS:
        for iteration in CHECKPOINTS:
            for suite in ("styles", "random"):
                arena = plans["first"].arena(plans["first"].scenarios[0])
                if suite == "random":
                    arena = replace(arena, opponents=("random",))
                outcomes, behavior = [], {}
                for arm in ARMS:
                    job = jobs[f"{arm}-{seed}"]
                    rows = evaluation_rows(job, arena, iteration, suite, seed)
                    outcomes.append(rows)
                    behavior[arm] = decision_counts(rows)
                item = {
                    "seed": seed,
                    "iteration": iteration,
                    "suite": suite,
                    **paired_rates(arena, *outcomes),
                    "behavior": behavior,
                }
                comparisons.append(item)
                if iteration == 512 and suite == "styles":
                    item["ci_family95"] = family_interval(
                        item["block_differences_bb100"]
                    )
                    primary.append(item)
    cost_comparisons = []
    for seed in SEEDS:
        selected = {
            arm: training[f"{arm}-{seed}"]["checkpoint_at_1800_seconds"] for arm in ARMS
        }
        if any(i is None for i in selected.values()):
            cost_comparisons.append(
                {"seed": seed, "selected_iterations": selected, "status": "unavailable"}
            )
            continue
        arena = plans["first"].arena(plans["first"].scenarios[0])
        outcomes = [
            read(jobs[f"{arm}-{seed}"] / f"outcomes-{selected[arm]}.json")
            for arm in ARMS
        ]
        cost_comparisons.append(
            {
                "seed": seed,
                "selected_iterations": selected,
                "status": "descriptive",
                "unused_seconds": {
                    arm: 1800
                    - training[f"{arm}-{seed}"]["cumulative_seconds"][selected[arm]]
                    for arm in ARMS
                },
                **paired_rates(arena, *outcomes),
            }
        )
    gain = mean(r["difference"]["bb_per_100"] for r in primary)
    passed = gain >= 25 and all(
        r["ci_family95"] is not None and r["ci_family95"][0] > 0 for r in primary
    )
    return {
        "format": "holdem-branching-comparison-v1",
        "manifests_sha256": {k: digest(v) for k, v in manifests.items()},
        "training": training,
        "comparisons": comparisons,
        "training_work_ceiling_seconds": 1800,
        "cost_comparisons_styles": cost_comparisons,
        "primary": {
            "mean_seed_gain_bb100": gain,
            "consistent_improvement": passed,
            "family_size": len(SEEDS),
            "material_mean_gain_bb100": 25,
        },
        "promoted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, report(args.results))


if __name__ == "__main__":
    main()
