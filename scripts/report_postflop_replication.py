"""Audit all retained runs and calculate block-clustered paired estimates."""

import argparse
import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

from scripts.run_postflop_replication import file_hash
from src.arena.schedule import digest
from src.blueprint.artifact import load_training


def estimate(values, confidence=0.95):
    if not values:
        return {"blocks": 0, "bb_per_100": None, "interval": None}
    center = mean(values)
    interval = None
    if len(values) >= 30:
        radius = float(t.ppf((1+confidence)/2, len(values)-1)) * stdev(values)/sqrt(len(values))
        interval = [center-radius, center+radius]
    return {"blocks": len(values), "bb_per_100": center,
            "confidence": confidence, "interval": interval}


def verified(run):
    checks = json.loads((run / "checksums.json").read_text())
    mismatches = [name for name, expected in checks.items()
                  if not (run / name).is_file() or file_hash(run / name) != expected]
    return {"checksums": len(checks), "mismatches": mismatches,
            "checksums_sha256": digest(checks)}


def density(checkpoint, contributions, keys):
    trainer = load_training(checkpoint)
    updates = json.loads(contributions.read_text()) if contributions else {}
    result = {}
    for source, entries in keys.items():
        by_street = defaultdict(list)
        for street, key in entries:
            node = trainer.nodes.get(key)
            by_street[street].append((None if node is None else node.visits,
                                      updates.get(key, 0)))
        result[source] = {}
        for street, rows in by_street.items():
            found = [visits for visits, _ in rows if visits is not None]
            result[source][street] = {
                "distinct_keys": len(rows), "found": len(found),
                "one_raw_visit": sum(v == 1 for v in found),
                "mean_raw_visits_if_found": mean(found) if found else None,
                "additional_outer_updates": sum(amount for _, amount in rows),
                "keys_with_additional_outer_update": sum(amount > 0 for _, amount in rows),
                "one_additional_outer_update": sum(amount == 1 for _, amount in rows),
            }
    return result


def analyze(plan, root, observation_set, parent):
    arms = plan["arms"]
    seeds = plan["continuation_seeds"]
    report = {"schema": "postflop-replication-report-v1",
              "plan_sha256": digest(plan), "training": {}, "evaluation": {},
              "resource": {}, "failures": [], "suites": {}}
    observation = json.loads(observation_set.read_text())
    key_sources = {"independent_observations":
                   [(street, key) for street, values in observation.items()
                    for key in values]}
    for seed in seeds:
        for mode in (1, 4):
            arm = f"k{mode}-{seed}"
            run = root / "training" / arm
            if not run.is_dir():
                report["failures"].append(f"Missing training run {arm}")
                continue
            checked = verified(run)
            result = json.loads((run / "result.json").read_text())
            lineage = json.loads((run / "lineage.json").read_text()) if (run / "lineage.json").exists() else None
            report["training"][arm] = {"result": result, "verification": checked,
                                       "lineage": lineage}
            if checked["mismatches"] or result["status"] != "complete":
                report["failures"].append(f"Training {arm} incomplete or checksum mismatch")
            if (result["completed_nodes"] < plan["additional_nodes_per_run"]
                    or result["overshoot_nodes"] > plan["limits"]["max_overshoot_nodes"]
                    or result["peak_process_rss_bytes"] >= plan["limits"]["max_rss_gib"] * 1024**3
                    or lineage is None or lineage["output_checkpoint_sha256"] != result["output_checkpoint_sha256"]):
                report["failures"].append(f"Training {arm} violated a frozen work, memory or lineage condition")
    rows = {}
    for arm in arms:
        run = root / "evaluation" / arm
        if not run.is_dir():
            report["failures"].append(f"Missing evaluation run {arm}")
            continue
        checked = verified(run)
        result = json.loads((run / "result.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        records = [json.loads(line) for line in (run / "hands.jsonl").read_text().splitlines()]
        coverage = json.loads((run / "coverage.json").read_text())
        report["evaluation"][arm] = {"result": result, "verification": checked,
                                      "manifest": manifest,
                                      "coverage_counts": {suite: data["counts"]
                                                          for suite, data in coverage.items()}}
        if (manifest.get("plan_sha256") != digest(plan)
                or result["peak_process_rss_bytes"] >= plan["limits"]["max_rss_gib"] * 1024**3):
            report["failures"].append(f"Evaluation {arm} violated frozen plan or memory cap")
        if arm != "TAG":
            checkpoint = (root / "training" / arm / "checkpoint.json.gz") if arm not in ("parent", "U_safe") else parent
            contributions = (root / "training" / arm / "outer-contributions.json") if arm not in ("parent", "U_safe") else None
            sources = dict(key_sources)
            for suite, data in coverage.items():
                sources[f"reached_{suite}"] = list({(r["street"], r["key"])
                                                    for r in data["distinct"]})
            if checkpoint.exists():
                report["evaluation"][arm]["density"] = density(checkpoint, contributions, sources)
            else:
                report["failures"].append(f"Missing density checkpoint for {arm}")
        if checked["mismatches"] or result["status"] != "complete":
            report["failures"].append(f"Evaluation {arm} incomplete or checksum mismatch")
        rows[arm] = {(r["suite"], r["block"], r["rotation"]): r for r in records}
        if len(rows[arm]) != len(records):
            report["failures"].append(f"Duplicate evaluation attempts in {arm}")
    for suite, definition in plan["suites"].items():
        expected = {(suite, block, rotation)
                    for block in range(definition["blocks"]) for rotation in range(6)}
        by_arm = {}
        for arm in arms:
            if arm not in rows:
                continue
            selected = {key: row for key, row in rows[arm].items() if key[0] == suite}
            if set(selected) != expected or any(row["status"] != "completed" or
                                                    row["candidate_chips"] is None for row in selected.values()):
                report["failures"].append(f"Incomplete {suite} hands in {arm}")
                continue
            by_arm[arm] = [mean(selected[(suite, block, rotation)]["candidate_chips"]
                                for rotation in range(6))
                           for block in range(definition["blocks"])]
        if len(by_arm) != len(arms):
            continue
        for key in expected:
            paired = [rows[arm][key] for arm in arms]
            if len({(r["deal_seed"], r["button"], tuple(r["opponents"]))
                    for r in paired}) != 1:
                report["failures"].append(f"Schedule mismatch at {key}")
                break
        effects = {}
        for seed in seeds:
            c, b = by_arm[f"k4-{seed}"], by_arm[f"k1-{seed}"]
            effects[f"k4-k1-{seed}"] = estimate([x-y for x, y in zip(c, b)], 0.975 if suite == "styles" else 0.95)
            effects[f"k4-u-{seed}"] = estimate([x-y for x, y in zip(c, by_arm["U_safe"])],
                                                0.975 if suite == "styles" else 0.95)
            effects[f"k4-parent-{seed}"] = estimate([x-y for x, y in zip(c, by_arm["parent"])])
            effects[f"k4-tag-{seed}"] = estimate([x-y for x, y in zip(c, by_arm["TAG"])])
        effects["aggregate_k4_minus_k1"] = estimate([
            mean(by_arm[f"k4-{seed}"][i] - by_arm[f"k1-{seed}"][i] for seed in seeds)
            for i in range(definition["blocks"])], 0.975 if suite == "styles" else 0.95)
        effects["aggregate_k4_minus_u_safe"] = estimate([
            mean(by_arm[f"k4-{seed}"][i] - by_arm["U_safe"][i] for seed in seeds)
            for i in range(definition["blocks"])], 0.975 if suite == "styles" else 0.95)
        report["suites"][suite] = {"blocks": definition["blocks"],
                                    "absolute": {arm: estimate(values) for arm, values in by_arm.items()},
                                    "effects": effects}
    report["status"] = "complete" if not report["failures"] else "incomplete"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--observation-set", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(json.loads(args.plan.read_text()), args.run, args.observation_set, args.parent)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "failures": report["failures"]}))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
