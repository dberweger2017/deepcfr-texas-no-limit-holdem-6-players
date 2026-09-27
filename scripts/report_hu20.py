"""Audit HU20 campaign outputs and estimate cluster-paired learning effects."""

import argparse
import json
from pathlib import Path
from statistics import mean
from collections import Counter

from scripts.report_postflop_replication import estimate, verified
from src.blueprint.artifact import load_training
from src.arena.schedule import digest
from src.blueprint.solver import HU20_GAME
from src.blueprint.windowed import _hash


def read_run(path, plan, expected_blocks, failures):
    if not path.is_dir():
        failures.append(f"Missing run {path}")
        return None
    check = verified(path)
    result = json.loads((path / "result.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    if (check["mismatches"] or result["status"] != "complete"
            or result["completed_blocks"] != expected_blocks
            or result["peak_process_rss_bytes"] >= plan["limits"]["max_rss_gib"]*1024**3
            or manifest.get("plan_sha256") != digest(plan)
            or manifest.get("game") != HU20_GAME):
        failures.append(f"Failed verification or resource bound: {path}")
    rows = [json.loads(line) for line in (path / "hands.jsonl").read_text().splitlines()]
    indexed = {(row["block"], row["rotation"]): row for row in rows}
    if (len(indexed) != expected_blocks*2 or len(rows) != len(indexed)
            or any(row["status"] != "completed" or row["candidate_chips"] is None
                   for row in indexed.values())):
        failures.append(f"Missing, duplicate or illegal hand rows: {path}")
        return None
    values = [mean(indexed[(block, rotation)]["candidate_chips"]
                   for rotation in range(2)) for block in range(expected_blocks)]
    return {"values": values, "rows": indexed, "result": result,
            "manifest": manifest, "verification": check}


def summarize_evaluation(plan, root, phase, arms):
    failures = []
    n = plan[phase]["blocks_per_opponent"]
    runs = {}
    for opponent in plan["opponents"]:
        for arm in arms:
            name = f"{opponent}-{arm}"
            runs[(opponent, arm)] = read_run(root / phase / name, plan, n, failures)
        complete = [runs[(opponent, arm)] for arm in arms]
        if any(item is None for item in complete):
            continue
        for block in range(n):
            for rotation in range(2):
                paired = [item["rows"][(block, rotation)] for item in complete]
                if len({(row["deal_seed"], row["button"]) for row in paired}) != 1:
                    failures.append(f"Coupled schedule mismatch: {opponent} {block} {rotation}")
                    break
        if len({item["manifest"]["schedule_sha256"] for item in complete}) != 1:
            failures.append(f"Schedule digest mismatch for {opponent}")
    compact = {}
    for (opponent, arm), item in runs.items():
        if item is None:
            continue
        compact[f"{opponent}-{arm}"] = {
            "absolute": estimate(item["values"]),
            "result": item["result"], "manifest": item["manifest"],
            "verification": item["verification"]}
    return runs, compact, failures


def analyze_development(plan, root):
    arms = ["uniform"]
    for seed in plan["training_seeds"]:
        arms.extend([f"E{seed}-{i}" for i in range(3)])
        arms.extend([f"C{seed}", f"A{seed}"])
    runs, compact, failures = summarize_evaluation(plan, root, "development", arms)
    n = plan["development"]["blocks_per_opponent"]
    effects = {}
    if not failures:
        for extraction in "CA":
            clusters = []
            for opponent in plan["opponents"]:
                uniform = runs[(opponent, "uniform")]["values"]
                for block in range(n):
                    clusters.append(mean(
                        runs[(opponent, f"{extraction}{seed}")]["values"][block]-uniform[block]
                        for seed in plan["training_seeds"]))
            effects[f"{extraction}-uniform"] = estimate(clusters)
        for seed in plan["training_seeds"]:
            for index in range(3):
                clusters = []
                for opponent in plan["opponents"]:
                    early = runs[(opponent, f"E{seed}-{index}")]["values"]
                    uniform = runs[(opponent, "uniform")]["values"]
                    clusters.extend(x-y for x,y in zip(early, uniform))
                effects[f"early-{index}-{seed}-uniform"] = estimate(clusters)
    return {"schema": "hu20-development-summary-v2", "plan_sha256": digest(plan),
            "status": "complete" if not failures else "incomplete", "failures": failures,
            "arms": compact, "effects": effects}


def analyze_final(plan, root, decision):
    extraction = decision["primary_extraction"]
    arms = ["uniform", *(f"{extraction}{seed}" for seed in plan["training_seeds"])]
    runs, compact, failures = summarize_evaluation(plan, root, "confirmation", arms)
    n = plan["confirmation"]["blocks_per_opponent"]
    effects = {}
    if not failures:
        clusters = []
        for opponent in plan["opponents"]:
            uniform = runs[(opponent, "uniform")]["values"]
            opponent_clusters = [mean(
                runs[(opponent, f"{extraction}{seed}")]["values"][block]-uniform[block]
                for seed in plan["training_seeds"]) for block in range(n)]
            effects[f"{opponent}:aggregate-trained-minus-uniform"] = estimate(opponent_clusters)
            clusters.extend(opponent_clusters)
            for seed in plan["training_seeds"]:
                candidate = runs[(opponent, f"{extraction}{seed}")]["values"]
                effects[f"{opponent}:{seed}-uniform"] = estimate(
                    [x-y for x,y in zip(candidate, uniform)])
        effects["primary-aggregate-trained-minus-uniform"] = estimate(clusters)
    training = {}
    for seed in plan["training_seeds"]:
        path = root / "training" / str(seed)
        if not path.is_dir():
            failures.append(f"Missing training seed {seed}")
            continue
        result = json.loads((path / "result.json").read_text())
        check = verified(path)
        captures = [json.loads(line) for line in (path / "captures.jsonl").read_text().splitlines()]
        checkpoints = [json.loads(line) for line in (path / "checkpoints.jsonl").read_text().splitlines()]
        identity = json.loads((path / "policy-manifest.json").read_text())
        if (result["status"] != "complete" or result["completed_nodes"] < plan["training_nodes"]
                or result["peak_process_rss_bytes"] >= plan["limits"]["max_rss_gib"]*1024**3
                or len(captures) != 8 or len(checkpoints) != 4 or check["mismatches"]
                or identity["artifact_sha256"] != _hash(path / "policy-index.sqlite")):
            failures.append(f"Training seed {seed} failed work, artifact or resource verification")
        training[str(seed)] = {"result": result, "verification": check,
                               "captures": captures, "checkpoints": checkpoints,
                               "identity": identity}
    density = {}
    if not failures:
        for seed in plan["training_seeds"]:
            trainer = load_training(root / "training" / str(seed) / "checkpoint-3.json.gz")
            histogram = Counter()
            decisions = found = revisited = 0
            for opponent in plan["opponents"]:
                path = root / "confirmation" / f"{opponent}-{extraction}{seed}" / "reached.json"
                for row in json.loads(path.read_text()):
                    count = row["decisions"]
                    node = trainer.nodes.get(row["key"])
                    visit_count = node.visits if node else 0
                    histogram[visit_count] += count
                    decisions += count
                    found += count * (node is not None)
                    revisited += count * (visit_count > 1)

            def quantile(fraction):
                threshold = max(1, int(decisions*fraction+0.999999))
                cumulative = 0
                for visits, count in sorted(histogram.items()):
                    cumulative += count
                    if cumulative >= threshold:
                        return visits
                return None

            density[str(seed)] = {"decisions": decisions, "trained_decisions": found,
                "revisited_decisions": revisited,
                "visit_quantiles": {"p25": quantile(.25), "p50": quantile(.5),
                                    "p75": quantile(.75), "p90": quantile(.9)},
                "mean_visits": sum(visits*count for visits,count in histogram.items())/decisions
                               if decisions else None}
            del trainer
    crossplay = {}
    seeds = plan["training_seeds"]
    for i, hero_seed in enumerate(seeds):
        opponent_seed = seeds[(i+1) % len(seeds)]
        for arm in (f"E{hero_seed}-0", f"{extraction}{hero_seed}"):
            opponent = f"{extraction}{opponent_seed}"
            name = f"{arm}-versus-{opponent}"
            item = read_run(root / "crossplay" / name, plan,
                            plan["crossplay"]["blocks_per_opponent"], failures)
            if item:
                crossplay[name] = {"absolute": estimate(item["values"]),
                                   "result": item["result"],
                                   "verification": item["verification"]}
    campaign = json.loads((root / "campaign.json").read_text())
    if campaign["status"] != "complete" or campaign.get("decision_sha256") is None:
        failures.append("Campaign or decision is incomplete")
    return {"schema": "hu20-final-report-v2", "plan_sha256": digest(plan),
            "decision": decision, "campaign": campaign,
            "status": "complete" if not failures else "incomplete", "failures": failures,
            "training": training, "confirmation": compact,
            "effects": effects, "crossplay": crossplay,
            "decision_weighted_density": density}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "final"), required=True)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.stage == "development":
        result = analyze_development(plan, args.root)
    else:
        if args.decision is None:
            parser.error("--decision is required for final report")
        result = analyze_final(plan, args.root, json.loads(args.decision.read_text()))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "failures": result["failures"]}), flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
