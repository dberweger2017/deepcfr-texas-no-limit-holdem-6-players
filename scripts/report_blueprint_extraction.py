"""Verify a complete extraction campaign and calculate paired block intervals."""

import argparse
import json
from pathlib import Path
from statistics import mean

from scripts.report_postflop_replication import estimate, verified
from src.arena.schedule import digest
from src.blueprint.windowed import _hash


def analyze(plan, root):
    campaign = json.loads((root / "campaign.json").read_text())
    report = {"schema": "windowed-blueprint-report-v1", "plan_sha256": digest(plan),
              "campaign": campaign, "extraction": {}, "evaluation": {},
              "suites": {}, "failures": []}
    for seed in plan["continuation_seeds"]:
        run = root / "extraction" / str(seed)
        if not run.exists():
            report["failures"].append(f"Missing extraction {seed}")
            continue
        result = json.loads((run / "result.json").read_text())
        checked = verified(run)
        captures = [json.loads(line) for line in (run / "captures.jsonl").read_text().splitlines()]
        manifest = json.loads((run / "policy-manifest.json").read_text()) if (run / "policy-manifest.json").exists() else None
        report["extraction"][str(seed)] = {"result": result, "verification": checked,
                                            "captures": captures, "manifest": manifest}
        if (checked["mismatches"] or result["status"] != "complete"
                or not result["checkpoint_byte_identical"] or len(captures) != 8
                or result["peak_process_rss_bytes"] >= plan["limits"]["max_rss_gib"]*1024**3
                or manifest is None or manifest["source_checkpoint_sha256"] != plan["source_checkpoints"][str(seed)]
                or manifest["artifact_sha256"] != _hash(run / "policy-index.sqlite")):
            report["failures"].append(f"Extraction {seed} failed identity, capture or resource verification")
        for index, capture in enumerate(captures):
            if capture["requested_nodes"] != plan["capture_nodes"][index] or capture["completed_nodes"] < capture["requested_nodes"]:
                report["failures"].append(f"Extraction {seed} milestone {index} differs from protocol")
    by_arm = {}
    for arm in plan["arms"]:
        run = root / "evaluation" / arm
        if not run.exists():
            report["failures"].append(f"Missing evaluation {arm}")
            continue
        result = json.loads((run / "result.json").read_text())
        checked = verified(run)
        manifest = json.loads((run / "manifest.json").read_text())
        rows = [json.loads(line) for line in (run / "hands.jsonl").read_text().splitlines()]
        indexed = {(row["suite"], row["block"], row["rotation"]): row for row in rows}
        if len(indexed) != len(rows):
            report["failures"].append(f"Duplicate attempt in {arm}")
        by_arm[arm] = indexed
        report["evaluation"][arm] = {"result": result, "verification": checked,
                                     "manifest": manifest, "hands": len(rows)}
        if (checked["mismatches"] or result["status"] != "complete"
                or manifest.get("plan_sha256") != digest(plan)
                or result["peak_process_rss_bytes"] >= plan["limits"]["max_rss_gib"]*1024**3):
            report["failures"].append(f"Evaluation {arm} failed plan, completion or resource verification")
    for suite, definition in plan["suites"].items():
        expected = {(suite, block, rotation) for block in range(definition["blocks"])
                    for rotation in range(6)}
        values = {}
        for arm in plan["arms"]:
            rows = by_arm.get(arm, {})
            selected = {key: row for key, row in rows.items() if key[0] == suite}
            if (set(selected) != expected or any(row["status"] != "completed" or row["candidate_chips"] is None
                                             for row in selected.values())):
                report["failures"].append(f"Incomplete {suite} schedule for {arm}")
                continue
            values[arm] = [mean(selected[(suite, block, rotation)]["candidate_chips"]
                                for rotation in range(6)) for block in range(definition["blocks"])]
        if len(values) != len(plan["arms"]):
            continue
        for key in expected:
            paired = [by_arm[arm][key] for arm in plan["arms"]]
            if len({(row["deal_seed"], row["button"], tuple(row["opponents"])) for row in paired}) != 1:
                report["failures"].append(f"Schedule mismatch at {key}")
                break
        confidence = 0.975 if suite == "styles" else 0.95
        effects = {}
        for seed in plan["continuation_seeds"]:
            for arm in "APF":
                effects[f"{arm}-C-{seed}"] = estimate([
                    x-y for x,y in zip(values[f"{arm}{seed}"], values[f"C{seed}"])], confidence)
            effects[f"A-U_safe-{seed}"] = estimate([
                x-y for x,y in zip(values[f"A{seed}"], values["U_safe"])], confidence)
        n = definition["blocks"]
        effects["aggregate_A_minus_C"] = estimate([
            mean(values[f"A{seed}"][i]-values[f"C{seed}"][i]
                 for seed in plan["continuation_seeds"]) for i in range(n)], confidence)
        effects["aggregate_A_minus_U_safe"] = estimate([
            mean(values[f"A{seed}"][i]-values["U_safe"][i]
                 for seed in plan["continuation_seeds"]) for i in range(n)], confidence)
        report["suites"][suite] = {"blocks": n,
                                   "absolute": {arm: estimate(rows) for arm, rows in values.items()},
                                   "effects": effects}
    if campaign["status"] != "complete" or len(campaign["attempts"]) != 18:
        report["failures"].append("Campaign did not complete all 18 sequential attempts")
    report["status"] = "complete" if not report["failures"] else "incomplete"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(json.loads(args.plan.read_text()), args.run)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "failures": report["failures"]}), flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
