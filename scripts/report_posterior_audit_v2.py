"""Independent arithmetic, completeness and immutable publication on M4."""

import argparse
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from time import time

from scripts.run_exact_ranker_experiment import file_hash, guard
from src.diagnostics.posterior_audit_v2 import atomic_json


def read_optional(path):
    return json.loads(path.read_text()) if path.exists() else {"status": "pending"}


def verify_journals(root):
    totals = {"likelihood": 0, "values": 0, "suit-values": 0, "timer": 0, "river": 0}
    failures, recovery = [], []
    for index_path in sorted(root.rglob("index.json")):
        index = json.loads(index_path.read_text())
        ids = set()
        for name, record in index["segments"].items():
            path = index_path.parent / name
            if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
                raise ValueError(f"Committed final shard/hash differs: {path}")
            for payload in path.read_bytes().splitlines(keepends=True):
                if not payload.endswith(b"\n"):
                    continue  # retained interrupted tail, explicitly indexed
                row = json.loads(payload)
                if row["id"] in ids:
                    raise ValueError("Duplicated committed deterministic ID")
                ids.add(row["id"])
                if row.get("status") == "failed":
                    failures.append({"shard": str(index_path.parent), "id": row["id"], "failure": row["failure"]})
        if len(ids) != index["completed_ids"]:
            raise ValueError("Final durable index/count differs")
        stage = index_path.relative_to(root).parts[0]
        totals[stage] = totals.get(stage, 0)+len(ids)
        recovery.extend(index["recovery"])
    return {"completed_rows": totals, "failed_rows": failures, "torn_tail_recovery": recovery}


def independent_arithmetic(root):
    posteriors, summaries = 0, 0
    for path in sorted((root / "likelihood").glob("*/*/posterior.json")):
        saved = json.loads(path.read_text())
        counts = [json.loads(p.read_text()) for p in sorted(path.parent.glob("event-*-counts.json"))]
        weights = [1/len(saved["holdings"])]*len(saved["holdings"])
        zero = []
        for event in counts:
            products = [weight*count for weight, count in zip(weights, event["counts"])]
            denominator = sum(products)
            if denominator == 0:
                weights = [0.0]*len(weights)
                zero.append(event["public_event_index"])
            else:
                weights = [p/denominator for p in products]
        if max(abs(a-b) for a, b in zip(weights, saved["weights"])) > 1e-10 or zero != saved["zero_evidence_events"]:
            raise ValueError(f"Independent count/posterior arithmetic differs: {path}")
        posteriors += 1
    values = read_optional(root / "values" / "result.json")
    for decision in values.get("decisions", []):
        for name, summary in decision["ranges"].items():
            rows = {}
            for path in sorted((root / "values" / decision["selection"]["rank"] / name).glob("segment-*.jsonl")):
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    rows[int(row["id"])] = row["values_bb"]
            if set(rows) != set(range(96)) or summary["selection_worlds"] != 48 or summary["evaluation_worlds"] != 48:
                raise ValueError("Primary world count/split differs")
            p = decision["probabilities"]
            select = [mean(rows[i][action] for i in range(48)) for action in range(len(p))]
            best = max(range(len(p)), key=lambda i: select[i])
            differences = [rows[i][best]-sum(weight*value for weight, value in zip(p, rows[i])) for i in range(48, 96)]
            gap, width = mean(differences), 1.96*stdev(differences)/sqrt(48)
            if best != summary["selected_action_index"] or abs(gap-summary["evaluation_policy_gap_bb"]) > 1e-10:
                raise ValueError("Independent held-out selected-action gap differs")
            if max(abs(a-b) for a, b in zip([gap-width, gap+width], summary["evaluation_policy_gap_95_interval_bb"])) > 1e-10:
                raise ValueError("Independent 48-world interval differs")
            summaries += 1
    return {"posteriors_recomputed": posteriors, "held_out_summaries_recomputed": summaries,
            "status": "passed"}


def report(root):
    clock = json.loads((root / "scientific-clock.json").read_text())
    guard(root, clock)
    state = json.loads((root / "coordinator.json").read_text())
    if state["status"] == "running":
        raise ValueError("Coordinator must finish before final reporting")
    timer = read_optional(root / "timer" / "result.json")
    stability = read_optional(root / "stability" / "result.json")
    main = read_optional(root / "main" / "result.json")
    suit = read_optional(root / "suit-likelihood" / "result.json")
    suit_values = read_optional(root / "suit-values" / "result.json")
    river = read_optional(root / "river" / "result.json")
    values = read_optional(root / "values" / "result.json")
    checks = verify_journals(root)
    arithmetic = independent_arithmetic(root)
    inputs = json.loads((root / "verified-inputs.json").read_text())
    for filename, expected in inputs["files"].items():
        guard(root, clock)
        if file_hash(Path(filename)) != expected:
            raise ValueError(f"Frozen audit input changed: {filename}")
    decisions = []
    for row in values.get("decisions", []):
        uniform, posterior = row["ranges"]["uniform"], row["ranges"]["posterior"]
        decisions.append({"selection": row["selection"], "key": row["key"], "trained": row["trained"],
            "visits": row["node"]["visits"], "uniform_gap_bb": uniform["evaluation_policy_gap_bb"],
            "uniform_interval_bb": uniform["evaluation_policy_gap_95_interval_bb"],
            "posterior_gap_bb": posterior["evaluation_policy_gap_bb"],
            "posterior_interval_bb": posterior["evaluation_policy_gap_95_interval_bb"],
            "identity_control": row["no_prior_action_identity_control"]})
    if stability.get("status") == "stopped" or main.get("status") == "stopped":
        next_experiment = ("One prospectively specified larger-likelihood stability assessment of the same five frozen decisions, "
                           "with independent higher-count references and an outcome-free M4 cost preflight. "
                           "Determine a sufficient likelihood budget before another conditional-value or training experiment; "
                           "do not reuse this attempt to choose favorable coordinates or seeds.")
    elif timer.get("status") != "passed":
        next_experiment = ("One revised original-ranker likelihood feasibility and timer-faithfulness plan for these same "
                           "recorded prefixes, for owner approval before any conditional values.")
    else:
        next_experiment = ("Review the completed paired posterior-conditioned gaps and controls to freeze one mechanism-discriminating "
                           "follow-up; no training choice is established automatically by local gaps.")
    result = {"status": state["status"], "reason": state.get("reason"), "source_head": clock["source_head"],
        "merged_base": clock["merged_base"], "merged_pr_head": inputs["merged_pr_head"], "clock": clock,
        "timer": {k: v for k, v in timer.items() if k != "checks"}, "stability": stability, "main_stability": main,
        "suit_likelihood": suit, "suit_values": suit_values, "river": river,
        "primary_values_status": values["status"], "primary_decisions": decisions,
        "posterior_clearly_positive": sum(r["posterior_interval_bb"][0] > 0 for r in decisions),
        "posterior_gap_shrunk": sum(r["posterior_gap_bb"] < r["uniform_gap_bb"] for r in decisions),
        "posterior_gap_increased": sum(r["posterior_gap_bb"] > r["uniform_gap_bb"] for r in decisions),
        "verification": checks, "independent_arithmetic": arithmetic, "inputs_verified": len(inputs["files"]),
        "resources": {k: state[k] for k in ("peak_owned_rss_bytes", "minimum_free_disk_bytes", "maximum_swap_growth_mib", "environment", "recoveries")},
        "scientific_elapsed_seconds": state["finished"]-clock["started"], "projection_hours": 8.66,
        "likelihood_calls_planned": 691988, "primary_worlds_planned": 4608, "suit_worlds_planned": 768,
        "phases": [{k: v for k, v in p.items() if k != "process_snapshot"} for p in state["phases"]],
        "recommendation": next_experiment,
        "limitations": ["Stratified local diagnostics, not exact exploitability or a decomposition of -73.14 BB/100.",
                        "Held-out intervals condition on estimated posterior; posterior-estimation uncertainty is separate.",
                        "Five stability cases do not certify the other 19; no causal card/history/menu diagnosis follows.",
                        "Empirical zero matches are finite-sample zeros, not proven impossible actions.",
                        "No training, paid host, promotion or release criterion change."], "reported_at": time()}
    atomic_json(root / "report.json", result)
    lines = ["# Stability-gated B100M posterior audit on M4", "", f"Status: **{result['status']}**.",
             f"Reason: {result['reason'] or 'all required stages completed'}", "",
             f"Merged #126: `{result['merged_base']}`; scientific source: `{result['source_head']}`.",
             f"Immutable UTC start/cutoff: {clock['started_utc']} / {clock['deadline_utc']}.",
             f"Real-clock gate: {timer['status']}; initial stability: {stability['status']}; main stability: {main['status']}.", "",
             "## Frozen stability findings", "", "| Rank / street / seed / position | 4-vs-4 TV | 4-vs-16 TV | ESS ratios | 16 mass on four-zero support | Pass |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in (main.get("cases") or stability.get("cases", [])):
        entry = row["selection"]
        render = lambda values: ", ".join("undefined" if v is None else f"{v:.6f}" for v in values)
        comparisons = row["four_vs_higher"]
        lines.append(f"| {entry['rank'][:12]} / {entry['street']} / {entry['seed']} / {entry['position']} | "
                     f"{render([r['tv'] for r in row['pairwise_four_tv']])} | {render([r['tv_vs_16'] for r in comparisons])} | "
                     f"{render([r['ess_ratio'] for r in comparisons])} | {render([r['higher_mass_on_four_zero_support'] for r in comparisons])} | {row['passed']} |")
        if row["failures"]:
            lines.extend(["", f"Failures at `{entry['rank']}`: " + "; ".join(row["failures"]) + ".", ""])
    lines += ["", "## Conditional values and controls", "", f"Primary values: {values['status']}, {len(decisions)}/24 decisions.",
              f"Suit likelihood/value controls: {suit['status']} / {suit_values['status']}. River references: {river['status']}.",
              "Pending conditional values are not zero gaps or negative findings. A failed gate prohibits the primary values.", "",
              "## Cost, verification and next measurement", "",
              f"Scientific elapsed: {result['scientific_elapsed_seconds']/3600:.3f} h versus the 8.66 h full-work projection.",
              f"Committed likelihood rows: {checks['completed_rows']['likelihood']:,}; primary worlds: {checks['completed_rows']['values']:,}; suit worlds: {checks['completed_rows']['suit-values']:,}.",
              f"Peak aggregate owned RSS: {state['peak_owned_rss_bytes']/1024**3:.3f} GiB; maximum swap growth: {state['maximum_swap_growth_mib']:.2f} MiB; minimum free disk: {state['minimum_free_disk_bytes']/1024**3:.3f} GiB.",
              f"Independent arithmetic: {arithmetic['posteriors_recomputed']} posteriors and {arithmetic['held_out_summaries_recomputed']} held-out summaries; {len(inputs['files'])} input hashes unchanged.",
              f"Coordinator recoveries: {len(state['recoveries'])}; failed durable rows: {len(checks['failed_rows'])}.", "",
              "**Exactly one recommended next experiment:** " + next_experiment, "", "M4 remains the default. No paid host was used or quoted; current credit is unknown.", "",
              "## Interpretation limits", "", *["- "+item for item in result["limitations"]], "",
              "## Retained raw artifacts", "", f"M4 root: `{root.resolve()}`. All raw likelihood/world journals remain here.",
              "Retrieve with `scp -o HostName=100.122.216.94 m4:<absolute-root>/<relative-path> <destination>` and verify against `final-manifest.json`.", ""]
    (root / "report.md").write_text("\n".join(lines))
    return result


def seal(root):
    clock = json.loads((root / "scientific-clock.json").read_text())
    guard(root, clock)
    state = json.loads((root / "coordinator.json").read_text())
    if state["status"] == "running":
        raise ValueError("Cannot seal running coordinator")
    output = root.parent / (root.name+"-final-manifest.json")
    if output.exists():
        raise FileExistsError("Final inventory already sealed; do not replace")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            guard(root, clock)
            files[str(path.resolve())] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    manifest = {"source_head": clock["source_head"], "merged_base": clock["merged_base"],
                "clock": clock, "files": files, "file_count": len(files), "sealed_at": time()}
    atomic_json(output, manifest)
    for filename, record in files.items():
        guard(root, clock)
        if file_hash(Path(filename)) != record["sha256"]:
            raise ValueError("Final hash inventory changed during verification")
    atomic_json(root.parent / (root.name+"-seal-verification.json"), {
        "manifest_sha256": file_hash(output), "report_sha256": file_hash(root / "report.json"),
        "verified_files": len(files), "verified_at": time(), "status": "passed"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    seal(args.root) if args.seal else report(args.root)


if __name__ == "__main__":
    main()
