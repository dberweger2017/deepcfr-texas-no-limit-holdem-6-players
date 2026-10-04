"""Conditional paired-board intervals; incomplete campaigns never classify."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from src.diagnostics.board_pooling import readout
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash

METRICS = ("e_bp", "e_root_v1", "e_cross_v1", "e_cross_eq50", "e_board_v1", "e_board_eq50")


def coverage_by_fold(rows):
    totals = defaultdict(lambda: [0., 0.])
    for r in rows:
        for street, values in r.get("fallback_by_metric", {}).get("e_cross_v1", {}).items():
            key = (r["evaluation_fold"], r["lineage"], street)
            totals[key][0] += r["board_weight"] * values[0]
            totals[key][1] += r["board_weight"] * values[1]
    return [{"fold": f, "lineage": l, "street": s, "target_decision_reach_mass": t,
             "missing_key_reach_mass": m, "fraction": m/t if t else None}
            for (f,l,s), (t,m) in sorted(totals.items())]


def summarize(rows, *, seed=202610030304, resamples=2000):
    grouped = defaultdict(list); weights = {}
    for row in rows:
        grouped[row["spot"]].append(row); weights[row["spot"]] = row["board_weight"]
    keys = sorted(grouped)
    if not keys:
        return {"boards": 0, "metrics": {}, "placement": None}
    columns = [m + suffix for suffix in ("", "_pct_pot") for m in METRICS]
    matrix = np.asarray([[np.mean([row[m] for row in grouped[k]]) for m in columns] for k in keys])
    w = np.asarray([weights[k] for k in keys])
    if not np.isfinite(matrix).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Invalid finite weighted board losses")
    point = np.average(matrix, axis=0, weights=w)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(keys), (resamples, len(keys)))
    means = (matrix[draw] * w[draw, None]).sum(axis=1) / w[draw].sum(axis=1)[:, None]
    summary = {"boards": len(keys), "metrics": {m: {"mean": float(point[i]),
               "ci95": np.quantile(means[:, i], [.025, .975]).tolist() if len(keys) >= 2 else None}
               for i, m in enumerate(columns)}, "readout": readout(*point[:3]),
               "interval_scope": "conditional on fixed fitted training-half policies/codebooks; no refitting"}
    ratios = {"per_root_over_bp": (1, 0), "crossfit_over_bp": (2, 0), "crossfit_equity50_over_v1": (3, 2),
              "in_sample_over_bp": (4, 0), "in_sample_equity50_over_v1": (5, 4)}
    summary["ratios"] = {}
    for name, (a, b) in ratios.items():
        valid = means[:, b] > 0
        summary["ratios"][name] = {"point": float(point[a] / point[b]) if point[b] > 0 else None,
            "ci95": np.quantile(means[valid, a] / means[valid, b], [.025, .975]).tolist()
                     if len(keys) >= 2 and valid.all() else None}
    gap = means[:, 0] - means[:, 1]; valid = gap >= .1
    summary["placement"] = {"point": summary["readout"]["placement"],
                            "ci95": np.quantile((means[valid, 2] - means[valid, 1]) / gap[valid], [.025, .975]).tolist()
                                     if len(keys) >= 2 and valid.all() else None,
                            "undefined_bootstrap_draws": int((~valid).sum())}
    coverage = defaultdict(lambda: [0., 0.])
    for row in rows:
        for metric, streets in row.get("fallback_by_metric", {}).items():
            for street, values in streets.items():
                total, missing, _ = values
                if not np.isfinite([total, missing]).all() or not 0 <= missing <= total + 1e-5:
                    raise ValueError("Invalid held-out fallback reach coverage")
                item = coverage[metric + "/" + street]
                item[0] += row["board_weight"]*total; item[1] += row["board_weight"]*missing
    summary["fallback_coverage"] = {key: {"decision_reach_mass": value[0], "fallback_reach_mass": value[1],
        "fraction": value[1]/value[0] if value[0] else None,
        "scope": "per-street own-policy decision reach; not joint/chance-weighted frequency"}
        for key, value in sorted(coverage.items())}
    return summary


def report(plan_path, run, out):
    plan = json.loads(plan_path.read_text()); corpus_path = Path(plan["corpus"]["path"])
    if file_hash(corpus_path) != plan["corpus"]["sha256"]:
        raise ValueError("Report corpus hash differs")
    corpus = json.loads(corpus_path.read_text())
    found = {}; manifest_path = run / "admission.json"
    for path in (run / "collect").glob("*/result.json"):
        result = json.loads(path.read_text()); job = result["job"]
        relock = run / "relock" / job["job"] / "result.json"
        if not relock.exists():
            continue
        second = json.loads(relock.read_text())
        if not result["eligible"] or not second["eligible"] or not second["replay_gate"]["passed"]:
            continue
        values = result["metrics"] + second["metrics"]
        for seat in (0, 1):
            selected = [m for m in values if m["target_solver_seat"] == seat]
            if not set(METRICS) <= {m["metric"] for m in selected} or len({m["metric"] for m in selected}) != len(selected):
                raise ValueError("Incomplete held-out/secondary seat record")
            found.setdefault((job["spot"], job["policy_index"]), []).append({
                "spot": job["spot"], "lineage": job["lineage"], "seat": seat,
                "board_weight": job["board_weight"],
                "evaluation_fold": job["evaluation_fold"],
                "fallback_by_metric": {m["metric"]: m.get("fallback_coverage", {}) for m in selected},
                **{m["metric"]: m["gain_bb"] for m in selected},
                **{m["metric"] + "_pct_pot": m["gain_pct_pot"] for m in selected}})
    eligible = {r["spot"] for r in corpus["roots"] if all((r["spot"], i) in found for i in range(3))}
    rows = [row for key, bundle in found.items() if key[0] in eligible for row in bundle]
    totals = {r["spot"]: r["board_weight"] for r in corpus["roots"]}
    weight_fraction = sum(totals[k] for k in eligible) / sum(totals.values())
    completed = (run / "completion.json").exists() and not (run / "failure.json").exists()
    summaries = {"pooled": summarize(rows, seed=plan["bootstrap_seed"], resamples=plan["bootstrap_resamples"])}
    for lineage in (p["seed"] for p in plan["policies"]):
        summaries[str(lineage)] = summarize([r for r in rows if r["lineage"] == lineage], seed=plan["bootstrap_seed"], resamples=plan["bootstrap_resamples"])
    for seat in (0, 1):
        summaries[f"seat-{seat}"] = summarize([r for r in rows if r["seat"] == seat], seed=plan["bootstrap_seed"], resamples=plan["bootstrap_resamples"])
    split_path = Path(plan["crossfit"]["path"])
    if file_hash(split_path) != plan["crossfit"]["sha256"]:
        raise ValueError("Report crossfit split hash differs")
    split = json.loads(split_path.read_text())
    fold_counts = {str(f): sum(split["folds"][spot] == f for spot in eligible) for f in (0, 1)}
    admitted = (completed and len(eligible) >= plan["minimum_common_boards"]
                and weight_fraction >= plan["minimum_common_board_weight_fraction"]
                and min(fold_counts.values()) >= plan["minimum_common_boards_per_fold"])
    fallback = coverage_by_fold(rows)
    coverage_ok = bool(fallback) and all(r["fraction"] is not None and r["fraction"] <= plan.get("missing_key_reach_threshold", .05) for r in fallback)
    covered_rows = [dict(r, e_cross_v1=r["e_cross_v1_covered"], e_cross_v1_pct_pot=r["e_cross_v1_covered_pct_pot"])
                    for r in rows if "e_cross_v1_covered" in r]
    covered_summary = summarize(covered_rows, seed=plan["bootstrap_seed"], resamples=plan["bootstrap_resamples"])
    classification = summaries["pooled"].get("readout", {}).get("classification") if admitted else "incomplete or insufficient common coverage; no hypothesis decision"
    if admitted and not coverage_ok:
        classification = "missing-key reach exceeds 5% or unavailable; D descriptive only"
    exclusions = [{"spot": r["spot"], "missing_policy_indices": [i for i in range(3) if (r["spot"], i) not in found]}
                  for r in corpus["roots"] if r["spot"] not in eligible]
    failures = [{"path": str(p.relative_to(run)), "evidence": json.loads(p.read_text())} for p in run.rglob("failure.json")]
    inventory = [{"path": str(p.relative_to(run)), "bytes": p.stat().st_size, "sha256": file_hash(p)}
                 for p in sorted(run.rglob("*")) if p.is_file()]
    billing = json.loads((run / "billing.json").read_text()) if (run / "billing.json").exists() else {"actual_cost": None, "reason": "itemized provider billing unavailable"}
    result = {"completed": completed, "common_boards": len(eligible), "common_weight_fraction": weight_fraction,
              "classification": classification, "missing_key_coverage_passed": coverage_ok,
              "missing_key_reach_by_fold_lineage": fallback, "covered_context_sensitivity": covered_summary,
              "common_fold_counts": fold_counts, "summaries": summaries, "rows": rows, "exclusions": exclusions,
              "failures": failures, "billing": billing, "inventory": inventory, "plan_sha256": file_hash(plan_path),
              "admission": json.loads(manifest_path.read_text()) if manifest_path.exists() else None}
    out.mkdir(parents=True, exist_ok=False); atomic_json(out / "summary.json", result)
    lines = ["# HU20 cross-board pooling diagnostic", "", f"Status: {'completed' if completed else 'incomplete'}. {len(eligible)}/40 common three-export boards; retained weight {weight_fraction:.1%}.",
             "", f"Primary held-out readout: **{classification}**.", "", "| Group | Blueprint BB | Per-root v1 BB | Held-out v1 BB | Held-out equity50 BB | In-sample v1 BB | In-sample equity50 BB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        values = []
        for metric in METRICS:
            value = summary["metrics"].get(metric)
            values.append("unavailable" if value is None else f'{value["mean"]:.4f}' + (f' [{value["ci95"][0]:.4f}, {value["ci95"][1]:.4f}]' if value["ci95"] else ""))
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    lines += ["", "Covered-context D uses the held-out strategy on covered keys and the per-root witness on absent/zero-mass keys. This is a hybrid sensitivity, not conditional EV or a causal decomposition. The primary D is descriptive only if either fold/lineage exceeds 5% missing target decision reach on either street. Detailed coverage and sensitivity intervals are in summary.json.", "", "Intervals condition on the fitted pooled policies/codebook; they omit fitting uncertainty. Signed placement, ratios, pot-percent intervals, every seat result, exclusion, failure, resource/clock admission and hashes are retained in summary.json.",
              "", f"Cost evidence: `{json.dumps(billing, sort_keys=True)}`.",
              "", "These are conditional turn/river feasible witnesses on the seed-1-selected limped/check-through public line, not a verdict on raised pots. Ranges are taken as given; preflop errors and flop strategy are excluded. Projections are not abstraction equilibria or lower bounds. Primary policies and codebooks fit only the opposite frozen half; in-sample losses remain secondary. Absent or zero-mass training keys use uniform probabilities, with raw per-street fallback reach coverage retained. This does not establish coverage of all boards or full-game strength.",
              "", "The solver-free companion reports diagnostic board diversity, not unretained historical training occupancy. No training, promotion or automatic merge."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return {"classification": classification, "common_boards": len(eligible)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(report(a.plan, a.run, a.out)))


if __name__ == "__main__":
    main()
