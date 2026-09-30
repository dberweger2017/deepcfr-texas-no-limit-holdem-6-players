"""Seal outcome-free LBR equivalence, timing, resource and cost evidence."""

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from scripts.evaluate_hu20 import write_json
from src.arena.schedule import digest

STREETS = ("preflop", "flop", "turn", "river")
WORLDS_96_SECONDS = 1728  # #119's historical paired-range planning allowance.
CONTROL_REPORT_SECONDS = 1200  # #119's unmeasured suit/river/report reserve.


def _jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _kind(action_repr):
    return next((kind for kind in ("fold", "check", "call", "raise")
                 if f"ActionKind.{kind.upper()}" in action_repr), "unrecognized")


def report(root, artifact_dir):
    results = {phase: json.loads((root / phase / "result.json").read_text())
               for phase in ("validation", "bench-native", "bench-cached", "bench-large")}
    for phase, result in results.items():
        if result["status"] != "complete":
            raise ValueError(f"Incomplete {phase}: {result['status']}")
    validation, native, cached, large = (results[p] for p in
                                         ("validation", "bench-native", "bench-cached", "bench-large"))
    if len({validation["cases_completed"], native["cases_completed"],
            cached["cases_completed"]}) != 1:
        raise ValueError("Matched corpus case counts differ")
    if validation["mismatches"] or native["output_digest"] != cached["output_digest"]:
        raise ValueError("Exact LBR equivalence or fresh-process output digest failed")
    if len({r["case_digest"] for r in results.values() if "case_digest" in r}) != 1:
        # The larger workload uses #119 selected coordinates rather than the
        # 336-case equivalence corpus; it deliberately has no case_digest.
        raise ValueError("Frozen corpus digest changed")
    attempts = {phase: _jsonl(root / phase / "attempts.jsonl") for phase in results}
    if any(len(attempts[p]) != 336 for p in ("validation", "bench-native", "bench-cached")):
        raise ValueError("A frozen 336-case attempt row is missing")
    if len(attempts["bench-large"]) != large["calls_completed"]:
        raise ValueError("A larger-workload attempt row is missing")
    by_case = {phase: {r["case_id"]: r for r in attempts[phase]}
               for phase in ("validation", "bench-native", "bench-cached")}
    if any(len(rows) != 336 for rows in by_case.values()):
        raise ValueError("Duplicate matched case ID")
    for case_id, row in by_case["validation"].items():
        output = {"case_id": case_id, "seed": row["seed"], "street": row["street"],
                  "action": row["native_action"], "values_chips": row["values_chips"],
                  "samples": row["completed_samples"]}
        expected = digest(output)
        if (by_case["bench-native"][case_id]["output_digest"] != expected or
                by_case["bench-cached"][case_id]["output_digest"] != expected):
            raise ValueError(f"Fresh-process native/cached repeat differs: {case_id}")
    large_ids = [(r["selected_rank"], r["event_index"], r["holding_rank"], r["sample"])
                 for r in attempts["bench-large"]]
    if len(large_ids) != len(set(large_ids)):
        raise ValueError("Duplicate larger-workload call")
    if large["full_one_sample_calls"] != 58047:
        raise ValueError("#119 complete one-sample inventory changed")
    first_rows = [r for r in attempts["bench-large"] if r["sample"] == 0]
    extra_rows = [r for r in attempts["bench-large"] if r["sample"] == 1]
    if (len(first_rows) != large["selected_first_sample_calls"] or
            len(extra_rows) != large["selected_incremental_calls"]):
        raise ValueError("Larger-workload first/incremental sample counts differ")
    validation_rows = attempts["validation"]
    case_coverage = {
        "seeds": dict(Counter(r["seed"] for r in validation_rows)),
        "streets": dict(Counter(r["street"] for r in validation_rows)),
        "positions": dict(Counter(r["position"] for r in validation_rows)),
        "observed_attacker_actions": dict(Counter(r["attacker_action_kind"] for r in validation_rows)),
        "chosen_actions": dict(Counter(_kind(r["native_action"]) for r in validation_rows)),
        "legal_menu_actions": dict(Counter(_kind(a) for r in validation_rows for a in r["menu"])),
        "complete_sample_cases": sum(r["completed_samples"] == r["requested_samples"]
                                     for r in validation_rows),
        "soft_limited_cases": sum(r["completed_samples"] < r["requested_samples"]
                                  for r in validation_rows),
        "zero_likelihood_cases": sum(bool(r["zero_likelihood"]) for r in validation_rows),
        "minimum_value_margin_chips": min(r["near_tie_margin_chips"]
                                          for r in validation_rows
                                          if r["near_tie_margin_chips"] is not None),
        "ties_within_tolerance": sum(r["near_tie_margin_chips"] is not None and
                                     r["near_tie_margin_chips"] <= 1e-10
                                     for r in validation_rows),
    }
    full_by_street = large["full_calls_by_street"]
    estimates = {}
    measured = {}
    for street in STREETS:
        first = [r["wall_seconds"] for r in first_rows if r["street"] == street]
        extra = [r["wall_seconds"] for r in extra_rows if r["street"] == street]
        if full_by_street.get(street, 0) and (not first or not extra):
            raise ValueError(f"No paired timing coverage for {street}")
        measured[street] = {
            "full_calls": full_by_street.get(street, 0),
            "first_measured": len(first), "incremental_measured": len(extra),
            "first_mean_seconds": sum(first)/len(first) if first else None,
            "first_p95_seconds": _percentile(first, .95) if first else None,
            "incremental_mean_seconds": sum(extra)/len(extra) if extra else None,
            "incremental_p95_seconds": _percentile(extra, .95) if extra else None,
        }
        if first:
            count = full_by_street[street]
            estimates[street] = {
                "first_mean": count * measured[street]["first_mean_seconds"],
                "additional_conservative_mean": count * max(
                    measured[street]["first_mean_seconds"],
                    measured[street]["incremental_mean_seconds"]),
                "first_p95": count * measured[street]["first_p95_seconds"],
                "additional_p95": count * max(
                    measured[street]["first_p95_seconds"],
                    measured[street]["incremental_p95_seconds"]),
            }
    first_mean = sum(v["first_mean"] for v in estimates.values())
    additional_mean = sum(v["additional_conservative_mean"] for v in estimates.values())
    first_p95 = sum(v["first_p95"] for v in estimates.values())
    additional_p95 = sum(v["additional_p95"] for v in estimates.values())
    scenarios = []
    for likelihood_samples, worlds in ((4, 96), (4, 192), (4, 384), (8, 96)):
        likelihood = first_mean + (likelihood_samples - 1) * additional_mean
        total = 1.25 * (likelihood + WORLDS_96_SECONDS * worlds/96 + CONTROL_REPORT_SECONDS)
        upper = 1.25 * (first_p95 + (likelihood_samples - 1) * additional_p95
                        + WORLDS_96_SECONDS * worlds/96 + CONTROL_REPORT_SECONDS)
        scenarios.append({"likelihood_samples": likelihood_samples, "worlds": worlds,
                          "likelihood_mean_projection_seconds": likelihood,
                          "complete_mean_projection_with_headroom_seconds": total,
                          "complete_p95_style_projection_seconds": upper})
    report_data = {
        "schema": "reverse-lbr-acceleration-report-v1", "status": "complete",
        "source_commit_at_report": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "expanded_focused_test_log": (root / "focused-tests-expanded.log").read_text().strip(),
        "scientific_outputs_generated": False,
        "equivalence": {"cases": 336, "mismatches": 0,
                        "value_absolute_tolerance_chips": 1e-10,
                        "fresh_process_native_repeat": True,
                        "cached_repeat": True, "suit_control_passed": True,
                        "output_digest": native["output_digest"],
                        "coverage": case_coverage},
        "timing": {"native_calls_seconds": native["call_wall_seconds"],
                   "cached_calls_seconds": cached["call_wall_seconds"],
                   "algorithmic_speedup": native["call_wall_seconds"] / cached["call_wall_seconds"],
                   "native_end_to_end_seconds": native["wall_seconds"],
                   "cached_end_to_end_seconds": cached["wall_seconds"],
                   "end_to_end_speedup": native["wall_seconds"] / cached["wall_seconds"],
                   "matched_calls": 336, "large_first_calls": len(first_rows),
                   "large_incremental_calls": len(extra_rows),
                   "large_wall_seconds": large["wall_seconds"],
                   "large_cpu_utilization": (
                       (large["first_sample"]["cpu_seconds"] + large["incremental_sample"]["cpu_seconds"]) /
                       (large["first_sample"]["wall_seconds"] + large["incremental_sample"]["wall_seconds"])),
                   "per_street": measured},
        "cache": cached["cache"], "large_cache": large["cache"],
        "resources": {phase: {k: value.get(k) for k in
                             ("peak_rss_bytes", "swap_start_mib", "swap_end_mib", "free_disk_bytes")}
                      for phase, value in results.items()},
        "projection": {"full_one_sample_calls": 58047,
                       "measured_fraction": len(first_rows)/58047,
                       "first_mean_seconds": first_mean,
                       "additional_mean_seconds_conservative": additional_mean,
                       "first_p95_style_seconds": first_p95,
                       "additional_p95_style_seconds": additional_p95,
                       "historical_two_range_96_world_seconds": WORLDS_96_SECONDS,
                       "unmeasured_suit_river_report_reserve_seconds": CONTROL_REPORT_SECONDS,
                       "safety_multiplier": 1.25,
                       "scenarios": scenarios,
                       "limits": "Sampled subset and historical value/control allowances; full-cache scaling and independent river reference cost remain unmeasured."},
        "inputs": {"corpus_sha256": validation["corpus_sha256"],
                   "selection_digest": large["selection_digest"],
                   "model_spec_sha256": large["model_spec_sha256"]},
        "decision": "No posterior-conditioned scientific run launched; determine feasibility from measured projection only."}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "report.json", report_data)
    inventory = {}
    for path in [root / "research-clock.json", root / "corpus.json",
                 *[root / p / name for p in results for name in ("result.json", "attempts.jsonl")],
                 *[root / f"{p}.log" for p in results],
                 root / "focused-tests.log", root / "focused-tests-expanded.log",
                 artifact_dir / "report.json"]:
        if path.exists():
            inventory[str(path.resolve())] = {"sha256": sha256(path.read_bytes()).hexdigest(),
                                             "bytes": path.stat().st_size}
    write_json(artifact_dir / "manifest.json", {"schema": "reverse-lbr-acceleration-inventory-v1",
                                                   "files": inventory,
                                                   "retained_m4_root": str(root.resolve())})
    return report_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report_data = report(args.root, args.out)
    print(json.dumps({"status": report_data["status"],
                      "algorithmic_speedup": report_data["timing"]["algorithmic_speedup"],
                      "end_to_end_speedup": report_data["timing"]["end_to_end_speedup"],
                      "minimum_total_seconds": report_data["projection"]["scenarios"][0][
                          "complete_mean_projection_with_headroom_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
