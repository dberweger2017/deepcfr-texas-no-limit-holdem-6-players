"""Independent arithmetic/input verification and sealed engineering report.

Run only after the supervisor and all benchmark children/logs have closed.
This consumes engineering outputs and public inventory, never new game outcomes.
"""

import argparse
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from random import Random
from time import time

import pokers

from scripts.diagnose_hu20_decisions import _selected_views
from scripts.benchmark_reverse_lbr_workload import _work
from scripts.evaluate_hu20 import rss, write_json
from scripts.run_exact_ranker_experiment import file_hash, guard
from src.arena.schedule import digest
from src.diagnostics.reverse_lbr import compatible_holdings, observed_lbr_actions
from src.game.observation import BoardDealt


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_checks(original, candidate):
    if len(original) != len(candidate):
        raise ValueError("Fresh timing case count changed")
    differences = []
    for left, right in zip(original, candidate):
        if left["case_id"] != right["case_id"]:
            raise ValueError("Fresh timing order/case identity changed")
        a, b = left["output"], right["output"]
        fields = [key for key in a if key != "values" and a[key] != b[key]]
        if len(a["values"]) != len(b["values"]) or any(abs(x - y) > 1e-10 for x, y in zip(a["values"], b["values"])):
            fields.append("values")
        if fields:
            differences.append({"case_id": left["case_id"], "fields": fields,
                                "original": a, "candidate": b})
    return differences


def run(args):
    clock = read(args.root / "engineering-clock.json")
    guard(args.root, clock)
    supervisor = read(args.root / "supervisor.json")
    if supervisor["status"] != "complete" or any(p["exit_code"] for p in supervisor["phases"]):
        raise ValueError("All frozen phases must finish before reporting")
    for phase in supervisor["phases"]:
        try:
            import os
            os.kill(phase["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("Benchmark child still exists; do not seal mutable logs")
    verified = read(args.root / "verified-inputs.json")
    for name, expected in verified["files"].items():
        guard(args.root, clock)
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Frozen input changed after execution: {name}")
    phases = {name: read(args.root / name / "result.json") for name in
              ("ranks", "validation", "bench336-original", "bench336-candidate", "large-original", "large-candidate")}
    if any(r["status"] != "complete" for r in phases.values()):
        raise ValueError("Incomplete engineering phase")
    if len({r["source_head"] for r in phases.values()}) != 1:
        raise ValueError("Runtime source changed across phases")
    if phases["ranks"]["hands"] != 100000 or phases["validation"]["cases_completed"] != 336 or phases["validation"]["mismatches"]:
        raise ValueError("Frozen correctness counts/gates failed")
    timings = {name: rows(args.root / name / "attempts.jsonl") for name in
               ("bench336-original", "bench336-candidate", "large-original", "large-candidate")}
    comparison = {panel: paired_checks(timings[panel + "-original"], timings[panel + "-candidate"])
                  for panel in ("bench336", "large")}
    if len(timings["bench336-original"]) != 336 or len(timings["large-original"]) != 4263:
        raise ValueError("Frozen performance sample count changed")
    for executor in ("original", "candidate"):
        if phases["bench336-" + executor]["output_digest"] != phases["validation"]["output_digest"]:
            raise ValueError("Fresh real-clock output differs from fixed-work oracle")
        large = phases["large-" + executor]
        if large["selected_first_calls"] != 3591 or large["selected_incremental_calls"] != 672 or large["full_one_sample_calls"] != 58047:
            raise ValueError("Frozen larger inventory changed")
    if any(comparison.values()):
        write_json(args.root / "real-clock-differences.json", comparison)
        raise ValueError("Real-clock difference retained; no unconditional equivalence claim")
    resource_rows = rows(args.root / "resources.jsonl") + rows(args.previous_root / "resources.jsonl")
    if not resource_rows:
        raise ValueError("Missing external resource samples")
    if any(r["owned_rss_bytes"] > 10.5 * 1024**3 or r["swap_mib"] - clock["swap_start_mib"] > 512
           or r["free_disk_bytes"] < 8 * 1024**3 or r["foreign_large_processes"] for r in resource_rows):
        raise ValueError("External resource guard evidence failed")
    if supervisor["finished"] > clock["deadline"]:
        raise ValueError("Engineering exceeded original deadline")
    inventory = []
    actual_full_counts = Counter()
    selection = read(args.selection)
    for entry, _, view in _selected_views(args.raw_dir, selection):
        actions = observed_lbr_actions(view)
        holding_count = len(compatible_holdings(view))
        for _, prefix, _ in actions:
            street = next((event.street.value for event in reversed(prefix)
                           if isinstance(event, BoardDealt)), "preflop")
            actual_full_counts[street] += holding_count
        inventory.append({"selected_rank": entry["rank"], "seed": entry["seed"],
            "street": entry["street"], "position": entry["position"],
            "compatible_holdings": len(compatible_holdings(view)),
            "prior_lbr_actions": len(observed_lbr_actions(view)),
            "one_sample_calls": len(compatible_holdings(view)) * len(observed_lbr_actions(view))})
    if len(inventory) != 24 or sum(r["one_sample_calls"] for r in inventory) != 58047:
        raise ValueError("Independent full inventory mismatch")
    tasks, _, _, _, _ = _work(args.raw_dir, args.selection)
    actual_street_by_id = {}
    for task in tasks:
        street = next((event.street.value for event in reversed(task["prefix"])
                       if isinstance(event, BoardDealt)), "preflop")
        for sample in (0, 1) if task["incremental"] else (0,):
            identifier = digest((task["selected_rank"], task["event_index"], task["pair"], sample))
            actual_street_by_id[identifier] = street
    actual_timings = {}
    for executor in ("original", "candidate"):
        actual_timings[executor] = {}
        for street in ("preflop", "flop", "turn", "river"):
            actual_timings[executor][street] = {
                str(sample): {"calls": len(selected),
                    "wall_seconds": sum(row["wall_seconds"] for row in selected),
                    "cpu_seconds": sum(row["cpu_seconds"] for row in selected),
                    "max_seconds": max(row["wall_seconds"] for row in selected)}
                for sample in (0, 1) if (selected := [row for row in timings["large-" + executor]
                    if row["sample"] == sample and actual_street_by_id[row["case_id"]] == street])}
    proposal = read(args.proposal_selection)
    stability = {row["rank"] for row in proposal["stability_selected"]}
    suit = set(proposal["suit_selected_ranks"])
    if len(stability) != 5 or len(suit) != 4 or not stability.issuperset(suit):
        raise ValueError("Prospective stability/control selection changed")
    costs = {}
    for executor in ("original", "candidate"):
        large = phases["large-" + executor]
        per_street = {}
        for street, parts in large["per_street"].items():
            first = parts["0"]["wall_seconds"] / parts["0"]["calls"]
            extra = parts["1"]["wall_seconds"] / parts["1"]["calls"]
            per_street[street] = {"first_mean_seconds": first, "additional_mean_seconds": extra,
                                 "conservative_additional_seconds": max(first, extra)}
        def likelihood(count, subset=None):
            return sum(row["one_sample_calls"] * (per_street[row["street"]]["first_mean_seconds"]
                + (count - 1) * per_street[row["street"]]["conservative_additional_seconds"])
                for row in inventory if subset is None or row["selected_rank"] in subset)
        startup = sum(row["seconds"] for row in large["model_load"]) + large["preprocessing_seconds"]
        minimum = {"main_four_sample_likelihood_seconds": likelihood(4),
                   "main_eight_sample_likelihood_seconds": likelihood(8),
                   "stability_28_sample_likelihood_seconds": likelihood(28, stability),
                   "coupled_suit_four_sample_likelihood_seconds": likelihood(4, suit),
                   "timer_crosscheck_allowance_seconds": 240,
                   "primary_two_range_96_world_historical_proxy_seconds": 1728,
                   "controls_reporting_historical_reserve_seconds": 1200,
                   "five_phase_load_preprocessing_allowance_seconds": 5 * startup}
        raw_four = sum(value for key, value in minimum.items() if key != "main_eight_sample_likelihood_seconds")
        four_total = 1.25 * raw_four
        eight_total = 1.25 * (raw_four - likelihood(4) + likelihood(8))
        base_four = 1.25 * (likelihood(4) + 1728 + 1200)
        costs[executor] = {"per_street": per_street, "one_sample_seconds": likelihood(1),
            "historical_minimum_four_sample_96_world_hours": base_four / 3600,
            "components": minimum, "headroom_multiplier": 1.25,
            "proposed_four_sample_total_hours": four_total / 3600,
            "proposed_eight_sample_total_hours": eight_total / 3600,
            "proposed_safety_window_hours": math.ceil(four_total / 1800) / 2 + .5}
    speedups = {panel: {"lbr_call_wall": phases[panel + "-original"]["call_wall_seconds"] / phases[panel + "-candidate"]["call_wall_seconds"],
                       "startup_inclusive_wall": phases[panel + "-original"]["wall_seconds"] / phases[panel + "-candidate"]["wall_seconds"]}
                for panel in ("bench336", "large")}
    # Resample the paired engineering calls as a descriptive timing interval.
    # Shared prefixes/cache and sequential host order limit generalization.
    grouped = defaultdict(lambda: [0.0, 0.0])
    corpus_cases = read(args.corpus)["cases"]
    by_case = {c["case_id"]: c["selected"]["rank"] for c in corpus_cases}
    for a, b in zip(timings["bench336-original"], timings["bench336-candidate"]):
        key = by_case[a["case_id"]]
        grouped[key][0] += a["wall_seconds"]
        grouped[key][1] += b["wall_seconds"]
    values, random = list(grouped.values()), Random(202610040126)
    boot = []
    for _ in range(2000):
        selected_groups = [random.choice(values) for _ in values]
        boot.append(sum(x[0] for x in selected_groups) / sum(x[1] for x in selected_groups))
    speedups["bench336"]["descriptive_95pct_decision_cluster_bootstrap"] = [quantile(boot, .025), quantile(boot, .975)]
    rank_micro = phases["ranks"]["microbenchmark"]
    result = {"status": "complete", "engineering_source_head": phases["ranks"]["source_head"],
        "report_source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "clock": clock, "rank_counts": {k: phases["ranks"][k] for k in
            ("hands", "hand_digest", "rank_comparisons", "suit_comparisons", "order_comparisons")},
        "focused_test_result": (args.root / "focused-tests.log").read_text().strip().splitlines()[-1],
        "fixed_cases": 336, "real_clock_differences": comparison,
        "microbenchmark": rank_micro,
        "microbenchmark_distinct_wall_speedup": rank_micro["original"]["distinct_wall_seconds"] / rank_micro["candidate"]["distinct_wall_seconds"],
        "microbenchmark_distinct_cpu_speedup": rank_micro["original"]["distinct_cpu_seconds"] / rank_micro["candidate"]["distinct_cpu_seconds"],
        "repeated_microtiming_note": "Timing stops before output-digest bookkeeping in the corrected attempt. Original attempt's repeated timings are retained but excluded.",
        "phase_results": phases, "speedups": speedups, "inventory": inventory,
        "actual_attacker_prefix_full_one_sample_counts": dict(actual_full_counts),
        "actual_attacker_prefix_timing": actual_timings,
        "timing_strata_note": "#121 primary projection strata are the selected target decision's street. Public prior attacker prefixes can be earlier streets; their independently reconstructed counts/timings are reported separately.",
        "stability_one_sample_calls": sum(r["one_sample_calls"] for r in inventory if r["selected_rank"] in stability),
        "suit_one_sample_calls": sum(r["one_sample_calls"] for r in inventory if r["selected_rank"] in suit),
        "cost_projections": costs,
        "resources": {"peak_aggregate_owned_rss_bytes": max(r["owned_rss_bytes"] for r in resource_rows),
            "swap_start_mib": clock["swap_start_mib"], "max_swap_mib": max(r["swap_mib"] for r in resource_rows),
            "minimum_free_disk_bytes": min(r["free_disk_bytes"] for r in resource_rows),
            "resource_samples": len(resource_rows), "heavy_elapsed_seconds": supervisor["finished"] - clock["started"]},
        "proposal_selection_sha256": file_hash(args.proposal_selection), "finished": time()}
    result["environment"] = {"python": sys.version, "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "physical_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "pokers_files": {str(Path(module.__file__).resolve()): file_hash(Path(module.__file__))
                         for name, module in list(sys.modules.items())
                         if name == "pokers" or name.startswith("pokers.") if getattr(module, "__file__", None)},
        "source_files": {str(path): file_hash(path) for path in
            (Path("src/game/showdown.py"), Path("src/diagnostics/robustness.py"),
             Path("src/diagnostics/cached_lbr.py"), Path("src/diagnostics/exact_ranker.py"))}}
    previous = read(args.previous_root / "supervisor.json")
    if previous["status"] != "complete":
        raise ValueError("Pre-correction attempt has not closed")
    result["retained_attempts"] = {"startup_before_heavy": str(args.startup_log),
        "pre_correction_root": str(args.previous_root), "pre_correction_source": previous["clock"]["source_head"],
        "correction": "Active suit-context binding and actual-executor validation; repeat-rank timer excludes digest bookkeeping. Original attempt retained, not used for final performance.",
        "original_clock_reused": read(args.previous_root / "engineering-clock.json") == clock}
    if not result["retained_attempts"]["original_clock_reused"]:
        raise ValueError("Engineering deadline was reset")
    reporting_guard = guard(args.root, clock)
    reporting_peak = rss()
    if reporting_peak > 10.5 * 1024**3:
        raise MemoryError("Reporting peak RSS exceeds the unchanged ceiling")
    result["resources"]["reporting_peak_process_rss_bytes"] = reporting_peak
    result["resources"]["reporting_guard_sample"] = reporting_guard
    write_json(args.root / "report.json", result)
    files = {str(path.resolve()): {"sha256": file_hash(path), "bytes": path.stat().st_size}
             for path in sorted(args.root.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    for path in sorted(args.previous_root.rglob("*")):
        if path.is_file():
            files[str(path.resolve())] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    for path in (args.wrapper_log, args.previous_wrapper_log, args.startup_log, args.proposal_selection):
        files[str(path.resolve())] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    manifest = {"retained_m4_root": str(args.root.resolve()), "files": files,
                "pre_correction_root": str(args.previous_root.resolve()),
                "engineering_source_head": result["engineering_source_head"],
                "report_source_head": result["report_source_head"], "sealed": time()}
    guard(args.root, clock)
    write_json(args.root / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "files_sealed": len(files), "speedups": speedups,
                      "proposal_hours": costs["candidate"]["proposed_four_sample_total_hours"]}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "corpus", "selection", "raw-dir", "proposal-selection", "wrapper-log", "previous-wrapper-log", "startup-log", "previous-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
