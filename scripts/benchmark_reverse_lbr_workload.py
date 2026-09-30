"""Hash-ranked, outcome-free acceleration timing for #119 likelihood calls.

This runs the selectable cached LBR only. It does not combine likelihoods into
a posterior and does not evaluate the target's actions or playing returns.
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time, time

from scripts.diagnose_hu20_decisions import _selected_views
from scripts.evaluate_hu20 import rss, write_json
from scripts.evaluate_hu20_reopening import Target
from scripts.validate_reverse_lbr_acceleration import _guard, _swap_mib
from src.arena.schedule import digest, stream_seed
from src.diagnostics.cached_lbr import CachedLocalBestResponse, SharedProbabilityCache
from src.diagnostics.reverse_lbr import compatible_holdings, observed_lbr_actions
from src.diagnostics.robustness import LBRConfig
from src.game.observation import replay

ROOT = 202610030101
PER_DECISION_FIRST = 171
PER_DECISION_INCREMENTAL = 32
EXPECTED_FULL_CALLS = 58047
CONFIG = LBRConfig(4, 5)


def _work(raw_dir, selection_path):
    selection = json.loads(selection_path.read_text())
    if digest(selection["selected"]) != selection["selection_digest"]:
        raise ValueError("#119 selection digest mismatch")
    if selection["selection_digest"] != "578b8b67f21a15d2961e88cf498aeeb64831c9827e578e5ce1db62f85d43e322":
        raise ValueError("Unexpected #119 decision selection")
    tasks, full = [], 0
    for entry, _, target_view in _selected_views(raw_dir, selection):
        options = []
        holdings = compatible_holdings(target_view)
        for event_index, prefix, _observed in observed_lbr_actions(target_view):
            full += len(holdings)
            for pair in holdings:
                key = sha256(repr(("reverse-lbr-large-v1", entry["rank"],
                                   event_index, pair)).encode()).hexdigest()
                options.append((key, event_index, prefix, pair))
        for rank, event_index, prefix, pair in sorted(options)[:PER_DECISION_FIRST]:
            tasks.append({"seed": entry["seed"], "street": entry["street"],
                          "position": entry["position"], "selected_rank": entry["rank"],
                          "attacker_seat": 1 - target_view.seat,
                          "event_index": event_index, "prefix": prefix,
                          "pair": pair, "rank": rank,
                          "incremental": len([t for t in tasks if t["selected_rank"] == entry["rank"]])
                                         < PER_DECISION_INCREMENTAL})
    if full != EXPECTED_FULL_CALLS:
        raise ValueError(f"#119 complete-workload call count changed: {full}")
    return tasks, full, selection["selection_digest"]


def run(raw_dir, selection_path, models_path, validation_path, out, deadline):
    validation = json.loads(validation_path.read_text())
    if validation["status"] != "complete" or validation["mismatches"] or validation["cases_completed"] != 336:
        raise ValueError("Complete frozen equivalence gate must pass first")
    out.mkdir(parents=True, exist_ok=False)
    swap_start = _swap_mib()
    started = time()
    summary = {"status": "running", "started": started, "deadline": deadline,
               "per_decision_first": PER_DECISION_FIRST,
               "per_decision_incremental": PER_DECISION_INCREMENTAL,
               "swap_start_mib": swap_start, "calls_completed": 0,
               "peak_rss_bytes": rss()}
    write_json(out / "result.json", summary)
    try:
        tasks, full, selection_digest = _work(raw_dir, selection_path)
        specs = {s["seed"]: s for s in json.loads(models_path.read_text())
                 if s.get("arm") == "B" and s.get("milestone") == 100000000}
        if set(specs) != {2026093001, 2026093002, 2026093003}:
            raise ValueError("Missing saved B100M model lineage")
        summary.update({"full_one_sample_calls": full,
                        "selected_first_sample_calls": len(tasks),
                        "selected_incremental_calls": sum(t["incremental"] for t in tasks),
                        "selection_digest": selection_digest,
                        "model_spec_sha256": sha256(models_path.read_bytes()).hexdigest()})
        by_seed = defaultdict(list)
        for task in tasks:
            by_seed[task["seed"]].append(task)
        load_rows, cache_rows, timing_rows, action_digests = [], [], [], []
        with (out / "attempts.jsonl").open("w") as handle:
            for seed in sorted(by_seed):
                _guard(out, deadline, swap_start)
                before = perf_counter()
                source = Target(specs[seed])
                load_rows.append({"seed": seed, "seconds": perf_counter() - before,
                                  "rss_bytes": rss()})
                cache = SharedProbabilityCache(source)
                for task in by_seed[seed]:
                    for sample in (0, 1) if task["incremental"] else (0,):
                        _guard(out, deadline, swap_start)
                        internal_seed = stream_seed(ROOT, "validation", "opponent",
                                                    "large", task["selected_rank"],
                                                    task["event_index"], task["pair"], sample)
                        attacker = CachedLocalBestResponse(source, internal_seed, cache, CONFIG)
                        view = replay(task["prefix"], task["attacker_seat"], task["pair"])
                        wall_start, cpu_start = perf_counter(), process_time()
                        action = attacker.choose_action(view)
                        row = {"seed": seed, "street": task["street"],
                               "position": task["position"], "selected_rank": task["selected_rank"],
                               "event_index": task["event_index"], "holding_rank": task["rank"],
                               "sample": sample, "wall_seconds": perf_counter() - wall_start,
                               "cpu_seconds": process_time() - cpu_start,
                               "rss_bytes": rss(),
                               "requested_samples": attacker.telemetry[-1]["requested_samples"],
                               "completed_samples": attacker.telemetry[-1]["samples"],
                               "action_digest": sha256(repr(action).encode()).hexdigest()}
                        action_digests.append(row["action_digest"])
                        timing_rows.append(row)
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        handle.flush()
                        summary["calls_completed"] += 1
                        summary["peak_rss_bytes"] = max(summary["peak_rss_bytes"], row["rss_bytes"])
                        if summary["calls_completed"] % 16 == 0:
                            write_json(out / "result.json", summary)
                shallow = (sys.getsizeof(cache.entries) +
                           sum(sys.getsizeof(k) + sys.getsizeof(v)
                               for k, v in cache.entries.items()))
                cache_rows.append({"seed": seed, **cache.telemetry(),
                                   "shallow_cache_bytes": shallow,
                                   "rss_after_seed_bytes": rss()})
                del source, cache
        summary["status"] = "complete"
        summary["model_load"] = load_rows
        summary["cache"] = cache_rows
        summary["action_digest"] = digest(action_digests)
        summary["first_sample"] = {
            "calls": sum(r["sample"] == 0 for r in timing_rows),
            "wall_seconds": sum(r["wall_seconds"] for r in timing_rows if r["sample"] == 0),
            "cpu_seconds": sum(r["cpu_seconds"] for r in timing_rows if r["sample"] == 0)}
        summary["incremental_sample"] = {
            "calls": sum(r["sample"] == 1 for r in timing_rows),
            "wall_seconds": sum(r["wall_seconds"] for r in timing_rows if r["sample"] == 1),
            "cpu_seconds": sum(r["cpu_seconds"] for r in timing_rows if r["sample"] == 1)}
        summary["per_street"] = {
            street: {"calls": len(rows), "wall_seconds": sum(r["wall_seconds"] for r in rows),
                     "cpu_seconds": sum(r["cpu_seconds"] for r in rows),
                     "max_call_seconds": max(r["wall_seconds"] for r in rows)}
            for street in ("preflop", "flop", "turn", "river")
            if (rows := [r for r in timing_rows if r["street"] == street])}
        summary["limited_calls"] = sum(r["completed_samples"] < r["requested_samples"]
                                       for r in timing_rows)
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure"] = f"{type(exc).__name__}: {exc}"
    summary["finished"] = time()
    summary["wall_seconds"] = summary["finished"] - started
    summary["swap_end_mib"] = _swap_mib()
    summary["free_disk_bytes"] = shutil.disk_usage(out).free
    write_json(out / "result.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    result = run(args.raw_dir, args.selection, args.models, args.validation,
                 args.out, args.deadline)
    print(json.dumps({k: result.get(k) for k in
                      ("status", "calls_completed", "failure", "wall_seconds")}, sort_keys=True))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
