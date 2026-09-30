"""Outcome-free function profile of frozen #121 cached LBR calls."""

import argparse
import cProfile
import json
import pstats
import shutil
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time, time

from scripts.evaluate_hu20 import rss, write_json
from scripts.evaluate_hu20_reopening import Target
from scripts.validate_reverse_lbr_acceleration import _inputs, _swap_mib, _view
from src.diagnostics.cached_lbr import CachedLocalBestResponse, SharedProbabilityCache
from src.diagnostics.robustness import LBRConfig


CONFIG = LBRConfig(4, 5)
RULE = "lbr-kernel-profile-v1|"
EXPECTED_CORPUS = "1559b5aac31016a9db92cfc2b319c28abf110446866d50dba0f45126eacadc4e"
EXPECTED_SELECTION = "578b8b67f21a15d2961e88cf498aeeb64831c9827e578e5ce1db62f85d43e322"


def select(corpus, selection):
    """Verify the committed outcome-blind 24-stratum selection exactly."""
    if corpus["case_digest"] != EXPECTED_CORPUS or corpus["selection_digest"] != EXPECTED_SELECTION:
        raise ValueError("Frozen source selection changed")
    by_stratum = defaultdict(list)
    for case in corpus["cases"]:
        item = case["selected"]
        by_stratum[(item["seed"], item["street"], item["position"])].append(case)
    if len(by_stratum) != 24:
        raise ValueError("Expected all 24 seed/street/position strata")
    expected = []
    for (seed, street, position), cases in sorted(by_stratum.items()):
        case = min(cases, key=lambda row: sha256((RULE + row["case_id"]).encode()).hexdigest())
        expected.append({"seed": seed, "street": street, "position": position,
                         "case_id": case["case_id"]})
    if selection["cases"] != expected or selection["corpus_case_digest"] != EXPECTED_CORPUS:
        raise ValueError("Committed profile case list changed")
    return {row["case_id"]: row for row in corpus["cases"] if row["case_id"] in
            {item["case_id"] for item in expected}}


def guard(out, deadline, swap_start):
    if time() >= deadline:
        raise TimeoutError("Absolute profile deadline")
    if rss() > 10.5 * 1024**3:
        raise MemoryError("10.5-GiB RSS guard")
    if _swap_mib() - swap_start > 512:
        raise MemoryError("0.5-GiB swap-growth guard")
    if shutil.disk_usage(out).free < 8 * 1024**3:
        raise OSError("8-GiB free-disk guard")


def function_rows(profile):
    stats = pstats.Stats(profile).stats
    rows = [{"file": file, "line": line, "function": function, "primitive_calls": primitive,
             "calls": calls, "self_seconds": self_time, "cumulative_seconds": cumulative}
            for (file, line, function), (primitive, calls, self_time, cumulative, _) in stats.items()]
    return sorted(rows, key=lambda row: row["self_seconds"], reverse=True)


def run(corpus_path, selection_path, raw_dir, models_path, out, deadline):
    if out.exists():
        raise FileExistsError(f"Preserve existing profile attempt: {out}")
    out.mkdir(parents=True)
    swap_start, started = _swap_mib(), time()
    summary = {"status": "running", "started": started, "deadline": deadline,
               "swap_start_mib": swap_start, "cases_completed": 0, "peak_rss_bytes": rss()}
    write_json(out / "result.json", summary)
    try:
        corpus, views, models = _inputs(corpus_path, raw_dir, models_path)
        selection = json.loads(selection_path.read_text())
        if selection["corpus_sha256"] != sha256(corpus_path.read_bytes()).hexdigest():
            raise ValueError("Profile corpus byte hash mismatch")
        cases = select(corpus, selection)
        summary.update({"source_head": "profile committed separately before execution",
                        "corpus_sha256": selection["corpus_sha256"],
                        "profile_selection_sha256": sha256(selection_path.read_bytes()).hexdigest(),
                        "model_spec_sha256": sha256(models_path.read_bytes()).hexdigest(),
                        "case_ids": [row["case_id"] for row in selection["cases"]]})
        attempts = out / "attempts.jsonl"
        with attempts.open("w") as handle:
            for seed in sorted(models):
                guard(out, deadline, swap_start)
                load_start = perf_counter()
                source = Target(models[seed])
                summary.setdefault("model_load", []).append({"seed": seed,
                    "seconds": perf_counter() - load_start, "rss_bytes": rss()})
                for item in (r for r in selection["cases"] if r["seed"] == seed):
                    guard(out, deadline, swap_start)
                    case = cases[item["case_id"]]
                    view = _view(case, views)
                    cache = SharedProbabilityCache(source)
                    timings = []
                    for phase in ("cold", "warm"):
                        attacker = CachedLocalBestResponse(source, case["internal_seed"], cache, CONFIG)
                        profiler = cProfile.Profile() if phase == "warm" else None
                        wall_start, cpu_start = perf_counter(), process_time()
                        if profiler:
                            profiler.enable()
                        attacker.choose_action(view)
                        if profiler:
                            profiler.disable()
                        timings.append({"phase": phase, "wall_seconds": perf_counter() - wall_start,
                                        "cpu_seconds": process_time() - cpu_start,
                                        "requested_samples": attacker.telemetry[-1]["requested_samples"],
                                        "completed_samples": attacker.telemetry[-1]["samples"],
                                        "cache": cache.telemetry()})
                    functions = function_rows(profiler)
                    row = {**item, "timings": timings, "profile_total_self_seconds":
                           sum(f["self_seconds"] for f in functions), "top_functions": functions[:25],
                           "all_functions": functions, "rss_bytes": rss()}
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    summary["cases_completed"] += 1
                    summary["peak_rss_bytes"] = max(summary["peak_rss_bytes"], row["rss_bytes"])
                    write_json(out / "result.json", summary)
                del source
        summary["status"] = "complete"
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
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    result = run(args.corpus, args.selection, args.raw_dir, args.models, args.out, args.deadline)
    print(json.dumps({key: result.get(key) for key in
                      ("status", "cases_completed", "wall_seconds", "failure")}, sort_keys=True))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
