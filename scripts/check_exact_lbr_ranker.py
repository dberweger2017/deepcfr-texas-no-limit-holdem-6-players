"""Frozen rank correctness, fixed-work equivalence and fresh LBR timings."""

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from time import perf_counter, process_time, time

from scripts.benchmark_reverse_lbr_workload import CONFIG, ROOT, _work
from scripts.evaluate_hu20 import rss, write_json
from scripts.evaluate_hu20_reopening import Target
from scripts.validate_reverse_lbr_acceleration import (
    _PairedDeck, _compare, _guard, _inputs, _swap_mib, _view,
)
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import choices
from src.diagnostics.cached_lbr import CachedLocalBestResponse, SharedProbabilityCache
from src.diagnostics.exact_ranker import (
    RankedCachedLocalBestResponse, _bind_choose_action, exact_seven_card,
)
from src.diagnostics.ranker_fixtures import FIXTURES, distinct_hands
from src.game.observation import replay
from src.game.showdown import hand_value

EXPECTED_CORPUS = "1559b5aac31016a9db92cfc2b319c28abf110446866d50dba0f45126eacadc4e"


def execute(source, view, seed, cache, executor, fixed=False):
    cls = CachedLocalBestResponse if executor == "original" else RankedCachedLocalBestResponse
    if fixed:
        ranker = hand_value if executor == "original" else exact_seven_card
        cls = type("FixedWorkLBR", (CachedLocalBestResponse,),
                   {"choose_action": _bind_choose_action(ranker, clock=lambda: 0)})
    lbr = cls(source, seed, cache, CONFIG)
    wall, cpu = perf_counter(), process_time()
    action = lbr.choose_action(view)
    wall, cpu = perf_counter() - wall, process_time() - cpu
    row = lbr.telemetry[-1]
    output = {"action": repr(action), "menu": [repr(c.action) for c in choices(view, free_fold=False)],
              "requested_samples": row["requested_samples"], "completed_samples": row["samples"],
              "zero_likelihood": lbr.zero_likelihood,
              "positive_range_holdings": row["positive_range_holdings"],
              "posterior_mass": row["posterior_mass"], "weights": tuple(map(float, lbr.weights)),
              "values": tuple(row["values_chips"]), "rng_digest": digest(lbr.random.getstate())}
    record = {k: v for k, v in output.items() if k != "weights"}
    record["weights_digest"] = digest(output["weights"])
    return output, record, {"wall_seconds": wall, "cpu_seconds": cpu,
                            "over_soft_budget": row["over_soft_budget"]}


def compare(a, b, label):
    differences = _compare(a, b, label)
    if a["rng_digest"] != b["rng_digest"]:
        differences.append(label + "/rng")
    return differences


def rank_check(out, deadline, swap_start):
    hands = list(distinct_hands())
    summary = {"hands": len(hands), "hand_digest": digest(hands),
               "rank_comparisons": 0, "suit_comparisons": 0, "order_comparisons": 0}
    def check(cards, expected=None, category="rank_comparisons"):
        reference = hand_value(cards)
        candidate = exact_seven_card(cards)
        if reference != candidate or (expected is not None and reference != expected):
            write_json(out / "mismatch.json", {"cards": cards, "reference": reference,
                       "candidate": candidate, "expected": expected, "category": category})
            raise ValueError("Independent rank mismatch; minimal reproduction retained")
        summary[category] += 1
    for cards, expected in FIXTURES:
        check(tuple(cards.split()), expected)
    for index, cards in enumerate(hands):
        if index % 1000 == 0:
            _guard(out, deadline, swap_start)
        check(cards)
    for cards in [tuple(cards.split()) for cards, _ in FIXTURES] + hands[:128]:
        expected = hand_value(cards)
        for suits in permutations("cdhs"):
            mapping = dict(zip("cdhs", suits))
            check(tuple(c[0] + mapping[c[1]] for c in cards), expected, "suit_comparisons")
    for cards in hands[:512]:
        check(tuple(reversed(cards)), hand_value(cards), "order_comparisons")
        check(cards[1:] + cards[:1], hand_value(cards), "order_comparisons")
    # Uninstrumented passes: same frozen corpus, each cache cold at start.
    summary["microbenchmark"] = {}
    for name, ranker in (("original", hand_value), ("candidate", exact_seven_card)):
        ranker.cache_clear()
        start_wall, start_cpu = perf_counter(), process_time()
        values = []
        for index, cards in enumerate(hands):
            if index % 1000 == 0:
                _guard(out, deadline, swap_start)
            values.append(ranker(cards))
        cold_wall, cold_cpu = perf_counter() - start_wall, process_time() - start_cpu
        cold_cache = ranker.cache_info()._asdict()
        start_wall, start_cpu = perf_counter(), process_time()
        repeated = [ranker(cards) for cards in hands[-4096:]]
        summary["microbenchmark"][name] = {"distinct_wall_seconds": cold_wall,
            "distinct_cpu_seconds": cold_cpu, "distinct_output_digest": digest(values),
            "cold_cache": cold_cache, "repeated_calls": len(repeated),
            "repeated_wall_seconds": perf_counter() - start_wall,
            "repeated_cpu_seconds": process_time() - start_cpu,
            "final_cache": ranker.cache_info()._asdict()}
    if summary["microbenchmark"]["original"]["distinct_output_digest"] != summary["microbenchmark"]["candidate"]["distinct_output_digest"]:
        raise ValueError("Rank timing digest mismatch")
    return summary


def validation(corpus, views, models, out, deadline, swap_start):
    summary = {"cases_completed": 0, "mismatches": [], "model_load": [], "outputs": []}
    with (out / "attempts.jsonl").open("w") as handle:
        for seed in sorted(models):
            _guard(out, deadline, swap_start)
            before = perf_counter()
            source = Target(models[seed])
            summary["model_load"].append({"seed": seed, "seconds": perf_counter() - before})
            cache = SharedProbabilityCache(source)
            for case in (c for c in corpus["cases"] if c["selected"]["seed"] == seed):
                _guard(out, deadline, swap_start)
                view = _view(case, views)
                original, record, _ = execute(source, view, case["internal_seed"], cache, "original", True)
                candidate, _, _ = execute(source, view, case["internal_seed"], cache, "candidate", True)
                repeated, _, _ = execute(source, view, case["internal_seed"], cache, "candidate", True)
                differences = compare(original, candidate, "fixed-original-candidate")
                differences += compare(candidate, repeated, "repeat")
                with _PairedDeck(True):
                    suit_view = _view(case, views, True)
                    suit_original, _, _ = execute(source, suit_view, case["internal_seed"], cache, "original", True)
                    suit_candidate, _, _ = execute(source, suit_view, case["internal_seed"], cache, "candidate", True)
                differences += compare(suit_original, suit_candidate, "suit-original-candidate")
                # Coupled card relabeling preserves numeric range/values; RNG unchanged.
                differences += compare(original, suit_original, "suit-coupling")
                output = {"case_id": case["case_id"], **record}
                summary["outputs"].append(output)
                row = {"case_id": case["case_id"], "seed": seed,
                       "street": case["selected"]["street"], "output": record,
                       "differences": differences, "rss_bytes": rss()}
                handle.write(json.dumps(row, sort_keys=True) + "\n"); handle.flush()
                summary["cases_completed"] += 1
                if differences:
                    summary["mismatches"].append(row)
                    write_json(out / "mismatch.json", {"case": case, "original": original,
                               "candidate": candidate, "suit_original": suit_original,
                               "suit_candidate": suit_candidate, "differences": differences})
                    raise ValueError("Fixed-work LBR mismatch; no performance gate accepted")
            del source, cache
    summary["output_digest"] = digest(summary.pop("outputs"))
    return summary


def benchmark(corpus, views, models, out, deadline, swap_start, executor, tasks=None):
    summary = {"calls_completed": 0, "limited_calls": 0, "over_soft_calls": 0,
               "model_load": [], "cache": [], "per_street": {}, "rank_cache": []}
    rows, outputs = [], []
    ranker = hand_value if executor == "original" else exact_seven_card
    ranker.cache_clear()
    if tasks is None:
        tasks = [{"seed": case["selected"]["seed"], "street": case["selected"]["street"],
                  "case": case, "sample": 0} for case in corpus["cases"]]
    with (out / "attempts.jsonl").open("w") as handle:
        for seed in sorted(models):
            _guard(out, deadline, swap_start)
            before = perf_counter()
            source = Target(models[seed])
            summary["model_load"].append({"seed": seed, "seconds": perf_counter() - before,
                                          "rss_bytes": rss()})
            cache = SharedProbabilityCache(source)
            for task in (t for t in tasks if t["seed"] == seed):
                samples = (0, 1) if task.get("incremental") else (0,)
                for sample in samples:
                    _guard(out, deadline, swap_start)
                    if "case" in task:
                        case = task["case"]
                        view, internal_seed = _view(case, views), case["internal_seed"]
                        identifier = case["case_id"]
                    else:
                        view = replay(task["prefix"], task["attacker_seat"], task["pair"])
                        internal_seed = stream_seed(ROOT, "validation", "opponent", "large",
                            task["selected_rank"], task["event_index"], task["pair"], sample)
                        identifier = digest((task["selected_rank"], task["event_index"], task["pair"], sample))
                    _, record, timing = execute(source, view, internal_seed, cache, executor)
                    output = {"case_id": identifier, **record}
                    outputs.append(output)
                    row = {"case_id": identifier, "seed": seed, "street": task["street"],
                           "sample": sample, **timing, "output": record, "rss_bytes": rss()}
                    handle.write(json.dumps(row, sort_keys=True) + "\n"); handle.flush()
                    rows.append(row)
                    summary["calls_completed"] += 1
                    summary["limited_calls"] += record["completed_samples"] < record["requested_samples"]
                    summary["over_soft_calls"] += timing["over_soft_budget"]
                    if summary["calls_completed"] % 32 == 0:
                        write_json(out / "progress.json", {"calls_completed": summary["calls_completed"],
                            "peak_rss_bytes": max(r["rss_bytes"] for r in rows)})
            summary["cache"].append({"seed": seed, **cache.telemetry()})
            summary["rank_cache"].append({"seed": seed, **ranker.cache_info()._asdict()})
            del source, cache
    summary["output_digest"] = digest(outputs)
    summary["peak_rss_bytes"] = max(r["rss_bytes"] for r in rows)
    summary["call_wall_seconds"] = sum(r["wall_seconds"] for r in rows)
    summary["call_cpu_seconds"] = sum(r["cpu_seconds"] for r in rows)
    for street in ("preflop", "flop", "turn", "river"):
        street_rows = [r for r in rows if r["street"] == street]
        summary["per_street"][street] = {
            str(sample): {"calls": len(selected),
                "wall_seconds": sum(r["wall_seconds"] for r in selected),
                "cpu_seconds": sum(r["cpu_seconds"] for r in selected),
                "max_seconds": max((r["wall_seconds"] for r in selected), default=0)}
            for sample in (0, 1) if (selected := [r for r in street_rows if r["sample"] == sample])}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("ranks", "validation", "bench336", "large"))
    parser.add_argument("--executor", choices=("original", "candidate"), default="candidate")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started, swap = time(), _swap_mib()
    summary = {"status": "running", "phase": args.phase, "executor": args.executor,
        "started": started, "deadline": args.deadline, "swap_start_mib": swap,
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    write_json(args.out / "result.json", summary)
    try:
        _guard(args.out, args.deadline, swap)
        if args.phase == "ranks":
            summary.update(rank_check(args.out, args.deadline, swap))
        else:
            before = perf_counter()
            corpus, views, models = _inputs(args.corpus, args.raw_dir, args.models)
            if corpus["case_digest"] != EXPECTED_CORPUS or len(corpus["cases"]) != 336:
                raise ValueError("Frozen #121 corpus changed")
            summary.update({"case_digest": corpus["case_digest"],
                "corpus_sha256": sha256(args.corpus.read_bytes()).hexdigest(),
                "model_spec_sha256": sha256(args.models.read_bytes()).hexdigest()})
            tasks = None
            if args.phase == "large":
                tasks, full, selected, streets, seeds = _work(args.raw_dir, args.selection)
                summary.update({"full_one_sample_calls": full, "selection_digest": selected,
                    "full_calls_by_street": streets, "full_calls_by_seed": seeds,
                    "selected_first_calls": len(tasks),
                    "selected_incremental_calls": sum(t["incremental"] for t in tasks)})
            summary["preprocessing_seconds"] = perf_counter() - before
            if args.phase == "validation":
                summary.update(validation(corpus, views, models, args.out, args.deadline, swap))
            else:
                summary.update(benchmark(corpus, views, models, args.out, args.deadline, swap,
                                         args.executor, tasks))
        summary["status"] = "complete"
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure"] = f"{type(exc).__name__}: {exc}"
    summary.update({"finished": time(), "swap_end_mib": _swap_mib(),
                    "free_disk_bytes": shutil.disk_usage(args.out).free})
    summary["wall_seconds"] = summary["finished"] - started
    write_json(args.out / "result.json", summary)
    print(json.dumps({k: summary.get(k) for k in ("status", "phase", "failure", "wall_seconds", "calls_completed", "cases_completed")}))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
