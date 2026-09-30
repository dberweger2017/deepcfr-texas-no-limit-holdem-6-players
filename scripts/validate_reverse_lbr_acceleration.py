"""Validate and time a selectable LBR cache against the frozen native corpus.

This script never estimates a posterior or target action value. The raw archive
is used only to reconstruct public prefixes and the target's own cards.
"""

import argparse
import gzip
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time, time

import numpy as np

from scripts.diagnose_hu20_decisions import _trace
from scripts.evaluate_hu20 import rss, write_json
from scripts.evaluate_hu20_reopening import Target
from src.arena.schedule import digest
from src.blueprint import search
from src.blueprint.abstraction import choices
from src.diagnostics import robustness
from src.diagnostics.cached_lbr import CachedLocalBestResponse, SharedProbabilityCache
from src.diagnostics.robustness import LBRConfig, LocalBestResponse
from src.game.observation import BoardDealt, replay

CONFIG = LBRConfig(4, 5)
SUITS = {"c": "d", "d": "h", "h": "s", "s": "c"}
TOLERANCE = 1e-10


def _swap_mib():
    output = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    return float(output.split("used = ", 1)[1].split("M", 1)[0])


def _guard(out, deadline, swap_start):
    if time() >= deadline:
        raise TimeoutError("Frozen three-hour M4 deadline")
    if rss() > 10.5 * 1024**3:
        raise MemoryError("10.5-GiB RSS guard")
    if shutil.disk_usage(out).free < 8 * 1024**3:
        raise OSError("8-GiB free-disk guard")
    if _swap_mib() - swap_start > 512:
        raise MemoryError("0.5-GiB swap-growth guard")


def _inputs(corpus_path, raw_dir, models_path):
    corpus = json.loads(corpus_path.read_text())
    if corpus["case_digest"] != digest(corpus["cases"]) or corpus["case_count"] != len(corpus["cases"]):
        raise ValueError("Frozen corpus digest/count mismatch")
    if corpus["selection_digest"] != "578b8b67f21a15d2961e88cf498aeeb64831c9827e578e5ce1db62f85d43e322":
        raise ValueError("#119 selected-list digest mismatch")
    wanted = defaultdict(lambda: defaultdict(set))
    for case in corpus["cases"]:
        selected = case["selected"]
        wanted[selected["file"]][selected["line"]].add(case["attacker_action_index"])
    views = {}
    for filename, lines in sorted(wanted.items()):
        path = raw_dir / filename
        if sha256(path.read_bytes()).hexdigest() != corpus["source_hashes"][filename]:
            raise ValueError(f"Retained raw source hash mismatch: {filename}")
        with gzip.open(path, "rt") as handle:
            for line, payload in enumerate(handle, 1):
                if line not in lines:
                    continue
                row = json.loads(payload)
                for item, view in _trace(row):
                    if item["index"] in lines[line] and item["logical_player"] == 1:
                        views[(filename, line, item["index"])] = view
    if len(views) != sum(len(indices) for lines in wanted.values() for indices in lines.values()):
        raise ValueError("Missing or non-attacker public prefix")
    models = {s["seed"]: s for s in json.loads(models_path.read_text())
              if s.get("arm") == "B" and s.get("milestone") == 100000000}
    if set(models) != {2026093001, 2026093002, 2026093003}:
        raise ValueError("Expected exactly three saved B100M lineages")
    return corpus, views, models


def _view(case, views, suit=False):
    selected = case["selected"]
    original = views[(selected["file"], selected["line"], case["attacker_action_index"])]
    if digest([repr(e) for e in original.history]) != case["prefix_digest"]:
        raise ValueError("Frozen public-prefix digest mismatch")
    pair = tuple(case["attacker_holding"])
    history = original.history
    if suit:
        card = lambda c: c[0] + SUITS[c[1]]
        pair = tuple(map(card, pair))
        history = tuple(replace(e, cards=tuple(map(card, e.cards)))
                        if isinstance(e, BoardDealt) else e for e in history)
    view = replay(history, original.seat, pair)
    expected_board = tuple(case["public_board_at_prefix"])
    if suit:
        expected_board = tuple(c[0] + SUITS[c[1]] for c in expected_board)
    if view.board != expected_board or view.hole_cards != pair:
        raise ValueError("Hypothetical attacker-visible view mismatch")
    if set(pair).intersection(case["target_cards"] if not suit else
                              (c[0] + SUITS[c[1]] for c in case["target_cards"])):
        raise ValueError("Hypothetical holding conflicts with target cards")
    return view


class _PairedDeck:
    def __init__(self, enabled):
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            self.native = robustness.DECK
            self.search = search.DECK
            permuted = tuple(c[0] + SUITS[c[1]] for c in self.native)
            robustness.DECK = permuted
            search.DECK = permuted

    def __exit__(self, *_):
        if self.enabled:
            robustness.DECK = self.native
            search.DECK = self.search


def _execute(source, case, view, mode, cache):
    seed = case["internal_seed"]
    if mode == "native":
        lbr = LocalBestResponse(source, seed, CONFIG)
    else:
        lbr = CachedLocalBestResponse(source, seed, cache, CONFIG)
    start_wall, start_cpu = perf_counter(), process_time()
    action = lbr.choose_action(view)
    wall, cpu = perf_counter() - start_wall, process_time() - start_cpu
    row = lbr.telemetry[-1]
    return {"action": repr(action),
            "menu": [repr(c.action) for c in choices(view, free_fold=False)],
            "requested_samples": row["requested_samples"], "completed_samples": row["samples"],
            "zero_likelihood": lbr.zero_likelihood,
            "positive_range_holdings": row["positive_range_holdings"],
            "posterior_mass": row["posterior_mass"],
            "weights": tuple(map(float, lbr.weights)),
            "values": tuple(row["values_chips"]), "wall_seconds": wall, "cpu_seconds": cpu}


def _compare(a, b, prefix):
    differences = []
    for key in ("action", "menu", "requested_samples", "completed_samples",
                "zero_likelihood", "positive_range_holdings"):
        if a[key] != b[key]:
            differences.append(f"{prefix}/{key}")
    for key in ("weights", "values"):
        if len(a[key]) != len(b[key]) or not np.allclose(a[key], b[key], atol=TOLERANCE, rtol=0):
            differences.append(f"{prefix}/{key}")
    if abs(a["posterior_mass"] - b["posterior_mass"]) > TOLERANCE:
        differences.append(f"{prefix}/posterior_mass")
    return differences


def run(corpus_path, raw_dir, models_path, out, *, mode, deadline):
    out.mkdir(parents=True, exist_ok=False)
    swap_start = _swap_mib()
    started = time()
    summary = {"status": "running", "mode": mode, "started": started,
               "deadline": deadline, "swap_start_mib": swap_start,
               "cases_completed": 0, "mismatches": [], "peak_rss_bytes": rss()}
    write_json(out / "result.json", summary)
    try:
        corpus, views, models = _inputs(corpus_path, raw_dir, models_path)
        summary["case_digest"] = corpus["case_digest"]
        summary["corpus_sha256"] = sha256(corpus_path.read_bytes()).hexdigest()
        summary["model_spec_sha256"] = sha256(models_path.read_bytes()).hexdigest()
        by_seed = defaultdict(list)
        for case in corpus["cases"]:
            by_seed[case["selected"]["seed"]].append(case)
        by_street = defaultdict(list)
        cache_data = []
        model_load = []
        with (out / "attempts.jsonl").open("w") as handle:
            for seed in sorted(by_seed):
                _guard(out, deadline, swap_start)
                before = perf_counter()
                source = Target(models[seed])
                model_load.append({"seed": seed, "seconds": perf_counter() - before, "rss_bytes": rss()})
                cache = SharedProbabilityCache(source)
                for case in by_seed[seed]:
                    _guard(out, deadline, swap_start)
                    base = _view(case, views)
                    with _PairedDeck(False):
                        native = _execute(source, case, base, "native", cache)
                        optimized = _execute(source, case, base, "cached", cache)
                        repeat = _execute(source, case, base, "cached", cache)
                    diffs = _compare(native, optimized, "native-vs-cache")
                    diffs += _compare(optimized, repeat, "cached-repeat")
                    suit = _view(case, views, suit=True)
                    with _PairedDeck(True):
                        native_suit = _execute(source, case, suit, "native", cache)
                        optimized_suit = _execute(source, case, suit, "cached", cache)
                    diffs += _compare(native_suit, optimized_suit, "suit-native-vs-cache")
                    diffs += _compare(native, native_suit, "native-suit-coupling")
                    vals = native["values"]
                    margin = (sorted(vals, reverse=True)[0] - sorted(vals, reverse=True)[1]
                              if len(vals) > 1 else None)
                    row = {"case_id": case["case_id"], "seed": seed,
                           "street": case["selected"]["street"], "position": case["selected"]["position"],
                           "attacker_action_kind": case["attacker_action_kind"],
                           "native_action": native["action"], "suit_action": native_suit["action"],
                           "menu": native["menu"], "requested_samples": native["requested_samples"],
                           "completed_samples": native["completed_samples"],
                           "positive_range_holdings": native["positive_range_holdings"],
                           "zero_likelihood": native["zero_likelihood"],
                           "values_chips": vals, "near_tie_margin_chips": margin,
                           "native_seconds": native["wall_seconds"],
                           "cached_seconds": optimized["wall_seconds"],
                           "repeat_seconds": repeat["wall_seconds"],
                           "native_suit_seconds": native_suit["wall_seconds"],
                           "cached_suit_seconds": optimized_suit["wall_seconds"],
                           "differences": diffs}
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    by_street[row["street"]].append(row)
                    summary["cases_completed"] += 1
                    summary["peak_rss_bytes"] = max(summary["peak_rss_bytes"], rss())
                    if diffs:
                        summary["mismatches"].append({"case_id": case["case_id"], "fields": diffs})
                    if summary["cases_completed"] % 12 == 0:
                        write_json(out / "result.json", summary)
                cache_data.append({"seed": seed, **cache.telemetry()})
                del source, cache
        summary["model_load"] = model_load
        summary["cache"] = cache_data
        summary["street_counts"] = dict(Counter(row["street"] for rows in by_street.values() for row in rows))
        summary["menu_actions"] = dict(Counter(action.split("(")[0]
                              for rows in by_street.values() for row in rows for action in row["menu"]))
        summary["chosen_actions"] = dict(Counter(row["native_action"].split("(")[0]
                              for rows in by_street.values() for row in rows))
        summary["native_call_seconds"] = sum(row["native_seconds"] for rows in by_street.values() for row in rows)
        summary["cached_call_seconds"] = sum(row["cached_seconds"] for rows in by_street.values() for row in rows)
        summary["per_street_seconds"] = {street: {"native": sum(r["native_seconds"] for r in rows),
                                                   "cached": sum(r["cached_seconds"] for r in rows)}
                                         for street, rows in by_street.items()}
        summary["status"] = "complete" if not summary["mismatches"] else "equivalence_failed"
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
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    result = run(args.corpus, args.raw_dir, args.models, args.out,
                 mode="validation", deadline=args.deadline)
    print(json.dumps({k: result.get(k) for k in
                      ("status", "cases_completed", "failure", "mismatches", "wall_seconds")}, sort_keys=True))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
