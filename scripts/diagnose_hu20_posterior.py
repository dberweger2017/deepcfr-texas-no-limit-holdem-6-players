"""Outcome-blind B100M reverse-LBR selection and cost preflight on M4."""

import argparse
import gzip
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter, time

from scripts.diagnose_hu20_decisions import _selected_views
from scripts.evaluate_hu20 import rss, write_json
from scripts.evaluate_hu20_reopening import Target
from src.arena.schedule import digest, stream_seed
from src.diagnostics.reverse_lbr import compatible_holdings, likelihood_sample, observed_lbr_actions

SEEDS = (2026093001, 2026093002, 2026093003)
STREETS = ("preflop", "flop", "turn", "river")
MILESTONE = 100000000
SELECT_SALT = "hu20-b100-posterior-v1"
ROOT = 202610020101
THRESHOLDS = (1000, 300, 100)
LIKELIHOOD_COUNTS = (1, 2, 4, 8)
WORLD_COUNTS = (96, 192, 384)


def _coord(row):
    return (row["seed"], row["block"], row["rotation"], row["action_index"])


def _rank(row):
    token = "|".join(str(part) for part in (SELECT_SALT, row["seed"], row["file"],
                                              row["block"], row["rotation"],
                                              row["action_index"]))
    return sha256(token.encode()).hexdigest()


def _exclusions(selection_path, collisions_path):
    selected = json.loads(selection_path.read_text())["selected"]
    collisions = json.loads(collisions_path.read_text())["cases"]
    return {_coord(entry) for entry in selected} | {
        _coord(case["selection"]) for case in collisions}


def select(raw_dir, exclude_selection, exclude_collisions, output):
    if output.exists():
        raise FileExistsError(output)
    excludes = _exclusions(exclude_selection, exclude_collisions)
    cells = defaultdict(list)
    source_hashes = {}
    counts = Counter()
    for seed in SEEDS:
        path = raw_dir / f"B-{seed}-{MILESTONE}.jsonl.gz"
        source_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, "rt") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if row["policy"] != f"B-{seed}-{MILESTONE}" or row["status"] != "complete":
                    raise ValueError("Unexpected or incomplete retained LBR row")
                for action in row["actions"]:
                    if action["logical_player"] != 0 or action["target_trained"] is not True:
                        continue
                    visits = action.get("target_visits")
                    if type(visits) is not int or visits < 0:
                        raise ValueError("Trained decision lacks visit count")
                    position = "button" if action["seat"] == row["button"] else "big_blind"
                    entry = {"seed": seed, "file": path.name, "line": line_number,
                             "block": row["block"], "rotation": row["rotation"],
                             "action_index": action["index"], "street": action["street"],
                             "seat": action["seat"], "position": position,
                             "visits": visits, "trained": True}
                    if _coord(entry) in excludes:
                        counts["excluded"] += 1
                        continue
                    entry["rank"] = _rank(entry)
                    cells[(seed, action["street"], position)].append(entry)
                    counts["candidate"] += 1
    thresholds = {threshold: {
        f"{seed}/{street}/{position}": sum(row["visits"] >= threshold for row in cells[(seed, street, position)])
        for seed in SEEDS for street in STREETS for position in ("button", "big_blind")}
        for threshold in THRESHOLDS}
    eligible_full = [t for t in THRESHOLDS if all(thresholds[t].values())]
    threshold = eligible_full[0] if eligible_full else 100
    selected, secondary, empty = [], [], []
    for seed in SEEDS:
        for street in STREETS:
            for position in ("button", "big_blind"):
                cell = (seed, street, position)
                eligible = [row for row in cells[cell] if row["visits"] >= threshold]
                if eligible:
                    selected.append(min(eligible, key=lambda row: (row["rank"], _coord(row))))
                else:
                    empty.append({"seed": seed, "street": street, "position": position})
                    if cells[cell]:
                        secondary.append(min(cells[cell], key=lambda row: (-row["visits"], row["rank"])))
    selected.sort(key=lambda row: (row["seed"], STREETS.index(row["street"]), row["position"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "hu20-b100-posterior-selection-v1",
              "source": "#117 fresh LBR curve root 202610010300; aggregate returns previously opened, selector outcome-blind",
              "source_hashes": source_hashes,
              "excluded_coordinates": sorted(excludes), "excluded_encountered": counts["excluded"],
              "threshold_candidates": thresholds, "threshold_rule": "largest of 1000/300/100 filling all cells, else 100",
              "visits_threshold": threshold, "selected": selected,
              "secondary_empty_cell_cases": secondary, "empty_cells": empty,
              "selection_digest": digest(selected), "selected_count": len(selected),
              "candidate_count": counts["candidate"], "finished": time()}
    write_json(output, report)
    return {"status": "complete", "selected": len(selected), "empty_cells": len(empty),
            "threshold": threshold, "selection_digest": report["selection_digest"]}


def _swap_gib():
    output = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    return float(output.split("used = ", 1)[1].split("M", 1)[0]) / 1024


def _guard(output, deadline, swap_before):
    if time() >= deadline:
        raise TimeoutError("Original two-hour heavy-compute allowance")
    if rss() > 10.5 * 1024**3:
        raise MemoryError("10.5-GiB RSS guard")
    if shutil.disk_usage(output).free < 8 * 1024**3:
        raise OSError("8-GiB free-disk guard")
    if _swap_gib() - swap_before > .5:
        raise MemoryError("0.5-GiB swap-growth guard")


def preflight(raw_dir, selection_path, models_path, output, *, deadline):
    """Time actual LBR likelihood calls; discard their action outcomes."""
    output.mkdir(parents=True, exist_ok=False)
    swap_before = _swap_gib()
    selection = json.loads(selection_path.read_text())
    if digest(selection["selected"]) != selection["selection_digest"]:
        raise ValueError("Selection digest changed")
    views = list(_selected_views(raw_dir, selection))
    if len(views) != len(selection["selected"]):
        raise ValueError("Could not replay every selected decision")
    specs = {s["seed"]: s for s in json.loads(models_path.read_text())
             if s.get("arm") == "B" and s.get("milestone") == MILESTONE}
    measurements = []
    projected_calls = 0
    try:
        for seed in SEEDS:
            _guard(output, deadline, swap_before)
            source = Target(specs[seed])
            for street in STREETS:
                options = [(entry, view) for entry, _, view in views
                           if entry["seed"] == seed and entry["street"] == street]
                if not options:
                    continue
                # Highest public action count is a conservative timing case;
                # hash rank breaks ties without any value/return information.
                entry, view = min(options, key=lambda item:
                    (-len(observed_lbr_actions(item[1])), item[0]["rank"]))
                actions = observed_lbr_actions(view)
                n_holdings = len(compatible_holdings(view))
                projected_calls += sum(len(compatible_holdings(case_view)) * len(observed_lbr_actions(case_view))
                                       for e, _, case_view in views if e["seed"] == seed and e["street"] == street)
                for event_index, prefix, observed in actions:
                    holdings = sorted(compatible_holdings(view), key=lambda pair:
                                      sha256(repr((entry["rank"], event_index, pair)).encode()).hexdigest())[:4]
                    for pair in holdings:
                        _guard(output, deadline, swap_before)
                        before = perf_counter()
                        seed_value = stream_seed(ROOT, "validation", "opponent", "timing",
                                                 entry["rank"], event_index, pair)
                        # Intentionally discard equality and telemetry; timing only.
                        likelihood_sample(source, prefix, 1-view.seat, pair, observed, seed_value)
                        measurements.append({"seed": seed, "street": street,
                                             "event_index": event_index,
                                             "seconds": perf_counter()-before,
                                             "rss_bytes": rss()})
                        write_json(output / "measurements.json", measurements)
            del source
        if measurements:
            by_street = {street: max((m["seconds"] for m in measurements if m["street"] == street), default=0)
                         for street in STREETS}
            projected_seconds = sum(
                len(compatible_holdings(view)) * len(observed_lbr_actions(view)) * by_street[entry["street"]]
                for entry, _, view in views)
        else:
            projected_seconds = 0
        # #117 measured 60×96 uniform value worlds in 36 minutes. Keep an
        # explicit generous allowance for two ranges, controls and reporting.
        world_seconds_96 = 36 * 60 * len(views) / 60 * 2
        reserve_seconds = 1200
        remaining = deadline - time()
        feasible = [(l, w) for l in LIKELIHOOD_COUNTS for w in WORLD_COUNTS
                    if 1.25 * (l * projected_seconds + world_seconds_96 * (w / 96))
                    + reserve_seconds < remaining]
        scientific = [(l, w) for l, w in feasible if l >= 4]
        choice = max(scientific, key=lambda pair: (pair[1], pair[0])) if scientific else None
        report = {"status": "feasible" if choice else "insufficient_budget",
                  "likelihood_samples": choice[0] if choice else None,
                  "worlds": choice[1] if choice else None,
                  "root": ROOT, "deadline": deadline,
                  "remaining_seconds_at_freeze": remaining,
                  "one_sample_projected_seconds": projected_seconds,
                  "projected_likelihood_calls": projected_calls,
                  "world_time_allowance_seconds_at_96": world_seconds_96,
                  "report_control_reserve_seconds": reserve_seconds,
                  "safety_multiplier": 1.25,
                  "candidates_likelihood": LIKELIHOOD_COUNTS,
                  "candidates_worlds": WORLD_COUNTS,
                  "scientific_minimum": {"likelihood_samples": 4, "worlds": 96},
                  "measurements": measurements,
                  "selection_digest": selection["selection_digest"],
                  "rss_peak_sampled_bytes": max([rss()] + [m["rss_bytes"] for m in measurements]),
                  "swap_before_gib": swap_before, "swap_after_gib": _swap_gib(),
                  "finished": time()}
    except Exception as exc:
        report = {"status": "failed", "failure": f"{type(exc).__name__}: {exc}",
                  "measurements": measurements, "selection_digest": selection["selection_digest"],
                  "finished": time()}
    write_json(output / "frozen.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    pick = sub.add_parser("select")
    pick.add_argument("--raw-dir", type=Path, required=True)
    pick.add_argument("--exclude-selection", type=Path, required=True)
    pick.add_argument("--exclude-collisions", type=Path, required=True)
    pick.add_argument("--out", type=Path, required=True)
    timing = sub.add_parser("preflight")
    timing.add_argument("--raw-dir", type=Path, required=True)
    timing.add_argument("--selection", type=Path, required=True)
    timing.add_argument("--models", type=Path, required=True)
    timing.add_argument("--out", type=Path, required=True)
    timing.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    result = (select(args.raw_dir, args.exclude_selection, args.exclude_collisions, args.out)
              if args.phase == "select" else
              preflight(args.raw_dir, args.selection, args.models, args.out, deadline=args.deadline))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("complete", "feasible", "insufficient_budget") else 1


if __name__ == "__main__":
    raise SystemExit(main())
