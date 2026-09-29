"""Fresh paired saved-policy LBR curve and exact-chip translation A/B on M4."""

import argparse
import gc
import gzip
import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from time import time

from scripts.evaluate_hu20 import rss, write_json
from scripts.evaluate_hu20_reopening import Target
from scripts.evaluate_robustness import play
from src.arena.heuristics import STYLES, StylePolicy
from src.arena.policies import make_policy
from src.arena.schedule import stream_seed
from src.diagnostics.action_translation import NearestOpponentRaiseLookup
from src.diagnostics.robustness import LBRConfig, ReactiveAttack

CURVE_ROOT = 202610010300
PREFLIGHT_ROOT = 202610010200
TRANSLATION_ROOT = 202610010400
SEEDS = (2026093001, 2026093002, 2026093003)
MILESTONES = (20000000, 40000000, 80000000, 100000000)
PANELS = (("pot_pressure", 256), ("one_third", 128),
          ("two_thirds", 128), ("native_minraise", 128), ("passive", 128))


def _swap_gib():
    output = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    matched = re.search(r"used = ([0-9.]+)M", output)
    if not matched:
        raise ValueError("Cannot parse M4 swap usage")
    return float(matched.group(1)) / 1024


def _guard(path, deadline, swap_before):
    if time() >= deadline:
        raise TimeoutError("Ten-hour absolute research deadline")
    if rss() > 10.5 * 1024**3:
        raise MemoryError("10.5-GiB process RSS guard")
    if shutil.disk_usage(path).free < 8 * 1024**3:
        raise OSError("8-GiB free disk guard")
    if _swap_gib() - swap_before > .5:
        raise MemoryError("0.5-GiB swap-growth guard")


def _specs(models):
    result = {(s["seed"], s["milestone"]): s for s in json.loads(models.read_text())
              if s.get("arm") == "B" and s.get("milestone") in MILESTONES}
    if set(result) != {(seed, milestone) for seed in SEEDS for milestone in MILESTONES}:
        raise ValueError("Missing saved B20/B40/B80/B100 models")
    return result


def curve_preflight(models, output, deadline):
    """Outcome-free timings on a disjoint validation root, with no result rows."""
    output.mkdir(parents=True, exist_ok=False)
    swap_before = _swap_gib()
    measurements = []
    for coordinate, spec in sorted(_specs(models).items()):
        _guard(output, deadline, swap_before)
        source = Target(spec)
        start = time()
        for block in range(8):
            for rotation in (0, 1):
                play(source, spec, ("lbr",), "menu", block, rotation,
                     PREFLIGHT_ROOT, "resource", LBRConfig(4, 5),
                     lambda row: None, resource_only=True)
        measurements.append({"seed": coordinate[0], "milestone": coordinate[1],
                             "blocks": 8, "seconds": time()-start, "rss_bytes": rss()})
        write_json(output / "measurements.json", measurements)
        del source
        gc.collect()
    per_block_total = sum(row["seconds"] / row["blocks"] for row in measurements)
    remaining = deadline - time()
    feasible = [count for count in (64, 128, 256, 512)
                if count * per_block_total * 1.2 + 5400 < remaining]
    selected = max(feasible) if feasible else None
    frozen = {"status": "feasible" if selected else "insufficient_time",
              "selected_blocks": selected, "candidates": [64, 128, 256, 512],
              "rule": "largest with 1.2x measured workload and 90-minute report reserve",
              "remaining_seconds_at_freeze": remaining,
              "per_block_total_seconds": per_block_total,
              "measurements": measurements, "root": CURVE_ROOT,
              "chance_samples": 4, "soft_seconds": 5,
              "swap_before_gib": swap_before, "frozen_at": time()}
    write_json(output / "frozen.json", frozen)
    return frozen


def run_curve(models, frozen_path, output, deadline):
    frozen = json.loads(frozen_path.read_text())
    count = frozen["selected_blocks"]
    if frozen["status"] != "feasible" or count not in (64, 128, 256, 512):
        raise ValueError("No frozen feasible LBR block count")
    output.mkdir(parents=True, exist_ok=False)
    swap_before = _swap_gib()
    attempts = []
    total_hands = 0
    try:
        for coordinate, spec in sorted(_specs(models).items()):
            _guard(output, deadline, swap_before)
            source = Target(spec)
            attempt = {"policy": spec["name"], "requested_blocks": count,
                       "completed_blocks": 0, "status": "running", "started": time()}
            attempts.append(attempt)
            write_json(output / "attempts.json", attempts)
            with gzip.open(output / f"{spec['name']}.jsonl.gz", "wt") as handle:
                def emit(row):
                    nonlocal total_hands
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    handle.flush()
                    total_hands += 1
                for block in range(count):
                    _guard(output, deadline, swap_before)
                    for rotation in (0, 1):
                        play(source, spec, ("lbr",), "menu", block, rotation,
                             CURVE_ROOT, "diagnosis-curve", LBRConfig(4, 5), emit)
                    attempt["completed_blocks"] += 1
                    if block % 16 == 15:
                        write_json(output / "attempts.json", attempts)
            attempt.update(status="complete", finished=time())
            write_json(output / "attempts.json", attempts)
            del source
            gc.collect()
        status, failure = "complete", None
    except Exception as exc:
        status, failure = "incomplete", f"{type(exc).__name__}: {exc}"
        if attempts and attempts[-1]["status"] == "running":
            attempts[-1].update(status="failed", failure=failure)
    result = {"status": status, "failure": failure, "hands": total_hands,
              "peak_rss_bytes": rss(), "swap_before_gib": swap_before,
              "swap_after_gib": _swap_gib(), "finished": time()}
    write_json(output / "attempts.json", attempts)
    write_json(output / "result.json", result)
    return result


def _opponent(panel, seed):
    if panel == "pot_pressure":
        return make_policy("pot_pressure", seed)
    if panel in ("one_third", "two_thirds"):
        numerator = 1 if panel == "one_third" else 2
        return StylePolicy(replace(STYLES["pot_pressure"], pot_numerator=numerator,
                                   pot_denominator=3), seed)
    if panel == "native_minraise":
        return ReactiveAttack("minraise", "native")
    if panel == "passive":
        return ReactiveAttack("passive", "menu")
    raise ValueError("Unknown frozen translation control")


def run_translation(models, output, deadline):
    output.mkdir(parents=True, exist_ok=False)
    swap_before = _swap_gib()
    specs = _specs(models)
    attempts = []
    hands = 0
    try:
        for seed in SEEDS:
            spec = specs[(seed, 100000000)]
            _guard(output, deadline, swap_before)
            base = Target(spec)
            for variant in ("exact", "nearest"):
                source = base if variant == "exact" else NearestOpponentRaiseLookup(base)
                for panel_index, (panel, blocks) in enumerate(PANELS):
                    root = TRANSLATION_ROOT + panel_index
                    attempt = {"seed": seed, "variant": variant, "panel": panel,
                               "requested_blocks": blocks, "completed_blocks": 0,
                               "status": "running", "started": time()}
                    attempts.append(attempt)
                    write_json(output / "attempts.json", attempts)
                    with gzip.open(output / f"{seed}-{variant}-{panel}.jsonl.gz", "wt") as handle:
                        def emit(row):
                            nonlocal hands
                            row.update(translation_variant=variant, translation_panel=panel)
                            handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                            handle.flush()
                            hands += 1
                        for block in range(blocks):
                            _guard(output, deadline, swap_before)
                            opponent_seed = stream_seed(root, "test", "action", 2, block, 1)
                            for rotation in (0, 1):
                                rivals = {1: _opponent(panel, opponent_seed)}
                                play(source, spec, (panel,), "secondary", block,
                                     rotation, root, "diagnosis-translation",
                                     LBRConfig(4, 5), emit, opponent_policies=rivals)
                            attempt["completed_blocks"] += 1
                            if block % 32 == 31:
                                write_json(output / "attempts.json", attempts)
                    attempt.update(status="complete", finished=time())
                    write_json(output / "attempts.json", attempts)
            del base
            gc.collect()
        status, failure = "complete", None
    except Exception as exc:
        status, failure = "incomplete", f"{type(exc).__name__}: {exc}"
        if attempts and attempts[-1]["status"] == "running":
            attempts[-1].update(status="failed", failure=failure)
    result = {"status": status, "failure": failure, "hands": hands,
              "peak_rss_bytes": rss(), "swap_before_gib": swap_before,
              "swap_after_gib": _swap_gib(), "finished": time()}
    write_json(output / "attempts.json", attempts)
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("curve-preflight", "curve", "translation"))
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    if args.phase == "curve-preflight":
        result = curve_preflight(args.models, args.out, args.deadline)
    elif args.phase == "curve":
        result = run_curve(args.models, args.frozen, args.out, args.deadline)
    else:
        result = run_translation(args.models, args.out, args.deadline)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("complete", "feasible") else 1


if __name__ == "__main__":
    raise SystemExit(main())
