"""Frozen small-game curves and varied full-range river CFR measurements."""

import argparse
import json
import shutil
from collections import Counter
from hashlib import sha256
from math import fsum
from pathlib import Path
from time import monotonic, time

import numpy as np
import pokers

from scripts.evaluate_river_quality import (
    _append, _hash, _memory_pressure, _ranges, _resource_guard, _rss_bytes, _swap,
)
from src.arena.artifacts import environment, git, write_json
from src.arena.endgame_quality import TinyRiverGame, fixture_hand
from src.blueprint.artifact import load_training
from src.blueprint.river_cfr import RiverCFR, profile_quality
from src.blueprint.river_game import RiverGame, river_root_history
from src.blueprint.river_player import active_public_ranges
from src.blueprint.search import LiveBlueprint


def _shape_ranges(ranges, shape):
    if shape not in {"blueprint", "uniform", "squared",
                     "suited-bias", "pair-bias", "high-card-bias"}:
        raise ValueError(f"Unknown declared range shape: {shape}")
    shaped = {}
    for seat, rows in ranges.items():
        weights = []
        for pair, mass in rows:
            if shape == "uniform":
                weight = 1.0
            elif shape == "squared":
                weight = mass * mass
            elif shape == "suited-bias":
                weight = mass * (4 if pair[0][1] == pair[1][1] else 1)
            elif shape == "pair-bias":
                weight = mass * (4 if pair[0][0] == pair[1][0] else 1)
            elif shape == "high-card-bias":
                weight = mass * (3 if any(card[0] in "JQKA" for card in pair) else 1)
            else:
                weight = mass
            weights.append(weight)
        total = fsum(weights)
        shaped[seat] = tuple((pair, weight / total)
                             for (pair, _), weight in zip(rows, weights, strict=True))
    return shaped


def _range_summary(ranges):
    return {str(seat): {
        "holdings": len(rows),
        "effective_holdings": 1 / fsum(mass * mass for _, mass in rows),
        "maximum_mass": max(mass for _, mass in rows),
    } for seat, rows in ranges.items()}


def _save_profile(path, result):
    arrays = {f"average_{node}": policy for node, policy in result.average.items()}
    arrays.update({f"current_{node}": policy for node, policy in result.current.items()})
    np.savez_compressed(path, **arrays)


def run(plan: dict, fixtures_path: Path, checkpoint: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    if _hash(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    fixtures = json.loads(fixtures_path.read_text())["cases"]
    amendment = plan["schema"] == "river-range-amendment-m4-v1"
    tiny = [] if amendment else [case for case in fixtures if case["survivors"] == 2]
    if (len(tiny) != (0 if amendment else 12) or len(plan["full_range_cases"]) < 3
            or plan["tiny_sweeps"] != ([] if amendment else [8192, 16384, 32768])
            or plan["full_range_time_snapshots_seconds"] != [5, 15, 30, 60]):
        raise ValueError("Expected the frozen development design")
    out.parent.mkdir(parents=True, exist_ok=True)
    if _rss_bytes() >= plan["max_rss_gib"] * 1024**3:
        raise MemoryError("Process RSS already exceeds the development limit")
    if shutil.disk_usage(out.parent).free < plan["min_free_gib"] * 1024**3:
        raise RuntimeError("Development free-disk guard failed")
    out.mkdir()
    (out / "profiles").mkdir()
    started = monotonic()
    binary = next(Path(pokers.__file__).parent.glob("pokers*.so"))
    write_json(out / "manifest.json", {
        "plan": plan,
        "plan_sha256": sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "fixtures_sha256": _hash(fixtures_path),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "native_engine_binary_sha256": _hash(binary),
        "requirements_sha256": _hash(Path("requirements.txt")),
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "float_precision": "float64",
        "action_abstraction": "existing choices, two river raises",
        "root_laws": ["stipulated-product-compatible-v1",
                      "active-marginals-ignore-folded-removal-v1 with declared shape"],
        "extraction_measured": ["own-reach weighted average", "last played"],
        "environment": environment(),
        "started_unix_seconds": time(),
        "swap_before": _swap(),
        "memory_pressure_before": _memory_pressure(),
    })
    rows = []
    phase = "tiny"
    case = None
    solver = None
    status = "valid"
    error = None
    try:
        for case in tiny:
            _resource_guard(plan, out, started)
            begun = monotonic()
            hand = fixture_hand(case)
            view = hand.observe(hand.actor)
            root = river_root_history(view.history)
            ranges = _ranges(view)
            oracle = TinyRiverGame(root, ranges)
            game = RiverGame(root, ranges)
            solver = RiverCFR(game)
            row = {
                "phase": "tiny", "case_id": case["id"],
                "board": list(view.board), "root_pot_bb": view.pot / view.big_blind,
                "joint_deals": len(oracle.deals),
                "snapshots": [],
            }
            pending = None
            try:
                for target in plan["tiny_sweeps"]:
                    pending = target
                    result = solver.solve(
                        max_sweeps=target - solver.completed_sweeps,
                        deadline=started + plan["max_wall_seconds"],
                        rss_limit_bytes=int(plan["max_rss_gib"] * 1024**3),
                    )
                    reached = result.completed_sweeps >= target
                    batched = profile_quality(game, result.average)
                    independent = oracle.quality(result.average)
                    row["snapshots"].append({
                        "requested_sweeps": target,
                        "completed_sweeps": result.completed_sweeps,
                        "milestone_reached": reached,
                        "stop_reason": result.stop_reason,
                        "seconds_since_case_start": monotonic() - begun,
                        "average_quality": batched,
                        "current_quality": profile_quality(game, result.current),
                        "independent_average_quality": independent,
                    })
                    if not reached:
                        raise TimeoutError(
                            f"Tiny {case['id']} completed {result.completed_sweeps}/{target}"
                        )
                    pending = None
            except Exception as exc:
                row.update(status="incomplete", requested_sweeps=pending,
                           completed_sweeps=solver.completed_sweeps,
                           error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                row["seconds"] = monotonic() - begun
                row["peak_process_rss_bytes"] = _rss_bytes()
                rows.append(row)
                _append(out / "rows.jsonl", row)
        _resource_guard(plan, out, started)
        phase = "checkpoint_load"
        case = None
        solver = None
        begun = monotonic()
        trainer = load_training(checkpoint)
        blueprint = LiveBlueprint(trainer)
        load_seconds = monotonic() - begun
        _resource_guard(plan, out, started)
        phase = "full_range"
        for case in plan["full_range_cases"]:
            solver = None
            _resource_guard(plan, out, started)
            begun = monotonic()
            hand = fixture_hand(case)
            view = hand.observe(hand.actor)
            root = river_root_history(view.history)
            live = tuple(p.seat for p in view.players if not p.folded)
            coverage = Counter()
            row = {
                "phase": "full_range", "case_id": case["id"],
                "board": list(view.board), "active_seats": list(live),
                "root_pot_bb": view.pot / view.big_blind,
                "remaining_stacks_bb": [view.players[seat].stack / view.big_blind
                                        for seat in live],
                "range_shape": case["range_shape"],
                "street_bets": case["street_bets"],
                "snapshots": [],
            }
            try:
                deadline = min(started + plan["max_wall_seconds"], begun + 60)
                range_started = monotonic()
                ranges = active_public_ranges(blueprint, root, live, deadline, coverage)
                ranges = _shape_ranges(ranges, case["range_shape"])
                row["range_seconds"] = monotonic() - range_started
                row["range_summary"] = _range_summary(ranges)
                row["range_trained_lookups"] = coverage[("range", "trained")]
                row["range_untrained_lookups"] = coverage[("range", "untrained")]
                tree_started = monotonic()
                game = RiverGame(root, ranges, law_label=(
                    f"active-marginals-ignore-folded-removal-v1/{case['range_shape']}"
                ))
                row["tree_and_payoff_seconds"] = monotonic() - tree_started
                row["public_nodes"] = len(game.nodes)
                row["holdings_per_seat"] = [len(holdings) for holdings in game.holdings]
                row["compatible_joint_deals"] = int((game.joint > 0).sum())
                solver = RiverCFR(game)
                previous_sweeps = 0
                for seconds in plan["full_range_time_snapshots_seconds"]:
                    result = solver.solve(
                        max_sweeps=plan["full_range_max_sweeps"] - solver.completed_sweeps,
                        deadline=min(started + plan["max_wall_seconds"], begun + seconds),
                        rss_limit_bytes=int(plan["max_rss_gib"] * 1024**3),
                    )
                    increment = result.completed_sweeps - previous_sweeps
                    row["snapshots"].append({
                        "requested_seconds": seconds,
                        "actual_seconds_since_case_start": monotonic() - begun,
                        "completed_sweeps": result.completed_sweeps,
                        "additional_sweeps": increment,
                        "segment_seconds": result.elapsed_seconds,
                        "seconds_per_additional_sweep": (
                            result.elapsed_seconds / increment if increment else None
                        ),
                        "stop_reason": result.stop_reason,
                        "average_quality": profile_quality(game, result.average),
                        "current_quality": profile_quality(game, result.current),
                    })
                    previous_sweeps = result.completed_sweeps
                    _resource_guard(plan, out, started)
                row["zero_external_reach_entries"] = result.zero_external_reach_entries
                row["zero_average_denominators"] = result.zero_average_denominators
                row["profile_file"] = f"profiles/{case['id']}.npz"
                _save_profile(out / row["profile_file"], result)
            except Exception as exc:
                row.update(status="incomplete", error=f"{type(exc).__name__}: {exc}",
                           completed_sweeps=solver.completed_sweeps if solver else 0)
                raise
            finally:
                row["seconds"] = monotonic() - begun
                row["peak_process_rss_bytes"] = _rss_bytes()
                rows.append(row)
                _append(out / "rows.jsonl", row)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        failure = {
            "phase": phase, "case_id": case["id"] if case else None,
            "status": "failed", "error": error,
            "completed_sweeps": solver.completed_sweeps if solver else 0,
            "elapsed_seconds": monotonic() - started,
            "peak_process_rss_bytes": _rss_bytes(),
        }
        rows.append(failure)
        _append(out / "rows.jsonl", failure)
    result = {
        "status": status, "error": error,
        "planned_case_rows": len(tiny) + len(plan["full_range_cases"]),
        "retained_rows": len(rows),
        "elapsed_seconds": monotonic() - started,
        "checkpoint_load_seconds": load_seconds if phase == "full_range" else None,
        "peak_process_rss_bytes": _rss_bytes(),
        "swap_after": _swap(),
        "memory_pressure_after": _memory_pressure(),
    }
    write_json(out / "result.json", result)
    write_json(out / "checksums.json", {
        str(path.relative_to(out)): _hash(path) for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(json.loads(args.plan.read_text()), args.fixtures,
                 args.checkpoint, args.out)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
