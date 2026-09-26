"""Bounded M4 resource and exact-quality preflight for river CFR."""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from time import monotonic, time

import pokers

from src.arena.artifacts import environment, git, write_json
from src.arena.endgame_quality import TinyRiverGame, fixture_hand
from src.blueprint.artifact import load_training
from src.blueprint.river_cfr import RiverCFR, profile_quality
from src.blueprint.river_game import RiverGame, river_root_history
from src.blueprint.river_player import active_public_ranges
from src.blueprint.search import LiveBlueprint


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def _swap():
    if sys.platform != "darwin":
        return None
    value = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                           text=True, check=False)
    return value.stdout.strip() if value.returncode == 0 else value.stderr.strip()


def _memory_pressure():
    if sys.platform != "darwin":
        return None
    value = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                           text=True, check=False)
    return value.stdout.strip() if value.returncode == 0 else value.stderr.strip()


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())


def _ranges(view):
    from src.blueprint.search import DECK
    available = [card for card in DECK if card not in view.board]
    result = {}
    for index, player in enumerate(p for p in view.players if not p.folded):
        pairs = [tuple(sorted((available[(index * 7 + 2 * offset) % 31],
                               available[(index * 7 + 2 * offset + 1) % 31])))
                 for offset in range(4)]
        result[player.seat] = tuple((pair, float(mass))
                                    for pair, mass in zip(pairs, (1, 2, 3, 0)))
    return result


def _resource_guard(plan, out, started):
    if monotonic() - started >= plan["max_wall_seconds"]:
        raise TimeoutError("Preflight reached its overall wall limit")
    if _rss_bytes() >= plan["max_rss_gib"] * 1024**3:
        raise MemoryError("Preflight reached its process RSS limit")
    if shutil.disk_usage(out).free < plan["min_free_gib"] * 1024**3:
        raise RuntimeError("Preflight reached its free-disk limit")


def run(plan: dict, fixtures_path: Path, checkpoint: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    cases = json.loads(fixtures_path.read_text())["cases"]
    if len(cases) != 16 or {c["survivors"] for c in cases} != {2, 3}:
        raise ValueError("Expected the 12+4 frozen river fixtures")
    if _hash(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < plan["min_free_gib"] * 1024**3:
        raise RuntimeError("M4 free-disk guard failed")
    out.mkdir()
    started = monotonic()
    native_binary = next(Path(pokers.__file__).parent.glob("pokers*.so"))
    write_json(out / "manifest.json", {
        "plan": plan, "plan_sha256": sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "fixtures_sha256": _hash(fixtures_path),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "native_engine_binary_sha256": _hash(native_binary),
        "requirements_sha256": _hash(Path("requirements.txt")),
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "float_precision": "float64",
        "action_abstraction": "existing choices, two river raises",
        "root_laws": ["stipulated-product-compatible-v1",
                      "active-marginals-ignore-folded-removal-v1"],
        "extraction": "both current and own-reach weighted average measured in preflight",
        "environment": environment(), "started_unix_seconds": time(),
        "swap_before": _swap(), "memory_pressure_before": _memory_pressure(),
    })
    rows = []
    status = "valid"
    error = None
    phase = "reference"
    solver = None
    partial_metrics = {}
    try:
        for case in cases:
            _resource_guard(plan, out, started)
            begun = monotonic()
            hand = fixture_hand(case)
            view = hand.observe(hand.actor)
            root = river_root_history(view.history)
            ranges = _ranges(view)
            reference = TinyRiverGame(root, ranges)
            row = {
                "phase": "reference", "case_id": case["id"],
                "board": list(view.board), "active_seats": list(reference.seats),
                "joint_deals": len(reference.deals),
                "public_nodes": len(reference.nodes),
                "uniform_quality": reference.quality(reference.uniform_profile()),
            }
            if case["survivors"] == 2:
                game = RiverGame(root, ranges)
                solver = RiverCFR(game)
                snapshots = []
                for target in plan["reference_sweeps"]:
                    result = solver.solve(
                        max_sweeps=target - solver.completed_sweeps,
                        deadline=started + plan["max_wall_seconds"],
                        rss_limit_bytes=int(plan["max_rss_gib"] * 1024**3),
                    )
                    snapshots.append({
                        "sweeps": target,
                        "average_quality": profile_quality(game, result.average),
                        "current_quality": profile_quality(game, result.current),
                        "reference_average_quality": reference.quality(result.average),
                        "seconds_since_case_start": monotonic() - begun,
                    })
                row["work_quality"] = snapshots
            row["seconds"] = monotonic() - begun
            row["peak_process_rss_bytes"] = _rss_bytes()
            rows.append(row)
            _append(out / "rows.jsonl", row)
        _resource_guard(plan, out, started)
        phase = "checkpoint_load"
        solver = None
        begun = monotonic()
        trainer = load_training(checkpoint)  # one large checkpoint load
        blueprint = LiveBlueprint(trainer)
        checkpoint_seconds = monotonic() - begun
        partial_metrics["checkpoint_load_seconds"] = checkpoint_seconds
        partial_metrics["rss_after_checkpoint_bytes"] = _rss_bytes()
        _resource_guard(plan, out, started)
        phase = "full_range"
        solver = None
        case = next(case for case in cases if case["id"] == plan["full_range_fixture_id"])
        hand = fixture_hand(case)
        view = hand.observe(hand.actor)
        root = river_root_history(view.history)
        live = tuple(p.seat for p in view.players if not p.folded)
        deadline = min(started + plan["max_wall_seconds"],
                       monotonic() + plan["full_range_max_seconds"])
        begun = monotonic()
        ranges = active_public_ranges(blueprint, root, live, deadline)
        range_seconds = monotonic() - begun
        partial_metrics["range_construction_seconds"] = range_seconds
        partial_metrics["rss_after_ranges_bytes"] = _rss_bytes()
        _resource_guard(plan, out, started)
        begun = monotonic()
        game = RiverGame(root, ranges,
                         law_label="active-marginals-ignore-folded-removal-v1")
        construction_seconds = monotonic() - begun
        partial_metrics["tree_and_payoff_construction_seconds"] = construction_seconds
        partial_metrics["rss_after_tree_bytes"] = _rss_bytes()
        _resource_guard(plan, out, started)
        solver = RiverCFR(game)
        begun = monotonic()
        result = solver.solve(
            max_sweeps=plan["full_range_max_sweeps"], deadline=deadline,
            rss_limit_bytes=int(plan["max_rss_gib"] * 1024**3),
        )
        solve_seconds = monotonic() - begun
        row = {
            "phase": "full_range", "case_id": case["id"],
            "checkpoint_load_seconds": checkpoint_seconds,
            "range_construction_seconds": range_seconds,
            "tree_and_payoff_construction_seconds": construction_seconds,
            "solve_seconds": solve_seconds,
            "total_decision_seconds": range_seconds + construction_seconds + solve_seconds,
            "completed_sweeps": result.completed_sweeps,
            "stop_reason": result.stop_reason,
            "public_nodes": len(game.nodes),
            "holdings_per_seat": [len(h) for h in game.holdings],
            "compatible_joint_deals": int((game.joint > 0).sum()),
            "average_quality": profile_quality(game, result.average),
            "current_quality": profile_quality(game, result.current),
            "zero_external_reach_entries": result.zero_external_reach_entries,
            "zero_average_denominators": result.zero_average_denominators,
            "peak_process_rss_bytes": _rss_bytes(),
            "swap_after": _swap(),
            "memory_pressure_after": _memory_pressure(),
        }
        rows.append(row)
        _append(out / "rows.jsonl", row)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        failure = {
            "phase": phase, "status": "failed", "error": error,
            "case_id": case["id"] if phase in {"reference", "full_range"} else None,
            "completed_sweeps": solver.completed_sweeps if solver is not None else 0,
            "elapsed_seconds": monotonic() - started,
            "peak_process_rss_bytes": _rss_bytes(),
            "swap_after": _swap(), "memory_pressure_after": _memory_pressure(),
            "partial_metrics": partial_metrics,
        }
        rows.append(failure)
        _append(out / "rows.jsonl", failure)
    result = {
        "status": status, "error": error,
        "retained_rows": len(rows), "planned_rows": len(cases) + 1,
        "elapsed_seconds": monotonic() - started,
        "peak_process_rss_bytes": _rss_bytes(),
        "swap_after": _swap(), "memory_pressure_after": _memory_pressure(),
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
