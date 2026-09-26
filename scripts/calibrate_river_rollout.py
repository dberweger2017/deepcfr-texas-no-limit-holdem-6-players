"""Resource-only calibration of same-Q corrected river rollout world counts."""

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import monotonic, time

import pokers

from scripts.evaluate_river_development import _shape_ranges
from scripts.evaluate_river_quality import (
    _append, _hash, _memory_pressure, _resource_guard, _rss_bytes, _swap,
)
from src.arena.artifacts import environment, git, write_json
from src.arena.endgame_quality import fixture_hand
from src.arena.river_conditional import ConditionalRiverRollout
from src.blueprint.artifact import load_training
from src.blueprint.river_game import RiverGame, river_root_history
from src.blueprint.river_player import active_public_ranges
from src.blueprint.search import LiveBlueprint, SearchConfig


def run(plan: dict, checkpoint: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    if _hash(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    source_plans = {path: json.loads(Path(path).read_text())
                    for path in plan["source_plans"]}
    cases = {case["id"]: case for source in source_plans.values()
             for case in source["full_range_cases"]}
    if (len(plan["case_ids"]) != 3 or len(set(plan["case_ids"])) != 3
            or any(case_id not in cases for case_id in plan["case_ids"])):
        raise ValueError("Calibration cases must be frozen development roots")
    out.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    if shutil.disk_usage(out.parent).free < plan["min_free_gib"] * 1024**3:
        raise RuntimeError("Calibration free-disk guard failed")
    out.mkdir()
    started = monotonic()
    binary = next(Path(pokers.__file__).parent.glob("pokers*.so"))
    write_json(out / "manifest.json", {
        "plan": plan,
        "plan_sha256": sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "source_plan_hashes": {path: _hash(Path(path)) for path in source_plans},
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "native_engine_binary_sha256": _hash(binary),
        "requirements_sha256": _hash(Path("requirements.txt")),
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "environment": environment(),
        "started_unix_seconds": time(),
        "swap_before": _swap(),
        "memory_pressure_before": _memory_pressure(),
    })
    rows = []
    status = "valid"
    error = None
    case_id = None
    phase = "checkpoint_load"
    try:
        trainer = load_training(checkpoint)
        blueprint = LiveBlueprint(trainer)
        _resource_guard(plan, out, started)
        phase = "calibration"
        for case_index, case_id in enumerate(plan["case_ids"]):
            case = cases[case_id]
            hand = fixture_hand(case)
            view = hand.observe(hand.actor)
            root = river_root_history(view.history)
            live = tuple(p.seat for p in view.players if not p.folded)
            deadline = started + plan["max_wall_seconds"]
            ranges = _shape_ranges(
                active_public_ranges(blueprint, root, live, deadline, Counter()),
                case["range_shape"],
            )
            game = RiverGame(root, ranges)
            for worlds in plan["world_counts"]:
                _resource_guard(plan, out, started)
                config = SearchConfig(
                    max_seconds=plan["per_decision_max_seconds"],
                    worlds=worlds, range_samples=plan["range_samples"],
                    styles=tuple(plan["styles"]), variant="corrected",
                )
                control = ConditionalRiverRollout(
                    blueprint, game, plan["seed"] + case_index * 10_000 + worlds,
                    config,
                )
                begun = monotonic()
                row = {"phase": "calibration", "case_id": case_id,
                       "worlds_requested": worlds, "range_shape": case["range_shape"],
                       "public_nodes": len(game.nodes)}
                try:
                    action = control.choose_action(view)
                    view.legal_actions.validate(action)
                    row.update(action_kind=action.kind.value,
                               action_raise_to=action.raise_to,
                               worlds_completed=control.worlds_completed[-1],
                               fallback=control.fallbacks == 1,
                               seconds=monotonic() - begun)
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                               seconds=monotonic() - begun)
                    raise
                finally:
                    row["peak_process_rss_bytes"] = _rss_bytes()
                    rows.append(row)
                    _append(out / "rows.jsonl", row)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        failure = {"phase": phase, "case_id": case_id, "status": "failed",
                   "error": error, "elapsed_seconds": monotonic() - started,
                   "peak_process_rss_bytes": _rss_bytes()}
        rows.append(failure)
        _append(out / "rows.jsonl", failure)
    result = {"status": status, "error": error,
              "planned_rows": 3 * len(plan["world_counts"]),
              "retained_rows": len(rows),
              "elapsed_seconds": monotonic() - started,
              "peak_process_rss_bytes": _rss_bytes(),
              "swap_after": _swap(),
              "memory_pressure_after": _memory_pressure()}
    write_json(out / "result.json", result)
    write_json(out / "checksums.json", {
        str(path.relative_to(out)): _hash(path) for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(json.loads(args.plan.read_text()), args.checkpoint, args.out)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
