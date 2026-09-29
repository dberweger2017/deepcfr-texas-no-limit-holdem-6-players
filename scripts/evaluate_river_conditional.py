"""Frozen, paired conditional river returns with the same declared card law.

This evaluator samples hidden deals only for simulation. Each policy receives
its own Observation and independent private random stream. It retains every
attempt and does not select a model from the confirmation outcomes.
"""

import argparse
import json
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random
from time import monotonic, time

import numpy as np
import pokers

from scripts.evaluate_river_development import _shape_ranges
from scripts.evaluate_river_quality import (
    _append, _hash, _memory_pressure, _resource_guard, _rss_bytes, _swap,
)
from src.arena.artifacts import environment, git, write_json
from src.arena.endgame_quality import _world, fixture_hand
from src.arena.policies import make_policy
from src.arena.river_conditional import ConditionalRiverRollout, FrozenRiverProfile
from src.blueprint.artifact import load_training
from src.blueprint.river_cfr import RiverCFR
from src.blueprint.river_game import RiverGame, river_root_history
from src.blueprint.river_player import active_public_ranges
from src.blueprint.search import LiveBlueprint, SearchConfig
from src.game.observation import HandFinished


ARMS = ("river_cfr", "rollout_normal", "rollout_matched")


def _seed(master: int, case_id: str, repetition: int, purpose: str) -> int:
    value = f"{master}:{case_id}:{repetition}:{purpose}".encode()
    return int.from_bytes(sha256(value).digest()[:8], "big")


def _draw(game: RiverGame, seed: int):
    rng = np.random.default_rng(seed)
    index = int(rng.choice(game.joint.size, p=game.joint.ravel()))
    ids = np.unravel_index(index, game.joint.shape)
    return {seat: game.holdings[player][ids[player]]
            for player, seat in enumerate(game.seats)}


def _play(game, holes, hero, opponent_style, opponent_seed, player):
    hand = _world(game.nodes[0].history, game.board, holes)
    root_stack = hand.observe(hero).players[hero].stack
    opponent = make_policy(opponent_style, opponent_seed)
    actions = []
    while not hand.finished:
        actor = hand.actor
        view = hand.observe(actor)
        action = (player if actor == hero else opponent).choose_action(view)
        view.legal_actions.validate(action)
        actions.append({"seat": actor, "kind": action.kind.value,
                        "raise_to": action.raise_to})
        hand = hand.apply(action)
    finish = next(event for event in hand.events if isinstance(event, HandFinished))
    return {"payoff_bb": (finish.stacks[hero] - root_stack) / game.big_blind,
            "actions": actions, "finished": True}


def _guard(plan, out, started, reserve_seconds=0):
    _resource_guard(plan, out, started)
    if monotonic() - started + reserve_seconds + 2 >= plan["max_wall_seconds"]:
        raise TimeoutError("Insufficient overall wall time for another bounded unit")


def run(plan: dict, cases_path: Path, checkpoint: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    if _hash(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    cases_source = json.loads(cases_path.read_text())["cases"]
    cases = {case["id"]: case for case in cases_source}
    if (len(cases) != len(cases_source) or len(set(plan["case_ids"])) != len(plan["case_ids"])
            or any(case_id not in cases for case_id in plan["case_ids"])):
        raise ValueError("Cases do not match the frozen plan")
    if plan["arms"] != list(ARMS) or plan["extraction"] != "average":
        raise ValueError("Expected the frozen average-profile three-arm comparison")
    if plan["matched_worlds"] < 1 or plan["matched_worlds"] > 16384:
        raise ValueError("Invalid matched world count")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < plan["min_free_gib"] * 1024**3:
        raise RuntimeError("Conditional evaluation free-disk guard failed")
    out.mkdir()
    (out / "profiles").mkdir()
    started = monotonic()
    binary = next(Path(pokers.__file__).parent.glob("pokers*.so"))
    write_json(out / "manifest.json", {
        "plan": plan,
        "plan_sha256": sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "cases_sha256": _hash(cases_path),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "native_engine_binary_sha256": _hash(binary),
        "requirements_sha256": _hash(Path("requirements.txt")),
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "environment": environment(),
        "started_unix_seconds": time(),
        "swap_before": _swap(),
        "memory_pressure_before": _memory_pressure(),
        "scope": "conditional two-player river returns, not six-player full-hand strength",
        "law": "active-marginals-ignore-folded-removal-v1 with declared shape",
    })
    attempts = 0
    status = "valid"
    error = None
    phase = "checkpoint_load"
    case_id = None
    repetition = None
    arm = None
    root_rows = 0
    try:
        trainer = load_training(checkpoint)
        blueprint = LiveBlueprint(trainer)
        for case_id in plan["case_ids"]:
            _guard(plan, out, started, plan["solver_seconds"])
            phase = "root"
            case = cases[case_id]
            case_started = monotonic()
            row = {"phase": "root", "case_id": case_id, "status": "started"}
            try:
                fixture = fixture_hand(case)
                view = fixture.observe(fixture.actor)
                root = river_root_history(view.history)
                live = tuple(p.seat for p in view.players if not p.folded)
                deadline = min(case_started + plan["solver_seconds"],
                               started + plan["max_wall_seconds"])
                coverage = Counter()
                ranges = _shape_ranges(
                    active_public_ranges(blueprint, root, live, deadline, coverage),
                    case["range_shape"],
                )
                game = RiverGame(root, ranges)
                solver = RiverCFR(game)
                result = solver.solve(
                    max_sweeps=plan["solver_max_sweeps"], deadline=deadline,
                    rss_limit_bytes=int(plan["max_rss_gib"] * 1024**3),
                )
                if result.completed_sweeps < plan["solver_min_sweeps"]:
                    raise TimeoutError("River CFR completed fewer than minimum sweeps")
                hero = (fixture.actor if case["hero_position"] == "first" else
                        next(seat for seat in game.seats if seat != fixture.actor))
                solve_seconds = monotonic() - case_started
                profile_name = sha256(case_id.encode()).hexdigest()[:16] + ".npz"
                profile_path = out / "profiles" / profile_name
                np.savez_compressed(profile_path, **{
                    str(node): strategy for node, strategy in result.average.items()
                })
                row.update(status="completed", seconds=monotonic() - case_started,
                           solver_seconds=solve_seconds,
                           profile_path=str(profile_path.relative_to(out)),
                           profile_sha256=_hash(profile_path),
                           completed_sweeps=result.completed_sweeps,
                           stop_reason=result.stop_reason,
                           public_nodes=len(game.nodes), joint_deals=int(np.count_nonzero(game.joint)),
                           root_pot_bb=game.root_pot / game.big_blind,
                           range_trained_lookups=coverage[("range", "trained")],
                           range_untrained_lookups=coverage[("range", "untrained")],
                           hero_seat=hero)
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                           seconds=monotonic() - case_started)
                raise
            finally:
                row["peak_process_rss_bytes"] = _rss_bytes()
                _append(out / "rows.jsonl", row)
                root_rows += 1
            for repetition in range(plan["deals_per_root"]):
                _guard(plan, out, started, plan["matched_seconds"])
                phase = "deal"
                holes = _draw(game, _seed(plan["seed"], case_id, repetition, "deal"))
                opponent_seed = _seed(plan["seed"], case_id, repetition, "opponent")
                # The same hidden deal and opponent seed are used for all arms.
                for arm in ARMS:
                    reserve = (plan["matched_seconds"] if arm == "rollout_matched"
                               else plan["normal_seconds"])
                    _guard(plan, out, started, reserve)
                    phase = "play"
                    begun = monotonic()
                    attempt = {"phase": "play", "case_id": case_id,
                               "repetition": repetition, "arm": arm,
                               "holes": {str(seat): list(pair) for seat, pair in holes.items()},
                               "opponent_style": case["opponent_style"],
                               "opponent_seed": opponent_seed,
                               "status": "started"}
                    player = None
                    try:
                        seed = _seed(plan["seed"], case_id, repetition, arm)
                        normal = SearchConfig(
                            max_seconds=plan["normal_seconds"],
                            worlds=plan["normal_worlds"],
                            styles=tuple(plan["styles"]), variant="corrected",
                        )
                        if arm == "river_cfr":
                            player = FrozenRiverProfile(blueprint, game, result.average,
                                                        seed, normal)
                        else:
                            config = (normal if arm == "rollout_normal" else SearchConfig(
                                max_seconds=plan["matched_seconds"],
                                worlds=min(plan["matched_worlds"], 4096),
                                styles=tuple(plan["styles"]), variant="corrected",
                            ))
                            player = ConditionalRiverRollout(
                                blueprint, game, seed, config,
                                worlds_override=(plan["matched_worlds"]
                                                 if arm == "rollout_matched" else None),
                            )
                        attempt.update(_play(game, holes, hero, case["opponent_style"],
                                             opponent_seed, player))
                        attempt["status"] = "completed"
                    except Exception as exc:
                        attempt.update(status="failed",
                                       error=f"{type(exc).__name__}: {exc}")
                        raise
                    finally:
                        attempt["seconds"] = monotonic() - begun
                        attempt["peak_process_rss_bytes"] = _rss_bytes()
                        if isinstance(player, FrozenRiverProfile):
                            attempt["delegations"] = player.delegations
                            player = player.fallback
                        if isinstance(player, ConditionalRiverRollout):
                            attempt.update(rollout_attempts=player.attempts,
                                           rollout_completed=player.completed,
                                           rollout_fallbacks=player.fallbacks,
                                           rollout_search_seconds=player.search_seconds,
                                           rollout_worlds_completed=player.worlds_completed,
                                           continuation_coverage={
                                               f"{kind}:{status}": count
                                               for (kind, status), count in player.coverage.items()
                                           })
                        _append(out / "rows.jsonl", attempt)
                        attempts += 1
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        _append(out / "rows.jsonl", {
            "phase": "failure", "case_id": case_id, "repetition": repetition,
            "arm": arm, "at_phase": phase, "error": error,
            "elapsed_seconds": monotonic() - started,
            "peak_process_rss_bytes": _rss_bytes(),
        })
    result_summary = {
        "status": status, "error": error,
        "planned_roots": len(plan["case_ids"]), "completed_or_failed_root_rows": root_rows,
        "planned_play_attempts": len(plan["case_ids"]) * plan["deals_per_root"] * 3,
        "retained_play_attempts": attempts,
        "elapsed_seconds": monotonic() - started,
        "peak_process_rss_bytes": _rss_bytes(),
        "swap_after": _swap(),
        "memory_pressure_after": _memory_pressure(),
    }
    write_json(out / "result.json", result_summary)
    write_json(out / "checksums.json", {
        str(path.relative_to(out)): _hash(path) for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    })
    return result_summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(json.loads(args.plan.read_text()), args.cases, args.checkpoint, args.out)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
