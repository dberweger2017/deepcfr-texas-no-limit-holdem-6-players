"""Measure local CFR on frozen eligible flop observations, without arena play."""

import argparse
import gc
import json
import os
import resource
import shutil
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random
from statistics import mean, median
from time import monotonic

from src.arena.artifacts import environment, git, write_json
from src.blueprint.artifact import load_training
from src.blueprint.local_cfr import (
    LocalCFRConfig, _LocalSolver, _eligible, _public_history,
)
from src.blueprint.search import LiveBlueprint, SearchUnavailable
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan_hash(plan: dict) -> str:
    return sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())


def _view(case: dict):
    table = Table(tuple(f"player-{seat}" for seat in range(6)), (10_000,) * 6,
                  button=case["button"])
    hand = Hand.start(table, hand_id=f"conditional-{case['id']}", seed=case["deal_seed"])
    while hand.observe(hand.actor).street == Street.PREFLOP:
        view = hand.observe(hand.actor)
        live = sum(not player.folded for player in view.players)
        if live > 3 and ActionKind.FOLD in view.legal_actions.kinds:
            kind = ActionKind.FOLD
        else:
            kind = (ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                    else ActionKind.CALL)
        hand = hand.apply(Action(kind))
    for index in range(case["position"]):
        view = hand.observe(hand.actor)
        if case["prefix"] == "raise" and index == 0:
            action = Action(ActionKind.RAISE, view.legal_actions.min_raise_to)
        else:
            action = Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                            else ActionKind.CALL)
        hand = hand.apply(action)
    view = hand.observe(hand.actor)
    if not _eligible(view):
        raise ValueError(f"Frozen situation {case['id']} is not eligible")
    return view


def prepare(plan: dict, cases_path: Path) -> None:
    if cases_path.exists():
        raise FileExistsError(cases_path)
    cases = []
    for deal_index in range(plan["deals"]):
        for pattern_index, pattern in enumerate(plan["patterns"]):
            index = len(cases)
            case = {
                "id": f"{deal_index:02d}-{pattern_index}",
                "deal_seed": plan["deal_seed_start"] + deal_index,
                "button": deal_index % 6,
                "position": pattern["position"],
                "prefix": pattern["prefix"],
                "search_seed": plan["search_seed_start"] + index,
            }
            view = _view(case)
            case["hero_seat"] = view.seat
            case["observation_sha256"] = sha256(repr(view).encode()).hexdigest()
            cases.append(case)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cases_path, {"plan_sha256": _plan_hash(plan), "cases": cases})


def _measure(blueprint, view, case: dict, targeted: bool, plan: dict) -> dict:
    config = LocalCFRConfig(**plan["local_cfr"], targeted_traversal=targeted)
    coverage = Counter()
    started = monotonic()
    solver = None
    status = "completed"
    policy = None
    try:
        solver = _LocalSolver(blueprint, view, Random(case["search_seed"]), config,
                              started + config.max_seconds, coverage)
        _, policy = solver.solve()
    except TimeoutError:
        status = "timeout"
    except SearchUnavailable as exc:
        status = ("target_unvisited" if "hero information set was not visited" in str(exc)
                  else "search_unavailable")
    elapsed = monotonic() - started
    target_visits = 0
    prior_mass = 0.0
    holdings = 0
    hero_infosets = 0
    diagnostics = []
    nodes = leaf_choices = sampled_nodes = cycles = 0
    if solver is not None:
        target_menu = solver._menu(view)
        target_key = solver._action_key(view, target_menu)
        target = solver.nodes.get(target_key)
        target_visits = target.visits if target is not None else 0
        public = _public_history(view)
        seen = {
            key[2] for key in solver.nodes
            if key[0] == "action" and key[1] == view.seat
            and key[3] == view.board and key[4] == public
        }
        holdings = len(seen)
        prior_mass = sum(mass for pair, mass in solver.root_ranges[view.seat]
                         if tuple(sorted(pair)) in seen)
        hero_infosets = sum(key[0] == "action" and key[1] == view.seat
                            for key in solver.nodes)
        diagnostics = solver.diagnostics
        nodes = len(solver.nodes)
        leaf_choices = solver.leaf_choices
        sampled_nodes = solver.sampled_nodes
        cycles = solver.cycles
    result = {
        "case_id": case["id"], "targeted": targeted, "status": status,
        "seconds": elapsed, "cycles": cycles,
        "target_visits": target_visits,
        "target_public_range_holdings_visited": holdings,
        "target_public_range_prior_mass_visited": prior_mass,
        "hero_action_infosets": hero_infosets,
        "infosets": nodes, "sampled_nodes": sampled_nodes,
        "leaf_choices": leaf_choices,
        "continuation_trained_lookups": coverage[("continuation", "trained")],
        "continuation_untrained_lookups": coverage[("continuation", "untrained")],
        "continuation_off_tree_lookups": coverage[("continuation", "off_tree")],
        "policy": list(policy) if policy is not None else None,
        "diagnostics": diagnostics,
        "peak_process_rss_bytes": _rss_bytes(),
    }
    del solver
    gc.collect()
    return result


def _summary(rows: list[dict]) -> dict:
    result = {}
    for targeted in (False, True):
        selected = [row for row in rows if row["targeted"] == targeted]
        completed = [row for row in selected if row["status"] == "completed"]
        latencies = sorted(row["seconds"] for row in selected)
        trained = sum(row["continuation_trained_lookups"] for row in selected)
        untrained = sum(row["continuation_untrained_lookups"] for row in selected)
        movements = [snapshot["target_l1_from_previous"]
                     for row in completed for snapshot in row["diagnostics"]
                     if snapshot["cycle"] == 128
                     and snapshot["target_l1_from_previous"] is not None]
        result["targeted_on" if targeted else "targeted_off"] = {
            "attempts": len(selected), "completed": len(completed),
            "target_unvisited": sum(row["status"] == "target_unvisited" for row in selected),
            "search_unavailable": sum(row["status"] == "search_unavailable"
                                      for row in selected),
            "timeouts": sum(row["status"] == "timeout" for row in selected),
            "minimum_completed_cycles": min((row["cycles"] for row in completed), default=None),
            "maximum_seconds": max(latencies, default=None),
            "p95_seconds": latencies[int(.95 * (len(latencies) - 1))] if latencies else None,
            "median_target_visits": median(row["target_visits"] for row in selected)
            if selected else None,
            "median_target_holdings_visited": median([
                row["target_public_range_holdings_visited"] for row in selected
            ]) if selected else None,
            "median_prior_mass_visited": median([
                row["target_public_range_prior_mass_visited"] for row in selected
            ]) if selected else None,
            "mean_target_policy_l1_64_to_128": mean(movements) if movements else None,
            "continuation_trained_lookups": trained,
            "continuation_untrained_lookups": untrained,
            "continuation_trained_fraction": trained / (trained + untrained)
            if trained + untrained else None,
        }
    return result


def run(plan: dict, cases_path: Path, checkpoint: Path, out: Path) -> dict:
    frozen = json.loads(cases_path.read_text())
    if frozen["plan_sha256"] != _plan_hash(plan):
        raise ValueError("Frozen cases do not match the diagnostic plan")
    if _hash(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    if out.exists():
        raise FileExistsError(out)
    limits = plan["execution"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < limits["min_free_gib"] * 1024**3:
        raise RuntimeError("M4 free-disk guard failed")
    started = monotonic()
    blueprint = LiveBlueprint(load_training(checkpoint))
    if _rss_bytes() >= limits["max_rss_gib"] * 1024**3:
        raise RuntimeError("Checkpoint load exceeded the RSS cap")
    if monotonic() - started >= limits["max_wall_seconds"]:
        raise RuntimeError("Checkpoint load exceeded the wall cap")
    out.mkdir(parents=True)
    write_json(out / "manifest.json", {
        "plan": plan, "plan_sha256": _plan_hash(plan),
        "cases_sha256": _hash(cases_path),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "environment": environment(),
    })
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(out / "tensorboard"))
    rows = []
    status = "valid"
    try:
        for index, case in enumerate(frozen["cases"]):
            view = _view(case)
            if view.seat != case["hero_seat"] or sha256(repr(view).encode()).hexdigest() \
                    != case["observation_sha256"]:
                raise ValueError(f"Frozen observation changed for {case['id']}")
            for targeted in ((True, False) if index % 2 == 0 else (False, True)):
                if monotonic() - started + plan["local_cfr"]["max_seconds"] \
                        >= limits["max_wall_seconds"]:
                    status = "wall_limit"
                    break
                row = _measure(blueprint, view, case, targeted, plan)
                rows.append(row)
                _append(out / "attempts.jsonl", row)
                if _rss_bytes() >= limits["max_rss_gib"] * 1024**3:
                    status = "rss_limit"
                    break
                if shutil.disk_usage(out).free < limits["min_free_gib"] * 1024**3:
                    status = "disk_limit"
                    break
            if status != "valid":
                break
            if (index + 1) % limits["tensorboard_every_cases"] == 0 \
                    or index + 1 == len(frozen["cases"]):
                summary = _summary(rows)
                for mode, values in summary.items():
                    for key in ("completed", "target_unvisited", "search_unavailable", "timeouts",
                                "p95_seconds", "median_target_visits",
                                "median_target_holdings_visited", "median_prior_mass_visited",
                                "mean_target_policy_l1_64_to_128",
                                "continuation_trained_fraction"):
                        if values[key] is not None:
                            writer.add_scalar(f"{mode}/{key}", values[key], index + 1)
                writer.add_scalar("peak_rss_gib", _rss_bytes() / 1024**3, index + 1)
                writer.flush()
    finally:
        writer.close()
    pairs = {}
    for row in rows:
        pairs.setdefault(row["case_id"], {})[row["targeted"]] = row
    comparable = [pair for pair in pairs.values() if len(pair) == 2
                  and pair[True]["policy"] is not None
                  and pair[False]["policy"] is not None]
    result = {
        "status": status, "case_count": len(frozen["cases"]),
        "completed_case_pairs": len(rows) // 2,
        "elapsed_seconds": monotonic() - started,
        "peak_process_rss_bytes": _rss_bytes(),
        "modes": _summary(rows),
        "both_modes_completed": len(comparable),
        "mean_final_policy_l1_on_off": mean(
            sum(abs(a - b) for a, b in zip(pair[True]["policy"],
                                            pair[False]["policy"], strict=True))
            for pair in comparable
        ) if comparable else None,
    }
    write_json(out / "result.json", result)
    write_json(out / "checksums.json", {
        str(path.relative_to(out)): _hash(path)
        for path in sorted(out.rglob("*")) if path.is_file()
    })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if args.prepare:
        if args.checkpoint or args.out:
            parser.error("--prepare only needs --plan and --cases")
        prepare(plan, args.cases)
        return 0
    if not args.checkpoint or not args.out:
        parser.error("Running needs --checkpoint and --out")
    result = run(plan, args.cases, args.checkpoint, args.out)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
