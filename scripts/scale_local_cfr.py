"""Frozen-range, sequential compute scaling for PR #107 conditional flop cases."""

import argparse
import gzip
import json
import os
import shutil
import traceback
from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random
from statistics import mean, median
from time import monotonic, time

from scripts.diagnose_local_cfr import _append, _hash, _rss_bytes, _view
from src.arena.artifacts import environment, git, write_json
from src.blueprint.artifact import load_training
from src.blueprint.local_cfr import (
    LocalCFRConfig, _LocalSolver, _flop_root, _public_history, _root_ranges,
)
from src.blueprint.search import LiveBlueprint


def _seed(namespace: str, case_id: str, purpose: str, repetition: int = 0) -> int:
    value = f"{namespace}:{case_id}:{purpose}:{repetition}".encode()
    return int.from_bytes(sha256(value).digest()[:8], "big")


def _cases(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())["cases"]
    if len(rows) != 48 or len({row["id"] for row in rows}) != 48:
        raise ValueError("Scaling requires the 48 distinct frozen cases")
    return rows


def _observation(case: dict):
    view = _view(case)
    if view.seat != case["hero_seat"] or sha256(repr(view).encode()).hexdigest() != case["observation_sha256"]:
        raise ValueError(f"Frozen observation changed for {case['id']}")
    return view


def _load_blueprint(checkpoint: Path, plan: dict):
    if _hash(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    blueprint = LiveBlueprint(load_training(checkpoint))
    if _rss_bytes() >= plan["execution"]["max_rss_gib"] * 1024**3:
        raise MemoryError("Checkpoint load exceeded the RSS cap")
    return blueprint


def prepare_ranges(plan: dict, cases_path: Path, checkpoint: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    cases = _cases(cases_path)
    blueprint = _load_blueprint(checkpoint, plan)
    frozen = []
    for case in cases:
        view = _observation(case)
        seed = _seed(plan["seed_namespace"], case["id"], "public-range")
        coverage = Counter()
        ranges = _root_ranges(
            blueprint, view, _flop_root(view)[0], Random(seed),
            plan["local_cfr"]["range_samples"], float("inf"), coverage,
        )
        frozen.append({
            "case_id": case["id"], "observation_sha256": case["observation_sha256"],
            "range_seed": seed,
            "ranges": {str(seat): [[list(pair), mass] for pair, mass in rows]
                       for seat, rows in ranges.items()},
        })
        if _rss_bytes() >= plan["execution"]["max_rss_gib"] * 1024**3:
            raise MemoryError("Range preparation exceeded the RSS cap")
    artifact = {
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "source_cases_sha256": _hash(cases_path),
        "range_samples": plan["local_cfr"]["range_samples"],
        "seed_namespace": plan["seed_namespace"], "cases": frozen,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            zipped.write(json.dumps(artifact, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode())
    return {"range_file_sha256": _hash(out), "cases": len(frozen),
            "peak_process_rss_bytes": _rss_bytes()}


def _read_ranges(path: Path, plan: dict, cases_path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        artifact = json.load(source)
    if (artifact["checkpoint_sha256"] != plan["checkpoint_sha256"]
            or artifact["source_cases_sha256"] != _hash(cases_path)
            or artifact["range_samples"] != plan["local_cfr"]["range_samples"]
            or artifact["seed_namespace"] != plan["seed_namespace"]):
        raise ValueError("Frozen public ranges do not match the experiment")
    return {row["case_id"]: {
        int(seat): tuple((tuple(pair), mass) for pair, mass in entries)
        for seat, entries in row["ranges"].items()
    } for row in artifact["cases"]}


def _work(solver: _LocalSolver, view) -> dict:
    target_key = solver._action_key(view, solver._menu(view))
    target = solver.nodes.get(target_key)
    public = _public_history(view)
    seen = {key[2] for key in solver.nodes if key[0] == "action"
            and key[1] == view.seat and key[3] == view.board and key[4] == public}
    return {
        "target_visits": target.visits if target else 0,
        "target_public_range_holdings_visited": len(seen),
        "target_public_range_prior_mass_visited": sum(
            mass for pair, mass in solver.root_ranges[view.seat]
            if tuple(sorted(pair)) in seen),
        "hero_action_infosets": sum(key[0] == "action" and key[1] == view.seat
                                    for key in solver.nodes),
    }


def _attempt(blueprint, view, ranges, case: dict, repetition: int,
             targeted: bool, plan: dict, max_seconds: float, max_cycles: int) -> dict:
    config = LocalCFRConfig(**{**plan["local_cfr"], "max_seconds": max_seconds,
                               "max_cycles": max_cycles}, targeted_traversal=targeted)
    seed = _seed(plan["seed_namespace"], case["id"], "traversal", repetition)
    coverage = Counter()
    started = monotonic()
    solver = None
    policy = None
    status = "completed"
    error = None
    try:
        solver = _LocalSolver(
            blueprint, view, Random(seed), config, started + max_seconds, coverage,
            root_ranges=ranges, snapshot_seconds=tuple(plan["snapshot_seconds"]),
            rss_limit_bytes=int(plan["execution"]["max_rss_gib"] * 1024**3),
        )
        _, policy = solver.solve()
    except Exception as exc:
        status = ("rss_limit" if isinstance(exc, MemoryError)
                  else "timeout" if isinstance(exc, TimeoutError)
                  else "target_unvisited" if "hero information set was not visited" in str(exc)
                  else "error")
        error = f"{type(exc).__name__}: {exc}"
        if status == "error":
            error += "\n" + traceback.format_exc()
    elapsed = monotonic() - started
    row = {
        "case_id": case["id"], "repetition": repetition, "targeted": targeted,
        "traversal_seed": seed, "status": status, "error": error,
        "seconds": elapsed, "cycles": solver.cycles if solver else 0,
        "stop_reason": solver.stop_reason if solver else None,
        "sampled_nodes": solver.sampled_nodes if solver else 0,
        "leaf_choices": solver.leaf_choices if solver else 0,
        "infosets": len(solver.nodes) if solver else 0,
        "policy": list(policy) if policy is not None else None,
        "diagnostics": solver.diagnostics if solver else [],
        "time_snapshots": solver.time_snapshots if solver else [],
        "continuation_trained_lookups": coverage[("continuation", "trained")],
        "continuation_untrained_lookups": coverage[("continuation", "untrained")],
        "continuation_off_tree_lookups": coverage[("continuation", "off_tree")],
        "peak_process_rss_bytes": _rss_bytes(),
        **(_work(solver, view) if solver else {
            "target_visits": 0, "target_public_range_holdings_visited": 0,
            "target_public_range_prior_mass_visited": 0.0, "hero_action_infosets": 0,
        }),
    }
    if solver is not None:
        # A deadline normally interrupts the cycle before the exact 60-second
        # threshold. Preserve its last fully published state separately.
        row["final_completed_cycle_snapshot"] = {
            "elapsed_seconds": elapsed, "cycles": solver.cycles,
            "sampled_nodes": solver.sampled_nodes,
            "target_visits": row["target_visits"],
            "target_holdings_visited": row["target_public_range_holdings_visited"],
            "target_prior_mass_visited": row["target_public_range_prior_mass_visited"],
            "target_policy": list(solver._policy(
                solver._action_key(view, solver._menu(view)),
                tuple(item.name for item in solver._menu(view)),
            )) if row["target_visits"] else None,
            "continuation_trained_lookups": row["continuation_trained_lookups"],
            "continuation_untrained_lookups": row["continuation_untrained_lookups"],
        }
    else:
        row["final_completed_cycle_snapshot"] = None
    return row


def _summary(rows: list[dict]) -> dict:
    modes = {}
    for targeted in (False, True):
        selected = [row for row in rows if row["targeted"] == targeted]
        trained = sum(row["continuation_trained_lookups"] for row in selected)
        untrained = sum(row["continuation_untrained_lookups"] for row in selected)
        modes["targeting_on" if targeted else "targeting_off"] = {
            "attempts": len(selected),
            "statuses": dict(Counter(row["status"] for row in selected)),
            "median_cycles": median([row["cycles"] for row in selected]) if selected else None,
            "max_cycles": max([row["cycles"] for row in selected], default=None),
            "median_target_visits": median([row["target_visits"] for row in selected]) if selected else None,
            "median_target_prior_mass_visited": median([
                row["target_public_range_prior_mass_visited"] for row in selected
            ]) if selected else None,
            "trained_lookup_fraction": trained / (trained + untrained) if trained + untrained else None,
        }
    groups = {}
    for row in rows:
        if row["policy"] is not None:
            groups.setdefault((row["case_id"], row["targeted"]), []).append(row["policy"])
    variability = []
    for policies in groups.values():
        if len(policies) < 2:
            continue
        for left in range(len(policies)):
            for right in range(left + 1, len(policies)):
                variability.append(sum(abs(a - b) for a, b in zip(
                    policies[left], policies[right], strict=True)))
    return {"modes": modes, "mean_within_case_policy_l1": mean(variability)
            if variability else None, "policy_pairs": len(variability)}


def run(plan: dict, cases_path: Path, ranges_path: Path, checkpoint: Path,
        out: Path, preflight: bool = False) -> dict:
    if out.exists():
        raise FileExistsError(out)
    cases = _cases(cases_path)
    ranges = _read_ranges(ranges_path, plan, cases_path)
    if set(ranges) != {case["id"] for case in cases}:
        raise ValueError("Frozen ranges do not cover every case")
    limits = plan["execution"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < limits["min_free_gib"] * 1024**3:
        raise RuntimeError("M4 free-disk guard failed")
    started = monotonic()
    blueprint = _load_blueprint(checkpoint, plan)  # exactly one load per process
    out.mkdir()
    write_json(out / "manifest.json", {
        "plan": plan, "preflight": preflight, "cases_sha256": _hash(cases_path),
        "ranges_sha256": _hash(ranges_path), "checkpoint_sha256": plan["checkpoint_sha256"],
        "revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
        "environment": environment(), "started_unix_seconds": time(),
    })
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(out / "tensorboard"))
    rows = []
    status = "valid"
    max_seconds = 20.0 if preflight else plan["local_cfr"]["max_seconds"]
    max_cycles = 2048 if preflight else plan["local_cfr"]["max_cycles"]
    selected = cases[:2] if preflight else cases
    repetitions = 1 if preflight else plan["repetitions"]
    try:
        for case in selected:
            view = _observation(case)
            for repetition in range(repetitions):
                modes = (True, False) if (len(rows) // 2) % 2 == 0 else (False, True)
                for targeted in modes:
                    if monotonic() - started + max_seconds + 5 >= limits["max_wall_seconds"]:
                        status = "wall_limit"
                        break
                    row = _attempt(blueprint, view, ranges[case["id"]], case, repetition,
                                   targeted, plan, max_seconds, max_cycles)
                    rows.append(row)
                    _append(out / "attempts.jsonl", row)
                    if len(rows) % limits["tensorboard_every_attempts"] == 0:
                        current = _summary(rows)
                        for mode, values in current["modes"].items():
                            for key in ("median_cycles", "max_cycles", "median_target_visits",
                                        "median_target_prior_mass_visited", "trained_lookup_fraction"):
                                if values[key] is not None:
                                    writer.add_scalar(f"{mode}/{key}", values[key], len(rows))
                        writer.add_scalar("peak_rss_gib", _rss_bytes() / 1024**3, len(rows))
                        writer.flush()
                    if row["status"] in ("rss_limit", "error"):
                        status = row["status"]
                        break
                    if _rss_bytes() >= limits["max_rss_gib"] * 1024**3:
                        status = "rss_limit"
                        break
                    if shutil.disk_usage(out).free < limits["min_free_gib"] * 1024**3:
                        status = "disk_limit"
                        break
                if status != "valid":
                    break
            if status != "valid":
                break
    finally:
        writer.close()
        result = {
            "status": status, "preflight": preflight,
            "planned_attempts": len(selected) * repetitions * 2,
            "retained_attempts": len(rows), "elapsed_seconds": monotonic() - started,
            "peak_process_rss_bytes": _rss_bytes(), **_summary(rows),
        }
        write_json(out / "result.json", result)
        write_json(out / "checksums.json", {str(path.relative_to(out)): _hash(path)
                                          for path in sorted(out.rglob("*")) if path.is_file()})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--ranges", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--prepare-ranges", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if args.prepare_ranges:
        print(json.dumps(prepare_ranges(plan, args.cases, args.checkpoint, args.ranges)))
        return 0
    if args.out is None:
        parser.error("--out is required to run")
    result = run(plan, args.cases, args.ranges, args.checkpoint, args.out, args.preflight)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
