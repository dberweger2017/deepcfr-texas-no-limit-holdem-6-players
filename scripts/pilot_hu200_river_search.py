"""Freeze outcome-blind real river spots, then measure complete HU200 decisions."""

import argparse
from dataclasses import asdict
import fcntl
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import subprocess
from time import monotonic
from zipfile import ZipFile

import numpy as np
import psutil

from scripts.evaluate_hu100_direct import put
from src.arena.external.slumbot import SlumbotAdapter
from src.arena.schedule import digest
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy
from src.blueprint.hu200_river_search import HU200RiverSearchPlayer, HU200RiverSearchPolicy
from src.blueprint.hu20_turn_search import TurnSearchConfig
from src.blueprint.hu20_turn_solver import ExternalTurnSolver
from src.blueprint.river_ledger import FamilyPeak
from src.blueprint.river_evidence import COMPACT_RETENTION_RULE
from src.blueprint.search import DECK
from src.game.observation import ActionTaken, replay
from src.game.types import Street
from src.policies.files import file_hash


MODEL_SHA256 = "ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485"
PR304_SHA256 = "51f4587dc3fec12a65537be51e27572c3f1eec147952c8ed383f1c7689c57137"
CONFIG = TurnSearchConfig(iterations=50, threads=6, compress=False,
    opponent_likelihood_floor=0, decision_seconds=120,
    memory_budget_bytes=2 * 1024**3, cache_entries=2)


def candidates(lines, source):
    previous = "0" * 64
    result = []
    for sequence, line in enumerate(lines):
        row = json.loads(line)
        checksum = row.pop("sha256")
        if row["sequence"] != sequence or row["previous"] != previous or digest(row) != checksum:
            raise ValueError("Real-spot source journal hash chain differs")
        previous = checksum
        decision = row["payload"].get("decision")
        if decision is None or len(decision["public"]["board"]) != 5:
            continue
        # No terminal outcome, revealed opponent holding or recorded action enters selection.
        view = SlumbotAdapter().observation(decision["public"], decision["hand_id"])
        payload = dict(hand_id=decision["hand_id"], public=decision["public"])
        result.append(dict(source=source, sequence=sequence, decision=payload,
            key=digest(payload), facing=bool(view.legal_actions.call_amount)))
    return result, previous


def freeze(archive, blend, out):
    if file_hash(archive) != PR304_SHA256:
        raise ValueError("PR304 source archive SHA256 differs")
    groups = {"pr304": [], "blend": []}
    sources = {}
    with ZipFile(archive) as zipped:
        for name in sorted(zipped.namelist()):
            if name.endswith("journal.jsonl"):
                body = zipped.read(name)
                rows, tail = candidates(body.splitlines(), "pr304/" + name)
                groups["pr304"].extend(rows)
                sources[name] = dict(bytes=len(body), sha256=sha256(body).hexdigest(), tail_sha256=tail)
    rows, tail = candidates(blend.read_text().splitlines(), "blend/verified-prefix")
    groups["blend"] = rows
    sources["blend/verified-prefix"] = dict(bytes=blend.stat().st_size,
        sha256=file_hash(blend), tail_sha256=tail)
    selected = []
    for label, rows in groups.items():
        used_hands = set()
        for facing in (False, True):
            cell = []
            for row in sorted(rows, key=lambda r: (r["key"], r["source"], r["sequence"])):
                hand_key = row["source"], row["decision"]["hand_id"]
                if row["facing"] == facing and hand_key not in used_hands:
                    used_hands.add(hand_key)
                    cell.append(row)
                    if len(cell) == 6:
                        break
            if len(cell) != 6:
                raise ValueError("Insufficient real river decisions in fixed source/price stratum")
            selected.extend(cell)
    put(out, dict(format="hu200-river-cost-sample-v1", config=asdict(CONFIG),
        model_sha256=MODEL_SHA256, sources=sources, sample=selected,
        selection="six lowest public-decision SHA256s per source/free-or-facing stratum; distinct hands",
        outcomes_used=False, strength_hands=0))


def pin_earlier_blueprint(policy, view):
    """Real journals used blueprint policies before this sampled decision."""
    policy._start_live_hand(view.hand_id)
    for i, event in enumerate(view.history):
        if isinstance(event, ActionTaken) and event.street == Street.RIVER and event.seat == view.seat:
            prior = view.history[:i]
            public = replay(prior, 0, ())
            pairs = tuple(tuple(sorted(h)) for h in combinations((c for c in DECK if c not in public.board), 2))
            matrix = policy._base_matrix(prior, pairs)
            policy._remember_play(replay(prior, view.seat, view.hole_cards), matrix, True,
                                  cause="earlier-blueprint-river-action")


def measure_spot(index, spot, base, binary, out, *, compact_profiles=False, cancel_check=None, profile_retention=None):
    view = SlumbotAdapter().observation(spot["decision"]["public"], spot["decision"]["hand_id"])
    solver = ExternalTurnSolver(binary, out / str(index) / "native",
        compact_profiles=compact_profiles, profile_retention=profile_retention, **({"resource_check": cancel_check} if cancel_check else {}))
    policy = HU200RiverSearchPolicy(base, solver, CONFIG, ledger=out / str(index) / "ledger")
    if cancel_check is not None:
        original_check = policy._check

        def check(deadline):
            cancel_check()
            original_check(deadline)

        policy._check = check
    player = HU200RiverSearchPlayer(policy, seed=20261016317000 + index)
    t0 = monotonic()
    with FamilyPeak() as memory:
        pin_earlier_blueprint(policy, view)
        player.choose_action(view)
    row = dict(player.last_decision, sample_index=index, source=spot["source"],
               pre_output_seconds=monotonic() - t0,
               sampled_peak_family_rss_bytes=memory.peak,
               minimum_host_available_bytes=memory.minimum_available_bytes,
               stats=dict(policy.stats), native_records=solver.records)
    # Encode action dataclasses while retaining exact integer targets.
    row["blueprint_action"] = asdict(row["blueprint_action"])
    row["search_action"] = asdict(row["search_action"])
    put(out / str(index) / "decision.json", row)
    row["complete_seconds"] = monotonic() - t0
    put(out / str(index) / "timing.json", dict(complete_seconds=row["complete_seconds"],
        scope="through decision serialization and write; this timing receipt excluded"))
    return row


def run(plan_path, model, cache, tables, binary, out, *, compact_profiles=False,
        latency_target_seconds=3, profile_retention=None):
    plan = json.loads(plan_path.read_text())
    declaration = json.loads((Path(__file__).resolve().parents[1] /
        "docs/reports/hu200-river-search-artifacts/cost-declaration.json").read_text())
    if file_hash(plan_path) != declaration["full_plan_sha256"]:
        raise ValueError("Real-spot sample differs from its committed declaration")
    if (plan["config"] != asdict(CONFIG) or plan["model_sha256"] != MODEL_SHA256
            or len(plan["sample"]) != 24):
        raise ValueError("Cost sample or search work differs from declaration")
    out.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    base = AveragePolicy(model, MODEL_SHA256, expected_schema=HU200_EQUITY_SCHEMA,
        card_buckets=tables,
        cache=cache, cache_read_only=True, translation=TranslationOptions(512, 128))
    if len(base.entries) != 156208331:
        raise ValueError("Verified play cache has the wrong entry count")
    put(out / "inputs.json", dict(plan_sha256=file_hash(plan_path),
        model_bytes=model.stat().st_size, model_sha256=MODEL_SHA256,
        binary_sha256=file_hash(binary), blueprint=base.description,
        load_and_validation_seconds=monotonic() - started, profile_retention=profile_retention,
        source_tree=subprocess.check_output(["git","rev-parse","HEAD:src"],text=True).strip()))
    rows = []
    for index, spot in enumerate(plan["sample"]):
        row = measure_spot(index, spot, base, binary, out, compact_profiles=compact_profiles, profile_retention=profile_retention)
        rows.append(row)
        print(json.dumps(dict(index=index, seconds=row["complete_seconds"], fallback=row["fallback"])), flush=True)
    times = [r["complete_seconds"] for r in rows]
    peak = max(r["sampled_peak_family_rss_bytes"] for r in rows)
    available = min(r["minimum_host_available_bytes"] for r in rows)
    missing_ev = sum(not r["fallback"] and r["model_ev_unavailable"] is not None for r in rows)
    summary = dict(status="complete", decisions=len(rows),
        seconds=dict(p50=float(np.percentile(times, 50)), p95=float(np.percentile(times, 95)), max=max(times)),
        peak_family_rss_bytes=peak, physical_memory_bytes=16 * 1024**3,
        physical_minus_family_bytes=16 * 1024**3 - peak,
        minimum_host_available_bytes=available,
        host_memory_definition="psutil.virtual_memory().available; available includes reclaimable memory",
        successful_decisions_missing_ev=missing_ev,
        fallback_decisions=sum(r["fallback"] for r in rows),
        latency_target_seconds=latency_target_seconds,
        latency_target_met=float(np.percentile(times, 95)) <= latency_target_seconds,
        memory_target_met=16 * 1024**3 - peak >= 3_000_000_000 and available >= 3_000_000_000,
        telemetry_target_met=missing_ev == 0,
        all_search_work_completed=not any(r["fallback"] for r in rows),
        no_strength_hands=True, no_automatic_budget_weakening=True)
    put(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--sample-native-bodies", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--blend", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--compact-profiles", action="store_true")
    parser.add_argument("--latency-target-seconds", type=int, choices=(3, 10), default=3)
    parser.add_argument("--parity", type=Path, help="Required exact-parity receipt for amended gate")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip() != "Apple M4":
        raise ValueError("All scientific work is restricted to the authorized M4")
    if not any("scripts.guard_m4_research" in parent.cmdline() for parent in psutil.Process().parents()):
        raise ValueError("Run the cost pilot inside scripts.guard_m4_research")
    with (Path.home() / "Local/.hu20-m4-research.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "freeze":
            freeze(args.archive, args.blend, args.out)
        else:
            if args.latency_target_seconds == 10:
                declaration = json.loads((Path(__file__).resolve().parents[1] /
                    "docs/reports/hu200-river-search-artifacts/optimization-declaration.json").read_text())
                parity = json.loads(args.parity.read_text()) if args.parity else {}
                if (not args.compact_profiles or parity.get("status") != "exact-parity"
                        or parity.get("decisions") != 24
                        or parity.get("plan_sha256") != declaration["full_plan_sha256"]
                        or file_hash(args.binary) != declaration["external_binary_sha256"]):
                    raise ValueError("Amended cost gate lacks matching exact-parity admission")
                if args.sample_native_bodies and parity.get("profile_retention") != COMPACT_RETENTION_RULE:
                    raise ValueError("Retention cost lacks matching retention parity")
                inputs = json.loads((args.parity.parent / "inputs.json").read_text())
                if args.sample_native_bodies and inputs.get("source_tree") != subprocess.check_output(
                        ["git","rev-parse","HEAD:src"],text=True).strip():
                    raise ValueError("Retention parity and cost source differ")
                if inputs["binary_sha256"] != file_hash(args.binary):
                    raise ValueError("Parity and cost executable identities differ")
            run(args.plan, args.model, args.cache, args.tables, args.binary, args.out,
                compact_profiles=args.compact_profiles, latency_target_seconds=args.latency_target_seconds,
                profile_retention=COMPACT_RETENTION_RULE if args.sample_native_bodies else None)


if __name__ == "__main__":
    main()
