"""Outcome-blind #116 decision selection and information-safe B100M audit."""

import argparse
import gzip
import json
import shutil
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from time import time

from scripts.evaluate_hu20 import rss, system, write_json
from scripts.evaluate_hu20_reopening import Target
from src.arena.schedule import digest
from src.blueprint.abstraction import _postflop, _preflop, information_key
from src.diagnostics.conditional_values import summarize, visible_fingerprint, world_action_returns
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind

SEEDS = (2026093001, 2026093002, 2026093003)
STREETS = ("preflop", "flop", "turn", "river")
ROOT = 202610010117
MODEL_MILESTONE = 100000000


def _rank(seed, block, rotation, action_index):
    token = f"hu20-b100-audit-v1|{seed}|{block}|{rotation}|{action_index}"
    return sha256(token.encode()).hexdigest()


def _start(row, *, synthetic=False):
    rotation = row["rotation"]
    ids = tuple(f"player-{(seat-rotation)%2}" for seat in range(2))
    return Hand.start(
        Table(ids, (2000, 2000), button=row["block"] % 2 if synthetic else row["button"]),
        hand_id=(f"selection-{row['block']}" if synthetic else
                 f"robustness-{row['phase']}-2-{row['block']}"),
        seed=0 if synthetic else row["deal_seed"],
    )


def _trace(row, *, synthetic=False):
    """Yield player-visible prefixes; no terminal payoff is consulted."""
    hand = _start(row, synthetic=synthetic)
    for item in row["actions"]:
        view = hand.observe(hand.actor)
        if (view.seat != item["seat"] or view.street.value != item["street"]):
            raise ValueError("Raw trace differs from native public replay")
        yield item, view
        hand = hand.apply(Action(ActionKind(item["kind"]), item["raise_to"]))
    if not synthetic and not hand.finished and row["status"] == "complete":
        raise ValueError("Completed raw hand has an incomplete action trace")


def select(raw_dir, output):
    candidates = defaultdict(list)
    fallback = defaultdict(list)
    source_hashes = {}
    counts = Counter()
    for seed in SEEDS:
        path = raw_dir / f"B-{seed}-{MODEL_MILESTONE}--LBR-original-cap2.jsonl.gz"
        source_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, "rt") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if row["policy"] != f"B-{seed}-{MODEL_MILESTONE}":
                    raise ValueError("Unexpected policy in retained LBR archive")
                for item, view in _trace(row, synthetic=True):
                    if item["logical_player"] != 0:
                        continue
                    context = "free" if view.legal_actions.call_amount == 0 else "facing"
                    coordinate = {
                        "seed": seed, "block": row["block"],
                        "rotation": row["rotation"], "action_index": item["index"],
                        "street": item["street"], "seat": item["seat"],
                        "context": context, "trained": item["target_trained"],
                        "file": path.name, "line": line_number,
                    }
                    if item["target_trained"] not in (True, False):
                        raise ValueError("Missing recorded target lookup status")
                    coordinate["rank"] = _rank(seed, row["block"], row["rotation"], item["index"])
                    cell = (seed, item["street"], item["seat"], context)
                    candidates[cell].append(coordinate)
                    counts[cell] += 1
                    if not item["target_trained"]:
                        fallback[seed].append(coordinate)
    selected = []
    cell_report = []
    for seed in SEEDS:
        for street in STREETS:
            for seat in (0, 1):
                picks = []
                for context in ("free", "facing"):
                    cell = (seed, street, seat, context)
                    ordered = sorted(candidates[cell], key=lambda c: (c["rank"], c["block"], c["rotation"], c["action_index"]))
                    cell_report.append({"seed": seed, "street": street, "seat": seat,
                                        "context": context, "eligible": len(ordered)})
                    if ordered:
                        picks.append(ordered[0])
                if len(picks) < 2:
                    all_here = candidates[(seed, street, seat, "free")] + candidates[(seed, street, seat, "facing")]
                    used = {(p["block"], p["rotation"], p["action_index"]) for p in picks}
                    for candidate in sorted(all_here, key=lambda c: (c["rank"], c["block"], c["rotation"], c["action_index"])):
                        key = (candidate["block"], candidate["rotation"], candidate["action_index"])
                        if key not in used:
                            picks.append(candidate)
                            break
                selected.extend(picks)
    used = {(c["seed"], c["block"], c["rotation"], c["action_index"]) for c in selected}
    for seed in SEEDS:
        count = 0
        for candidate in sorted(fallback[seed], key=lambda c: (c["rank"], c["block"], c["rotation"], c["action_index"])):
            key = (seed, candidate["block"], candidate["rotation"], candidate["action_index"])
            if key not in used:
                selected.append(candidate)
                used.add(key)
                count += 1
                if count == 4:
                    break
    selected.sort(key=lambda c: (c["seed"], c["street"], c["seat"], c["rank"]))
    report = {"schema": "hu20-b100-outcome-blind-selection-v1", "protocol": "docs/hu20-b100-diagnosis-protocol.md",
              "source_hashes": source_hashes, "cells": cell_report,
              "base_cap": 48, "fallback_extra_cap": 12,
              "selected": selected, "selection_digest": digest(selected),
              "fallback_selected": sum(not c["trained"] for c in selected)}
    write_json(output, report)
    return {"selected": len(selected), "fallback": report["fallback_selected"],
            "selection_digest": report["selection_digest"]}


def _selected_views(raw_dir, selection):
    wanted = defaultdict(dict)
    for entry in selection["selected"]:
        wanted[entry["file"]][(entry["line"], entry["action_index"])] = entry
    for filename, targets in sorted(wanted.items()):
        path = raw_dir / filename
        if sha256(path.read_bytes()).hexdigest() != selection["source_hashes"][filename]:
            raise ValueError("Retained LBR raw-hand hash changed")
        with gzip.open(path, "rt") as handle:
            for line_number, line in enumerate(handle, 1):
                relevant = {index: entry for (line_no, index), entry in targets.items() if line_no == line_number}
                if not relevant:
                    continue
                row = json.loads(line)
                for item, view in _trace(row):
                    if item["index"] in relevant:
                        yield relevant[item["index"]], item, view


def _node_data(path, keys):
    found = {}
    with gzip.open(path, "rt") as handle:
        json.loads(handle.readline())
        for line in handle:
            key, names, regrets, average, visits = json.loads(line)
            if key in keys:
                found[key] = {"names": names, "regrets": regrets,
                              "average": average, "visits": visits}
    return found


def _guard(output, deadline):
    if time() >= deadline:
        raise TimeoutError("Absolute ten-hour research deadline")
    if rss() > 10.5 * 1024**3:
        raise MemoryError("10.5-GiB process RSS guard")
    if shutil.disk_usage(output).free < 8 * 1024**3:
        raise OSError("8-GiB free-disk guard")


def values(raw_dir, selection_path, models_path, output, deadline):
    output.mkdir(parents=True, exist_ok=False)
    peak_rss = rss()
    selection = json.loads(selection_path.read_text())
    if digest(selection["selected"]) != selection["selection_digest"]:
        raise ValueError("Selection digest mismatch")
    specs = {s["seed"]: s for s in json.loads(models_path.read_text())
             if s.get("arm") == "B" and s.get("milestone") == MODEL_MILESTONE}
    if set(specs) != set(SEEDS):
        raise ValueError("Missing B100M model lineage")
    views = list(_selected_views(raw_dir, selection))
    if len(views) != len(selection["selected"]):
        raise ValueError("Selected decision could not be replayed")
    attempts = []
    world_path = output / "worlds.jsonl.gz"
    with gzip.open(world_path, "wt") as worlds_file:
        for seed in SEEDS:
            _guard(output, deadline)
            source = Target(specs[seed])
            peak_rss = max(peak_rss, rss())
            cases = [(entry, item, view) for entry, item, view in views if entry["seed"] == seed]
            keys = {information_key(view, source.distribution(view)[0], schema=specs[seed]["abstraction"])
                    for _, _, view in cases}
            nodes = _node_data(Path(specs[seed]["checkpoint_path"]), keys)
            for entry, item, view in cases:
                _guard(output, deadline)
                menu, probabilities, trained = source.distribution(view)
                key = information_key(view, menu, schema=specs[seed]["abstraction"])
                if trained != entry["trained"] or key != item.get("target_key"):
                    raise ValueError("Selected lookup differs from retained trace")
                abstraction = _preflop(view.hole_cards) if not view.board else _postflop(view.hole_cards, view.board)
                record = {"selection": entry, "model_sha256": specs[seed]["sha256"],
                          "checkpoint_sha256": specs[seed]["checkpoint_sha256"],
                          "public_history": [repr(e) for e in view.history],
                          "hole_cards": list(view.hole_cards), "board": list(view.board),
                          "visible_fingerprint": visible_fingerprint(view),
                          "abstraction": abstraction, "key": key,
                          "menu": [{"name": c.name, "kind": c.action.kind.value,
                                    "raise_to": c.action.raise_to} for c in menu],
                          "probabilities": probabilities, "selected_action": {"kind": item["kind"], "raise_to": item["raise_to"]},
                          "trained": trained, "node": nodes.get(key),
                          "worlds_requested": 96, "worlds_completed": 0,
                          "status": "running", "started": time()}
                attempts.append(record)
                write_json(output / "attempts.json", attempts)
                returns = []
                try:
                    decision_seed = int(entry["rank"][:16], 16) ^ ROOT
                    for world_index in range(96):
                        if world_index % 8 == 0:
                            _guard(output, deadline)
                            peak_rss = max(peak_rss, rss())
                        result, limited = world_action_returns(view, source, decision_seed, world_index)
                        worlds_file.write(json.dumps({"selection": entry, "world_index": world_index,
                            "action_returns_bb": result, "limited_lbr_batches": limited}) + "\n")
                        worlds_file.flush()
                        returns.append(result)
                        record["worlds_completed"] += 1
                        if world_index % 8 == 7:
                            write_json(output / "attempts.json", attempts)
                    record["summary"] = summarize(returns, probabilities)
                    record["status"] = "complete"
                except Exception as exc:
                    record["status"] = "failed"
                    record["failure"] = f"{type(exc).__name__}: {exc}"
                    write_json(output / "attempts.json", attempts)
                    raise
                record["finished"] = time()
                write_json(output / "attempts.json", attempts)
            del source
    write_json(output / "result.json", {"status": "complete", "decisions": len(attempts),
               "worlds": sum(row["worlds_completed"] for row in attempts),
               "peak_rss_bytes": max(peak_rss, rss()),
               "swap_after": system(["sysctl", "vm.swapusage"]),
               "finished": time()})
    return {"status": "complete", "decisions": len(attempts)}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("select")
    pick.add_argument("--raw-dir", type=Path, required=True)
    pick.add_argument("--out", type=Path, required=True)
    audit = sub.add_parser("values")
    audit.add_argument("--raw-dir", type=Path, required=True)
    audit.add_argument("--selection", type=Path, required=True)
    audit.add_argument("--models", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    result = (select(args.raw_dir, args.out) if args.command == "select" else
              values(args.raw_dir, args.selection, args.models, args.out, args.deadline))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
