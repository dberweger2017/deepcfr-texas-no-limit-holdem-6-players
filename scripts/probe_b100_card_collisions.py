"""Prespecified same-key concrete-hand probes; never use played hidden deals."""

import argparse
import atexit
import gc
import gzip
import json
from itertools import combinations
from pathlib import Path
from time import time

from scripts.diagnose_hu20_decisions import ROOT, _guard, _selected_views
from scripts.evaluate_hu20 import write_json
from scripts.evaluate_hu20_reopening import Target
from src.blueprint.abstraction import information_key
from src.blueprint.search import DECK
from src.diagnostics.conditional_values import summarize, world_action_returns
from src.game.observation import replay
from src.game.showdown import hand_value


def _alternatives(view, source):
    menu, _, _ = source.distribution(view)
    key = information_key(view, menu, schema=source.source.abstraction)
    compatible = []
    for pair in combinations((c for c in DECK if c not in view.board), 2):
        if pair == view.hole_cards or pair == tuple(reversed(view.hole_cards)):
            continue
        alternative = replay(view.history, view.seat, pair)
        alt_menu, _, _ = source.distribution(alternative)
        if information_key(alternative, alt_menu, schema=source.source.abstraction) == key:
            rank = hand_value(pair + view.board) if view.board else ()
            compatible.append((rank, pair))
    if len(compatible) < 2:
        return []
    compatible.sort(key=lambda item: (item[0], item[1]))
    return [compatible[0][1], compatible[-1][1]]


def run(raw_dir, selection_path, models_path, output, deadline):
    output.mkdir(parents=True, exist_ok=False)
    selection = json.loads(selection_path.read_text())
    views = list(_selected_views(raw_dir, selection))
    picked = {}
    for entry, item, view in views:
        if entry["trained"] and (entry["street"] not in picked or entry["rank"] < picked[entry["street"]][0]["rank"]):
            picked[entry["street"]] = (entry, view)
    specs = {s["seed"]: s for s in json.loads(models_path.read_text())
             if s.get("arm") == "B" and s.get("milestone") == 100000000}
    results = []
    world_file = gzip.open(output / "worlds.jsonl.gz", "wt")
    atexit.register(world_file.close)
    for street in ("preflop", "flop", "turn", "river"):
        if street not in picked:
            results.append({"street": street, "status": "no_trained_primary_root"})
            continue
        entry, view = picked[street]
        _guard(output, deadline)
        source = Target(specs[entry["seed"]])
        original_menu, original_probabilities, original_hit = source.distribution(view)
        original_key = information_key(view, original_menu, schema=source.source.abstraction)
        case = {"street": street, "selection": entry, "key": original_key,
                "original_cards": list(view.hole_cards), "original_trained": original_hit,
                "alternatives": [], "status": "running"}
        results.append(case)
        write_json(output / "attempts.json", results)
        for alternate_index, pair in enumerate(_alternatives(view, source)):
            _guard(output, deadline)
            alternative = replay(view.history, view.seat, pair)
            menu, probabilities, hit = source.distribution(alternative)
            alt_key = information_key(alternative, menu, schema=source.source.abstraction)
            if alt_key != original_key or probabilities != original_probabilities or hit != original_hit:
                raise ValueError("Concrete holdings do not share policy information set")
            values = []
            attempt = {"cards": list(pair), "worlds_requested": 96,
                       "worlds_completed": 0, "status": "running"}
            case["alternatives"].append(attempt)
            write_json(output / "attempts.json", results)
            seed = int(entry["rank"][:16], 16) ^ ROOT ^ (alternate_index + 1)
            for index in range(96):
                if index % 8 == 0:
                    _guard(output, deadline)
                returns, limited = world_action_returns(alternative, source, seed, index)
                world_file.write(json.dumps({"street": street, "cards": pair,
                    "world_index": index, "action_returns_bb": returns,
                    "limited_lbr_batches": limited}) + "\n")
                world_file.flush()
                values.append(returns)
                attempt["worlds_completed"] += 1
                attempt["limited_lbr_batches"] = attempt.get("limited_lbr_batches", 0) + limited
                if index % 8 == 7:
                    write_json(output / "attempts.json", results)
            attempt["summary"] = summarize(values, probabilities)
            attempt["status"] = "complete"
            write_json(output / "attempts.json", results)
        case["status"] = "complete"
        write_json(output / "attempts.json", results)
        del source
        gc.collect()
    world_file.close()
    write_json(output / "result.json", {"status": "complete", "cases": results,
                                         "finished": time()})
    return {"status": "complete", "roots": len(results)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    try:
        result = run(args.raw_dir, args.selection, args.models, args.out, args.deadline)
    except Exception as exc:
        if args.out.exists():
            write_json(args.out / "failure.json", {"status": "incomplete",
                       "error": f"{type(exc).__name__}: {exc}", "time": time()})
        raise
    print(json.dumps(result))


if __name__ == "__main__":
    main()
