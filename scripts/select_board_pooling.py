"""Choose one occupied exact public line and forty boards before solver values."""

import argparse
from collections import Counter, defaultdict
from random import Random
from pathlib import Path
import json

from src.diagnostics.board_pooling import public_line
from src.diagnostics.board_pooling_policy import build_index, DiskAverage
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.turn_check import root_record
from src.game.hand import Hand, Table
from src.game.types import Street


def select(source, *, deals=3000, boards=40, deal_seed=202610030301,
           action_seed=202610030302, selection_seed=202610030303):
    deals_rng = Random(deal_seed)
    actions_rng = Random(action_seed)
    populations = defaultdict(dict)
    occupancy = Counter()
    counts = Counter()
    for index in range(deals):
        hand = Hand.start(Table(("a", "b"), (2000, 2000), button=0),
                          hand_id="board-pooling-selection", seed=deals_rng.getrandbits(64))
        while not hand.finished and hand.observe(hand.actor).street != Street.TURN:
            menu, p, trained = source.distribution(hand.observe(hand.actor))
            counts["trained" if trained else "missing"] += 1
            hand = hand.apply(actions_rng.choices(menu, weights=p, k=1)[0].action)
        if hand.finished:
            counts["earlier_terminal"] += 1
            continue
        key = public_line(hand.events, 0)
        record = root_record(hand.events)
        record["multiplicity"] = populations[key].get(record["spot"], {}).get("multiplicity", 0) + 1
        populations[key][record["spot"]] = record
        occupancy[key] += 1
        counts["live_turn"] += 1
    eligible = [line for line in occupancy if len(populations[line]) >= boards]
    if not eligible:
        raise ValueError("No public line has forty unique boards; no adaptive deal extension")
    chosen = sorted(eligible, key=lambda line: (-occupancy[line], line))[0]
    population = sorted(populations[chosen].values(), key=lambda root: root["spot"])
    selected = Random(selection_seed).sample(population, boards)
    for root in selected:
        root["inclusion_probability"] = boards / len(population)
        root["board_weight"] = root["multiplicity"] / root["inclusion_probability"]
    return {"format": "hu20-board-pooling-corpus-v1", "roots": selected,
            "chosen_line": chosen, "occupancy": dict(occupancy),
            "unique_boards_by_line": {k: len(v) for k, v in populations.items()},
            "counts": dict(counts), "deals": deals, "source": source.description,
            "deal_seed": deal_seed, "action_seed": action_seed, "selection_seed": selection_seed,
            "button": 0, "turn_actions_sampled": 0, "selection_uses_payoffs": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve a frozen corpus")
    spec = next(s for s in json.loads(Path("configs/diagnostics/hu20-exact-turn-check.json").read_text())["policies"]
                if s["strategy"] == "stored-average" and s["seed"] == 2026093001)
    inventory = build_index(spec, args.inputs, args.index)
    corpus = select(DiskAverage(args.index, spec))
    corpus["index"] = inventory
    atomic_json(args.out, corpus)


if __name__ == "__main__":
    main()
