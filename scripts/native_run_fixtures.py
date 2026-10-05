"""Record a whole Python HU20 training run as a parity fixture for the native trainer.

Runs `BlueprintTrainer.step` from an empty table while logging every root's deck
and every sampled opponent action. Given the same decks and draws, the native
trainer must reproduce each iteration's node count and the final table bit for bit.
Decks come from `Random(deal_seed).shuffle`, so the recorded run is an ordinary
Python run with a different, still uniform, deal stream.
"""

import argparse
import json
from pathlib import Path
from random import Random

from src.blueprint import solver
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.blueprint.solver import BlueprintTrainer, HU20_UNCAPPED_GAME, PilotConfig
from src.game.hand import Hand, Table

DECK = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")


class RecordingRandom(Random):
    """Logs each opponent sample as the index `random.choices` returned."""

    def __init__(self, seed, log):
        super().__init__(seed)
        self.log = log

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        picked = super().choices(population, weights=weights, cum_weights=cum_weights, k=k)
        self.log.extend(picked)
        return picked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--seed", type=int, default=202610051500)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    table = Table(("player-0", "player-1"), (2000, 2000))
    config = PilotConfig(seed=a.seed, raise_cap=None, abstraction=HU20_UNCAPPED_SCHEMA, game=HU20_UNCAPPED_GAME,
                         roots_per_seat=1, postflop_replicates=1, max_nodes=10**9, max_entries=10**9, max_seconds=900)
    trainer = BlueprintTrainer(table, config)
    roots = []

    def start(cls, t, *, hand_id, seed):
        deck = list(DECK)
        Random(seed).shuffle(deck)
        roots.append({"deck": deck, "draws": []})
        return Hand.from_deck(t, hand_id=hand_id, deck=tuple(deck))

    original_start, original_random = solver.Hand.start, solver.Random
    iterations = []
    try:
        solver.Hand.start = classmethod(start)
        # `_collect_root` builds the hand before its action stream, so the newest root owns the draws.
        solver.Random = lambda s: RecordingRandom(s, roots[-1]["draws"])
        for _ in range(a.iterations):
            roots.clear()
            report = trainer.step()
            iterations.append({"nodes": report.nodes, "roots": list(roots)})
    finally:
        solver.Hand.start, solver.Random = original_start, original_random
    nodes = {k: [list(n.names), n.regrets, n.average, n.visits] for k, n in trainer.nodes.items()}
    a.out.write_text(json.dumps({"table_button": table.button, "iterations": iterations, "nodes": nodes},
                                separators=(",", ":")))


if __name__ == "__main__":
    main()
