"""Record production CFR traversals as parity fixtures for the native trainer.

Trains a small Python HU20 table so policies are non-uniform, then runs the
real `_collect_root` traversal on recorded decks while logging every sampled
opponent action. The native trainer must reproduce the per-traversal regret,
average and visit deltas from the same table, deck and draws.
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
    p.add_argument("--train-iterations", type=int, default=400)
    p.add_argument("--cases", type=int, default=300)
    p.add_argument("--seed", type=int, default=202610050200)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    table = Table(("player-0", "player-1"), (2000, 2000))
    config = PilotConfig(seed=a.seed, raise_cap=None, abstraction=HU20_UNCAPPED_SCHEMA, game=HU20_UNCAPPED_GAME,
                         roots_per_seat=1, postflop_replicates=1, max_nodes=10**9, max_entries=10**9, max_seconds=900)
    trainer = BlueprintTrainer(table, config)
    for _ in range(a.train_iterations):
        trainer.step()
    rng = Random(a.seed)
    cases = []
    original_start, original_random = solver.Hand.start, solver.Random
    try:
        for index in range(a.cases):
            deck = list(DECK)
            rng.shuffle(deck)
            seat, iteration = index % 2, trainer.iteration + 1 + index // 2
            log = []
            solver.Hand.start = classmethod(lambda cls, t, *, hand_id, seed, d=tuple(deck): Hand.from_deck(t, hand_id=hand_id, deck=d))
            solver.Random = lambda s, log=log: RecordingRandom(s, log)
            result = solver._collect_root(table, config, trainer.nodes, iteration, seat, 0, float("inf"))
            cases.append({"deck": deck, "seat": seat, "iteration": iteration, "draws": list(log),
                          "nodes": result.nodes, "terminals": result.terminals,
                          "deltas": {k: [list(d.names), d.regrets, d.average, d.visits] for k, d in result.deltas.items()}})
    finally:
        solver.Hand.start, solver.Random = original_start, original_random
    nodes = {k: [list(n.names), n.regrets, n.average, n.visits] for k, n in trainer.nodes.items()}
    a.out.write_text(json.dumps({"table_button": table.button, "iteration": trainer.iteration,
                                 "nodes": nodes, "cases": cases}, separators=(",", ":")))


if __name__ == "__main__":
    main()
