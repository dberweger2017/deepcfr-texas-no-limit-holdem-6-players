import json
from collections import Counter
from math import fsum
from pathlib import Path
from random import Random

from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, choices, information_key
from src.diagnostics.subgame_bench import STRATEGIES, FrozenRoot, SubgameTrainer

CORPUS = Path("docs/reports/hu20-board-pooling-artifacts/corpus.json")


def record(index=0):
    return json.loads(CORPUS.read_text())["roots"][index]


def request(rec, ranges):
    return {"board": rec["board"], "spot": rec["spot"],
            "ranges": [[{"hand": list(h), "weight": w} for h, w in seat] for seat in ranges]}


def small_root(index=0):
    rec = record(index)
    free = [c for c in ("2c", "2d", "3c", "3d", "4h", "4s", "5h", "5s", "6c", "6d") if c not in rec["board"]]
    hands = [(free[i], free[i + 1]) for i in range(0, len(free) - 1, 2)]
    return FrozenRoot.from_request(rec, request(rec, [[(h, 1.0) for h in hands]] * 2))


def test_joint_holdings_follow_disjoint_product_law():
    rec = record()
    cards = [c for c in ("2c", "2d", "3c", "3d", "4h", "4s") if c not in rec["board"]][:5]
    seat0 = [((cards[0], cards[1]), 1.0), ((cards[2], cards[3]), 1.0)]
    seat1 = [((cards[0], cards[4]), 1.0), ((cards[3], cards[4]), 3.0)]
    root = FrozenRoot.from_request(rec, request(rec, [seat0, seat1]))
    rng = Random(3)
    counts = Counter(root.sample_holdings(rng) for _ in range(20000))
    # Weights 1*1 (disjoint), 1*3 (collides), 1*1 (collides), 1*3 (disjoint)? Enumerate exactly.
    law = {}
    for h0, w0 in seat0:
        for h1, w1 in seat1:
            if not set(h0) & set(h1):
                law[(h0, h1)] = w0 * w1
    total = fsum(law.values())
    for pair, weight in law.items():
        assert abs(counts[pair] / 20000 - weight / total) < .015
    assert set(counts) == set(law)


def test_root_hand_is_the_frozen_turn_root_with_given_cards():
    root = small_root()
    rng = Random(1)
    holdings = root.sample_holdings(rng)
    hand = root.hand(holdings, rng)
    view = hand.observe(hand.actor)
    assert view.street.value == "turn"
    assert tuple(view.board) == root.board
    for seat in (0, 1):
        assert set(hand.observe(seat).hole_cards) == set(holdings[seat])


def test_first_update_uses_production_weights_and_both_averages():
    rec = record()
    free = [c for c in ("2c", "2d", "3c", "3d") if c not in rec["board"]]
    root = FrozenRoot.from_request(rec, request(rec, [[((free[0], free[1]), 1.0)], [((free[2], free[3]), 1.0)]]))
    trainer = SubgameTrainer([root], seed=9)
    trainer.step()
    hand = root.hand(root.sample_holdings(Random(0)), Random(0))
    view = hand.observe(hand.actor)
    menu = choices(view, raise_cap=None, free_fold=False)
    entry = trainer.table[information_key(view, menu, schema=HU20_UNCAPPED_SCHEMA)]
    uniform = [1 / len(menu)] * len(menu)
    # Iteration 1 with uniform play: the root actor's traversal adds t * own reach (1) * policy,
    # and the other seat's traversal samples this node once as the opponent and adds t * policy.
    assert entry.traverser_average == uniform
    assert entry.opponent_average == uniform
    assert entry.visits == 1


def test_regrets_are_deterministic_and_exports_normalize():
    roots = [small_root(0), small_root(1)]
    left, right = SubgameTrainer(roots, seed=4), SubgameTrainer(roots, seed=4)
    for _ in range(25):
        left.step(); right.step()
    assert {k: e.regrets for k, e in left.table.items()} == {k: e.regrets for k, e in right.table.items()}
    for strategy in STRATEGIES:
        exported = left.export(2026093001, strategy)
        assert exported["format"] == "hu20-board-pooling-policy-v1" and exported["groups"]
        for group in exported["groups"]:
            assert group["lineage"] == 2026093001 and group["metric"] == "v1" and len(group["key"]) == 32
            assert abs(fsum(group["probabilities"]) - 1) < 1e-9 and group["mass"] >= 0
    opponent_only = [e for e in left.table.values() if e.visits == 0]
    assert all(fsum(e.traverser_average) == 0 for e in opponent_only)
