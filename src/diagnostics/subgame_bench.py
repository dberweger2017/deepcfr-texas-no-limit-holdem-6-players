"""Production CFR update rules on frozen turn roots, for exact held-out scoring.

The bench isolates the trainer from full-game budget and range drift: both
players start at a #149 limped turn root with that root's frozen ranges, and
the traversal applies the same external-sampling, linear-weighted updates as
`src.blueprint.solver` over the same v1 information keys. Exact exploitability
is then measured by the #149 lock-only evaluator on held-out boards.
"""

import bisect
from dataclasses import dataclass
from itertools import accumulate
import json
from math import fsum, isfinite
from random import Random

from src.blueprint import equity_buckets
from src.blueprint.abstraction import HU20_EQUITY_SCHEMA, HU20_UNCAPPED_SCHEMA, choices, information_key
from src.blueprint.solver import regret_match
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind

STRATEGIES = ("average-traverser-reach", "average-opponent-sampled", "current")
DECK = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")


def root_actions(record):
    events = json.loads(record["events"]) if isinstance(record["events"], str) else record["events"]
    return tuple((e["seat"], Action(ActionKind(e["action"]["kind"]), e["action"]["raise_to"]))
                 for e in events if "action" in e)


@dataclass(frozen=True)
class FrozenRoot:
    """One public turn root with the exact ranges its #149 equilibrium used."""

    spot: str
    button: int
    board: tuple[str, ...]
    actions: tuple
    hands: tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]
    cumulative: tuple[tuple[float, ...], tuple[float, ...]]

    @classmethod
    def from_request(cls, record, request):
        if tuple(request["board"]) != tuple(record["board"]) or request["spot"] != record["spot"]:
            raise ValueError("Request belongs to another root")
        hands, cumulative = [], []
        for seat in (0, 1):
            rows = [(tuple(r["hand"]), float(r["weight"])) for r in request["ranges"][seat]]
            if any(not isfinite(w) or w < 0 for _, w in rows) or fsum(w for _, w in rows) <= 0:
                raise ValueError("Invalid frozen range")
            if any(card in record["board"] for hand, _ in rows for card in hand):
                raise ValueError("A range holding collides with the board")
            hands.append(tuple(hand for hand, _ in rows))
            cumulative.append(tuple(accumulate(w for _, w in rows)))
        return cls(record["spot"], record["button"], tuple(record["board"]),
                   root_actions(record), tuple(hands), tuple(cumulative))

    def to_native(self):
        """This root for `hu20-trainer bench-train --roots`, running weight sums unchanged."""
        return {"spot": self.spot, "button": self.button, "board": list(self.board),
                "actions": [[seat, action.kind.value, action.raise_to] for seat, action in self.actions],
                "hands": [[list(hand) for hand in hands] for hands in self.hands],
                "cumulative": [list(cumulative) for cumulative in self.cumulative]}

    def sample_holdings(self, random):
        """Joint law proportional to w0(h0) w1(h1) for card-disjoint pairs."""
        while True:
            pair = []
            for seat in (0, 1):
                total = self.cumulative[seat][-1]
                index = bisect.bisect_right(self.cumulative[seat], random.random() * total)
                pair.append(self.hands[seat][min(index, len(self.hands[seat]) - 1)])
            if not set(pair[0]) & set(pair[1]):
                return tuple(pair)

    def hand(self, holdings, random, hand_id="bench"):
        """The engine at this turn root with the given hole cards and a fresh river."""
        table = Table(("a", "b"), (2000, 2000), button=self.button)
        order = tuple((self.button + offset + 1) % 2 for offset in range(2))
        dealt = tuple(holdings[seat][index] for index in range(2) for seat in order)
        used = set(dealt) | set(self.board)
        rest = [card for card in DECK if card not in used]
        random.shuffle(rest)
        hand = Hand.from_deck(table, hand_id=hand_id, deck=(*dealt, *self.board, *rest))
        for seat, action in self.actions:
            if hand.actor != seat:
                raise ValueError("Turn root actor mismatch")
            hand = hand.apply(action)
        return hand


class Entry:
    __slots__ = ("names", "regrets", "traverser_average", "opponent_average", "visits")

    def __init__(self, names):
        self.names = names
        self.regrets = [0.0] * len(names)
        self.traverser_average = [0.0] * len(names)
        self.opponent_average = [0.0] * len(names)
        self.visits = 0


class SubgameTrainer:
    """External-sampling CFR with the production linear weights and v1 keys.

    Regrets, and so the current policy, never depend on how the average is
    kept, so one run accumulates both averages. The traverser-reach average
    reproduces `solver.py`: it is accumulated at the traverser's nodes with
    weight t * own reach, and those nodes are reached through sampled opponent
    and chance actions, so it is also weighted by that sampled reach. The
    opponent-sampled average is the standard external-sampling rule: each
    sampled opponent node adds t * its current policy.
    """

    def __init__(self, roots, *, seed, schema=HU20_UNCAPPED_SCHEMA):
        if not roots:
            raise ValueError("An empty root set cannot be trained")
        self.roots = tuple(roots)
        self.seed = seed
        self.schema = schema
        self.iteration = 0
        self.table = {}
        self.nodes = 0

    def policy(self, key, names):
        entry = self.table.get(key)
        if entry is None:
            return (1.0 / len(names),) * len(names)
        if entry.names != names:
            raise ValueError("A v1 key changed its action menu")
        return regret_match(tuple(entry.regrets))

    def step(self):
        iteration = self.iteration + 1
        deltas = {}
        for traverser in (0, 1):
            random = Random(f"{self.seed}/{iteration}/{traverser}")
            root = self.roots[random.randrange(len(self.roots))]
            hand = root.hand(root.sample_holdings(random), random)
            self._visit(hand, traverser, 1.0, random, deltas, iteration)
        for key, (names, regrets, traverser, opponent, visits) in deltas.items():
            entry = self.table.get(key)
            if entry is None:
                entry = self.table[key] = Entry(names)
            for index in range(len(names)):
                entry.regrets[index] += regrets[index]
                entry.traverser_average[index] += traverser[index]
                entry.opponent_average[index] += opponent[index]
            entry.visits += visits[0]
        self.iteration = iteration

    def _delta(self, deltas, key, names):
        delta = deltas.get(key)
        if delta is None:
            delta = deltas[key] = (names, [0.0] * len(names), [0.0] * len(names), [0.0] * len(names), [0])
        return delta

    def _visit(self, hand, traverser, own_reach, random, deltas, iteration):
        self.nodes += 1
        if hand.finished:
            player = hand.observe(traverser).players[traverser]
            return (player.stack - player.starting_stack) / hand.table.big_blind
        view = hand.observe(hand.actor)
        menu = choices(view, raise_cap=None, free_fold=False)
        names = tuple(item.name for item in menu)
        key = information_key(view, menu, schema=self.schema)
        policy = self.policy(key, names)
        if hand.actor != traverser:
            delta = self._delta(deltas, key, names)
            for index in range(len(names)):
                delta[3][index] += iteration * policy[index]
            index = random.choices(range(len(menu)), weights=policy, k=1)[0]
            return self._visit(hand.apply(menu[index].action), traverser, own_reach,
                               random, deltas, iteration)
        values = tuple(self._visit(hand.apply(item.action), traverser, own_reach * policy[index],
                                   random, deltas, iteration)
                       for index, item in enumerate(menu))
        value = fsum(p * v for p, v in zip(policy, values))
        delta = self._delta(deltas, key, names)
        for index in range(len(names)):
            delta[1][index] += iteration * (values[index] - value)
            delta[2][index] += iteration * own_reach * policy[index]
        delta[4][0] += 1
        return value

    def export(self, lineage, strategy):
        """Groups in the #149 pooled-policy format read by the native lock pass."""
        groups = []
        for key in sorted(self.table):
            entry = self.table[key]
            if strategy == "current":
                p = list(regret_match(tuple(entry.regrets)))
                mass = float(entry.visits)
            elif strategy in STRATEGIES:
                average = (entry.traverser_average if strategy == "average-traverser-reach"
                           else entry.opponent_average)
                mass = fsum(average)
                p = ([v / mass for v in average] if mass > 0
                     else [1 / len(entry.names)] * len(entry.names))
            else:
                raise ValueError("Unknown exported strategy")
            if not all(isfinite(v) for v in p) or abs(fsum(p) - 1) > 2e-5:
                raise ValueError("Non-finite or unnormalized exported policy")
            groups.append({"lineage": lineage, "metric": "equity-k50" if self.schema == HU20_EQUITY_SCHEMA else "v1", "key": key, "names": list(entry.names),
                           "probabilities": p, "mass": mass, "roots": entry.visits})
        document = {"format": "hu20-board-pooling-policy-v1", "groups": groups, "lineage": lineage,
                    "strategy": strategy, "iteration": self.iteration,
                    "zero_mass_rule": "uniform within the actual menu"}
        if self.schema == HU20_EQUITY_SCHEMA:
            document["abstraction"] = self.schema
            document["card_tables"] = dict(equity_buckets.registered().sha256)
        return document
