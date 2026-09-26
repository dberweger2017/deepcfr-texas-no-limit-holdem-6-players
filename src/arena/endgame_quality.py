"""Small, independently enumerated Hold'em river quality oracle.

This reference intentionally replays the native engine for every compatible
private deal and terminal history. It is limited to small two- and three-seat
ranges and is independent of the batched river solver's payoff matrices.
"""

from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np

from src.blueprint.abstraction import Choice, choices
from src.blueprint.river_game import RiverUnsupported
from src.blueprint.search import DECK
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, BoardDealt, HandFinished, HandStarted, PublicEvent, replay
from src.game.types import Action, ActionKind, Street


@dataclass(frozen=True, slots=True)
class _Node:
    actor: int | None
    menu: tuple[Choice, ...]
    children: tuple[int, ...]
    actions: tuple


def _world(root, board, holes):
    start = root[0]
    if not isinstance(start, HandStarted):
        raise ValueError("Missing public hand start")
    used = set(board)
    cards = dict(holes)
    used.update(card for pair in cards.values() for card in pair)
    available = iter(card for card in DECK if card not in used)
    for seat in range(len(start.stacks)):
        if seat not in cards:
            cards[seat] = (next(available), next(available))
    order = tuple((start.button + offset) % len(start.stacks)
                  for offset in range(1, len(start.stacks) + 1))
    dealt = tuple(cards[seat][round_] for round_ in range(2) for seat in order)
    remaining = tuple(card for card in DECK if card not in set(dealt + board))
    table = Table(start.player_ids, start.stacks, start.button, start.small_blind,
                  start.big_blind, start.chip_unit, start.seat_numbers,
                  start.table_seats, start.capacity, start.session_profile)
    hand = Hand.from_deck(table, hand_id=start.hand_id, deck=dealt + board + remaining)
    for event in root:
        if isinstance(event, ActionTaken):
            hand = hand.apply(event.action)
    if hand.events != root:
        raise RuntimeError("Exact-deal replay differs from the public root")
    return hand


class TinyRiverGame:
    """An exact finite public-range game with two or three active seats."""

    def __init__(self, root: tuple[PublicEvent, ...], ranges, *, raise_cap=2,
                 max_deals=256, max_public_nodes=1000):
        start = root[0]
        board = tuple(card for event in root if isinstance(event, BoardDealt)
                      for card in event.cards)
        if not isinstance(start, HandStarted) or len(board) != 5:
            raise RiverUnsupported("Tiny river reference needs a river root")
        public = replay(root, 0, ())
        self.seats = tuple(player.seat for player in public.players if not player.folded)
        if len(self.seats) not in (2, 3):
            raise RiverUnsupported("Tiny reference needs two or three active seats")
        self.root = root
        self.board = board
        self.root_pot = public.pot
        self.big_blind = public.big_blind
        self.root_stacks = tuple(player.stack for player in public.players)
        self.holdings = tuple(tuple(tuple(sorted(pair)) for pair, _ in ranges[seat])
                              for seat in self.seats)
        weights = [tuple(float(mass) for _, mass in ranges[seat]) for seat in self.seats]
        if (set(ranges) != set(self.seats) or any(len(set(h)) != len(h)
             for h in self.holdings)):
            raise ValueError("Tiny reference ranges are invalid")
        self.deals = []
        unnormalized = []
        for ids in product(*(range(len(h)) for h in self.holdings)):
            pairs = [self.holdings[index][holding] for index, holding in enumerate(ids)]
            cards = [card for pair in pairs for card in pair]
            mass = prod(weights[index][holding] for index, holding in enumerate(ids))
            if mass > 0 and len(cards) == len(set(cards)) and not set(cards) & set(board):
                self.deals.append(ids)
                unnormalized.append(mass)
        if not self.deals or len(self.deals) > max_deals:
            raise RiverUnsupported("Tiny reference has no deals or exceeds its cap")
        self.deal_mass = np.asarray(unnormalized, dtype=np.float64)
        self.deal_mass /= self.deal_mass.sum()
        self.nodes: list[_Node] = []

        def compile_tree(hand: Hand, actions: tuple) -> int:
            if len(self.nodes) >= max_public_nodes:
                raise RiverUnsupported("Tiny public tree exceeded its cap")
            node_id = len(self.nodes)
            self.nodes.append(_Node(None, (), (), actions))
            if hand.finished:
                return node_id
            view = hand.observe(hand.actor)
            menu = choices(view, raise_cap=raise_cap)
            children = tuple(compile_tree(hand.apply(choice.action),
                                          actions + (choice.action,)) for choice in menu)
            self.nodes[node_id] = _Node(hand.actor, menu, children, actions)
            return node_id

        # Only the public tree is taken from this arbitrary representative.
        representative = _world(root, board, {})
        compile_tree(representative, ())
        self.payoffs = {}
        for node_id, node in enumerate(self.nodes):
            if node.actor is not None:
                continue
            values = []
            for ids in self.deals:
                holes = {seat: self.holdings[index][ids[index]]
                         for index, seat in enumerate(self.seats)}
                hand = _world(root, board, holes)
                for action in node.actions:
                    hand = hand.apply(action)
                if not hand.finished:
                    raise RuntimeError("Tiny terminal did not settle")
                finished = next(event for event in hand.events
                                if isinstance(event, HandFinished))
                values.append(tuple(
                    (finished.stacks[seat] - self.root_stacks[seat]
                     - self.root_pot / len(self.seats)) / self.big_blind
                    for seat in self.seats
                ))
            self.payoffs[node_id] = np.asarray(values, dtype=np.float64)

    def uniform_profile(self):
        return {node_id: np.full(
            (len(self.holdings[self.seats.index(node.actor)]), len(node.menu)),
            1 / len(node.menu), dtype=np.float64,
        ) for node_id, node in enumerate(self.nodes) if node.actor is not None}

    def quality(self, profile) -> dict:
        values = []
        gains = []
        for player, seat in enumerate(self.seats):
            for best_response in (False, True):
                total = 0.0
                for holding in range(len(self.holdings[player])):
                    # Weights over complete hidden deals are carried through
                    # opponent actions; maximization occurs only after their
                    # contributions have been summed at an information set.
                    mass = np.asarray([
                        self.deal_mass[index] if ids[player] == holding else 0.0
                        for index, ids in enumerate(self.deals)
                    ])

                    def visit(node_id, weights):
                        node = self.nodes[node_id]
                        if node.actor is None:
                            return float(np.dot(weights, self.payoffs[node_id][:, player]))
                        actor = self.seats.index(node.actor)
                        if actor == player:
                            child_values = [visit(child, weights)
                                            for child in node.children]
                            return (max(child_values) if best_response else
                                    float(np.dot(profile[node_id][holding], child_values)))
                        return sum(visit(child, weights * np.asarray([
                            profile[node_id][ids[actor], action]
                            for ids in self.deals
                        ])) for action, child in enumerate(node.children))

                    total += visit(0, mass)
                if best_response:
                    gains.append(max(0.0, total - values[-1]))
                else:
                    values.append(total)
        result = {
            "values_bb": values,
            "individual_deviation_gains_bb": gains,
            "nash_conv_bb": sum(gains),
        }
        if len(self.seats) == 2:
            result["exploitability_bb"] = sum(gains) / 2
        return result


def fixture_hand(case: dict) -> Hand:
    """Build a declared six-seat river root without using a sampled board."""
    board = tuple(case["board"])
    if len(board) != 5 or len(set(board)) != 5 or any(card not in DECK for card in board):
        raise ValueError("Fixture board must contain five distinct cards")
    table = Table(tuple(f"fixture-player-{seat}" for seat in range(6)),
                  (case["stack"],) * 6, button=case["button"])
    available = [card for card in DECK if card not in board]
    shift = case["deck_shift"] % len(available)
    available = available[shift:] + available[:shift]
    hand = Hand.from_deck(table, hand_id=f"river-fixture-{case['id']}",
                          deck=tuple(available[:12]) + board + tuple(available[12:]))
    while hand.observe(hand.actor).street != Street.RIVER:
        view = hand.observe(hand.actor)
        live = sum(not player.folded for player in view.players)
        kind = (ActionKind.FOLD if view.street == Street.PREFLOP
                and live > case["survivors"] and ActionKind.FOLD in view.legal_actions.kinds
                else ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                else ActionKind.CALL)
        hand = hand.apply(Action(kind))
    return hand
