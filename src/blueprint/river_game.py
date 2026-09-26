"""A public river betting tree with an explicit joint private-card law.

The representative native hand supplies betting legality and public accounting.
Its private cards are never used for policies or showdown values.  One joint
law and three board-specific matrices are shared by every public tree node.
"""

from dataclasses import dataclass
from itertools import combinations
from math import fsum

import numpy as np

from src.blueprint.abstraction import Choice, choices
from src.blueprint.search import DECK
from src.game.hand import Hand, Table
from src.game.observation import (
    ActionTaken, BoardDealt, HandFinished, HandStarted, PublicEvent, replay,
)
from src.game.showdown import hand_value
from src.game.types import Action, ActionKind, Street


class RiverUnsupported(ValueError):
    """The current public state is outside the first river solver's game."""


Range = tuple[tuple[tuple[str, str], float], ...]


def river_root_history(history: tuple[PublicEvent, ...]) -> tuple[PublicEvent, ...]:
    for index, event in enumerate(history):
        if isinstance(event, BoardDealt) and event.street == Street.RIVER:
            if index + 1 >= len(history):
                raise RiverUnsupported("River root has no decision")
            return history[:index + 2]
    raise RiverUnsupported("There is no river root")


def _representative(root: tuple[PublicEvent, ...]) -> Hand:
    start = root[0]
    if not isinstance(start, HandStarted):
        raise RiverUnsupported("River root has no public hand start")
    board = tuple(card for event in root if isinstance(event, BoardDealt)
                  for card in event.cards)
    if len(board) != 5 or len(set(board)) != 5:
        raise RiverUnsupported("River root needs five distinct board cards")
    available = iter(card for card in DECK if card not in board)
    holes = {seat: (next(available), next(available))
             for seat in range(len(start.stacks))}
    order = tuple((start.button + offset) % len(start.stacks)
                  for offset in range(1, len(start.stacks) + 1))
    dealt = tuple(holes[seat][round_] for round_ in range(2) for seat in order)
    remaining = tuple(card for card in DECK if card not in set(dealt + board))
    table = Table(
        start.player_ids, start.stacks, start.button, start.small_blind,
        start.big_blind, start.chip_unit, start.seat_numbers, start.table_seats,
        start.capacity, start.session_profile,
    )
    hand = Hand.from_deck(table, hand_id=start.hand_id, deck=dealt + board + remaining)
    for event in root:
        if isinstance(event, ActionTaken):
            hand = hand.apply(event.action)
    if hand.events != root:
        raise RiverUnsupported("Representative deal did not reproduce public root")
    return hand


def _law(board: tuple[str, ...], ranges: dict[int, Range], seats: tuple[int, int]):
    holdings = []
    weights = []
    for seat in seats:
        rows = ranges[seat]
        if not rows:
            raise RiverUnsupported("Private range is empty")
        pairs = tuple(tuple(sorted(pair)) for pair, _ in rows)
        mass = np.asarray([weight for _, weight in rows], dtype=np.float64)
        if (len(set(pairs)) != len(pairs) or any(len(pair) != 2 or pair[0] == pair[1]
                or set(pair) & set(board) or any(card not in DECK for card in pair)
                for pair in pairs) or not np.isfinite(mass).all()
                or (mass < 0).any() or mass.sum() <= 0):
            raise RiverUnsupported("Invalid public private-card range")
        holdings.append(pairs)
        weights.append(mass / mass.sum())
    compatible = np.fromiter(
        (not set(a) & set(b) for a in holdings[0] for b in holdings[1]),
        dtype=np.bool_, count=len(holdings[0]) * len(holdings[1]),
    ).reshape(len(holdings[0]), len(holdings[1]))
    joint = np.outer(*weights) * compatible
    mass = joint.sum()
    if mass <= 0:
        raise RiverUnsupported("Private ranges have no compatible joint deal")
    joint /= mass
    ranks = [[hand_value(pair + board) for pair in pairs] for pairs in holdings]
    outcome = np.fromiter(
        ((a > b) - (a < b) for a in ranks[0] for b in ranks[1]),
        dtype=np.int8, count=joint.size,
    ).reshape(joint.shape)
    return tuple(holdings), joint, joint * (outcome > 0), joint * (outcome == 0)


@dataclass(frozen=True, slots=True)
class RiverTerminal:
    # u0(h0,h1) = base + win0 * first_wins + tie0 * ties, in BB.
    base: float
    win0: float = 0.0
    tie0: float = 0.0


@dataclass(frozen=True, slots=True)
class RiverNode:
    id: int
    history: tuple[PublicEvent, ...]
    actor: int | None
    menu: tuple[Choice, ...] = ()
    children: tuple[int, ...] = ()
    terminal: RiverTerminal | None = None


class RiverGame:
    """Two live seats, one public tree, and every pair in the declared law.

    `root_history` and `observed_history` are public event sequences.  The
    constructor has no actual-hero-hand parameter.  For controlled fixtures,
    `law_label` is `stipulated-product-compatible-v1`.  The play adapter uses
    a separately named active-range approximation that ignores folded-card
    removal; it must not be presented as the true conditional belief.
    """

    def __init__(
        self, root_history: tuple[PublicEvent, ...], ranges: dict[int, Range],
        *, observed_history: tuple[PublicEvent, ...] | None = None,
        raise_cap: int = 2, max_public_nodes: int = 10_000,
        law_label: str = "stipulated-product-compatible-v1",
    ):
        hand = _representative(root_history)
        root = hand.observe(0)
        if root.street != Street.RIVER or root.finished or len(root.players) != 6:
            raise RiverUnsupported("Expected an active six-seat river root")
        live = tuple(p.seat for p in root.players if not p.folded)
        if len(live) != 2 or any(root.players[seat].all_in for seat in live):
            raise RiverUnsupported("River pilot needs two active non-all-in seats")
        contested = [pot for pot in root.pots if pot.refund_to is None]
        if len(contested) != 1 or contested[0].eligible_seats != live:
            raise RiverUnsupported("River pilot needs one contested root pot")
        if set(ranges) != set(live):
            raise RiverUnsupported("Ranges must name exactly the active seats")
        self.seats = live
        self.board = root.board
        self.root_pot = root.pot
        self.big_blind = root.big_blind
        self.root_stacks = tuple(p.stack for p in root.players)
        self.holdings, self.joint, self.first_wins, self.ties = _law(
            root.board, ranges, live,
        )
        self.law_label = law_label
        self.raise_cap = raise_cap
        self.nodes: list[RiverNode] = []
        self.history_to_node: dict[tuple[PublicEvent, ...], int] = {}
        self.max_public_nodes = max_public_nodes
        self.observed_raises = {}
        for index, event in enumerate(observed_history or root_history):
            if (isinstance(event, ActionTaken) and event.street == Street.RIVER
                    and event.action.kind == ActionKind.RAISE):
                self.observed_raises[(observed_history or root_history)[:index]] = event.action
        self._compile(hand)

    def _terminal(self, hand: Hand) -> RiverTerminal:
        finish = next((event for event in hand.events if isinstance(event, HandFinished)), None)
        if finish is None:
            raise RuntimeError("Native hand lacks settlement")
        first = self.seats[0]
        paid = self.root_stacks[first]
        if not finish.showdown:
            return RiverTerminal(
                (finish.stacks[first] - paid - self.root_pot / 2) / self.big_blind,
            )
        before = hand.events[:hand.events.index(finish)]
        public = replay(before, first, ())
        paid_since_root = paid - public.players[first].stack
        payout0 = contested = 0
        for pot in public.pots:
            if pot.refund_to == first or (pot.refund_to is None
                    and pot.eligible_seats == (first,)):
                payout0 += pot.amount
            elif pot.refund_to is None and set(pot.eligible_seats) == set(self.seats):
                contested += pot.amount
            elif pot.refund_to is None and pot.eligible_seats != (self.seats[1],):
                raise RiverUnsupported("Unexpected showdown pot eligibility")
        # Odd chips in a tied contested pot go to the first eligible seat
        # clockwise from the button, as in the pinned native engine.
        button = public.button
        n = len(public.players)
        first_left = min(self.seats, key=lambda seat: (seat - button - 1) % n)
        tie0 = contested // 2 + (contested % 2 if first_left == first else 0)
        return RiverTerminal(
            (payout0 - paid_since_root - self.root_pot / 2) / self.big_blind,
            contested / self.big_blind, tie0 / self.big_blind,
        )

    def _compile(self, hand: Hand) -> int:
        if len(self.nodes) >= self.max_public_nodes:
            raise RiverUnsupported("Public river tree exceeded its node cap")
        node_id = len(self.nodes)
        self.nodes.append(RiverNode(node_id, hand.events, None))
        if hand.finished:
            self.nodes[node_id] = RiverNode(
                node_id, hand.events, None, terminal=self._terminal(hand),
            )
            return node_id
        actor = hand.actor
        view = hand.observe(actor)
        menu = choices(view, raise_cap=self.raise_cap)
        observed = self.observed_raises.get(hand.events)
        if observed is not None and all(item.action != observed for item in menu):
            view.legal_actions.validate(observed)
            menu += (Choice(f"observed-{observed.raise_to}", observed),)
        children = tuple(self._compile(hand.apply(item.action)) for item in menu)
        self.nodes[node_id] = RiverNode(node_id, hand.events, actor, menu, children)
        self.history_to_node[hand.events] = node_id
        return node_id

    def infoset_key(self, node_id: int, seat: int, holding_id: int) -> tuple[int, int, int]:
        return (node_id, seat, holding_id)

    def utility_vectors(self, terminal: RiverTerminal, traverser: int,
                        other_reach: np.ndarray) -> np.ndarray:
        """Exact counterfactual values for all traverser holdings at one leaf."""
        if traverser == self.seats[0]:
            return (terminal.base * (self.joint @ other_reach)
                    + terminal.win0 * (self.first_wins @ other_reach)
                    + terminal.tie0 * (self.ties @ other_reach))
        if traverser == self.seats[1]:
            return (-terminal.base * (self.joint.T @ other_reach)
                    - terminal.win0 * (self.first_wins.T @ other_reach)
                    - terminal.tie0 * (self.ties.T @ other_reach))
        raise ValueError("Unknown traverser")
