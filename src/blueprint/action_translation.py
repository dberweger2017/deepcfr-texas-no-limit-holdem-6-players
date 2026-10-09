"""Bounded public-only witnesses for HU100/HU200 off-menu inference.

No simulator, hidden-world sampling or policy RNG is used here. Witnesses change
past raise labels only; the caller supplies the real current action menu.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from heapq import heappop, heappush
from itertools import count

from src.blueprint.abstraction import HU100_SCHEMA, HU200_SCHEMA, Choice, choices, information_key
from src.game.observation import ActionTaken, BlindPosted, BoardDealt, Observation
from src.game.types import Action, ActionKind, LegalActions, Player, Street, pots_for

VERSION = "hu100-public-menu-translation-v1"
HU200_VERSION = "hu200-public-menu-translation-v1"


def version_for_schema(schema: str) -> str:
    if schema == HU100_SCHEMA:
        return VERSION
    if schema == HU200_SCHEMA:
        return HU200_VERSION
    raise ValueError("Action translation requires an explicit HU100/HU200 schema")


@dataclass(frozen=True, slots=True)
class TranslationOptions:
    max_states: int = 512
    max_events: int = 128

    def __post_init__(self):
        if (type(self.max_states) is not int or not 1 <= self.max_states <= 4096
                or type(self.max_events) is not int or not 1 <= self.max_events <= 512):
            raise ValueError("Translation bounds must be small positive integers")


@dataclass(frozen=True, slots=True)
class Betting:
    """Two-seat public betting state; cards and engine state have no place here."""
    stacks: tuple[int, int]
    bets: tuple[int, int] = (0, 0)
    contributed: tuple[int, int] = (0, 0)
    folded: tuple[bool, bool] = (False, False)
    acted: tuple[bool, bool] = (False, False)
    increment: int = 100
    actor: int | None = 0
    street: Street = Street.PREFLOP

    @classmethod
    def start(cls, view: Observation):
        start = view.history[0]
        return cls(tuple(start.stacks), actor=start.button, increment=start.big_blind)

    @property
    def pot(self):
        return sum(self.contributed)

    def blind(self, event: BlindPosted):
        s = event.seat
        stacks = list(self.stacks); bets = list(self.bets); committed = list(self.contributed)
        stacks[s] -= event.amount; bets[s] += event.amount; committed[s] += event.amount
        return replace(self, stacks=tuple(stacks), bets=tuple(bets), contributed=tuple(committed))

    def board(self, event: BoardDealt, button: int, big_blind: int):
        if self.actor is not None or any(self.folded) or any(s == 0 for s in self.stacks):
            return None
        return replace(self, bets=(0, 0), acted=(False, False), increment=big_blind,
                       actor=1-button, street=event.street)

    def legal(self):
        if self.actor is None:
            return LegalActions()
        s = self.actor; other = 1-s
        call = min(self.stacks[s], max(self.bets)-self.bets[s])
        kinds = [ActionKind.FOLD, ActionKind.CALL if call else ActionKind.CHECK]
        low = high = None
        if self.stacks[other] and self.stacks[s] > call:
            high = self.bets[s]+self.stacks[s]
            low = min(high, max(self.bets)+self.increment)
            kinds.append(ActionKind.RAISE)
        return LegalActions(tuple(kinds), call, low, high)

    def menu(self, view: Observation):
        players = tuple(Player(s, view.players[s].player_id, view.history[0].stacks[s],
                               self.stacks[s], self.bets[s], self.contributed[s],
                               self.folded[s]) for s in (0, 1))
        public = replace(view, seat=self.actor, actor=self.actor, players=players,
                         legal_actions=self.legal(), pots=pots_for(players),
                         street=self.street)
        return choices(public, raise_cap=None, free_fold=False)

    def apply(self, action: Action):
        legal = self.legal(); legal.validate(action)
        s = self.actor; other = 1-s
        stacks = list(self.stacks); bets = list(self.bets); committed = list(self.contributed)
        folded = list(self.folded); acted = list(self.acted); increment = self.increment
        old_max = max(bets)
        paid = (action.raise_to-bets[s] if action.kind == ActionKind.RAISE else
                legal.call_amount if action.kind == ActionKind.CALL else 0)
        stacks[s] -= paid; bets[s] += paid; committed[s] += paid
        if action.kind == ActionKind.FOLD:
            folded[s] = True; actor = None
        elif action.kind == ActionKind.RAISE:
            increment = max(increment, bets[s]-old_max)
            acted = [False, False]; acted[s] = True; actor = other
        else:
            acted[s] = True
            closed = (all(acted) and bets[0] == bets[1]) or not all(stacks)
            actor = None if closed else other
        return replace(self, stacks=tuple(stacks), bets=tuple(bets),
                       contributed=tuple(committed), folded=tuple(folded),
                       acted=tuple(acted), increment=increment, actor=actor), paid

    @property
    def flags(self):
        return tuple((self.folded[s], not self.folded[s] and self.stacks[s] == 0)
                     for s in (0, 1))


def raise_label(paid: int, pot: int, remaining: int, big_blind: int):
    ratio = Fraction(paid, max(pot, big_blind))
    size = 0 if ratio < Fraction(1, 2) else 1 if ratio < Fraction(3, 2) else 2 if ratio < 3 else 3
    return f"raise-{size}" + ("-all-in" if paid == remaining else "")


def size_distance(paid: int, pot: int, other_paid: int, other_pot: int, big_blind: int):
    # x/(1+x) avoids arbitrary bin representatives and compresses oversized jams.
    return abs(Fraction(paid, max(pot, big_blind)+paid)
               - Fraction(other_paid, max(other_pot, big_blind)+other_paid))


@dataclass(frozen=True, slots=True)
class TranslationResult:
    key: str | None
    distance: float
    all_in_changes: int
    states: int
    bound_reached: bool
    overrides: tuple[tuple[int, str], ...] = ()
    witness_raise_to: tuple[int, ...] = ()


def translate(view: Observation, menu: tuple[Choice, ...], entries, zero_mass,
              options: TranslationOptions, *, schema: str = HU100_SCHEMA) -> TranslationResult:
    """Nearest positive-mass witness within a deterministic work bound."""
    version_for_schema(schema)
    # Validate game identity even when a work bound or on-menu path returns early.
    information_key(view, menu, schema=schema)
    empty = TranslationResult(None, 0., 0, 0, False)
    if len(view.history) > options.max_events:
        return replace(empty, bound_reached=True)
    state = Betting.start(view)
    steps = []
    off_menu = False
    for index, event in enumerate(view.history[1:], 1):
        if isinstance(event, BlindPosted):
            state = state.blind(event)
        elif isinstance(event, BoardDealt):
            state = state.board(event, view.button, view.big_blind)
            if state is None:
                raise ValueError("Public history reaches an impossible betting boundary")
            steps.append((index, event, None, None))
        elif isinstance(event, ActionTaken):
            if state.actor != event.seat or state.street != event.street:
                raise ValueError("Public betting actor/street differs")
            original_menu = state.menu(view)
            name = next((c.name for c in original_menu if c.action == event.action), None)
            off_menu |= event.action.kind == ActionKind.RAISE and name is None
            before = state
            state, paid = state.apply(event.action)
            if paid != event.paid:
                raise ValueError("Public action payment differs")
            steps.append((index, event, name, before))
    # On-menu controls have a single unchanged witness; missing keys stay uniform.
    if not off_menu:
        return empty
    initial = Betting.start(view)
    for event in view.history:
        if isinstance(event, BlindPosted):
            initial = initial.blind(event)
    serial = count()
    # Cost is monotone: suffix changes, rational size distance, raise-to path,
    # label path. Serial only disambiguates identical priorities.
    queue = [(0, Fraction(0), (), (), next(serial), 0, initial, ())]
    examined = 0
    target_names = tuple(c.name for c in menu)
    flags = tuple((p.folded, p.all_in) for p in view.players)
    while queue and examined < options.max_states:
        mismatches, distance, targets, labels, _, position, state, overrides = heappop(queue)
        examined += 1
        if position == len(steps):
            if state.actor != view.seat or state.flags != flags:
                continue
            if tuple(c.name for c in state.menu(view)) != target_names:
                continue
            key = information_key(view, menu, schema=schema,
                                  history_label_overrides=dict(overrides))
            saved = entries.get(key)
            if saved is not None and key not in zero_mass and saved[0] == target_names:
                return TranslationResult(key, float(distance), mismatches, examined, False,
                                         overrides, targets)
            continue
        index, event, name, observed = steps[position]
        if isinstance(event, BoardDealt):
            child = state.board(event, view.button, view.big_blind)
            if child is not None:
                heappush(queue, (mismatches, distance, targets, labels, next(serial),
                                 position+1, child, overrides))
            continue
        if state.actor != event.seat or state.street != event.street:
            continue
        candidate_menu = state.menu(view)
        candidates = (tuple(c for c in candidate_menu if c.action.kind == ActionKind.RAISE)
                      if event.action.kind == ActionKind.RAISE and name is None
                      else tuple(c for c in candidate_menu if c.name == name))
        for choice in candidates:
            child, paid = state.apply(choice.action)
            changed, cost, path, token_path, override = mismatches, distance, targets, labels, overrides
            if choice.action.kind == ActionKind.RAISE:
                token = raise_label(paid, state.pot, state.stacks[event.seat], view.big_blind)
                changed += int((paid == state.stacks[event.seat])
                               != (event.paid == observed.stacks[event.seat]))
                cost += size_distance(event.paid, observed.pot, paid, state.pot, view.big_blind)
                path += (choice.action.raise_to,); token_path += (token,)
                override += ((index, token),)
            heappush(queue, (changed, cost, path, token_path, next(serial),
                             position+1, child, override))
    return TranslationResult(None, 0., 0, examined, bool(queue))

