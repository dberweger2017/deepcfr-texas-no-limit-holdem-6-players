"""Observation-only, bounded postflop search against sampled public ranges.

This is a rollout search, not Pluribus's multiplayer equilibrium re-solver.
It keeps the blueprint fixed and tests whether reasoning from public ranges and
exact observed bet sizes improves its decisions.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import exp, log
from random import Random
from time import monotonic

from src.blueprint.abstraction import choices, information_key
from src.blueprint.solver import regret_match
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, HandStarted, Observation, replay
from src.game.types import Action, ActionKind, Street

DECK = tuple(rank + suit for suit in "cdhs" for rank in "23456789TJQKA")
STYLES = ("blueprint", "fold", "call", "raise")


class SearchUnavailable(RuntimeError):
    """A sampled range has no compatible hidden deal."""


class LiveBlueprint:
    """Read the loaded training table directly, avoiding a second full export."""

    def __init__(self, trainer):
        self.trainer = trainer

    def distribution(self, view: Observation):
        trainer = self.trainer
        if view.capacity != trainer.table.capacity:
            raise ValueError("Blueprint table size differs from the search table")
        menu = choices(view, raise_cap=trainer.config.raise_cap)
        key = information_key(view, menu, schema=trainer.config.abstraction)
        node = trainer.nodes.get(key)
        if node is None:
            return menu, (1 / len(menu),) * len(menu), False
        if node.names != tuple(item.name for item in menu):
            raise ValueError("Blueprint action labels differ from the observation")
        return menu, regret_match(tuple(node.regrets)), True


@dataclass(frozen=True, slots=True)
class SearchConfig:
    max_seconds: float = 0.5
    worlds: int = 8
    range_samples: int = 96
    styles: tuple[str, ...] = STYLES
    variant: str = "corrected"

    def __post_init__(self):
        object.__setattr__(self, "styles", tuple(self.styles))
        if not 0 < self.max_seconds <= 60:
            raise ValueError("Search time limit must be in (0, 60] seconds")
        if type(self.worlds) is not int or not 1 <= self.worlds <= 4096:
            raise ValueError("Search needs a bounded positive world count")
        if type(self.range_samples) is not int or not 2 <= self.range_samples <= 1326:
            raise ValueError("Search needs a bounded positive range sample count")
        if not self.styles or any(style not in STYLES for style in self.styles):
            raise ValueError("Unknown continuation style")
        if self.variant not in {"original", "corrected"}:
            raise ValueError("Unknown search variant")


def _observed_likelihood(
    blueprint, view: Observation, action: Action, *, variant: str = "corrected",
    coverage: Counter | None = None, off_tree: bool = False,
) -> float:
    menu, probabilities, trained = blueprint.distribution(view)
    if coverage is not None:
        coverage[("range", "off_tree" if off_tree else "trained" if trained else "untrained")] += 1
    if variant == "corrected":
        exact = sum(
            probability for choice, probability in zip(menu, probabilities)
            if choice.action == action
        )
        if action.kind != ActionKind.RAISE or any(choice.action == action for choice in menu):
            return exact
        if coverage is not None:
            coverage[("range", "off_menu_action")] += 1
    if action.kind != ActionKind.RAISE:
        mass = sum(
            probability
            for choice, probability in zip(menu, probabilities)
            if choice.action.kind == action.kind
        )
    else:
        # Observed off-menu raises remain possible. Nearby abstract raise sizes
        # contribute more evidence than distant sizes, but no size has zero mass.
        mass = sum(
            probability * exp(-abs(log(action.raise_to / choice.action.raise_to)))
            for choice, probability in zip(menu, probabilities)
            if choice.action.kind == ActionKind.RAISE
        )
    return max(0.01, mass)


def _off_tree_before(blueprint, view: Observation) -> tuple[bool, ...]:
    """Mark public histories after an action absent from the abstract menu."""
    known = set(view.hole_cards + view.board)
    available = (card for card in DECK if card not in known)
    pair = (next(available), next(available))
    result = []
    off_tree = False
    for index, event in enumerate(view.history):
        result.append(off_tree)
        if isinstance(event, ActionTaken):
            prior = replay(view.history[:index], event.seat, pair)
            menu, _, _ = blueprint.distribution(prior)
            off_tree |= all(item.action != event.action for item in menu)
    return tuple(result)


def public_ranges(
    blueprint,
    view: Observation,
    random: Random,
    samples: int,
    deadline: float,
    *, variant: str = "corrected", coverage: Counter | None = None,
) -> dict[int, tuple[tuple[tuple[str, str], float], ...]]:
    """Approximate each opponent's private range using only observed actions.

    Card candidates exclude the observer's hand and dealt board. The action
    likelihood is the fixed blueprint's public-history policy for that
    candidate holding; independent marginals are joined without card overlap
    when worlds are sampled.
    """
    known = set(view.hole_cards + view.board)
    available = tuple(card for card in DECK if card not in known)
    all_pairs = tuple(combinations(available, 2))
    off_tree_before = _off_tree_before(blueprint, view) if coverage is not None else ()
    result = {}
    for seat in range(len(view.players)):
        if seat == view.seat:
            continue
        candidates = random.sample(all_pairs, min(samples, len(all_pairs)))
        weighted = []
        for pair in candidates:
            if monotonic() >= deadline:
                raise TimeoutError("Public-range update exceeded the search limit")
            weight = 1.0
            for index, event in enumerate(view.history):
                if isinstance(event, ActionTaken) and event.seat == seat:
                    prior = replay(view.history[:index], seat, pair)
                    weight *= _observed_likelihood(
                        blueprint, prior, event.action, variant=variant,
                        coverage=coverage, off_tree=off_tree_before[index] if coverage is not None else False,
                    )
            weighted.append((pair, weight))
        total = sum(weight for _, weight in weighted)
        if total <= 0:
            raise SearchUnavailable("Opponent range has no legal holdings")
        result[seat] = tuple((pair, weight / total) for pair, weight in weighted)
    return result


def _sample_joint_holes(ranges, random: Random) -> dict[int, tuple[str, str]]:
    """Draw independent seat ranges conditional on having no shared cards."""
    options = {
        seat: (tuple(pair for pair, _ in rows), tuple(weight for _, weight in rows))
        for seat, rows in ranges.items()
    }
    for _ in range(256):
        sampled = {
            seat: random.choices(pairs, weights=weights, k=1)[0]
            for seat, (pairs, weights) in options.items()
        }
        cards = [card for pair in sampled.values() for card in pair]
        if len(cards) == len(set(cards)):
            return sampled
    raise SearchUnavailable("Compatible joint private range was not sampled")


def _sample_world(
    view: Observation, ranges, random: Random, *, variant: str = "corrected",
) -> Hand:
    start = view.history[0]
    if not isinstance(start, HandStarted):
        raise ValueError("Search needs a public hand start")
    n = len(view.players)
    holes = {view.seat: view.hole_cards}
    used = set(view.hole_cards + view.board)
    if variant == "corrected":
        # Rejection samples the product of seat marginals conditioned on card
        # compatibility. Sequential renormalization changes that joint law.
        sampled = _sample_joint_holes(ranges, random)
        holes.update(sampled)
        used.update(card for pair in sampled.values() for card in pair)
    else:
        for seat in random.sample(tuple(ranges), len(ranges)):
            eligible = [(pair, weight) for pair, weight in ranges[seat] if not used.intersection(pair)]
            if not eligible:
                raise SearchUnavailable("Sampled private ranges have no compatible deal")
            pair = random.choices(
                [pair for pair, _ in eligible],
                weights=[weight for _, weight in eligible], k=1,
            )[0]
            holes[seat] = pair
            used.update(pair)
    order = tuple((start.button + offset) % n for offset in range(1, n + 1))
    dealt = tuple(holes[seat][round_] for round_ in range(2) for seat in order)
    unused = [card for card in DECK if card not in used]
    random.shuffle(unused)
    deck = dealt + view.board + tuple(unused)
    table = Table(
        start.player_ids, start.stacks, start.button, start.small_blind,
        start.big_blind, start.chip_unit, start.seat_numbers,
        start.table_seats, start.capacity, start.session_profile,
    )
    hand = Hand.from_deck(table, hand_id=start.hand_id, deck=deck)
    for event in view.history:
        if isinstance(event, ActionTaken):
            hand = hand.apply(event.action)
    if hand.events != view.history or hand.observe(view.seat).hole_cards != view.hole_cards:
        raise RuntimeError("Sampled world does not reproduce the public observation")
    return hand


def _style_action(
    blueprint, view: Observation, random: Random, style: str,
    coverage: Counter | None = None, off_tree: bool = False,
) -> Action:
    menu, probabilities, trained = blueprint.distribution(view)
    if coverage is not None:
        coverage[("continuation", "off_tree" if off_tree else "trained" if trained else "untrained")] += 1
    if style == "blueprint":
        weights = probabilities
    else:
        preferred = {
            "fold": {ActionKind.FOLD},
            "call": {ActionKind.CHECK, ActionKind.CALL},
            "raise": {ActionKind.RAISE},
        }[style]
        weights = tuple(
            probability * (5 if item.action.kind in preferred else 1)
            for item, probability in zip(menu, probabilities)
        )
    return random.choices(menu, weights=weights, k=1)[0].action


def _continuation_active(
    root_street: Street, current_street: Street, flop_raises: int,
    flop_start_players: int, current_players: int, already_active: bool,
    variant: str,
) -> bool:
    if variant == "original":
        return current_street != root_street or (
            root_street == Street.FLOP and current_players > 2 and flop_raises >= 2
        )
    return already_active or current_street != root_street or (
        root_street == Street.FLOP and flop_start_players > 2 and flop_raises >= 2
    )


def _rollout(
    blueprint,
    hand: Hand,
    hero: int,
    random: Random,
    continuations: tuple[str, ...],
    root_street: Street,
    deadline: float,
    *, variant: str = "corrected", coverage: Counter | None = None,
    off_tree: bool = False,
) -> float:
    decisions = 0
    continuation_started = False
    flop_start_players = 0
    if variant == "corrected":
        start_view = hand.observe(hero)
        flop_start_players = len(start_view.players) - sum(
            isinstance(event, ActionTaken) and event.street == Street.PREFLOP
            and event.action.kind == ActionKind.FOLD
            for event in start_view.history
        )
    while not hand.finished:
        if monotonic() >= deadline:
            raise TimeoutError("Postflop search exceeded its time limit")
        if decisions >= 200:
            raise RuntimeError("Search continuation exceeded its decision limit")
        actor = hand.actor
        observation = hand.observe(actor)
        # Search the current betting round under the blueprint profile. At
        # the next street, or after a second multiway flop raise, sample the
        # selected continuation profile for the rest of the hand.
        flop_raises = sum(
            isinstance(event, ActionTaken)
            and event.street == Street.FLOP
            and event.action.kind == ActionKind.RAISE
            for event in observation.history
        )
        continuation_started = _continuation_active(
            root_street, observation.street, flop_raises, flop_start_players,
            sum(not player.folded for player in observation.players),
            continuation_started, variant,
        )
        continuation = continuations[actor] if continuation_started else "blueprint"
        action = _style_action(
            blueprint, observation, random, continuation, coverage, off_tree,
        )
        hand = hand.apply(action)
        decisions += 1
    player = hand.observe(hero).players[hero]
    return (player.stack - player.starting_stack) / hand.table.big_blind


class SearchPlayer:
    """Bounded public-range search with unchanged blueprint fallback."""

    def __init__(self, blueprint, seed: int, config: SearchConfig = SearchConfig()):
        self.blueprint = blueprint
        self.action_random = Random(seed)
        self.search_random = Random(seed ^ 0x505F4C5552494255)
        self.config = config
        self.attempts = 0
        self.completed = 0
        self.fallbacks = 0
        self.by_street = Counter()
        self.search_seconds = []
        self.coverage = Counter()

    def choose_action(self, view: Observation) -> Action:
        menu, probabilities, _ = self.blueprint.distribution(view)
        fallback = self.action_random.choices(menu, weights=probabilities, k=1)[0].action
        if view.street == Street.PREFLOP or len(menu) < 2:
            return fallback
        self.attempts += 1
        self.by_street[(view.street.value, "attempts")] += 1
        started = monotonic()
        deadline = started + self.config.max_seconds
        try:
            coverage = self.coverage if self.config.variant == "corrected" else None
            ranges = public_ranges(
                self.blueprint, view, self.search_random,
                self.config.range_samples, deadline, variant=self.config.variant,
                coverage=coverage,
            )
            off_tree = _off_tree_before(self.blueprint, view)[-1] if coverage is not None else False
            worlds = [
                _sample_world(view, ranges, self.search_random, variant=self.config.variant)
                for _ in range(self.config.worlds)
            ]
            values = [0.0] * len(menu)
            for world in worlds:
                for hero_style in self.config.styles:
                    profile = [self.search_random.choice(self.config.styles) for _ in view.players]
                    profile[view.seat] = hero_style
                    seed = self.search_random.getrandbits(64)
                    for index, item in enumerate(menu):
                        branch = world.apply(item.action)
                        values[index] += _rollout(
                            self.blueprint, branch, view.seat, Random(seed), tuple(profile),
                            view.street, deadline, variant=self.config.variant,
                            coverage=coverage, off_tree=off_tree,
                        )
            selected = menu[max(
                range(len(menu)), key=lambda index: (values[index], probabilities[index])
            )]
            view.legal_actions.validate(selected.action)
            self.completed += 1
            self.by_street[(view.street.value, "completed")] += 1
            self.search_seconds.append(monotonic() - started)
            return selected.action
        except (TimeoutError, SearchUnavailable):
            self.fallbacks += 1
            self.by_street[(view.street.value, "fallbacks")] += 1
            self.search_seconds.append(monotonic() - started)
            return fallback
