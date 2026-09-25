"""A bounded, observation-only three-way flop subgame pilot.

The public flop root and all private ranges are reconstructed from the acting
player's observation.  External sampling updates one player's regrets at a
time.  At a depth limit, continuation profiles are information-set actions,
not labels drawn before the search.
"""

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from math import fsum
from random import Random
from time import monotonic

from src.blueprint.abstraction import Choice, choices
from src.blueprint.search import (
    DECK, STYLES, SearchConfig, SearchPlayer, SearchUnavailable,
    _observed_likelihood, _rollout,
)
from src.blueprint.solver import regret_match
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, BoardDealt, Decision, HandStarted, Observation, replay
from src.game.types import Action, ActionKind, Street


@dataclass(frozen=True, slots=True)
class LocalCFRConfig:
    max_seconds: float = 5.0
    range_samples: int = 96
    min_cycles: int = 32
    max_cycles: int = 128
    max_nodes: int = 200_000
    targeted_traversal: bool = True

    def __post_init__(self):
        if not 0 < self.max_seconds <= 60:
            raise ValueError("Local CFR needs a positive bounded decision time")
        for name in ("range_samples", "min_cycles", "max_cycles", "max_nodes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Local CFR needs a positive {name}")
        if self.range_samples > 1326 or self.max_cycles < self.min_cycles:
            raise ValueError("Invalid local CFR sample or cycle bounds")
        if type(self.targeted_traversal) is not bool:
            raise ValueError("targeted_traversal must be a boolean")


@dataclass(slots=True)
class _Node:
    names: tuple[str, ...]
    regrets: list[float]
    average: list[float]
    visits: int = 0

    def policy(self) -> tuple[float, ...]:
        return regret_match(tuple(self.regrets))

    def average_policy(self) -> tuple[float, ...]:
        total = fsum(self.average)
        return tuple(value / total for value in self.average) if total > 0 else self.policy()


@dataclass(slots=True)
class _Delta:
    names: tuple[str, ...]
    regrets: list[float]
    average: list[float]
    visits: int = 0


def _record_delta(
    deltas: dict[tuple, _Delta], key: tuple, names: tuple[str, ...],
    policy: tuple[float, ...], values: tuple[float, ...],
    weight: float, own_reach: float,
) -> float:
    """Stage one external-sampling regret update; publish only full cycles."""
    expected = fsum(p * value for p, value in zip(policy, values, strict=True))
    delta = deltas.get(key)
    if delta is None:
        delta = _Delta(names, [0.0] * len(names), [0.0] * len(names))
        deltas[key] = delta
    elif delta.names != names:
        raise ValueError("Local information set changed its available actions")
    for index, probability in enumerate(policy):
        delta.regrets[index] += weight * (values[index] - expected)
        delta.average[index] += weight * own_reach * probability
    delta.visits += 1
    return expected


def _publish(nodes: dict[tuple, _Node], deltas: dict[tuple, _Delta]) -> None:
    for key, delta in deltas.items():
        node = nodes.get(key)
        if node is None:
            node = _Node(delta.names, [0.0] * len(delta.names), [0.0] * len(delta.names))
            nodes[key] = node
        elif node.names != delta.names:
            raise ValueError("Local information set changed its available actions")
        for index in range(len(delta.names)):
            node.regrets[index] += delta.regrets[index]
            node.average[index] += delta.average[index]
        node.visits += delta.visits


def _external_sampling_cycle(
    players: tuple[int, ...], cycle: int, draw_world, visit,
) -> dict[tuple, _Delta]:
    """One simultaneous, linearly weighted pass through every player role."""
    deltas: dict[tuple, _Delta] = {}
    for traverser in players:
        visit(draw_world(traverser), traverser, deltas, cycle)
    return deltas


def _flop_root(view: Observation) -> tuple[tuple, tuple[ActionTaken, ...]]:
    for index, event in enumerate(view.history):
        if isinstance(event, BoardDealt) and event.street == Street.FLOP:
            if index + 1 >= len(view.history) or not isinstance(view.history[index + 1], Decision):
                raise ValueError("The flop root needs its first public decision")
            actions = tuple(
                item for item in view.history[index + 2:]
                if isinstance(item, ActionTaken) and item.street == Street.FLOP
            )
            return view.history[:index + 2], actions
    raise ValueError("The observation has no flop root")


def _eligible(view: Observation) -> bool:
    if view.street != Street.FLOP or len(view.players) != 6 or view.actor != view.seat:
        return False
    if any(player.all_in for player in view.players):
        return False
    if sum(
        isinstance(event, ActionTaken) and event.street == Street.FLOP
        and event.action.kind == ActionKind.RAISE for event in view.history
    ) >= 2:
        return False
    live = tuple(player.seat for player in view.players if not player.folded)
    if len(live) != 3 or view.seat not in live:
        return False
    try:
        root_events, _ = _flop_root(view)
        root = replay(root_events, view.seat, view.hole_cards)
    except ValueError:
        return False
    root_live = tuple(player.seat for player in root.players if not player.folded)
    if len(root_live) != 3 or any(root.players[seat].all_in for seat in root_live):
        return False
    return all(
        pot.refund_to is not None or pot.eligible_seats == live
        for pot in view.pots
    )


def _root_ranges(
    blueprint, view: Observation, root_events: tuple, random: Random,
    samples: int, deadline: float, coverage: Counter,
) -> dict[int, tuple[tuple[tuple[str, str], float], ...]]:
    """Every seat, including hero, has a public preflop-conditioned range."""
    available = tuple(card for card in DECK if card not in view.board)
    pairs = tuple(combinations(available, 2))
    result = {}
    for seat in range(len(view.players)):
        weighted = []
        # The hero range is the complete public support.  Sampling it after
        # looking at the real hand would make its support private-information
        # dependent; exhaustive enumeration also guarantees target coverage.
        candidate_pairs = pairs if seat == view.seat else random.sample(pairs, min(samples, len(pairs)))
        for pair in candidate_pairs:
            if monotonic() >= deadline:
                raise TimeoutError("Root range construction exceeded the decision limit")
            probability = 1.0
            for index, event in enumerate(root_events):
                if isinstance(event, ActionTaken) and event.seat == seat:
                    prior = replay(root_events[:index], seat, pair)
                    # A blueprint can assign zero probability to an action
                    # that was nevertheless observed. Keep every legal hand
                    # in the public range with a small contamination mass.
                    probability *= max(1e-4, _observed_likelihood(
                        blueprint, prior, event.action, coverage=coverage,
                    ))
            weighted.append((pair, probability))
        total = fsum(probability for _, probability in weighted)
        if total <= 0:
            raise SearchUnavailable("A public flop-root range has no holdings")
        result[seat] = tuple((pair, probability / total) for pair, probability in weighted)
    return result


def _sample_holes(ranges, random: Random, forced: tuple[int, tuple[str, str]] | None = None):
    for _ in range(256):
        holes = {seat: random.choices(
            [pair for pair, _ in rows], weights=[weight for _, weight in rows], k=1,
        )[0] for seat, rows in ranges.items()}
        if forced is not None:
            holes[forced[0]] = forced[1]
        cards = [card for pair in holes.values() for card in pair]
        if len(cards) == len(set(cards)):
            return holes
    raise SearchUnavailable("No collision-free public-root world was sampled")


def _world(view: Observation, root_events: tuple, holes, random: Random) -> Hand:
    start = root_events[0]
    if not isinstance(start, HandStarted):
        raise ValueError("The public root has no hand start")
    n = len(start.stacks)
    order = tuple((start.button + offset) % n for offset in range(1, n + 1))
    dealt = tuple(holes[seat][round_] for round_ in range(2) for seat in order)
    used = set(dealt + view.board)
    if len(used) != len(dealt) + len(view.board):
        raise SearchUnavailable("A sampled root reused a public board card")
    remaining = [card for card in DECK if card not in used]
    random.shuffle(remaining)
    table = Table(
        start.player_ids, start.stacks, start.button, start.small_blind,
        start.big_blind, start.chip_unit, start.seat_numbers,
        start.table_seats, start.capacity, start.session_profile,
    )
    hand = Hand.from_deck(table, hand_id=start.hand_id, deck=dealt + view.board + tuple(remaining))
    for event in root_events:
        if isinstance(event, ActionTaken):
            hand = hand.apply(event.action)
    if hand.events != root_events:
        raise RuntimeError("A sampled flop root did not reproduce the public events")
    return hand


def _public_history(view: Observation) -> tuple:
    return tuple(
        (event.street.value, event.seat, event.action.kind.value, event.action.raise_to)
        for event in view.history if isinstance(event, ActionTaken)
    )


class _LocalSolver:
    def __init__(self, blueprint, view: Observation, random: Random, config: LocalCFRConfig,
                 deadline: float, coverage: Counter):
        self.blueprint = blueprint
        self.view = view
        self.random = random
        self.config = config
        self.deadline = deadline
        self.coverage = coverage
        self.nodes: dict[tuple, _Node] = {}
        self.final_strategy: tuple[float, ...] | None = None
        self.sampled_nodes = 0
        self.leaf_choices = 0
        self.cycles = 0
        self.range_updates = 0
        self.diagnostics: list[dict] = []
        self.root_events, self.past_actions = _flop_root(view)
        root_view = replay(self.root_events, view.seat, view.hole_cards)
        self.live = tuple(player.seat for player in root_view.players if not player.folded)
        self.root_ranges = _root_ranges(
            blueprint, view, self.root_events, random, config.range_samples, deadline, coverage,
        )
        self.hero_actual_prior = next(
            weight for pair, weight in self.root_ranges[view.seat]
            if set(pair) == set(view.hole_cards)
        )
        self.current_ranges = self.root_ranges
        self.observed_raises = {}
        for index, event in enumerate(view.history):
            if isinstance(event, ActionTaken) and event.street == Street.FLOP:
                if event.action.kind == ActionKind.RAISE:
                    self.observed_raises[view.history[:index]] = event.action

    def _check(self):
        self.sampled_nodes += 1
        if self.sampled_nodes > self.config.max_nodes or monotonic() >= self.deadline:
            raise TimeoutError("Local CFR reached its node or time limit")

    def _menu(self, view: Observation) -> tuple[Choice, ...]:
        menu = choices(view)
        observed = self.observed_raises.get(view.history)
        if observed is not None and all(item.action != observed for item in menu):
            view.legal_actions.validate(observed)
            menu += (Choice(f"observed-{observed.raise_to}", observed),)
        return menu

    def _action_key(self, view: Observation, menu: tuple[Choice, ...]) -> tuple:
        # Current-street information is lossless.  The blueprint's card
        # buckets are only used by the continuation policy.
        return ("action", view.seat, tuple(sorted(view.hole_cards)),
                view.board, _public_history(view),
                tuple((item.name, item.action.kind.value, item.action.raise_to)
                      for item in menu))

    def _leaf_key(self, view: Observation) -> tuple:
        # No opponent hand, future deck, or another player's continuation choice
        # enters this key. Thus indistinguishable leaf worlds share one choice.
        return ("leaf", view.seat, tuple(sorted(view.hole_cards)),
                view.board, _public_history(view))

    def _policy(self, key: tuple, names: tuple[str, ...]) -> tuple[float, ...]:
        node = self.nodes.get(key)
        if node is None:
            return (1.0 / len(names),) * len(names)
        if node.names != names:
            raise ValueError("Local information set changed its action names")
        return node.policy()

    def _leaf(
        self, hand: Hand, traverser: int, deltas: dict[tuple, _Delta],
        weight: float, random: Random, own_reach: float,
        index: int = 0, styles: tuple[str, ...] = (),
    ) -> float:
        self._check()
        if index == len(self.live):
            profiles = ["blueprint"] * len(self.view.players)
            for seat, style in zip(self.live, styles, strict=True):
                profiles[seat] = style
            return _rollout(
                self.blueprint, hand, traverser, random, tuple(profiles),
                Street.FLOP, self.deadline, coverage=self.coverage,
            )
        seat = self.live[index]
        if hand.observe(seat).players[seat].folded:
            return self._leaf(hand, traverser, deltas, weight, random, own_reach,
                              index + 1, styles + ("blueprint",))
        key = self._leaf_key(hand.observe(seat))
        policy = self._policy(key, STYLES)
        self.leaf_choices += 1
        if seat != traverser:
            choice = random.choices(STYLES, weights=policy, k=1)[0]
            return self._leaf(hand, traverser, deltas, weight, random, own_reach,
                              index + 1, styles + (choice,))
        values = tuple(
            self._leaf(hand, traverser, deltas, weight, Random(seed),
                       own_reach * policy[i], index + 1, styles + (style,))
            for i, style in enumerate(STYLES)
            for seed in (random.getrandbits(64),)
        )
        return _record_delta(deltas, key, STYLES, policy, values, weight, own_reach)

    def _visit(
        self, hand: Hand, traverser: int, deltas: dict[tuple, _Delta],
        weight: float, random: Random, own_reach: float,
    ) -> float:
        self._check()
        if hand.finished:
            player = hand.observe(traverser).players[traverser]
            return (player.stack - player.starting_stack) / hand.table.big_blind
        view = hand.observe(hand.actor)
        raises = sum(
            isinstance(event, ActionTaken) and event.street == Street.FLOP
            and event.action.kind == ActionKind.RAISE for event in view.history
        )
        if view.street != Street.FLOP or raises >= 2:
            return self._leaf(hand, traverser, deltas, weight, random, own_reach)
        menu = self._menu(view)
        names = tuple(item.name for item in menu)
        key = self._action_key(view, menu)
        policy = self._policy(key, names)
        if hand.actor != traverser:
            choice = random.choices(range(len(menu)), weights=policy, k=1)[0]
            return self._visit(hand.apply(menu[choice].action), traverser, deltas,
                               weight, random, own_reach)
        values = tuple(
            self._visit(hand.apply(item.action), traverser, deltas, weight,
                        Random(random.getrandbits(64)), own_reach * policy[index])
            for index, item in enumerate(menu)
        )
        return _record_delta(deltas, key, names, policy, values, weight, own_reach)

    def _observed_world(self, holes) -> Hand:
        hand = _world(self.view, self.root_events, holes, self.random)
        for event in self.past_actions:
            hand = hand.apply(event.action)
        if hand.events != self.view.history:
            raise RuntimeError("A sampled current world did not reproduce the observation")
        return hand

    def _observed_opponent_reach(self, holes) -> float:
        """Counterfactual reach of the observed flop path for hero updates."""
        reach = 1.0
        for index, event in enumerate(self.view.history):
            if not isinstance(event, ActionTaken) or event.street != Street.FLOP \
                    or event.seat == self.view.seat:
                continue
            prior = replay(self.view.history[:index], event.seat, holes[event.seat])
            menu = self._menu(prior)
            policy = self._policy(self._action_key(prior, menu),
                                  tuple(item.name for item in menu))
            reach *= fsum(probability for item, probability in zip(menu, policy, strict=True)
                          if item.action == event.action)
        return reach

    def _update_current_ranges(self) -> None:
        """Condition the public root on observed flop actions using average play."""
        result = {}
        for seat, rows in self.root_ranges.items():
            weighted = []
            for pair, prior in rows:
                if monotonic() >= self.deadline:
                    raise TimeoutError("Public range update exceeded the decision limit")
                weight = prior
                for index, event in enumerate(self.view.history):
                    if not isinstance(event, ActionTaken) or event.street != Street.FLOP \
                            or event.seat != seat:
                        continue
                    prior_view = replay(self.view.history[:index], seat, pair)
                    menu = self._menu(prior_view)
                    node = self.nodes.get(self._action_key(prior_view, menu))
                    if node is None:
                        likelihood = _observed_likelihood(
                            self.blueprint, prior_view, event.action, coverage=self.coverage,
                        )
                    else:
                        likelihood = fsum(
                            probability for item, probability in zip(menu, node.average_policy(), strict=True)
                            if item.action == event.action
                        )
                    weight *= max(1e-4, likelihood)
                weighted.append((pair, weight))
            total = fsum(weight for _, weight in weighted)
            if total <= 0:
                raise SearchUnavailable("A solved public range has no holdings")
            result[seat] = tuple((pair, weight / total) for pair, weight in weighted)
        self.current_ranges = result
        self.range_updates += 1

    def solve(self) -> tuple[tuple[Choice, ...], tuple[float, ...]]:
        target_menu = self._menu(self.view)
        target_key = self._action_key(self.view, target_menu)
        target_names = tuple(item.name for item in target_menu)
        while self.cycles < self.config.max_cycles:
            cycle = self.cycles + 1
            try:
                def draw_world(_traverser):
                    holes = _sample_holes(self.root_ranges, self.random)
                    return _world(self.view, self.root_events, holes, self.random)

                def visit(root, traverser, deltas, weight):
                    self._visit(root, traverser, deltas, weight,
                                Random(self.random.getrandbits(64)), 1.0)

                deltas = _external_sampling_cycle(
                    self.live, cycle, draw_world, visit,
                )
                if self.config.targeted_traversal:
                    # Stratify the rare actual-hand/current-path stratum.
                    # Opponent actions before the target are forced, so restore
                    # their counterfactual reach in the regret weight.  Hero's
                    # own earlier actions do not enter counterfactual reach.
                    holes = _sample_holes(
                        self.root_ranges, self.random,
                        (self.view.seat, self.view.hole_cards),
                    )
                    weight = cycle * self.hero_actual_prior * self._observed_opponent_reach(holes)
                    if weight > 0:
                        self._visit(self._observed_world(holes), self.view.seat, deltas,
                                    weight, Random(self.random.getrandbits(64)), 1.0)
            except TimeoutError:
                if self.cycles < self.config.min_cycles:
                    raise
                break
            if target_key in deltas or target_key in self.nodes:
                self.final_strategy = self._policy(target_key, target_names)
            _publish(self.nodes, deltas)
            self.cycles = cycle
            if cycle in (16, 32, 64, 128):
                policy = self._policy(target_key, target_names)
                previous = self.diagnostics[-1]["target_policy"] if self.diagnostics else None
                positive = [max(0.0, regret) for node in self.nodes.values()
                            for regret in node.regrets]
                leaf_policies = [node.policy() for key, node in self.nodes.items()
                                 if key[0] == "leaf"]
                self.diagnostics.append({
                    "cycle": cycle,
                    "target_policy": list(policy),
                    "target_names": list(target_names),
                    "target_l1_from_previous": (
                        fsum(abs(a - b) for a, b in zip(policy, previous, strict=True))
                        if previous is not None else None
                    ),
                    "mean_positive_regret": fsum(positive) / len(positive) if positive else 0.0,
                    "max_positive_regret": max(positive, default=0.0),
                    "action_infosets": sum(key[0] == "action" for key in self.nodes),
                    "leaf_infosets": len(leaf_policies),
                    "leaf_style_mean_policy": [
                        fsum(policy[index] for policy in leaf_policies) / len(leaf_policies)
                        if leaf_policies else 0.0 for index in range(len(STYLES))
                    ],
                })
            if self.cycles % 8 == 0 and self.past_actions:
                try:
                    self._update_current_ranges()
                except TimeoutError:
                    if self.cycles < self.config.min_cycles:
                        raise
                    break
        if self.cycles < self.config.min_cycles:
            raise TimeoutError("Local CFR did not complete the required cycles")
        node = self.nodes.get(target_key)
        if node is None or node.visits == 0 or self.final_strategy is None:
            raise SearchUnavailable("The actual hero information set was not visited")
        return target_menu, self.final_strategy


class LocalCFRPlayer:
    """Opt-in local solver, with corrected rollout outside its pilot target."""

    def __init__(
        self, blueprint, seed: int, config: LocalCFRConfig = LocalCFRConfig(),
        other_search_config: SearchConfig = SearchConfig(variant="corrected"),
    ):
        self.blueprint = blueprint
        self.seed = seed
        self.config = config
        self.action_random = Random(seed)
        self.search_random = Random(seed ^ 0x4C4F43414C434652)
        self.other = SearchPlayer(blueprint, seed, other_search_config)
        self.attempts = self.completed = self.fallbacks = 0
        self.by_street = Counter()
        self.cycles: list[int] = []
        self.search_seconds: list[float] = []
        self.attempt_records: list[dict] = []
        self.nodes = self.leaf_choices = self.range_updates = 0
        self.coverage = Counter()

    def choose_action(self, view: Observation) -> Action:
        if not _eligible(view):
            return self.other.choose_action(view)
        menu, probabilities, _ = self.blueprint.distribution(view)
        fallback = self.action_random.choices(menu, weights=probabilities, k=1)[0].action
        self.attempts += 1
        self.by_street[("flop", "attempts")] += 1
        started = monotonic()
        solver = None
        status = "fallback"
        reason = None
        try:
            solver = _LocalSolver(
                self.blueprint, view, self.search_random, self.config,
                started + self.config.max_seconds, self.coverage,
            )
            local_menu, policy = solver.solve()
            action = self.action_random.choices(local_menu, weights=policy, k=1)[0].action
            view.legal_actions.validate(action)
            self.completed += 1
            self.by_street[("flop", "completed")] += 1
            self.cycles.append(solver.cycles)
            status = "completed"
            return action
        except (TimeoutError, SearchUnavailable) as exc:
            self.fallbacks += 1
            self.by_street[("flop", "fallbacks")] += 1
            reason = f"{type(exc).__name__}: {exc}"
            return fallback
        finally:
            elapsed = monotonic() - started
            if solver is not None:
                self.nodes += solver.sampled_nodes
                self.leaf_choices += solver.leaf_choices
                self.range_updates += solver.range_updates
            self.search_seconds.append(elapsed)
            self.attempt_records.append({
                "status": status,
                "reason": reason,
                "cycles": solver.cycles if solver is not None else 0,
                "nodes": solver.sampled_nodes if solver is not None else 0,
                "leaf_choices": solver.leaf_choices if solver is not None else 0,
                "range_updates": solver.range_updates if solver is not None else 0,
                "root_range_sizes": ({str(seat): len(rows) for seat, rows in solver.root_ranges.items()}
                                     if solver is not None else None),
                "diagnostics": solver.diagnostics if solver is not None else [],
                "seconds": elapsed,
            })
