"""Bounded external-sampling tabular CFR over the pilot Hold'em abstraction."""

import resource
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
from math import fsum, isfinite
from multiprocessing import get_context
from os import getpid
from random import Random
from time import perf_counter

from src.blueprint.abstraction import (
    SCHEMA,
    HU20_SCHEMA,
    HU20_UNCAPPED_SCHEMA,
    HU20_COMPRESSED_SCHEMA,
    HU20_CARD_V2_SCHEMA,
    HU20_COMPRESSED_CARD_V2_SCHEMA,
    NATIVE_SCHEMAS,
    HU100_SCHEMA,
    HU200_SCHEMA,
    STACK_BY_SCHEMA,
    TP20_SCHEMA,
    SHORTSTACK_SEATS,
    SUPPORTED_SCHEMAS,
    Choice,
    choices,
    information_key,
)
from src.game.hand import Hand, Table, card_name
from src.game.observation import ActionTaken, Observation
from src.game.types import Street

FORMAT = "holdem-blueprint-v1"
HU20_GAME = "hu20-20bb-52card-no-ante-rake-v2"
HU20_UNCAPPED_GAME = "hu20-native-reopening-20bb-52card-no-ante-rake-v1"
HU100_GAME = "hu100-native-reopening-100bb-52card-no-ante-rake-v1"
HU200_GAME = "hu200-native-reopening-200bb-52card-no-ante-rake-v1"
TP20_GAME = "tp20-20bb-52card-no-ante-rake-v1"
SHORTSTACK_GAMES = {HU20_SCHEMA: HU20_GAME, HU20_UNCAPPED_SCHEMA: HU20_UNCAPPED_GAME, HU20_COMPRESSED_SCHEMA: HU20_UNCAPPED_GAME, HU20_CARD_V2_SCHEMA: HU20_UNCAPPED_GAME, HU20_COMPRESSED_CARD_V2_SCHEMA: HU20_UNCAPPED_GAME, TP20_SCHEMA: TP20_GAME}
SHORTSTACK_GAMES[HU100_SCHEMA] = HU100_GAME
SHORTSTACK_GAMES[HU200_SCHEMA] = HU200_GAME
LEGACY_GAME = "legacy-blueprint-game-v1"


def _seed(seed: int, iteration: int, seat: int, sample: int, stream: str) -> int:
    value = f"{FORMAT}/{seed}/{iteration}/{seat}/{sample}/{stream}".encode()
    return int.from_bytes(sha256(value).digest()[:8], "big")


def regret_match(regrets: tuple[float, ...]) -> tuple[float, ...]:
    positive = tuple(max(0.0, value) for value in regrets)
    total = fsum(positive)
    return (
        tuple(value / total for value in positive)
        if total > 0
        else (1.0 / len(regrets),) * len(regrets)
    )


@dataclass(slots=True)
class Node:
    names: tuple[str, ...]
    regrets: list[float]
    average: list[float]
    visits: int = 0


@dataclass(frozen=True, slots=True)
class PilotConfig:
    seed: int = 7
    raise_cap: int | None = 2
    roots_per_seat: int = 1
    max_nodes: int = 30_000
    max_entries: int = 100_000
    max_seconds: float = 300.0
    abstraction: str = SCHEMA
    postflop_replicates: int = 1
    game: str = LEGACY_GAME

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value < 0
                for value in (self.seed,)
            )
            or (self.raise_cap is not None and (type(self.raise_cap) is not int or self.raise_cap < 0))
            or (self.abstraction in NATIVE_SCHEMAS and self.raise_cap is not None)
            or (self.abstraction not in NATIVE_SCHEMAS and self.raise_cap is None)
            or any(
                type(value) is not int or value < 1
                for value in (self.roots_per_seat, self.max_nodes, self.max_entries)
            )
            or not isfinite(self.max_seconds)
            or not 0 < self.max_seconds <= 900
            or self.abstraction not in SUPPORTED_SCHEMAS
            or type(self.postflop_replicates) is not int
            or self.postflop_replicates not in (1, 4)
            or (self.abstraction in SHORTSTACK_SEATS and
                (self.game != SHORTSTACK_GAMES[self.abstraction] or self.postflop_replicates != 1))
            or (self.abstraction not in SHORTSTACK_SEATS and self.game != LEGACY_GAME)
        ):
            raise ValueError("Invalid bounded blueprint pilot configuration")


@dataclass(frozen=True, slots=True)
class FrozenTable:
    """A fixed policy profile for one complete iteration."""

    entries: dict[str, tuple[tuple[str, ...], tuple[float, ...]]]
    raise_cap: int | None
    abstraction: str = SCHEMA

    def distribution(
        self, view: Observation, menu: tuple[Choice, ...]
    ) -> tuple[float, ...]:
        key = information_key(view, menu, schema=self.abstraction)
        saved = self.entries.get(key)
        if saved is None:
            return (1.0 / len(menu),) * len(menu)
        names, regrets = saved
        if names != tuple(item.name for item in menu):
            raise ValueError("Abstract action schema changed within a policy profile")
        return regret_match(regrets)


class CollectionLimitExceeded(RuntimeError):
    """No partial iteration is published when a bound is reached."""


@dataclass(frozen=True, slots=True)
class IterationReport:
    iteration: int
    nodes: int
    terminals: int
    entries: int
    new_entries: int
    elapsed_seconds: float
    worker_rss_sum_bytes: int = 0
    coverage: dict[str, int] = field(default_factory=dict)
    schema: str = SCHEMA
    sampled_postflop_prefixes: int = 0
    continuation_samples: int = 0
    replay_actions: int = 0
    replay_seconds: float = 0.0
    raw_traverser_visits: int = 0
    normalized_update_mass: float = 0.0
    contributing_infosets: int = 0
    traverser_visits_by_street: dict[str, int] = field(default_factory=dict)
    normalized_mass_by_street: dict[str, float] = field(default_factory=dict)
    conditional_value_variance_sum: float = 0.0
    conditional_regret_variance_sum: float = 0.0
    updated_keys: tuple[str, ...] = ()
    new_entries_by_street: dict[str, int] = field(default_factory=dict)
    revisited_keys_by_street: dict[str, int] = field(default_factory=dict)
    attempted_work: dict = field(default_factory=dict)
    new_entries_after_cap: int = 0
    revisited_entries_after_cap: int = 0


@dataclass(slots=True)
class _Delta:
    names: tuple[str, ...]
    regrets: list[float]
    average: list[float]
    visits: int = 0
    street: str | None = None
    after_cap: bool = False


@dataclass(slots=True)
class _RootResult:
    nodes: int
    terminals: int
    deltas: dict[str, _Delta]
    worker_pid: int
    worker_peak_rss_bytes: int
    coverage: dict[str, int]
    sampled_postflop_prefixes: int = 0
    continuation_samples: int = 0
    replay_actions: int = 0
    replay_seconds: float = 0.0
    raw_traverser_visits: int = 0
    normalized_update_mass: float = 0.0
    traverser_visits_by_street: dict[str, int] = field(default_factory=dict)
    normalized_mass_by_street: dict[str, float] = field(default_factory=dict)
    conditional_value_variance_sum: float = 0.0
    conditional_regret_variance_sum: float = 0.0


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _distribution(
    nodes: dict[str, Node], key: str, menu: tuple[Choice, ...]
) -> tuple[tuple[float, ...], bool]:
    node = nodes.get(key)
    if node is None:
        return (1.0 / len(menu),) * len(menu), False
    if node.names != tuple(item.name for item in menu):
        raise ValueError("Abstract action schema changed within a policy profile")
    return regret_match(tuple(node.regrets)), True


def _merge_deltas(target: dict[str, _Delta], source: dict[str, _Delta],
                  *, weight: float = 1.0) -> None:
    """Scale strategy increments but retain visits as raw traverser samples."""
    for key, item in source.items():
        delta = target.get(key)
        if delta is None:
            delta = _Delta(item.names, [0.0] * len(item.names),
                           [0.0] * len(item.names), street=item.street)
            target[key] = delta
        elif delta.names != item.names:
            raise ValueError("An abstract infoset changed its action labels")
        if delta.street != item.street:
            raise ValueError("An abstract infoset changed streets")
        for index in range(len(item.names)):
            delta.regrets[index] += weight * item.regrets[index]
            delta.average[index] += weight * item.average[index]
        delta.visits += item.visits
        delta.after_cap |= item.after_cap


def _mean_continuations(samples, count: int) -> tuple[float, dict[str, _Delta]]:
    """Missing infosets contribute zero across the declared number of samples."""
    if type(count) is not int or count < 1:
        raise ValueError("A positive continuation count is required")
    total = 0.0
    deltas: dict[str, _Delta] = {}
    observed = 0
    for value, contribution in samples:
        observed += 1
        total += value
        _merge_deltas(deltas, contribution, weight=1.0 / count)
    if observed != count:
        raise ValueError("Incomplete continuation batch")
    return total / count, deltas


def _conditional_variance(samples, value: float,
                          averaged: dict[str, _Delta],
                          iteration: int) -> tuple[float, float]:
    """Population variability across draws, with absent infosets as zero."""
    count = len(samples)
    utility = sum((sample_value - value) ** 2
                  for sample_value, _ in samples) / count
    squared = sum(sum((regret / iteration) ** 2
                      for delta in contributions.values()
                      for regret in delta.regrets)
                  for _, contributions in samples) / count
    mean_squared = sum((regret / iteration) ** 2
                       for delta in averaged.values() for regret in delta.regrets)
    return utility, max(0.0, squared - mean_squared)


def _resample_flop_future(hand: Hand, seed: int) -> tuple[Hand, int]:
    """Rebuild the exact simulated prefix with a fresh conditional deck suffix."""
    if hand.finished or hand.observe(hand.actor).street != Street.FLOP:
        raise ValueError("A first-flop decision is required")
    board = tuple(card_name(card) for card in hand._state.public_cards)
    if len(board) != 3:
        raise ValueError("The boundary must precede all flop actions")
    holes = tuple(tuple(card_name(card) for card in player.hand)
                  for player in hand._state.players_state)
    seats = tuple((hand.table.button + offset + 1) % len(holes)
                  for offset in range(len(holes)))
    dealt = tuple(holes[seat][round_index]
                  for round_index in range(2) for seat in seats)
    remaining = [card_name(card) for card in hand._state.deck]
    if len(dealt) + len(board) + len(remaining) != 52 or len(set((*dealt, *board, *remaining))) != 52:
        raise ValueError("The prefix does not contain a complete collision-free deck")
    Random(seed).shuffle(remaining)
    replayed = Hand.from_deck(
        hand.table, hand_id=hand.events[0].hand_id,
        deck=(*dealt, *board, *remaining),
    )
    actions = tuple(event.action for event in hand.events
                    if isinstance(event, ActionTaken))
    for action in actions:
        replayed = replayed.apply(action)
    if replayed.events != hand.events:
        raise RuntimeError("Conditional deck replay changed the sampled prefix")
    return replayed, len(actions)


def _collect_root(
    table: Table,
    config: PilotConfig,
    frozen_nodes: dict[str, Node],
    iteration: int,
    seat: int,
    sample: int,
    deadline: float,
    work: dict | None = None,
) -> _RootResult:
    deltas: dict[str, _Delta] = {}
    coverage: dict[str, int] = {}
    nodes = terminals = 0
    prefixes = continuations = replay_actions = raw_visits = 0
    replay_seconds = normalized_mass = 0.0
    street_visits: dict[str, int] = {}
    street_mass: dict[str, float] = {}
    value_variance = regret_variance = 0.0

    def visit(hand: Hand, traverser: int, own_reach: float, random: Random,
              target: dict[str, _Delta], path: tuple[int, ...] = (),
              replicated: bool = False, sample_weight: float = 1.0) -> float:
        nonlocal nodes, terminals, prefixes, continuations, replay_actions
        nonlocal replay_seconds, raw_visits, normalized_mass
        nonlocal value_variance, regret_variance
        if (config.postflop_replicates > 1 and not replicated and not hand.finished
                and hand.observe(hand.actor).street == Street.FLOP):
            player = hand.observe(traverser).players[traverser]
            if not player.folded and not player.all_in:
                prefixes += 1
                count = config.postflop_replicates

                def samples():
                    nonlocal continuations, replay_actions, replay_seconds
                    for index in range(count):
                        stream = f"flop/{path}/{index}"
                        started = perf_counter()
                        world, action_count = _resample_flop_future(
                            hand, _seed(config.seed, iteration, seat, sample,
                                        stream + "/deck"))
                        replay_seconds += perf_counter() - started
                        replay_actions += action_count
                        continuations += 1
                        contribution: dict[str, _Delta] = {}
                        value = visit(
                            world, traverser, own_reach,
                            Random(_seed(config.seed, iteration, seat, sample,
                                         stream + "/actions")),
                            contribution, path, True, sample_weight / count,
                        )
                        yield value, contribution

                draws = list(samples())
                value, averaged = _mean_continuations(draws, count)
                value_var, regret_var = _conditional_variance(
                    draws, value, averaged, iteration)
                value_variance += value_var
                regret_variance += regret_var
                _merge_deltas(target, averaged)
                return value
        if work is not None and work.get("cancelled") is not None and work["cancelled"]():
            raise CollectionLimitExceeded("Blueprint iteration cancelled before publication")
        if nodes >= config.max_nodes or perf_counter() >= deadline:
            raise CollectionLimitExceeded(
                "Blueprint iteration reached its node or time bound"
            )
        nodes += 1
        if work is not None:
            work["nodes"] += 1
        if hand.finished:
            terminals += 1
            if work is not None:
                work["terminals"] += 1
            player = hand.observe(traverser).players[traverser]
            return (player.stack - player.starting_stack) / hand.table.big_blind
        view = hand.observe(hand.actor)
        raises = sum(isinstance(e, ActionTaken) and e.street == view.street
                     and e.action.kind.value == "raise" for e in getattr(view, "history", ()))
        work_label = f"{view.street.value}:{raises}"
        if work is not None:
            work["nodes_by_street"][view.street.value] = work["nodes_by_street"].get(view.street.value, 0) + 1
            counts = work["decisions_by_street_raise_count"]
            counts[work_label] = counts.get(work_label, 0) + 1
        menu = choices(view, raise_cap=config.raise_cap,
                       free_fold=config.abstraction not in SHORTSTACK_SEATS)
        key = information_key(view, menu, schema=config.abstraction)
        policy, trained = _distribution(frozen_nodes, key, menu)
        label = f"{view.street.value}:{'trained' if trained else 'fallback'}"
        coverage[label] = coverage.get(label, 0) + 1
        if hand.actor != traverser:
            index = random.choices(range(len(menu)), weights=policy, k=1)[0]
            return visit(hand.apply(menu[index].action), traverser, own_reach,
                         random, target, (*path, index), replicated, sample_weight)
        values = tuple(
            visit(
                hand.apply(item.action),
                traverser,
                own_reach * policy[index],
                random,
                target,
                (*path, index),
                replicated,
                sample_weight,
            )
            for index, item in enumerate(menu)
        )
        value = fsum(p * v for p, v in zip(policy, values))
        names = tuple(item.name for item in menu)
        delta = target.get(key)
        if delta is None:
            delta = _Delta(names, [0.0] * len(menu), [0.0] * len(menu),
                           street=view.street.value, after_cap=raises >= 2)
            target[key] = delta
        elif delta.names != names:
            raise ValueError("An abstract infoset changed its action labels")
        if delta.street != view.street.value:
            raise ValueError("An abstract infoset changed streets")
        for index in range(len(menu)):
            delta.regrets[index] += iteration * (values[index] - value)
            delta.average[index] += iteration * own_reach * policy[index]
        delta.visits += 1
        if work is not None:
            counts = work["updates_by_street_raise_count"]
            counts[work_label] = counts.get(work_label, 0) + 1
        raw_visits += 1
        normalized_mass += sample_weight
        street_visits[view.street.value] = street_visits.get(view.street.value, 0) + 1
        street_mass[view.street.value] = (
            street_mass.get(view.street.value, 0.0) + sample_weight)
        return value

    hand = Hand.start(
        table,
        hand_id=f"blueprint-{iteration}-{seat}-{sample}",
        seed=_seed(config.seed, iteration, seat, sample, "deal"),
    )
    random = Random(_seed(config.seed, iteration, seat, sample, "actions"))
    visit(hand, seat, 1.0, random, deltas)
    return _RootResult(nodes, terminals, deltas, getpid(), _peak_rss_bytes(),
                       coverage, prefixes, continuations, replay_actions,
                       replay_seconds, raw_visits, normalized_mass,
                       street_visits, street_mass, value_variance, regret_variance)


_worker_state: tuple[Table, PilotConfig, dict[str, Node], int, float] | None = None


def _initialize_worker(
    table: Table,
    config: PilotConfig,
    nodes: dict[str, Node],
    iteration: int,
    deadline: float,
) -> None:
    global _worker_state
    _worker_state = (table, config, nodes, iteration, deadline)


def _worker_root(task: tuple[int, int]) -> _RootResult:
    if _worker_state is None:
        raise RuntimeError("Blueprint worker was not initialized")
    table, config, nodes, iteration, deadline = _worker_state
    return _collect_root(table, config, nodes, iteration, *task, deadline)


class BlueprintTrainer:
    def __init__(self, table: Table, config: PilotConfig):
        if not isinstance(table, Table) or not isinstance(config, PilotConfig):
            raise TypeError("Provide a table and pilot configuration")
        if not 2 <= len(table.stacks) <= 6:
            raise ValueError("The blueprint pilot supports two to six players")
        if config.abstraction in SHORTSTACK_SEATS and (
            table.capacity != SHORTSTACK_SEATS[config.abstraction]
            or table.stacks != (STACK_BY_SCHEMA[config.abstraction],) * SHORTSTACK_SEATS[config.abstraction]
            or table.small_blind != 50 or table.big_blind != 100
            or table.chip_unit != "0.01"
        ):
            raise ValueError(f"Fixed-stack training requires the versioned {STACK_BY_SCHEMA[config.abstraction] // 100}BB table")
        self.table = table
        self.config = config
        self.iteration = 0
        self.nodes: dict[str, Node] = {}
        self.last_attempt_nodes = 0
        self.last_attempt_work = {}

    def frozen(self) -> FrozenTable:
        return FrozenTable(
            {
                key: (node.names, tuple(node.regrets))
                for key, node in self.nodes.items()
            },
            self.config.raise_cap,
            self.config.abstraction,
        )

    def step(self, *, workers: int = 1, cancelled=None) -> IterationReport:
        if type(workers) is not int or workers < 1:
            raise ValueError("Blueprint workers must be a positive integer")
        iteration = self.iteration + 1
        deltas: dict[str, _Delta] = {}
        started = perf_counter()
        deadline = started + self.config.max_seconds
        nodes = terminals = 0
        work = {"nodes": 0, "cancelled": cancelled, "nodes_by_street": {},
                "decisions_by_street_raise_count": {}, "updates_by_street_raise_count": {},
                "terminals": 0}
        self.last_attempt_nodes = 0
        worker_peaks: dict[int, int] = {}
        coverage: dict[str, int] = {}
        prefixes = continuations = replay_actions = raw_visits = 0
        replay_seconds = normalized_mass = 0.0
        street_visits: dict[str, int] = {}
        street_mass: dict[str, float] = {}
        value_variance = regret_variance = 0.0
        tasks = [
            (seat, sample)
            for seat in range(len(self.table.stacks))
            for sample in range(self.config.roots_per_seat)
        ]
        if workers == 1:
            results = (
                _collect_root(
                    self.table,
                    self.config,
                    self.nodes,
                    iteration,
                    seat,
                    sample,
                    deadline,
                    work,
                )
                for seat, sample in tasks
            )
            executor = None
        else:
            # Linux fork inherits the read-only table copy-on-write. macOS spawn
            # serializes it per worker, so the M4 measurement uses one worker.
            context = get_context("fork" if sys.platform == "linux" else "spawn")
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(self.table, self.config, self.nodes, iteration, deadline),
            )
            results = executor.map(_worker_root, tasks)
        try:
            for result in results:
                if workers > 1:
                    worker_peaks[result.worker_pid] = max(
                        worker_peaks.get(result.worker_pid, 0),
                        result.worker_peak_rss_bytes,
                    )
                nodes += result.nodes
                terminals += result.terminals
                prefixes += result.sampled_postflop_prefixes
                continuations += result.continuation_samples
                replay_actions += result.replay_actions
                replay_seconds += result.replay_seconds
                raw_visits += result.raw_traverser_visits
                normalized_mass += result.normalized_update_mass
                value_variance += result.conditional_value_variance_sum
                regret_variance += result.conditional_regret_variance_sum
                for street, amount in result.traverser_visits_by_street.items():
                    street_visits[street] = street_visits.get(street, 0) + amount
                for street, amount in result.normalized_mass_by_street.items():
                    street_mass[street] = street_mass.get(street, 0.0) + amount
                for label, count in result.coverage.items():
                    coverage[label] = coverage.get(label, 0) + count
                if nodes > self.config.max_nodes or perf_counter() >= deadline:
                    raise CollectionLimitExceeded(
                        "Blueprint iteration reached its node or time bound"
                    )
                _merge_deltas(deltas, result.deltas)
        finally:
            self.last_attempt_nodes = work["nodes"] if workers == 1 else nodes
            self.last_attempt_work = {k: v for k, v in work.items() if k != "cancelled"}
            if executor is not None:
                executor.shutdown(cancel_futures=True)
        new_entries = sum(key not in self.nodes for key in deltas)
        new_by_street: dict[str, int] = {}
        revisited_by_street: dict[str, int] = {}
        new_after_cap = sum(d.after_cap and k not in self.nodes for k, d in deltas.items())
        revisited_after_cap = sum(d.after_cap and k in self.nodes for k, d in deltas.items())
        for key, delta in deltas.items():
            bucket = new_by_street if key not in self.nodes else revisited_by_street
            street = delta.street or "unknown"
            bucket[street] = bucket.get(street, 0) + 1
        if len(self.nodes) + new_entries > self.config.max_entries:
            raise CollectionLimitExceeded("Blueprint iteration reached its entry bound")
        for delta in deltas.values():
            if not all(isfinite(value) for value in (*delta.regrets, *delta.average)):
                raise FloatingPointError("Non-finite blueprint update")
        for key, delta in deltas.items():
            node = self.nodes.get(key)
            if node is not None and not all(
                isfinite(node.regrets[index] + delta.regrets[index])
                and isfinite(node.average[index] + delta.average[index])
                for index in range(len(delta.names))
            ):
                raise FloatingPointError("Non-finite accumulated blueprint update")
        for key, delta in deltas.items():
            node = self.nodes.get(key)
            if node is None:
                node = Node(
                    delta.names, [0.0] * len(delta.names), [0.0] * len(delta.names)
                )
                self.nodes[key] = node
            if node.names != delta.names:
                raise ValueError("An abstract infoset changed its action labels")
            for index in range(len(delta.names)):
                node.regrets[index] += delta.regrets[index]
                node.average[index] += delta.average[index]
            node.visits += delta.visits
        self.iteration = iteration
        return IterationReport(
            iteration,
            nodes,
            terminals,
            len(self.nodes),
            new_entries,
            perf_counter() - started,
            sum(worker_peaks.values()),
            coverage,
            self.config.abstraction,
            prefixes,
            continuations,
            replay_actions,
            replay_seconds,
            raw_visits,
            normalized_mass,
            len(deltas),
            street_visits,
            street_mass,
            value_variance,
            regret_variance,
            tuple(deltas),
            new_by_street,
            revisited_by_street,
            self.last_attempt_work,
            new_after_cap,
            revisited_after_cap,
        )
