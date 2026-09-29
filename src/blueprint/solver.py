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
    SUPPORTED_SCHEMAS,
    Choice,
    choices,
    information_key,
)
from src.game.hand import Hand, Table
from src.game.observation import Observation

FORMAT = "holdem-blueprint-v1"


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
    raise_cap: int = 2
    roots_per_seat: int = 1
    max_nodes: int = 30_000
    max_entries: int = 100_000
    max_seconds: float = 300.0
    abstraction: str = SCHEMA

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value < 0
                for value in (self.seed, self.raise_cap)
            )
            or any(
                type(value) is not int or value < 1
                for value in (self.roots_per_seat, self.max_nodes, self.max_entries)
            )
            or not isfinite(self.max_seconds)
            or not 0 < self.max_seconds <= 900
            or self.abstraction not in SUPPORTED_SCHEMAS
        ):
            raise ValueError("Invalid bounded blueprint pilot configuration")


@dataclass(frozen=True, slots=True)
class FrozenTable:
    """A fixed policy profile for one complete iteration."""

    entries: dict[str, tuple[tuple[str, ...], tuple[float, ...]]]
    raise_cap: int
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


@dataclass(slots=True)
class _Delta:
    names: tuple[str, ...]
    regrets: list[float]
    average: list[float]
    visits: int = 0


@dataclass(slots=True)
class _RootResult:
    nodes: int
    terminals: int
    deltas: dict[str, _Delta]
    worker_pid: int
    worker_peak_rss_bytes: int
    coverage: dict[str, int]


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


def _collect_root(
    table: Table,
    config: PilotConfig,
    frozen_nodes: dict[str, Node],
    iteration: int,
    seat: int,
    sample: int,
    deadline: float,
) -> _RootResult:
    deltas: dict[str, _Delta] = {}
    coverage: dict[str, int] = {}
    nodes = terminals = 0

    def visit(hand: Hand, traverser: int, own_reach: float, random: Random) -> float:
        nonlocal nodes, terminals
        if nodes >= config.max_nodes or perf_counter() >= deadline:
            raise CollectionLimitExceeded(
                "Blueprint iteration reached its node or time bound"
            )
        nodes += 1
        if hand.finished:
            terminals += 1
            player = hand.observe(traverser).players[traverser]
            return (player.stack - player.starting_stack) / hand.table.big_blind
        view = hand.observe(hand.actor)
        menu = choices(view, raise_cap=config.raise_cap)
        key = information_key(view, menu, schema=config.abstraction)
        policy, trained = _distribution(frozen_nodes, key, menu)
        label = f"{view.street.value}:{'trained' if trained else 'fallback'}"
        coverage[label] = coverage.get(label, 0) + 1
        if hand.actor != traverser:
            index = random.choices(range(len(menu)), weights=policy, k=1)[0]
            return visit(hand.apply(menu[index].action), traverser, own_reach, random)
        values = tuple(
            visit(
                hand.apply(item.action),
                traverser,
                own_reach * policy[index],
                random,
            )
            for index, item in enumerate(menu)
        )
        value = fsum(p * v for p, v in zip(policy, values))
        names = tuple(item.name for item in menu)
        delta = deltas.get(key)
        if delta is None:
            delta = _Delta(names, [0.0] * len(menu), [0.0] * len(menu))
            deltas[key] = delta
        elif delta.names != names:
            raise ValueError("An abstract infoset changed its action labels")
        for index in range(len(menu)):
            delta.regrets[index] += iteration * (values[index] - value)
            delta.average[index] += iteration * own_reach * policy[index]
        delta.visits += 1
        return value

    hand = Hand.start(
        table,
        hand_id=f"blueprint-{iteration}-{seat}-{sample}",
        seed=_seed(config.seed, iteration, seat, sample, "deal"),
    )
    random = Random(_seed(config.seed, iteration, seat, sample, "actions"))
    visit(hand, seat, 1.0, random)
    return _RootResult(nodes, terminals, deltas, getpid(), _peak_rss_bytes(), coverage)


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
        self.table = table
        self.config = config
        self.iteration = 0
        self.nodes: dict[str, Node] = {}

    def frozen(self) -> FrozenTable:
        return FrozenTable(
            {
                key: (node.names, tuple(node.regrets))
                for key, node in self.nodes.items()
            },
            self.config.raise_cap,
            self.config.abstraction,
        )

    def step(self, *, workers: int = 1) -> IterationReport:
        if type(workers) is not int or workers < 1:
            raise ValueError("Blueprint workers must be a positive integer")
        iteration = self.iteration + 1
        deltas: dict[str, _Delta] = {}
        started = perf_counter()
        deadline = started + self.config.max_seconds
        nodes = terminals = 0
        worker_peaks: dict[int, int] = {}
        coverage: dict[str, int] = {}
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
                for label, count in result.coverage.items():
                    coverage[label] = coverage.get(label, 0) + count
                if nodes > self.config.max_nodes or perf_counter() >= deadline:
                    raise CollectionLimitExceeded(
                        "Blueprint iteration reached its node or time bound"
                    )
                for key, contribution in result.deltas.items():
                    delta = deltas.get(key)
                    if delta is None:
                        deltas[key] = contribution
                    else:
                        if delta.names != contribution.names:
                            raise ValueError(
                                "An abstract infoset changed its action labels"
                            )
                        for index in range(len(delta.names)):
                            delta.regrets[index] += contribution.regrets[index]
                            delta.average[index] += contribution.average[index]
                        delta.visits += contribution.visits
        finally:
            if executor is not None:
                executor.shutdown(cancel_futures=True)
        new_entries = sum(key not in self.nodes for key in deltas)
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
        )
