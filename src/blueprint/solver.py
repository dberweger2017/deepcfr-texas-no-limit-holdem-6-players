"""Bounded external-sampling tabular CFR over the pilot Hold'em abstraction."""

from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite
from random import Random
from time import perf_counter

from src.blueprint.abstraction import SCHEMA, Choice, choices, information_key
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
        ):
            raise ValueError("Invalid bounded blueprint pilot configuration")


@dataclass(frozen=True, slots=True)
class FrozenTable:
    """A fixed policy profile for one complete iteration."""

    entries: dict[str, tuple[tuple[str, ...], tuple[float, ...]]]
    raise_cap: int

    def distribution(
        self, view: Observation, menu: tuple[Choice, ...]
    ) -> tuple[float, ...]:
        key = information_key(view, menu)
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
    schema: str = SCHEMA


@dataclass(slots=True)
class _Delta:
    names: tuple[str, ...]
    regrets: list[float]
    average: list[float]
    visits: int = 0


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
        )

    def step(self) -> IterationReport:
        iteration = self.iteration + 1
        frozen = self.frozen()
        deltas: dict[str, _Delta] = {}
        started = perf_counter()
        deadline = started + self.config.max_seconds
        nodes = terminals = 0

        def visit(
            hand: Hand, traverser: int, own_reach: float, random: Random
        ) -> float:
            nonlocal nodes, terminals
            if nodes >= self.config.max_nodes or perf_counter() >= deadline:
                raise CollectionLimitExceeded(
                    "Blueprint iteration reached its node or time bound"
                )
            nodes += 1
            if hand.finished:
                terminals += 1
                player = hand.observe(traverser).players[traverser]
                return (player.stack - player.starting_stack) / hand.table.big_blind
            view = hand.observe(hand.actor)
            menu = choices(view, raise_cap=self.config.raise_cap)
            policy = frozen.distribution(view, menu)
            if hand.actor != traverser:
                index = random.choices(range(len(menu)), weights=policy, k=1)[0]
                return visit(
                    hand.apply(menu[index].action), traverser, own_reach, random
                )
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
            key = information_key(view, menu)
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

        for seat in range(len(self.table.stacks)):
            for sample in range(self.config.roots_per_seat):
                hand = Hand.start(
                    self.table,
                    hand_id=f"blueprint-{iteration}-{seat}-{sample}",
                    seed=_seed(self.config.seed, iteration, seat, sample, "deal"),
                )
                random = Random(
                    _seed(self.config.seed, iteration, seat, sample, "actions")
                )
                visit(hand, seat, 1.0, random)
        new_entries = len(deltas.keys() - self.nodes.keys())
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
        )
