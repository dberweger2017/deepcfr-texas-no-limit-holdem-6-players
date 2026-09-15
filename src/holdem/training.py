"""Transactional collect-and-fit iterations; averaging and persistence follow separately."""

from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter

from src.game.hand import Table
from src.holdem.betting import BettingNetwork
from src.holdem.collection import collect_phase
from src.holdem.fitting import FitConfig, FitMetrics, fit_role
from src.holdem.policy import FrozenProfile
from src.holdem.replay import RoleReservoir, split_collection
from src.solver.neural.network import stream_seed


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 0
    capacity: int = 10_000
    traversals_per_player: int = 1
    max_nodes: int = 50_000
    max_seconds: float = 60
    fit: FitConfig = field(default_factory=FitConfig)

    def __post_init__(self):
        if any(
            type(v) is not int or v < 1
            for v in (self.capacity, self.traversals_per_player, self.max_nodes)
        ):
            raise ValueError(
                "Replay, traversal and node budgets must be positive integers"
            )
        if (
            type(self.seed) is not int
            or self.seed < 0
            or not isfinite(self.max_seconds)
            or self.max_seconds <= 0
            or not isinstance(self.fit, FitConfig)
        ):
            raise ValueError("Invalid seed, time budget or fitting configuration")


@dataclass(frozen=True, slots=True)
class RoleUpdate:
    role: int
    new_samples: int
    seen: int
    stored: int
    fit: FitMetrics | None


@dataclass(frozen=True, slots=True)
class IterationReport:
    iteration: int
    collection_profile: str
    fitted_profile: str
    nodes: int
    roles: tuple[RoleUpdate, ...]


@dataclass(frozen=True, slots=True)
class _State:
    iteration: int
    models: tuple[BettingNetwork | None, ...]
    memories: tuple[RoleReservoir, ...]
    reports: tuple[IterationReport, ...] = ()


class HoldemTrainer:
    def __init__(self, table: Table, config: TrainConfig):
        if (
            not isinstance(table, Table)
            or not 2 <= table.capacity <= 6
            or not isinstance(config, TrainConfig)
        ):
            raise ValueError("Provide a supported table and training configuration")
        self.table, self.config = table, config
        self._state = _State(
            0,
            (None,) * table.capacity,
            tuple(
                RoleReservoir(
                    role,
                    config.capacity,
                    stream_seed(config.seed, "holdem-reservoir", player=role),
                )
                for role in range(table.capacity)
            ),
        )

    @property
    def iteration(self) -> int:
        return self._state.iteration

    @property
    def memories(self) -> tuple[RoleReservoir, ...]:
        return self._state.memories

    @property
    def reports(self) -> tuple[IterationReport, ...]:
        return self._state.reports

    def current_profile(self) -> FrozenProfile:
        return FrozenProfile(self._state.models)

    def step(self) -> IterationReport:
        config, old = self.config, self._state
        iteration = old.iteration + 1
        deadline = perf_counter() + config.max_seconds
        profile = self.current_profile()
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise TimeoutError("Iteration deadline expired before collection")
        batch = collect_phase(
            self.table,
            profile,
            iteration=iteration,
            seed=config.seed,
            traversals_per_player=config.traversals_per_player,
            max_nodes=config.max_nodes,
            max_seconds=remaining,
        )
        if (
            batch.table != self.table
            or batch.iteration != iteration
            or batch.seed != config.seed
            or batch.profile != profile.fingerprint
            or batch.traversals_per_player != config.traversals_per_player
        ):
            raise ValueError("Collection does not belong to this training iteration")
        samples = split_collection(batch)
        memories = tuple(memory.clone() for memory in old.memories)
        models, updates = list(old.models), []
        for role, memory in enumerate(memories):
            memory.extend(samples[role])
            metrics = None
            if len(memory) and role in self.table.seat_numbers:
                models[role], metrics = fit_role(
                    memory,
                    config.fit,
                    iteration=iteration,
                    seed=config.seed,
                    deadline=deadline,
                )
            updates.append(
                RoleUpdate(role, len(samples[role]), memory.seen, len(memory), metrics)
            )
        profile.assert_unchanged()
        fitted = FrozenProfile(models)
        if perf_counter() >= deadline:
            raise TimeoutError("Iteration deadline expired before publication")
        report = IterationReport(
            iteration,
            profile.fingerprint,
            fitted.fingerprint,
            sum(t.nodes for t in batch.traversals),
            tuple(updates),
        )
        # Publish once: failed admission or fitting cannot leave a mixed policy generation.
        self._state = _State(
            iteration, tuple(models), memories, old.reports + (report,)
        )
        return report
