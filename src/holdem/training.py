"""Transactional collect-and-fit iterations with collection-aligned strategy archives."""

from collections import Counter
from dataclasses import dataclass, field, replace
from math import isfinite
from time import perf_counter

from src.game.hand import Table
from src.holdem.average import AveragePolicy
from src.holdem.betting import BettingNetwork
from src.holdem.collection import collect_phase
from src.holdem.fitting import FitConfig, FitMetrics, fit_role
from src.holdem.policy import FrozenProfile
from src.holdem.replay import RoleReservoir, split_collection
from src.holdem.sampled_collection import (
    collect_sampled_phase,
    expansion_depth,
    split_sampled_collection,
)
from src.holdem.timing import measure, peak_rss_bytes
from src.solver.neural.network import stream_seed


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 0
    capacity: int = 10_000
    traversals_per_player: int = 1
    max_nodes: int = 50_000
    max_seconds: float = 60
    fit: FitConfig = field(default_factory=FitConfig)
    rotate_button: bool = True

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
            or type(self.rotate_button) is not bool
            or not isinstance(self.fit, FitConfig)
        ):
            raise ValueError("Invalid seed, time budget or fitting configuration")


@dataclass(frozen=True, slots=True)
class SampledTrainConfig(TrainConfig):
    sampler: str = "first-decision"
    exploration: float = 0.5

    def __post_init__(self):
        TrainConfig.__post_init__(self)
        expansion_depth(self.sampler)
        if type(self.exploration) not in (int, float) or not 0 < self.exploration <= 1:
            raise ValueError("Expected positive exploration for sampled training")


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
class SampledIterationReport(IterationReport):
    roots: tuple[int, ...]
    terminals: int
    max_inverse_reach: float
    max_regret_update_bb: float


@dataclass(frozen=True, slots=True)
class _State:
    iteration: int
    models: tuple[BettingNetwork | None, ...]
    memories: tuple[RoleReservoir, ...]
    reports: tuple[IterationReport, ...] = ()
    archive: tuple[FrozenProfile, ...] = ()


class HoldemTrainer:
    def __init__(self, table: Table, config: TrainConfig):
        if (
            not isinstance(table, Table)
            or not 2 <= table.capacity <= 6
            or not isinstance(config, TrainConfig)
        ):
            raise ValueError("Provide a supported table and training configuration")
        self.table, self.config = table, config
        self.last_timing = None
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

    def average_policy(self) -> AveragePolicy:
        return AveragePolicy(self._state.archive)

    def step(self) -> IterationReport:
        timing = {"iteration": self.iteration + 1, "status": "failed"}
        self.last_timing = timing
        with measure(timing, "total_seconds"):
            try:
                report = self._step(timing)
                timing.update(
                    status="complete",
                    nodes=report.nodes,
                    new_records=sum(r.new_samples for r in report.roles),
                    replay_seen=sum(m.seen for m in self.memories),
                    replay_stored=sum(len(m) for m in self.memories),
                    archive_profiles=len(self._state.archive),
                )
                return report
            finally:
                timing["peak_process_rss_bytes"] = peak_rss_bytes()

    def _step(self, timing) -> IterationReport:
        config, old = self.config, self._state
        iteration = old.iteration + 1
        deadline = perf_counter() + config.max_seconds
        profile = self.current_profile()
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise TimeoutError("Iteration deadline expired before collection")
        table = (
            replace(
                self.table,
                button=(self.table.button + iteration - 1) % len(self.table.stacks),
            )
            if config.rotate_button
            else self.table
        )
        sampled = isinstance(config, SampledTrainConfig)
        collector = collect_sampled_phase if sampled else collect_phase
        with measure(timing, "collection_seconds"):
            batch = collector(
                table,
                profile,
                iteration=iteration,
                seed=config.seed,
                traversals_per_player=config.traversals_per_player,
                max_nodes=config.max_nodes,
                max_seconds=remaining,
                **(
                    {"exploration": config.exploration, "sampler": config.sampler}
                    if sampled
                    else {}
                ),
            )
        if (
            batch.table != table
            or batch.iteration != iteration
            or batch.seed != config.seed
            or batch.profile != profile.fingerprint
            or batch.traversals_per_player != config.traversals_per_player
        ):
            raise ValueError("Collection does not belong to this training iteration")
        if sampled and batch.sampler != config.sampler:
            raise ValueError("Collection uses another sampler")
        timing["collection_coverage"] = []
        for seat, role in enumerate(table.seat_numbers):
            traversals = [t for t in batch.traversals if t.root.seat == seat]
            records = [t.decisions if sampled else t.targets for t in traversals]
            counts = Counter(
                d.candidates.decision.source.street.value
                for targets in records
                for d in targets
            )
            timing["collection_coverage"].append(
                {
                    "role": role,
                    "position_from_button": (seat - table.button) % len(table.stacks),
                    "roots": len(traversals),
                    "roots_with_postflop": sum(
                        any(
                            d.candidates.decision.source.street.value != "preflop"
                            for d in targets
                        )
                        for targets in records
                    ),
                    "records_by_street": dict(counts),
                    "nodes": sum(t.nodes for t in traversals),
                }
            )
        with measure(timing, "replay_seconds"):
            samples = (
                split_sampled_collection(batch) if sampled else split_collection(batch)
            )
            memories = tuple(memory.clone() for memory in old.memories)
        models, updates = list(old.models), []
        for role, memory in enumerate(memories):
            with measure(timing, "replay_seconds"):
                memory.extend(samples[role])
            metrics = None
            if len(memory) and role in self.table.seat_numbers:
                with measure(timing, "fitting_seconds"):
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
        report_type = SampledIterationReport if sampled else IterationReport
        details = {}
        if sampled:
            decisions = [d for t in batch.traversals for d in t.decisions]
            details = {
                "roots": tuple(
                    config.traversals_per_player if r in table.seat_numbers else 0
                    for r in range(table.capacity)
                ),
                "terminals": sum(t.terminals for t in batch.traversals),
                "max_inverse_reach": max(
                    (1 / d.own_sample_reach for d in decisions), default=0
                ),
                "max_regret_update_bb": max(
                    (abs(r) for d in decisions for r in d.regret_updates_bb), default=0
                ),
            }
        report = report_type(
            iteration,
            profile.fingerprint,
            fitted.fingerprint,
            sum(t.nodes for t in batch.traversals),
            tuple(updates),
            **details,
        )
        # Publish once: failed admission or fitting cannot leave a mixed policy generation.
        self._state = _State(
            iteration,
            tuple(models),
            memories,
            old.reports + (report,),
            old.archive + (profile,),
        )
        return report
