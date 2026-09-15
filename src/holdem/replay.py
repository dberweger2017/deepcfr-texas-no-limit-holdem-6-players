"""Uniform role reservoirs with immutable targets and collection provenance."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isclose, isfinite
from random import Random

from src.holdem.actions import bet_candidates
from src.holdem.collection import FORMAT, Collection, collection_seed
from src.holdem.targets import CandidateTargets


def _positive(value: int) -> bool:
    return type(value) is int and value > 0


def _digest(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True, slots=True)
class ReplaySample:
    role: int
    iteration: int
    profile: str
    action_seed: int
    target_index: int
    target: CandidateTargets

    def validate(self) -> None:
        if (
            type(self.role) is not int
            or not 0 <= self.role < 6
            or not _positive(self.iteration)
            or not _digest(self.profile)
        ):
            raise ValueError("Invalid replay role, iteration or profile")
        if any(
            type(v) is not int or v < 0 for v in (self.action_seed, self.target_index)
        ) or not isinstance(self.target, CandidateTargets):
            raise ValueError("Invalid replay provenance or target")
        target = self.target
        view = target.candidates.decision.source
        if (
            view.seat_numbers[view.seat] != self.role
            or bet_candidates(view) != target.candidates
        ):
            raise ValueError(
                "Replay candidates do not match their public observation and role"
            )
        count = len(target.candidates.actions)
        vectors = (target.policy, target.values_bb, target.regrets_bb)
        if any(
            type(v) is not tuple or len(v) != count or any(not isfinite(x) for x in v)
            for v in vectors
        ):
            raise ValueError(
                "Expected a finite policy, value and regret for every candidate"
            )
        if min(target.policy) < 0 or not isclose(
            fsum(target.policy), 1, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("Invalid replay policy")
        baseline = fsum(p * v for p, v in zip(target.policy, target.values_bb))
        if any(
            not isclose(r, v - baseline, rel_tol=1e-9, abs_tol=1e-9)
            for r, v in zip(target.regrets_bb, target.values_bb)
        ):
            raise ValueError(
                "Regret targets disagree with the recorded policy and values"
            )


class RoleReservoir:
    def __init__(self, role: int, capacity: int, seed: int):
        if (
            type(role) is not int
            or not 0 <= role < 6
            or not _positive(capacity)
            or type(seed) is not int
            or seed < 0
        ):
            raise ValueError(
                "Provide a physical role, positive capacity and nonnegative seed"
            )
        self.role, self.capacity, self.seen = role, capacity, 0
        self._items: list[ReplaySample] = []
        self._random = Random(seed)

    @property
    def items(self) -> tuple[ReplaySample, ...]:
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def clone(self) -> "RoleReservoir":
        result = RoleReservoir(self.role, self.capacity, 0)
        result.seen = self.seen
        result._items = self._items.copy()
        result._random.setstate(self._random.getstate())
        return result

    def extend(self, samples: Sequence[ReplaySample]) -> None:
        samples = tuple(samples)
        for sample in samples:
            sample.validate()
            if sample.role != self.role:
                raise ValueError("Cannot pool samples from different roles")
        for sample in samples:
            self.seen += 1
            if len(self._items) < self.capacity:
                self._items.append(sample)
            else:
                slot = self._random.randrange(self.seen)
                if slot < self.capacity:
                    self._items[slot] = sample

    def sample(self, count: int, random: Random) -> tuple[ReplaySample, ...]:
        if not _positive(count) or not self._items:
            raise ValueError(
                "Sampling needs a positive batch size and a nonempty memory"
            )
        return tuple(
            self._items[random.randrange(len(self._items))] for _ in range(count)
        )


def split_collection(batch: Collection) -> tuple[tuple[ReplaySample, ...], ...]:
    if not isinstance(batch, Collection) or batch.schema != FORMAT:
        raise ValueError("Expected a completed collection with the supported schema")
    table = batch.table
    if (
        not _positive(batch.iteration)
        or not _positive(batch.traversals_per_player)
        or not _digest(batch.profile)
    ):
        raise ValueError("Invalid collection provenance")
    expected = [
        (seat, physical, sample)
        for seat, physical in enumerate(table.seat_numbers)
        for sample in range(batch.traversals_per_player)
    ]
    if len(batch.traversals) != len(expected):
        raise ValueError("Collection does not contain every scheduled traversal")
    roles = [[] for _ in range(table.capacity)]
    for traversal, (seat, physical, sample_index) in zip(batch.traversals, expected):
        root = traversal.root
        if (
            traversal.iteration != batch.iteration
            or traversal.profile != batch.profile
            or traversal.action_seed
            != collection_seed(
                batch.seed, batch.iteration, physical, sample_index, "opponents"
            )
            or root.seat != seat
            or root.seat_numbers != table.seat_numbers
            or root.capacity != table.capacity
            or root.player_id != table.player_ids[seat]
            or root.history[0].stacks != table.stacks
            or root.hand_id != f"collection-{batch.iteration}-{physical}-{sample_index}"
            or traversal.nodes != len(traversal.executions) + 1
            or traversal.terminals < 1
        ):
            raise ValueError("Traversal does not match its collection schedule")
        decisions = {
            e.candidates.decision.source.history
            for e in traversal.executions
            if e.event.seat == seat
        }
        histories = [t.candidates.decision.source.history for t in traversal.targets]
        if len(histories) != len(decisions) or set(histories) != decisions:
            raise ValueError("Collection is missing or repeating traverser targets")
        for index, target in enumerate(traversal.targets):
            view = target.candidates.decision.source
            if (
                view.seat != seat
                or view.hole_cards != root.hole_cards
                or view.history[: len(root.history)] != root.history
            ):
                raise ValueError("Target belongs to another traversal or private owner")
            item = ReplaySample(
                physical,
                batch.iteration,
                batch.profile,
                traversal.action_seed,
                index,
                target,
            )
            item.validate()
            roles[physical].append(item)
    return tuple(tuple(samples) for samples in roles)
