"""Privileged external sampling; only public candidate inputs cross into policies."""

from dataclasses import dataclass, field
from hashlib import sha256
from math import fsum, isfinite, isnan
from random import Random
from time import perf_counter

from src.game.hand import Hand, Table
from src.game.observation import Observation
from src.holdem.actions import (
    BetCandidates,
    ExecutedBet,
    bet_candidates,
    record_execution,
)
from src.holdem.policy import FrozenProfile
from src.holdem.targets import CandidateTargets, action_targets
from src.solver.neural.network import deterministic_cpu

FORMAT = "holdem-external-sampling-v1"


def collection_seed(
    seed: int, iteration: int, seat: int, sample: int, stream: str
) -> int:
    for value in (seed, iteration, seat, sample):
        if type(value) is not int or value < 0:
            raise ValueError("Seed coordinates must be nonnegative integers")
    if stream not in ("deal", "opponents"):
        raise ValueError("Unknown collection random stream")
    key = f"{FORMAT}/{seed}/{iteration}/{seat}/{sample}/{stream}".encode()
    return int.from_bytes(sha256(key).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class Traversal:
    root: Observation
    iteration: int
    profile: str
    action_seed: int
    value_bb: float
    targets: tuple[CandidateTargets, ...]
    executions: tuple[ExecutedBet, ...]
    nodes: int
    terminals: int


@dataclass(frozen=True, slots=True)
class Collection:
    table: Table
    iteration: int
    seed: int
    traversals_per_player: int
    profile: str
    traversals: tuple[Traversal, ...]
    schema: str = FORMAT


@dataclass
class _Frame:
    hand: Hand
    candidates: BetCandidates
    probabilities: tuple[float, ...]
    indices: tuple[int, ...]
    values: list[float] = field(default_factory=list)


class CollectionLimitExceeded(RuntimeError):
    """The entire requested collection must be discarded, not its unfinished tail."""


def collect_traversal(
    hand: Hand,
    profile: FrozenProfile,
    traverser: int,
    *,
    iteration: int,
    action_seed: int,
    max_nodes: int = 10_000,
    deadline: float = float("inf"),
) -> Traversal:
    """Estimate values conditional on this root; collect_phase samples fresh roots."""
    if not isinstance(hand, Hand) or not isinstance(profile, FrozenProfile):
        raise TypeError("Collection needs a Hand and an isolated FrozenProfile")
    if type(traverser) is not int or not 0 <= traverser < len(hand.table.stacks):
        raise ValueError("Traverser must be a participant in this hand")
    if hand.table.capacity != profile.capacity:
        raise ValueError("Policy profile must cover the table's physical seats")
    if any(type(v) is not int or v < 1 for v in (iteration, max_nodes)):
        raise ValueError("Iteration and node budget must be positive integers")
    if type(action_seed) is not int or action_seed < 0 or isnan(deadline):
        raise ValueError("Invalid action seed or deadline")
    profile.assert_unchanged()
    root = hand.observe(traverser)
    random = Random(action_seed)
    targets, executions, stack = [], [], []
    nodes = terminals = 0

    def descend(frame):
        index = frame.indices[len(frame.values)]
        child = frame.hand.apply(frame.candidates.actions[index])
        event = child.events[len(frame.hand.events)]
        executions.append(record_execution(frame.candidates, index, event))
        return child

    with deterministic_cpu():
        while True:
            if nodes >= max_nodes or perf_counter() >= deadline:
                raise CollectionLimitExceeded(
                    f"Collection aborted at {root.hand_id}, traverser {traverser}, "
                    f"action seed {action_seed}, after {nodes} nodes; no batch returned"
                )
            nodes += 1
            if hand.finished:
                terminals += 1
                final = hand.observe(traverser).players[traverser]
                value = float(final.stack - final.starting_stack)
            else:
                candidates = bet_candidates(hand.observe(hand.actor))
                probabilities = profile.distribution(candidates)
                indices = (
                    tuple(range(len(candidates.actions)))
                    if hand.actor == traverser
                    else (
                        random.choices(
                            range(len(candidates.actions)), weights=probabilities, k=1
                        )[0],
                    )
                )
                frame = _Frame(hand, candidates, probabilities, indices)
                stack.append(frame)
                hand = descend(frame)
                continue

            # Explicit postorder traversal also handles histories beyond Python's call limit.
            while stack:
                frame = stack[-1]
                frame.values.append(value)
                if len(frame.values) < len(frame.indices):
                    hand = descend(frame)
                    break
                stack.pop()
                if frame.hand.actor == traverser:
                    target = action_targets(
                        frame.candidates,
                        frame.probabilities,
                        dict(zip(frame.candidates.actions, frame.values)),
                    )
                    targets.append(target)
                    value = fsum(
                        p * v for p, v in zip(frame.probabilities, frame.values)
                    )
                else:
                    value = frame.values[0]
            else:
                profile.assert_unchanged()
                return Traversal(
                    root,
                    iteration,
                    profile.fingerprint,
                    action_seed,
                    value / root.big_blind,
                    tuple(targets),
                    tuple(executions),
                    nodes,
                    terminals,
                )


def collect_phase(
    table: Table,
    profile: FrozenProfile,
    *,
    iteration: int,
    seed: int,
    traversals_per_player: int = 1,
    max_nodes: int = 50_000,
    max_seconds: float = 30,
) -> Collection:
    """Collect every participant against one frozen profile, with no fitting inside."""
    if not isinstance(table, Table) or not isinstance(profile, FrozenProfile):
        raise TypeError("Collection needs a Table and an isolated FrozenProfile")
    if table.capacity != profile.capacity:
        raise ValueError("Policy profile must cover the table's physical seats")
    if any(
        type(v) is not int or v < 1
        for v in (iteration, traversals_per_player, max_nodes)
    ):
        raise ValueError("Iteration, traversal count and node budget must be positive")
    if (
        type(seed) is not int
        or seed < 0
        or not isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("Provide a nonnegative seed and a positive finite time budget")
    profile.assert_unchanged()
    deadline = perf_counter() + max_seconds
    results = []
    remaining = max_nodes
    for traverser, physical in enumerate(table.seat_numbers):
        for sample in range(traversals_per_player):
            if remaining <= 0:
                raise CollectionLimitExceeded(
                    "Phase node budget exhausted; no batch returned"
                )
            # Public hand IDs describe the schedule; they never encode a deal seed.
            hand = Hand.start(
                table,
                hand_id=f"collection-{iteration}-{physical}-{sample}",
                seed=collection_seed(seed, iteration, physical, sample, "deal"),
            )
            result = collect_traversal(
                hand,
                profile,
                traverser,
                iteration=iteration,
                action_seed=collection_seed(
                    seed, iteration, physical, sample, "opponents"
                ),
                max_nodes=remaining,
                deadline=deadline,
            )
            remaining -= result.nodes
            results.append(result)
    profile.assert_unchanged()
    return Collection(
        table,
        iteration,
        seed,
        traversals_per_player,
        profile.fingerprint,
        tuple(results),
    )
