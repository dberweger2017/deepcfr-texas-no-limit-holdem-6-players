"""Diagnostic outcome sampling; records are not compatible with training replay."""

from dataclasses import dataclass
from math import fsum, isfinite, isnan
from random import Random
from time import perf_counter

from src.game.hand import Hand
from src.game.observation import Observation
from src.holdem.actions import (
    BetCandidates,
    ExecutedBet,
    bet_candidates,
    record_execution,
)
from src.holdem.collection import CollectionLimitExceeded
from src.holdem.policy import FrozenProfile
from src.solver.neural.network import deterministic_cpu

FORMAT = "holdem-outcome-sampling-diagnostic-v1"


@dataclass(frozen=True, slots=True)
class SampledDecision:
    candidates: BetCandidates
    policy: tuple[float, ...]
    sampling_policy: tuple[float, ...]
    sampled_action: int
    own_sample_reach: float
    values_bb: tuple[float, ...]
    regret_updates_bb: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OutcomeTraversal:
    root: Observation
    iteration: int
    profile: str
    action_seed: int
    exploration: float
    value_bb: float
    decisions: tuple[SampledDecision, ...]
    executions: tuple[ExecutedBet, ...]
    nodes: int
    schema: str = FORMAT


def collect_outcome(
    hand: Hand,
    profile: FrozenProfile,
    traverser: int,
    *,
    iteration: int,
    action_seed: int,
    exploration: float,
    max_nodes: int = 10_000,
    deadline: float = float("inf"),
) -> OutcomeTraversal:
    """Sample one terminal path and correct updates for own prefix and suffix reach."""
    if not isinstance(hand, Hand) or not isinstance(profile, FrozenProfile):
        raise TypeError("Sampling needs a Hand and an isolated FrozenProfile")
    if type(traverser) is not int or not 0 <= traverser < len(hand.table.stacks):
        raise ValueError("Traverser must be a participant in this hand")
    if hand.table.capacity != profile.capacity:
        raise ValueError("Policy profile must cover the table's physical seats")
    if any(type(v) is not int or v < 1 for v in (iteration, max_nodes)):
        raise ValueError("Iteration and node budget must be positive integers")
    if type(action_seed) is not int or action_seed < 0 or isnan(deadline):
        raise ValueError("Invalid action seed or deadline")
    if type(exploration) not in (int, float) or not 0 < exploration <= 1:
        raise ValueError("Exploration must be in (0, 1]")
    profile.assert_unchanged()
    root = hand.observe(traverser)
    random = Random(action_seed)
    frames, executions = [], []
    reach = 1.0
    nodes = 0
    with deterministic_cpu():
        while True:
            if nodes >= max_nodes or perf_counter() >= deadline:
                raise CollectionLimitExceeded(
                    f"Outcome sampling aborted at {root.hand_id}, traverser {traverser}, "
                    f"action seed {action_seed}, after {nodes} nodes; no sample returned"
                )
            nodes += 1
            if hand.finished:
                final = hand.observe(traverser).players[traverser]
                value = (final.stack - final.starting_stack) / root.big_blind
                break
            candidates = bet_candidates(hand.observe(hand.actor))
            policy = profile.distribution(candidates)
            updating = hand.actor == traverser
            sampling = (
                tuple((1 - exploration) * p + exploration / len(policy) for p in policy)
                if updating
                else policy
            )
            index = random.choices(range(len(policy)), weights=sampling, k=1)[0]
            if updating:
                frames.append((candidates, policy, sampling, index, reach))
                reach *= sampling[index]
                if reach == 0 or not isfinite(1 / reach):
                    raise FloatingPointError("Own sampling reach underflowed")
            child = hand.apply(candidates.actions[index])
            executions.append(
                record_execution(candidates, index, child.events[len(hand.events)])
            )
            hand = child

        decisions = []
        # Opponents use their target policy, so their likelihood ratios cancel.
        for candidates, policy, sampling, index, prefix in reversed(frames):
            values = tuple(
                value / sampling[index] if a == index else 0.0
                for a in range(len(policy))
            )
            value = fsum(p * v for p, v in zip(policy, values))
            updates = tuple((v - value) / prefix for v in values)
            if not all(isfinite(v) for v in (*values, value, *updates)):
                raise FloatingPointError("Non-finite outcome sampling estimate")
            decisions.append(
                SampledDecision(
                    candidates, policy, sampling, index, prefix, values, updates
                )
            )
        profile.assert_unchanged()
        return OutcomeTraversal(
            root,
            iteration,
            profile.fingerprint,
            action_seed,
            exploration,
            value,
            tuple(decisions),
            tuple(executions),
            nodes,
        )
