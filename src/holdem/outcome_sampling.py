"""Outcome traversal estimates; sampled replay supplies their training normalization."""

from dataclasses import dataclass, field
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

FORMAT = "holdem-outcome-sampling-diagnostic-v2"


@dataclass(frozen=True, slots=True)
class SampledDecision:
    candidates: BetCandidates
    policy: tuple[float, ...]
    inclusion_probabilities: tuple[float, ...]
    sampled_action: int | None
    own_sample_reach: float
    values_bb: tuple[float, ...]
    baselines_bb: tuple[float, ...]

    @property
    def regrets_bb(self) -> tuple[float, ...]:
        value = fsum(p * v for p, v in zip(self.policy, self.values_bb))
        return tuple(v - value for v in self.values_bb)

    @property
    def regret_updates_bb(self) -> tuple[float, ...]:
        return tuple(r / self.own_sample_reach for r in self.regrets_bb)


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
    terminals: int
    baseline: str
    branch_first: bool
    schema: str = FORMAT


@dataclass
class _Frame:
    hand: Hand
    candidates: BetCandidates
    policy: tuple[float, ...]
    inclusion: tuple[float, ...]
    indices: tuple[int, ...]
    prefix: float
    baseline: tuple[float, ...]
    updating: bool
    expand: bool
    may_expand: bool
    values: list[float] = field(default_factory=list)


def collect_outcome(
    hand: Hand,
    profile: FrozenProfile,
    traverser: int,
    *,
    iteration: int,
    action_seed: int,
    exploration: float,
    baseline: str = "zero",
    branch_first: bool = False,
    max_nodes: int = 10_000,
    deadline: float = float("inf"),
) -> OutcomeTraversal:
    """Sample paths, optionally expanding the first own decision, with frozen baselines."""
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
    if baseline not in ("zero", "frozen") or type(branch_first) is not bool:
        raise ValueError("Choose zero/frozen baseline and a boolean branching flag")
    profile.assert_unchanged()
    root = hand.observe(traverser)
    random = Random(action_seed)
    frames, executions, decisions = [], [], []
    reach, may_expand = 1.0, branch_first
    nodes = terminals = 0

    def descend(frame):
        index = frame.indices[len(frame.values)]
        prefix = frame.prefix
        if frame.updating and not frame.expand:
            prefix *= frame.inclusion[index]
        if prefix == 0 or not isfinite(1 / prefix):
            raise FloatingPointError("Own sampling reach underflowed")
        child = frame.hand.apply(frame.candidates.actions[index])
        executions.append(
            record_execution(
                frame.candidates, index, child.events[len(frame.hand.events)]
            )
        )
        return child, prefix, frame.may_expand and not frame.updating

    with deterministic_cpu():
        while True:
            if nodes >= max_nodes or perf_counter() >= deadline:
                raise CollectionLimitExceeded(
                    f"Outcome sampling aborted at {root.hand_id}, traverser {traverser}, "
                    f"action seed {action_seed}, after {nodes} nodes; no sample returned"
                )
            nodes += 1
            if hand.finished:
                terminals += 1
                final = hand.observe(traverser).players[traverser]
                value = (final.stack - final.starting_stack) / root.big_blind
            else:
                candidates = bet_candidates(hand.observe(hand.actor))
                policy = profile.distribution(candidates)
                updating = hand.actor == traverser
                expand = updating and may_expand
                sampling = (
                    tuple(
                        (1 - exploration) * p + exploration / len(policy)
                        for p in policy
                    )
                    if updating
                    else policy
                )
                bases = (
                    profile.action_values(candidates)
                    if updating and baseline == "frozen"
                    else (0.0,) * len(policy)
                )
                if len(bases) != len(policy) or not all(isfinite(b) for b in bases):
                    raise FloatingPointError("Invalid action baseline")
                indices = (
                    tuple(range(len(policy)))
                    if expand
                    else (random.choices(range(len(policy)), weights=sampling, k=1)[0],)
                )
                frame = _Frame(
                    hand,
                    candidates,
                    policy,
                    (1.0,) * len(policy) if expand else sampling,
                    indices,
                    reach,
                    bases,
                    updating,
                    expand,
                    may_expand,
                )
                frames.append(frame)
                hand, reach, may_expand = descend(frame)
                continue

            while frames:
                frame = frames[-1]
                frame.values.append(value)
                if len(frame.values) < len(frame.indices):
                    hand, reach, may_expand = descend(frame)
                    break
                frames.pop()
                if not frame.updating:
                    value = frame.values[0]
                    continue
                if frame.expand:
                    values = tuple(frame.values)
                    sampled_action = None
                else:
                    sampled_action = frame.indices[0]
                    # Baselines are fixed before the draw; only the residual is reweighted.
                    values = tuple(
                        b + (value - b) / frame.inclusion[a]
                        if a == sampled_action
                        else b
                        for a, b in enumerate(frame.baseline)
                    )
                value = fsum(p * v for p, v in zip(frame.policy, values))
                decision = SampledDecision(
                    frame.candidates,
                    frame.policy,
                    frame.inclusion,
                    sampled_action,
                    frame.prefix,
                    values,
                    frame.baseline,
                )
                if not all(
                    isfinite(v) for v in (*values, value, *decision.regret_updates_bb)
                ):
                    raise FloatingPointError("Non-finite outcome sampling estimate")
                decisions.append(decision)
            else:
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
                    terminals,
                    baseline,
                    branch_first,
                )
