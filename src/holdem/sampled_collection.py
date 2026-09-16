"""Complete first-decision branching phases and typed replay admission."""

from dataclasses import dataclass
from math import isclose, isfinite, prod
from time import perf_counter

from src.game.hand import Hand, Table
from src.holdem.actions import record_execution
from src.holdem.collection import (
    Collection,
    CollectionLimitExceeded,
    Traversal,
    collection_seed,
)
from src.holdem.outcome_sampling import FORMAT, OutcomeTraversal, collect_outcome
from src.holdem.policy import FrozenProfile
from src.holdem.replay import SampledReplaySample, split_collection
from src.holdem.targets import CandidateTargets


@dataclass(frozen=True, slots=True)
class SampledCollection:
    table: Table
    iteration: int
    seed: int
    traversals_per_player: int
    profile: str
    traversals: tuple[OutcomeTraversal, ...]
    exploration: float


def collect_sampled_phase(
    table,
    profile,
    *,
    iteration,
    seed,
    traversals_per_player=1,
    max_nodes=50_000,
    max_seconds=60,
    exploration=0.5,
):
    if not isinstance(table, Table) or not isinstance(profile, FrozenProfile):
        raise TypeError("Collection needs a table and frozen profile")
    if table.capacity != profile.capacity:
        raise ValueError("Profile capacity differs from the table")
    if any(
        type(n) is not int or n < 1
        for n in (iteration, traversals_per_player, max_nodes)
    ):
        raise ValueError("Collection counts must be positive integers")
    if (
        type(seed) is not int
        or seed < 0
        or not isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("Invalid collection seed or time budget")
    profile.assert_unchanged()
    deadline, remaining, traversals = perf_counter() + max_seconds, max_nodes, []
    for seat, role in enumerate(table.seat_numbers):
        for sample in range(traversals_per_player):
            if remaining <= 0:
                raise CollectionLimitExceeded(
                    "Phase node budget exhausted; no batch returned"
                )
            hand = Hand.start(
                table,
                hand_id=f"collection-{iteration}-{role}-{sample}",
                seed=collection_seed(seed, iteration, role, sample, "deal"),
            )
            traversal = collect_outcome(
                hand,
                profile,
                seat,
                iteration=iteration,
                action_seed=collection_seed(seed, iteration, role, sample, "opponents"),
                exploration=exploration,
                baseline="frozen",
                branch_first=True,
                max_nodes=remaining,
                deadline=deadline,
            )
            remaining -= traversal.nodes
            traversals.append(traversal)
    profile.assert_unchanged()
    return SampledCollection(
        table,
        iteration,
        seed,
        traversals_per_player,
        profile.fingerprint,
        tuple(traversals),
        exploration,
    )


def split_sampled_collection(batch):
    if not isinstance(batch, SampledCollection):
        raise TypeError("Expected a sampled collection")
    converted = []
    for traversal in batch.traversals:
        if (
            traversal.schema != FORMAT
            or traversal.branch_second
            or traversal.baseline != "frozen"
            or traversal.branch_first is not True
            or traversal.exploration != batch.exploration
        ):
            raise ValueError("Traversal uses another sampling design")
        by_history = {
            d.candidates.decision.source.history: d for d in traversal.decisions
        }
        if sum(d.sampled_action is None for d in traversal.decisions) != bool(
            by_history
        ):
            raise ValueError("Expected one expanded first decision per nonempty root")
        for decision in traversal.decisions:
            history = decision.candidates.decision.source.history
            expected_q = (
                (1.0,) * len(decision.policy)
                if decision.sampled_action is None
                else tuple(
                    (1 - batch.exploration) * p
                    + batch.exploration / len(decision.policy)
                    for p in decision.policy
                )
            )
            if decision.inclusion_probabilities != expected_q:
                raise ValueError("Inclusion probabilities disagree with exploration")
            edges = [
                e
                for e in traversal.executions
                if e.candidates.decision.source.history == history
            ]
            expected_indices = (
                set(range(len(decision.policy)))
                if decision.sampled_action is None
                else {decision.sampled_action}
            )
            if {e.index for e in edges} != expected_indices or len(edges) != len(
                expected_indices
            ):
                raise ValueError("Sampled decisions disagree with executed branches")
            for edge in edges:
                record_execution(edge.candidates, edge.index, edge.event)
            prefix = prod(
                by_history[
                    e.candidates.decision.source.history
                ].inclusion_probabilities[e.index]
                for e in traversal.executions
                if e.event.seat == traversal.root.seat
                and len(e.candidates.decision.source.history) < len(history)
                and history[: len(e.candidates.decision.source.history) + 1]
                == e.candidates.decision.source.history + (e.event,)
            )
            if not isclose(decision.own_sample_reach, prefix, rel_tol=1e-12, abs_tol=0):
                raise ValueError("Own sampling reach disagrees with executed history")
        converted.append(
            Traversal(
                traversal.root,
                traversal.iteration,
                traversal.profile,
                traversal.action_seed,
                traversal.value_bb,
                tuple(
                    CandidateTargets(d.candidates, d.policy, d.values_bb, d.regrets_bb)
                    for d in traversal.decisions
                ),
                traversal.executions,
                traversal.nodes,
                traversal.terminals,
            )
        )
    # Reuse the schedule, ownership and complete-history checks, never its loss.
    checked = split_collection(
        Collection(
            batch.table,
            batch.iteration,
            batch.seed,
            batch.traversals_per_player,
            batch.profile,
            tuple(converted),
        )
    )
    decisions = {
        (t.action_seed, index): d
        for t in batch.traversals
        for index, d in enumerate(t.decisions)
    }
    roles = []
    for samples in checked:
        items = tuple(
            SampledReplaySample(
                s.role,
                s.iteration,
                s.profile,
                s.action_seed,
                s.target_index,
                decisions[s.action_seed, s.target_index],
                batch.traversals_per_player,
            )
            for s in samples
        )
        for item in items:
            item.validate()
        roles.append(items)
    return tuple(roles)
