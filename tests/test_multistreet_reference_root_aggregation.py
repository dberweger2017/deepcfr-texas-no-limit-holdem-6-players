"""Equivalence checks for the root-only multi-street reference accumulator."""

from __future__ import annotations

import json
from math import fsum
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from src.holdem.actions import bet_candidates
from src.holdem.multistreet_reference import (
    MultiStreetReference,
    build_context,
    enumerate_reference,
)
from src.holdem.representation_reference import range_support
from src.holdem.river_reference import ReferenceProfile, check_deadline
from src.holdem.targets import CandidateTargets


def _original_enumerate_reference(context, profile, *, max_nodes, deadline):
    """Test-local copy of the pre-optimization infoset accumulator."""

    started = perf_counter()
    hero = context.worlds[0].actor
    totals, candidates_by_view = {}, {}
    per_world = []
    total_nodes = 0
    for root in context.worlds:
        nodes = 0
        root_values = None

        def visit(node, reach, root=root):
            nonlocal nodes, root_values
            check_deadline(deadline)
            nodes += 1
            if nodes > max_nodes:
                raise RuntimeError("Multi-street reference exceeded its node budget")
            if node.finished:
                stacks = node.events[-1].stacks
                return (stacks[hero] - node.table.stacks[hero]) / node.table.big_blind
            view = node.observe(node.actor)
            candidates = bet_candidates(view)
            probs = profile.distribution(candidates)
            values = tuple(
                visit(node.apply(action), reach * (p if node.actor != hero else 1.0))
                for action, p in zip(candidates.actions, probs, strict=True)
            )
            if node is root and node.actor == hero:
                root_values = values
            if node.actor == hero:
                if (
                    view in candidates_by_view
                    and candidates_by_view[view].actions != candidates.actions
                ):
                    raise ValueError("One information set has inconsistent actions")
                candidates_by_view[view] = candidates
                weight = reach / len(context.worlds)
                mass, sums = totals.setdefault(view, [0.0, np.zeros(len(values))])
                totals[view][0] = mass + weight
                sums += weight * np.asarray(values)
            return fsum(p * value for p, value in zip(probs, values))

        visit(root, 1.0)
        if root_values is None:
            raise ValueError("Reference root was not a hero decision")
        per_world.append(tuple(root_values))
        total_nodes += nodes

    root_view = context.worlds[0].observe(hero)
    candidates = candidates_by_view[root_view]
    values = tuple(totals[root_view][1] / totals[root_view][0])
    probs = profile.distribution(candidates)
    center = fsum(p * v for p, v in zip(probs, values))
    target = CandidateTargets(candidates, probs, values, tuple(v - center for v in values))
    arr = np.asarray(per_world, dtype=float)
    se = (
        tuple(float(x) for x in arr.std(axis=0, ddof=1) / np.sqrt(len(arr)))
        if len(arr) > 1
        else (0.0,) * len(values)
    )
    return MultiStreetReference(
        target,
        tuple(tuple(float(v) for v in row) for row in per_world),
        se,
        "estimated" if len(arr) > 1 else "insufficient_worlds",
        total_nodes,
        len(context.worlds),
        perf_counter() - started,
    )


@pytest.fixture(scope="module")
def support():
    plan = json.loads(Path("configs/holdem/representation.json").read_text())
    return range_support(plan["range_templates"])


@pytest.mark.parametrize(
    ("street", "board"),
    [
        ("flop", ("Ac", "Kd", "7h")),
        ("turn", ("Ac", "Kd", "7h", "4s")),
    ],
)
@pytest.mark.parametrize("facing", [False, True])
def test_root_only_reference_matches_original_infoset_accumulator(
    support, street, board, facing
):
    context = build_context(
        name=f"root-aggregation-{street}",
        split="train",
        street=street,
        board=board,
        holding=("Qs", "Jc"),
        support=support,
        samples=2,
        deals_per_sample=1,
        seed=173,
        facing=facing,
    )
    profile = ReferenceProfile("increasing")
    expected = _original_enumerate_reference(
        context, profile, max_nodes=100_000, deadline=float("inf")
    )
    actual = enumerate_reference(
        context, profile, max_nodes=100_000, deadline=float("inf")
    )

    assert actual.target.candidates.actions == expected.target.candidates.actions
    assert actual.target.policy == expected.target.policy
    assert actual.target.values_bb == expected.target.values_bb
    assert actual.target.regrets_bb == expected.target.regrets_bb
    assert actual.world_action_values_bb == expected.world_action_values_bb
    assert actual.action_standard_error_bb == expected.action_standard_error_bb
    assert actual.uncertainty_status == expected.uncertainty_status
    assert actual.nodes == expected.nodes
    assert actual.worlds == expected.worlds
