from dataclasses import replace
from itertools import combinations, product
from math import fsum

import pytest
import torch

from src.holdem import outcome_sampling
from src.holdem.actions import bet_candidates
from src.holdem.betting import ActionScores, betting_loss
from src.holdem.outcome_sampling import collect_outcome
from src.holdem.policy import FrozenProfile
from src.holdem.sampled_loss import sampled_betting_loss
from src.holdem.targets import CandidateTargets
from tests.test_holdem_outcome_sampling import enumerate_samples, moments, river


def full_tree(hand, profile, traverser):
    records = {}

    def visit(node, opponent_reach):
        if node.finished:
            return (
                node.events[-1].stacks[traverser] - node.table.stacks[traverser]
            ) / node.table.big_blind
        candidates = bet_candidates(node.observe(node.actor))
        policy = profile.distribution(candidates)
        values = tuple(
            visit(node.apply(a), opponent_reach * (p if node.actor != traverser else 1))
            for a, p in zip(candidates.actions, policy)
        )
        value = fsum(p * v for p, v in zip(policy, values))
        if node.actor == traverser:
            records[node.events] = (
                CandidateTargets(
                    candidates, policy, values, tuple(v - value for v in values)
                ),
                opponent_reach,
            )
        return value

    return visit(hand, 1), records


def predictions(candidates, parameter):
    features = torch.arange(1, len(candidates.actions) + 1, dtype=torch.float64)
    features = features + len(candidates.decision.source.history) / 10
    return ActionScores(
        candidates,
        parameter[0] * features + parameter[1],
        parameter[2] * features - parameter[1],
    )


def gradient(decisions, *, roots=1, population=None, iteration_weight=1):
    parameter = torch.tensor([0.7, -0.3, 1.1], dtype=torch.float64, requires_grad=True)
    if not decisions:
        return torch.zeros_like(parameter)
    scores = [predictions(d.candidates, parameter) for d in decisions]
    loss = sampled_betting_loss(
        scores,
        decisions,
        roots=roots,
        population=population,
        iteration_weight=iteration_weight,
    )
    return torch.autograd.grad(loss, parameter)[0]


@pytest.mark.parametrize("seat", [0, 1])
@pytest.mark.parametrize("branch", [False, True])
@pytest.mark.parametrize("baseline", ["zero", "inaccurate", "exact"])
@pytest.mark.parametrize("policy_kind", ["uniform", "nonuniform", "zero-reach"])
def test_expected_gradient_and_updates_match_full_tree(
    monkeypatch, seat, branch, baseline, policy_kind
):
    hand, profile = river(), FrozenProfile([None] * 2)
    if policy_kind != "uniform":

        def distribution(self, candidates):
            count = len(candidates.actions)
            if policy_kind == "zero-reach" and candidates.decision.source.seat == seat:
                return (1.0,) + (0.0,) * (count - 1)
            return tuple((a + 1) / (count * (count + 1) / 2) for a in range(count))

        monkeypatch.setattr(FrozenProfile, "distribution", distribution)
    exact_value, records = full_tree(hand, profile, seat)

    def action_values(self, candidates):
        if baseline == "exact":
            return records[candidates.decision.source.history][0].values_bb
        return tuple(0.3 - 0.7 * a for a in range(len(candidates.actions)))

    monkeypatch.setattr(FrozenProfile, "action_values", action_values)
    samples = enumerate_samples(
        monkeypatch,
        outcome_sampling,
        collect_outcome,
        hand,
        profile,
        seat,
        exploration=0.5,
        baseline="zero" if baseline == "zero" else "frozen",
        branch_first=branch,
    )
    parameter = torch.tensor([0.7, -0.3, 1.1], dtype=torch.float64, requires_grad=True)
    exact_loss = sum(
        reach * betting_loss([predictions(t.candidates, parameter)], [t])
        for t, reach in records.values()
    )
    expected = torch.autograd.grad(exact_loss, parameter)[0]
    actual = sum(p * gradient(s.decisions) for p, s in samples)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert fsum(p * s.value_bb for p, s in samples) == pytest.approx(
        exact_value, abs=1e-12, rel=0
    )
    means, _ = moments(samples, True)
    for history, (target, reach) in records.items():
        for a, regret in enumerate(target.regrets_bb):
            assert means[history, a] == pytest.approx(reach * regret, abs=1e-12, rel=0)
    if branch:
        for _, sample in samples:
            expanded = [d for d in sample.decisions if d.sampled_action is None]
            assert len(expanded) <= 1
            assert sample.terminals <= (len(expanded[0].policy) if expanded else 1)
            assert all(q == 1 for d in expanded for q in d.inclusion_probabilities)


def test_perfect_baseline_removes_action_sampling_noise_at_a_terminal_decision(
    monkeypatch,
):
    hand = river()
    hand = hand.apply(bet_candidates(hand.observe(hand.actor)).actions[-1])
    profile, seat = FrozenProfile([None] * 2), hand.actor
    _, records = full_tree(hand, profile, seat)
    assert len(records) == 1
    monkeypatch.setattr(
        FrozenProfile,
        "action_values",
        lambda self, c: records[c.decision.source.history][0].values_bb,
    )
    raw = enumerate_samples(
        monkeypatch,
        outcome_sampling,
        collect_outcome,
        hand,
        profile,
        seat,
        exploration=0.5,
    )
    corrected = enumerate_samples(
        monkeypatch,
        outcome_sampling,
        collect_outcome,
        hand,
        profile,
        seat,
        exploration=0.5,
        baseline="frozen",
    )
    assert sum(moments(raw, True)[1].values()) > 0
    assert sum(moments(corrected, True)[1].values()) == pytest.approx(0, abs=1e-12)
    assert all(
        s.decisions[0].values_bb == records[hand.events][0].values_bb
        for _, s in corrected
    )


def test_uniform_reservoir_and_minibatches_preserve_the_root_normalized_gradient(
    monkeypatch,
):
    hand, profile = river(), FrozenProfile([None] * 2)
    paths = enumerate_samples(
        monkeypatch,
        outcome_sampling,
        collect_outcome,
        hand,
        profile,
        hand.actor,
        exploration=0.5,
    )
    records = [d for _, path in paths for d in path.decisions]
    expected = gradient(records, roots=11, iteration_weight=3)
    estimates = []
    for reservoir in combinations(range(len(records)), 2):
        for indices in product(reservoir, repeat=2):
            estimates.append(
                gradient(
                    [records[i] for i in indices],
                    roots=11,
                    population=len(records),
                    iteration_weight=3,
                )
            )
    torch.testing.assert_close(
        torch.stack(estimates).mean(0), expected, atol=1e-12, rtol=0
    )
    # Empty scheduled roots change the denominator even though they create no records.
    torch.testing.assert_close(
        expected * 11, gradient(records, iteration_weight=3), atol=1e-12, rtol=0
    )


def test_weight_is_not_a_multiplier_on_the_regression_label(monkeypatch):
    hand, profile = river(), FrozenProfile([None] * 2)
    paths = enumerate_samples(
        monkeypatch,
        outcome_sampling,
        collect_outcome,
        hand,
        profile,
        hand.actor,
        exploration=0.5,
    )
    record = next(d for _, p in paths for d in p.decisions if d.own_sample_reach < 1)
    wrong = replace(
        record,
        values_bb=tuple(v / record.own_sample_reach for v in record.values_bb),
        own_sample_reach=1,
    )
    assert not torch.allclose(gradient([record]), gradient([wrong]))


@pytest.mark.parametrize(
    "changes",
    [
        {"own_sample_reach": 0},
        {"own_sample_reach": float("nan")},
        {"policy": (1.0,)},
        {"policy": (float("nan"),) * 2},
    ],
)
def test_loss_rejects_invalid_measure(changes):
    hand, profile = river(), FrozenProfile([None] * 2)
    result = collect_outcome(
        hand, profile, hand.actor, iteration=1, action_seed=0, exploration=0.5
    )
    with pytest.raises(ValueError):
        gradient([replace(result.decisions[0], **changes)])


def test_branch_limit_discards_the_whole_traversal_and_profile_is_reusable():
    from src.holdem.collection import CollectionLimitExceeded

    hand, profile = river(), FrozenProfile([None] * 2)
    args = {
        "iteration": 1,
        "action_seed": 0,
        "exploration": 0.5,
        "branch_first": True,
        "baseline": "frozen",
    }
    complete = collect_outcome(hand, profile, hand.actor, **args)
    with pytest.raises(CollectionLimitExceeded, match="no sample returned"):
        collect_outcome(hand, profile, hand.actor, max_nodes=complete.nodes - 1, **args)
    assert complete == collect_outcome(hand, profile, hand.actor, **args)
