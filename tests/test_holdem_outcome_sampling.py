import random
from collections import defaultdict
from math import fsum

import pytest
import torch

from src.game.hand import Hand
from src.game.types import Street
from src.holdem import collection, outcome_sampling
from src.holdem.actions import bet_candidates
from src.holdem.collection import CollectionLimitExceeded, collect_traversal
from src.holdem.outcome_sampling import collect_outcome
from src.holdem.policy import FrozenProfile
from tests.test_hand_observations import table
from tests.test_holdem_encoding import advance
from tests.test_holdem_policy import folding_model


def river():
    return advance(Hand.start(table(2, (3, 3)), hand_id="exact", seed=7), Street.RIVER)


def exact_tree(hand, profile, traverser):
    updates = {}

    def visit(node, opponent_reach):
        if node.finished:
            return (
                node.events[-1].stacks[traverser] - node.table.stacks[traverser]
            ) / node.table.big_blind
        candidates = bet_candidates(node.observe(node.actor))
        policy = profile.distribution(candidates)
        values = [
            visit(node.apply(a), opponent_reach * (p if node.actor != traverser else 1))
            for a, p in zip(candidates.actions, policy)
        ]
        value = fsum(p * v for p, v in zip(policy, values))
        if node.actor == traverser:
            updates[node.events] = tuple(opponent_reach * (v - value) for v in values)
        return value

    return visit(hand, 1.0), updates


def enumerate_samples(monkeypatch, module, collect, hand, profile, traverser, **kwargs):
    class NeedChoice(Exception):
        def __init__(self, weights):
            self.weights = weights

    class ScriptRandom:
        def __init__(self, seed):
            self.offset = 0

        def choices(self, population, weights, k):
            assert k == 1
            if self.offset == len(script):
                raise NeedChoice(weights)
            choice = script[self.offset]
            self.offset += 1
            return [population[choice]]

    completed = []
    with monkeypatch.context() as patch:
        patch.setattr(module, "Random", ScriptRandom)
        pending = [((), 1.0)]
        while pending:
            script, probability = pending.pop()
            try:
                sample = collect(
                    hand, profile, traverser, iteration=1, action_seed=0, **kwargs
                )
            except NeedChoice as needed:
                pending.extend(
                    (script + (a,), probability * p)
                    for a, p in enumerate(needed.weights)
                    if p > 0
                )
            else:
                completed.append((probability, sample))
    assert fsum(p for p, _ in completed) == pytest.approx(1, abs=1e-12, rel=0)
    return completed


def moments(samples, outcome):
    means, seconds = defaultdict(float), defaultdict(float)
    for p, sample in samples:
        records = sample.decisions if outcome else sample.targets
        for record in records:
            updates = record.regret_updates_bb if outcome else record.regrets_bb
            for a, value in enumerate(updates):
                key = (record.candidates.decision.source.history, a)
                means[key] += p * value
                seconds[key] += p * value * value
    return means, {key: seconds[key] - value * value for key, value in means.items()}


@pytest.mark.parametrize("traverser", [0, 1])
@pytest.mark.parametrize("exploration", [0.5, 1.0])
@pytest.mark.parametrize("policy_kind", ["uniform", "nonuniform", "zero-reach"])
def test_every_counterfactual_update_matches_exact_tree(
    monkeypatch, traverser, exploration, policy_kind
):
    hand = river()
    profile = FrozenProfile([None] * 2)
    if policy_kind != "uniform":

        def distribution(self, candidates):
            count = len(candidates.actions)
            if (
                policy_kind == "zero-reach"
                and candidates.decision.source.seat == traverser
            ):
                return (1.0,) + (0.0,) * (count - 1)
            total = count * (count + 1) / 2
            return tuple((a + 1) / total for a in range(count))

        monkeypatch.setattr(FrozenProfile, "distribution", distribution)
    expected_value, expected = exact_tree(hand, profile, traverser)
    samples = enumerate_samples(
        monkeypatch,
        outcome_sampling,
        collect_outcome,
        hand,
        profile,
        traverser,
        exploration=exploration,
    )
    means, _ = moments(samples, True)
    assert fsum(p * s.value_bb for p, s in samples) == pytest.approx(
        expected_value, abs=1e-12, rel=0
    )
    assert set(means) == {(h, a) for h, v in expected.items() for a in range(len(v))}
    for history, values in expected.items():
        for a, value in enumerate(values):
            assert means[history, a] == pytest.approx(value, abs=1e-12, rel=0)
    # This fixture contains later decisions, so missing own-prefix correction is detectable.
    if traverser == hand.actor:
        assert any(d.own_sample_reach < 1 for _, s in samples for d in s.decisions)


def test_exact_variance_comparison_includes_unvisited_decisions(monkeypatch):
    hand, profile = river(), FrozenProfile([None] * 2)
    external = enumerate_samples(
        monkeypatch, collection, collect_traversal, hand, profile, hand.actor
    )
    outcome = enumerate_samples(
        monkeypatch,
        outcome_sampling,
        collect_outcome,
        hand,
        profile,
        hand.actor,
        exploration=1.0,
    )
    external_mean, external_var = moments(external, False)
    outcome_mean, outcome_var = moments(outcome, True)
    assert outcome_mean == pytest.approx(external_mean, abs=1e-12, rel=0)
    assert sum(outcome_var.values()) > sum(external_var.values())
    assert sum(p * s.nodes for p, s in outcome) < sum(p * s.nodes for p, s in external)


@pytest.mark.parametrize("n", [4, 5, 6])
def test_neural_paths_are_reproducible_and_keep_public_policy_inputs(n):
    hand = Hand.start(
        table(n, tuple(3 + 4 * i for i in range(n))), hand_id="paths", seed=19
    )
    profile = FrozenProfile([folding_model()] * n)
    before = tuple(hand.observe(p) for p in range(n))
    rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    args = {"iteration": 2, "action_seed": 7, "exploration": 0.5}
    first = collect_outcome(hand, profile, hand.actor, **args)
    assert first == collect_outcome(hand, profile, hand.actor, **args)
    assert rng == random.getstate() and torch.equal(torch_rng, torch.get_rng_state())
    assert tuple(hand.observe(p) for p in range(n)) == before
    assert first.nodes == len(first.executions) + 1
    for e in first.executions:
        view = e.candidates.decision.source
        assert view.hole_cards == before[view.seat].hole_cards
        assert view.seat == view.actor == e.event.seat
        view.legal_actions.validate(e.event.action)
    for d in first.decisions:
        assert min(d.sampling_policy) > 0
        assert fsum(
            p * v for p, v in zip(d.policy, d.regret_updates_bb)
        ) == pytest.approx(0)


@pytest.mark.parametrize("kwargs", [{"max_nodes": 1}, {"deadline": 0}])
def test_limits_never_return_partial_estimates(kwargs):
    hand = river()
    profile = FrozenProfile([None] * 2)
    with pytest.raises(CollectionLimitExceeded, match="no sample returned"):
        collect_outcome(
            hand,
            profile,
            hand.actor,
            iteration=1,
            action_seed=0,
            exploration=1,
            **kwargs,
        )
    profile.assert_unchanged()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exploration": 0},
        {"exploration": float("nan")},
        {"exploration": 1.1},
        {"exploration": True},
        {"action_seed": -1},
        {"iteration": 0},
        {"max_nodes": True},
        {"deadline": float("nan")},
    ],
)
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        collect_outcome(
            river(),
            FrozenProfile([None] * 2),
            0,
            **dict({"iteration": 1, "action_seed": 0, "exploration": 0.5}, **kwargs),
        )


def test_profile_mutation_is_rejected(monkeypatch):
    profile = FrozenProfile([folding_model()] * 2)
    original = FrozenProfile.distribution

    def changed(self, candidates):
        result = original(self, candidates)
        with torch.no_grad():
            self._models[0].regret.bias.add_(0.01)
        return result

    monkeypatch.setattr(FrozenProfile, "distribution", changed)
    with pytest.raises(RuntimeError, match="changed"):
        collect_outcome(river(), profile, 0, iteration=1, action_seed=0, exploration=1)


def test_terminal_root_preserves_each_roles_payoff():
    hand = river()
    while not hand.finished:
        hand = hand.apply(bet_candidates(hand.observe(hand.actor)).actions[-1])
    profile = FrozenProfile([None] * 2)
    for seat in range(2):
        result = collect_outcome(
            hand, profile, seat, iteration=1, action_seed=0, exploration=1
        )
        assert (
            result.value_bb
            == (hand.events[-1].stacks[seat] - hand.table.stacks[seat])
            / hand.table.big_blind
        )
        assert result.nodes == 1 and not result.decisions and not result.executions
