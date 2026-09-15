import itertools
import random

import numpy as np
import pytest

from src.solver.cfr import CFR, normalize, regret_delta, sampled_delta
from src.solver.evaluate import best_response, evaluate
from src.solver.games import Action as A
from src.solver.sequence_form import solve
from src.solver.tree import GameTree


def uniform(tree):
    return tree.mask / tree.mask.sum(axis=1, keepdims=True)


def test_regret_matching_and_first_kuhn_updates():
    tree = GameTree("kuhn")
    weights = np.full(tree.mask.shape, -1.0)
    assert np.array_equal(normalize(weights, tree.mask), uniform(tree))
    weights[0, :2] = (2, 6)
    assert normalize(weights, tree.mask)[0].tolist() == [0.25, 0.75, 0]
    delta = regret_delta(tree, uniform(tree))
    for i, key in enumerate(tree.information_sets):
        if key.player == 0 and key.history == ((),):
            assert delta[i] == pytest.approx([-0.125, 0.125, 0])
        if key.player == 0 and key.card == 0 and key.history == ((A.CHECK, A.RAISE),):
            assert delta[i] == pytest.approx([1 / 12, -1 / 12, 0])
    solver = CFR(tree)
    solver.step()
    assert solver.regrets == pytest.approx(delta)


def test_counterfactual_regret_does_not_include_own_reach():
    tree = GameTree("kuhn")
    policy = uniform(tree)
    before = regret_delta(tree, policy)
    for i, key in enumerate(tree.information_sets):
        if key.player == 0 and key.history == ((),):
            policy[i] = (0, 1, 0)
    after = regret_delta(tree, policy)
    for i, key in enumerate(tree.information_sets):
        if key.player == 0 and key.history == ((A.CHECK, A.RAISE),):
            assert after[i] == pytest.approx(before[i])


def test_strategy_average_uses_own_reach_once_per_information_set():
    tree = GameTree("kuhn")
    solver = CFR(tree)
    profiles = [uniform(tree), uniform(tree)]
    for i, key in enumerate(tree.information_sets):
        if key.player == 0 and key.history == ((),):
            profiles[0][i], profiles[1][i] = (0.25, 0.75, 0), (0.75, 0.25, 0)
        if key.player == 0 and len(key.history[0]) == 2:
            profiles[0][i], profiles[1][i] = (1, 0, 0), (0, 1, 0)
    for policy in profiles:
        solver.regrets[:] = policy
        solver.step()
    for i, key in enumerate(tree.information_sets):
        if key.player == 0 and len(key.history[0]) == 2:
            assert solver.strategy_sum[i] == pytest.approx([0.25, 0.75, 0])
            assert solver.average_policy()[i] == pytest.approx([0.25, 0.75, 0])


@pytest.mark.parametrize("player", [0, 1])
def test_exact_best_response_matches_every_pure_kuhn_strategy(player):
    tree = GameTree("kuhn")
    rng = np.random.default_rng(29)
    policy = normalize(rng.random(tree.mask.shape), tree.mask)
    infos = np.flatnonzero(tree.owners == player)
    best = -np.inf
    for choices in itertools.product((0, 1), repeat=len(infos)):
        candidate = policy.copy()
        candidate[infos] = 0
        candidate[infos, choices] = 1
        value = tree.values(tree.edge_probabilities(candidate))[0] * (1 - 2 * player)
        best = max(best, value)
    value, response = best_response(tree, policy, player)
    assert value == pytest.approx(best, abs=1e-12)
    assert tree.values(tree.edge_probabilities(response))[0] * (
        1 - 2 * player
    ) == pytest.approx(best)
    assert np.array_equal(
        response[tree.owners != player], policy[tree.owners != player]
    )


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_sequence_form_equilibrium_agrees_with_exact_best_response(game):
    tree = GameTree(game)
    equilibrium = solve(tree)
    result = evaluate(tree, equilibrium.policy)
    assert equilibrium.upper_value - equilibrium.lower_value == pytest.approx(
        0, abs=1e-9
    )
    assert equilibrium.maximum_residual < 1e-9
    assert result.value_player0 == pytest.approx(equilibrium.lower_value, abs=1e-9)
    assert result.exploitability < 1e-9
    if game == "kuhn":
        assert result.value_player0 == pytest.approx(-1 / 18, abs=1e-12)
    # A perturbed policy is a useful check that evaluation does not just return zero.
    assert (
        evaluate(tree, 0.5 * equilibrium.policy + 0.5 * uniform(tree)).exploitability
        > 0.01
    )


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize("deterministic", [False, True])
def test_exhaustive_external_samples_have_the_full_cfr_expectation(
    player, deterministic
):
    tree = GameTree("kuhn")
    policy = normalize(np.random.default_rng(3).random(tree.mask.shape), tree.mask)
    if deterministic:
        policy[::2] = (0, 1, 0)
    expected = regret_delta(tree, policy)
    expected[tree.owners != player] = 0
    total = np.zeros_like(policy)
    mass = 0

    class ChoiceNeeded(Exception):
        def __init__(self, probabilities):
            self.probabilities = probabilities

    def enumerate_samples(tape=(), probability=1.0):
        nonlocal mass, total
        draws = iter(tape)

        def choose(probabilities):
            try:
                return next(draws)
            except StopIteration:
                raise ChoiceNeeded(probabilities) from None

        try:
            delta = sampled_delta(tree, policy, player, choose)
        except ChoiceNeeded as pending:
            for action, p in enumerate(pending.probabilities):
                if p > 0:
                    enumerate_samples(tape + (action,), probability * p)
        else:
            mass += probability
            total += probability * delta

    enumerate_samples()
    assert mass == pytest.approx(1, abs=1e-12)
    assert total == pytest.approx(expected, abs=1e-12)


def test_exploitability_is_half_nash_conv_in_ante_units():
    tree = GameTree("kuhn")
    result = evaluate(tree, uniform(tree))
    assert result.value_player0 == pytest.approx(1 / 8)
    assert result.nash_conv == pytest.approx(11 / 12)
    assert result.exploitability == pytest.approx(11 / 24)
    assert sum(result.improvements) == pytest.approx(result.nash_conv)


@pytest.mark.parametrize("method", ["full", "external"])
def test_seeded_training_repeats_without_touching_global_generators(method):
    tree = GameTree("kuhn")
    first, second = CFR(tree, method=method, seed=17), CFR(tree, method=method, seed=17)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    first.train(40)
    second.train(17)
    second.train(23)
    assert np.array_equal(first.regrets, second.regrets)
    assert np.array_equal(first.strategy_sum, second.strategy_sum)
    assert random.getstate() == python_state
    now = np.random.get_state()
    assert np.array_equal(numpy_state[1], now[1]) and numpy_state[2:] == now[2:]
    tree.validate_policy(first.average_policy())
    assert first.iterations == 40
