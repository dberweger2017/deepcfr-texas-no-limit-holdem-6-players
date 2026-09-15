import random

import numpy as np
import pytest
import torch

from src.solver.cfr import normalize, regret_delta
from src.solver.games import Action, new_game
from src.solver.neural.encoding import ACTION_SLOT
from src.solver.neural.network import deterministic_cpu, new_network
from src.solver.neural.solver import Config, DeepCFR, collect
from src.solver.tree import GameTree


def global_values(tree, local):
    result = np.zeros_like(local)
    for info, key in enumerate(tree.information_sets):
        for action, value in zip(key.actions, local[info]):
            result[info, ACTION_SLOT[action]] = value
    return result


@pytest.mark.parametrize("player", [0, 1])
def test_collected_targets_and_strategy_visits_have_the_exact_expectation(player):
    tree = GameTree("kuhn")
    local = normalize(np.random.default_rng(21).random(tree.mask.shape), tree.mask)
    policy = global_values(tree, local)
    advantages = np.zeros_like(policy)
    visits = np.zeros(len(policy))
    mass = 0

    class ChoiceNeeded(Exception):
        def __init__(self, probabilities):
            self.probabilities = probabilities

    def enumerate_samples(tape=(), probability=1.0):
        nonlocal mass
        draws = iter(tape)
        rows, strategies = [], []

        def choose(probabilities):
            try:
                return next(draws)
            except StopIteration:
                raise ChoiceNeeded(probabilities) from None

        try:
            collect(
                tree,
                policy,
                player,
                3,
                choose,
                lambda *row: rows.append(row),
                lambda *row: strategies.append(row),
            )
        except ChoiceNeeded as pending:
            for action, p in enumerate(pending.probabilities):
                if p > 0:
                    enumerate_samples(tape + (action,), probability * p)
        else:
            mass += probability
            for info, iteration, target in rows:
                assert tree.owners[info] == player and iteration == 3
                advantages[info] += probability * target
            for info, iteration, target in strategies:
                assert tree.owners[info] == 1 - player and iteration == 3
                assert target == pytest.approx(policy[info])
                visits[info] += probability

    enumerate_samples()
    expected = global_values(tree, regret_delta(tree, local))
    expected[tree.owners != player] = 0
    assert mass == pytest.approx(1)
    assert advantages == pytest.approx(expected, abs=1e-12)
    reach = tree.reaches(tree.edge_probabilities(local))
    expected_visits = np.zeros(len(policy))
    for node, actor in enumerate(tree.actor):
        if actor == 1 - player:
            expected_visits[tree.info[node]] += reach[node, 2] * reach[node, actor]
    assert visits == pytest.approx(expected_visits, abs=1e-12)


def test_second_player_collects_against_the_updated_first_player(monkeypatch):
    import src.solver.neural.solver as module

    tree = GameTree("kuhn")
    solver = DeepCFR(
        tree, Config(hidden=8, traversals=1, advantage_steps=1, strategy_steps=1)
    )
    profiles = []

    def record(tree, policy, player, *args):
        profiles.append((player, policy.copy()))

    def fitted(memory, iteration, player, strategy, deadline):
        net = new_network(8, player, zero_output=True)
        with torch.no_grad():
            net.layers[-1].bias[2] = 1
        return net.eval().requires_grad_(False), {}

    monkeypatch.setattr(module, "collect", record)
    monkeypatch.setattr(solver, "_fit", fitted)
    with deterministic_cpu():
        solver.step()
    root = next(
        i
        for i, key in enumerate(tree.information_sets)
        if key.player == 0 and key.history == ((),)
    )
    assert [p for p, _ in profiles] == [0, 1]
    assert profiles[0][1][root].tolist() == [0, 1, 0]
    assert profiles[1][1][root].tolist() == [0, 0, 1]


def test_neural_training_is_repeatable_and_diagnostics_do_not_feed_the_learner():
    tree = GameTree("kuhn")
    config = Config(
        hidden=16,
        traversals=16,
        advantage_steps=8,
        strategy_steps=12,
        capacity=64,
        batch_size=16,
        seed=9,
    )
    before_python, before_numpy, before_torch = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )
    with deterministic_cpu():
        first, second = DeepCFR(tree, config), DeepCFR(tree, config)
        first.step()
        first.fit_strategy()
        first.played_strategy_sum.fill(1e6)
        first.step()
        first.fit_strategy()
        second.step()
        second.step()
        second.fit_strategy()
        assert np.array_equal(first.average_policy(), second.average_policy())
        assert first.traversal_random.getstate() == second.traversal_random.getstate()
        for a, b in zip(
            first.advantage_memories + [first.strategy_memory],
            second.advantage_memories + [second.strategy_memory],
        ):
            assert a.seen == b.seen and a.size == b.size
            assert np.array_equal(a.infos[: a.size], b.infos[: b.size])
            assert np.array_equal(a.targets[: a.size], b.targets[: b.size])
            assert a.random.getstate() == b.random.getstate()
        assert first.fits == second.fits
        for net in first.advantages + [first.strategy]:
            assert not net.training and all(
                not p.requires_grad for p in net.parameters()
            )
    assert random.getstate() == before_python
    now = np.random.get_state()
    assert np.array_equal(now[1], before_numpy[1]) and now[2:] == before_numpy[2:]
    assert torch.equal(before_torch, torch.random.get_rng_state())


def test_exported_policy_only_accepts_its_own_public_information():
    with deterministic_cpu():
        solver = DeepCFR(
            GameTree("leduc"),
            Config(
                hidden=8,
                traversals=8,
                advantage_steps=2,
                strategy_steps=3,
                batch_size=8,
            ),
        )
        with pytest.raises(RuntimeError):
            solver.average_policy()
        solver.step()
        solver.fit_strategy()
        policy = solver.policy(0, 7)
        first = new_game("leduc").deal(0).deal(2)
        other = new_game("leduc").deal(0).deal(4)
        assert policy.distribution(first.information_set()) == policy.distribution(
            other.information_set()
        )
        probabilities = policy.distribution(first.information_set())
        assert sum(p for _, p in probabilities) == pytest.approx(1)
        assert {a for a, _ in probabilities} == set(first.actions())
        with pytest.raises(TypeError):
            policy.distribution(first)
        with pytest.raises(ValueError):
            policy.distribution(first.play(Action.CHECK).information_set())
        solver.step()
        with pytest.raises(RuntimeError):
            solver.average_policy()


def test_a_partial_failed_iteration_cannot_be_silently_continued():
    solver = DeepCFR(GameTree("kuhn"), Config(hidden=8))
    with pytest.raises(TimeoutError):
        solver.step(deadline=0)
    assert solver.failed and solver.iterations == 0
    with pytest.raises(RuntimeError):
        solver.step()
    with pytest.raises(RuntimeError):
        solver.fit_strategy()
