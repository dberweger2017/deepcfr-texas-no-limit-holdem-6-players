from collections import Counter
from itertools import product
from random import Random

import numpy as np
import pytest
import torch

from src.solver.games import new_game
from src.solver.neural.encoding import FEATURES, encode, legal_mask
from src.solver.neural.memory import Reservoir
from src.solver.neural.network import (
    deterministic_cpu,
    fitting_metrics,
    new_network,
    regret_matching,
    strategy_probabilities,
    weighted_loss,
)
from src.solver.tree import GameTree


def test_encoding_is_injective_and_contains_only_player_information():
    encoded = []
    for game in ("kuhn", "leduc"):
        tree = GameTree(game)
        for info in tree.information_sets:
            row = encode(info)
            assert row.shape == (FEATURES,)
            assert row.dtype == np.float32
            assert np.array_equal(row[-3:], legal_mask(info))
            encoded.append(row.tobytes())
    assert len(encoded) == len(set(encoded))
    first = new_game("leduc").deal(0).deal(2)
    second = new_game("leduc").deal(0).deal(4)
    assert np.array_equal(
        encode(first.information_set()), encode(second.information_set())
    )
    with pytest.raises(TypeError):
        encode(first)


def test_algorithm_r_gives_every_two_item_subset_the_same_probability():
    outcomes = Counter()
    for choices in product(range(3), range(4)):
        memory = Reservoir(2, 0)

        class Tape:
            def __init__(self, choices):
                self.draws = iter(choices)

            def randrange(self, n):
                value = next(self.draws)
                assert 0 <= value < n
                return value

        memory.random = Tape(choices)
        for value in range(4):
            memory.add(value, value + 1, np.array([value, 0, 0.0]))
        assert memory.seen == 4 and memory.size == 2
        outcomes[tuple(sorted(memory.infos))] += 1
        for info, iteration, target in zip(
            memory.infos, memory.iterations, memory.targets
        ):
            assert iteration == info + 1 and target.tolist() == [info, 0, 0]
    assert len(outcomes) == 6 and set(outcomes.values()) == {2}


def test_reservoir_streams_and_iteration_weighted_targets():
    state = Random(19).getstate()
    first, second = Reservoir(3, 19), Reservoir(3, 19)
    for memory in (first, second):
        for i in range(20):
            memory.add(i, 1, np.ones(3))
    assert np.array_equal(first.infos, second.infos)
    assert first.random.getstate() != state
    memory = Reservoir(3, 0)
    memory.add(0, 1, np.array([1.0, 0, 0]))
    memory.add(0, 3, np.array([3.0, 0, 0]))
    means, weights = memory.means(2)
    assert means == pytest.approx(np.array([[2.5, 0, 0], [0, 0, 0]]))
    assert weights.tolist() == [4, 0]


def test_loss_is_linear_in_iteration_weight_and_excludes_illegal_outputs():
    prediction = torch.tensor(
        [[1.0, 100.0, 2.0], [3.0, 100.0, 4.0]], requires_grad=True
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False, True]] * 2)
    loss = weighted_loss(prediction, target, mask, torch.tensor([1.0, 3.0]), 4)
    assert loss.item() == pytest.approx((0.5 * 5 + 1.5 * 25) / 2)
    loss.backward()
    assert prediction.grad.tolist() == [[0.5, 0, 1.0], [4.5, 0, 6.0]]
    probabilities = strategy_probabilities(prediction.detach(), mask)
    assert torch.equal(probabilities[:, 1], torch.zeros(2))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))


def test_regret_matching_uses_positive_values_then_highest_legal_prediction():
    scores = np.array([[2.0, 4, 100], [-3.0, -1, 100], [0.0, 0, 100]])
    mask = np.array([[True, True, False]] * 3)
    assert regret_matching(scores, mask) == pytest.approx(
        np.array([[1 / 3, 2 / 3, 0], [0, 1, 0], [1, 0, 0]])
    )
    with pytest.raises(ValueError):
        regret_matching(scores * np.nan, mask)


def test_fitting_diagnostics_separate_sample_noise_from_prediction_error():
    memory = Reservoir(2, 0)
    memory.add(0, 1, np.array([1.0, 0, 0]))
    memory.add(0, 3, np.array([3.0, 0, 0]))
    net = new_network(8, 0, zero_output=True)
    features = torch.zeros((1, FEATURES))
    mask = torch.ones((1, 3), dtype=torch.bool)
    with torch.no_grad():
        net.layers[-1].bias[0] = 2.5
    metrics = fitting_metrics(net, features, mask, memory, strategy=False)
    assert metrics["excess_mse"] == 0
    assert metrics["sample_noise_mse"] == pytest.approx(0.75)
    with torch.no_grad():
        net.layers[-1].bias[0] = 3.5
    assert (
        fitting_metrics(net, features, mask, memory, strategy=False)["excess_mse"] == 1
    )


def test_model_initialization_and_cpu_settings_do_not_escape_the_run():
    before = torch.random.get_rng_state().clone()
    settings = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    first, second = new_network(16, 7), new_network(16, 7)
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())
    )
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(RuntimeError), deterministic_cpu():
        assert torch.get_num_threads() == 1
        raise RuntimeError("interrupted")
    assert settings == (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
