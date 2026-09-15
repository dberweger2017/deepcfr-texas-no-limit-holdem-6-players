from dataclasses import replace

import pytest
import torch

from src.holdem.betting import ActionScores
from src.holdem.fitting import FitConfig, fit_role, weighted_betting_loss
from src.holdem.policy import FrozenProfile
from src.holdem.replay import RoleReservoir
from src.holdem.targets import action_targets
from src.solver.neural.network import deterministic_cpu
from tests.test_holdem_replay import example


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(717)
        yield


def contradictory_samples():
    first = example(1)
    candidates = first.target.candidates
    # Two different observed branch returns for exactly the same decision.
    early = (-1.0, 0.0, 1.0)
    late = (1.0, 0.0, -1.0)
    assert len(candidates.actions) == 3

    def target(values):
        return action_targets(
            candidates,
            (1 / 3,) * 3,
            {a: 2 * v for a, v in zip(candidates.actions, values)},
        )

    return replace(first, target=target(early)), replace(
        first, iteration=3, target=target(late)
    )


def test_iteration_weights_apply_to_each_action_and_both_heads():
    samples = contradictory_samples()
    scores = [
        ActionScores(
            s.target.candidates,
            torch.zeros(3, requires_grad=True),
            torch.zeros(3, requires_grad=True),
        )
        for s in samples
    ]
    error = weighted_betting_loss(scores, samples, iteration=4)
    expected = (
        sum(
            (2 * s.iteration / 4)
            * sum(
                r * r + v * v for r, v in zip(s.target.regrets_bb, s.target.values_bb)
            )
            for s in samples
        )
        / 2
    )
    assert float(error.detach()) == pytest.approx(expected)
    error.backward()
    for score, sample in zip(scores, samples):
        weight = 2 * sample.iteration / 4
        torch.testing.assert_close(
            score.regrets.grad, -weight * torch.tensor(sample.target.regrets_bb)
        )
        torch.testing.assert_close(
            score.values.grad, -weight * torch.tensor(sample.target.values_bb)
        )


def test_controlled_fit_learns_the_iteration_weighted_mean():
    samples = contradictory_samples()
    memory = RoleReservoir(3, 2, 5)
    memory.extend(samples)
    model, metrics = fit_role(
        memory,
        FitConfig(width=16, steps=400, batch_size=32, learning_rate=0.005),
        iteration=3,
        seed=17,
    )
    # Weight 1:3 gives +0.5, 0, -0.5; an unweighted fit would give all zeros.
    expected = torch.tensor([0.5, 0.0, -0.5])
    score = model([samples[0].target.candidates])[0]
    torch.testing.assert_close(score.regrets, expected, rtol=0, atol=0.12)
    torch.testing.assert_close(score.values, expected, rtol=0, atol=0.12)
    assert metrics.loss_after < metrics.loss_before
    assert metrics.diagnostic_samples == 2 and metrics.steps == 400


def test_fitting_reproduces_without_mutating_replay_or_global_rng():
    memory = RoleReservoir(3, 2, 11)
    memory.extend(contradictory_samples())
    before = memory.items, memory.seen, memory._random.getstate()
    random_state = torch.get_rng_state().clone()
    config = FitConfig(width=8, steps=4, batch_size=4)
    first, metrics = fit_role(memory, config, iteration=3, seed=7)
    assert torch.equal(torch.get_rng_state(), random_state)
    torch.rand(9)
    second, repeated = fit_role(memory, config, iteration=3, seed=7)
    assert metrics == repeated
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name])
    assert (memory.items, memory.seen, memory._random.getstate()) == before
    snapshot = FrozenProfile([first, None])
    with torch.no_grad():
        first.regret.bias.add_(1)
    snapshot.assert_unchanged()


def test_diagnostic_sample_count_does_not_change_optimizer_updates():
    memory = RoleReservoir(3, 2, 11)
    memory.extend(contradictory_samples())
    config = FitConfig(width=8, steps=4, batch_size=4, diagnostic_samples=1)
    first, _ = fit_role(memory, config, iteration=3, seed=7)
    second, _ = fit_role(
        memory, replace(config, diagnostic_samples=2), iteration=3, seed=7
    )
    assert all(
        torch.equal(p, q) for p, q in zip(first.parameters(), second.parameters())
    )


def test_future_targets_empty_memory_and_expired_deadlines_are_rejected():
    memory = RoleReservoir(3, 2, 1)
    with pytest.raises(ValueError):
        fit_role(memory, FitConfig(), iteration=1, seed=0)
    memory.extend([example(3)])
    with pytest.raises(ValueError, match="future"):
        fit_role(memory, FitConfig(), iteration=2, seed=0)
    with pytest.raises(TimeoutError):
        fit_role(memory, FitConfig(width=8, steps=1), iteration=3, seed=0, deadline=0)
    sample = example(3)
    score = ActionScores(sample.target.candidates, torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError, match="future"):
        weighted_betting_loss([score], [sample], iteration=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"steps": True},
        {"batch_size": 0},
        {"diagnostic_samples": 0},
        {"learning_rate": float("nan")},
        {"learning_rate": 0},
    ],
)
def test_fit_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        FitConfig(**kwargs)
