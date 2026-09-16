from dataclasses import replace
from time import perf_counter

import pytest
import torch

from scripts.check_holdem_fitting import replay_metrics
from src.holdem.fitting import FitConfig, fit_role
from src.holdem.replay import RoleReservoir
from src.holdem.sampled_loss import sampled_replay_loss
from src.holdem.training import HoldemTrainer, SampledTrainConfig
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import table


@pytest.fixture
def memory():
    with deterministic_cpu():
        trainer = HoldemTrainer(
            table(4, (20,) * 4),
            SampledTrainConfig(
                seed=31, capacity=64, fit=FitConfig(width=8, steps=1, batch_size=4)
            ),
        )
        trainer.step()
        return next(m for m in trainer.memories if m)


def test_explicit_default_preserves_fit_and_unclipped_arm_shares_batches(
    memory, monkeypatch
):
    config = FitConfig(width=8, steps=2, batch_size=4)
    batches = []
    original = RoleReservoir.sample

    def track(self, size, rng):
        samples = original(self, size, rng)
        batches.append(samples)
        return samples

    monkeypatch.setattr(RoleReservoir, "sample", track)
    first, baseline = fit_role(memory, config, iteration=1, seed=19)
    expected_batches = tuple(batches)
    batches.clear()
    explicit, metrics = fit_role(memory, config, iteration=1, seed=19, gradient_clip=1)
    assert baseline == metrics
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), explicit.parameters())
    )
    assert tuple(batches) == expected_batches
    batches.clear()
    before = memory.items, memory.seen, memory._random.getstate()
    _, unclipped = fit_role(
        memory, replace(config, steps=4), iteration=1, seed=19, gradient_clip=None
    )
    assert tuple(batches[:2]) == expected_batches
    assert unclipped.loss_before == baseline.loss_before
    assert unclipped.clipped_steps == 0
    assert unclipped.max_gradient_norm > 1
    assert (memory.items, memory.seen, memory._random.getstate()) == before


def test_full_replay_measurement_matches_production_weighted_objective(memory):
    with deterministic_cpu():
        model, _ = fit_role(
            memory, FitConfig(width=8, steps=2, batch_size=4), iteration=1, seed=19
        )
        measured = replay_metrics(model, memory, 1)
        with torch.inference_mode():
            scores = model([s.target.candidates for s in memory.items])
            expected = float(
                sampled_replay_loss(
                    scores, memory.items, iteration=1, population=memory.seen
                )
            )
        assert measured["loss"] == pytest.approx(expected, rel=1e-6)
        assert measured["loss"] == measured["regret_loss"] + measured["value_loss"]
        assert measured["records"] == len(memory)
        with pytest.raises(TimeoutError):
            replay_metrics(model, memory, 1, deadline=perf_counter() - 1)


@pytest.mark.parametrize("clip", [0, -1, True, float("nan"), float("inf")])
def test_invalid_clipping_rejected(memory, clip):
    with pytest.raises(ValueError, match="Gradient clipping"):
        fit_role(
            memory,
            FitConfig(width=8, steps=1),
            iteration=1,
            seed=19,
            gradient_clip=clip,
        )
