from dataclasses import replace

import numpy as np
import pytest
import torch

from src.solver.neural.experiment import Plan
from src.solver.neural.network import deterministic_cpu
from src.solver.neural.solver import Config, DeepCFR
from src.solver.tree import GameTree


def test_strategy_capacity_and_fit_schedule_do_not_change_collection():
    config = Config(
        hidden=8,
        traversals=16,
        advantage_steps=4,
        strategy_steps=6,
        capacity=32,
        batch_size=8,
        seed=311,
    )
    with deterministic_cpu():
        first = DeepCFR(GameTree("leduc"), config)
        second = DeepCFR(
            first.tree, replace(config, strategy_hidden=16, strategy_steps=12)
        )
        for _ in range(3):
            first.step()
            first.fit_strategy()
            second.step()
        second.fit_strategy()
    assert first.strategy.layers[0].out_features == 8
    assert second.strategy.layers[0].out_features == 16
    assert first.fits == second.fits
    assert first.traversal_random.getstate() == second.traversal_random.getstate()
    assert np.array_equal(first.played_strategy_sum, second.played_strategy_sum)
    for a, b in zip(first.advantages, second.advantages):
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    for a, b in zip(
        first.advantage_memories + [first.strategy_memory],
        second.advantage_memories + [second.strategy_memory],
    ):
        assert (a.size, a.seen, a.random.getstate()) == (
            b.size,
            b.seen,
            b.random.getstate(),
        )
        for field in ("infos", "iterations", "targets"):
            assert np.array_equal(
                getattr(a, field)[: a.size], getattr(b, field)[: b.size]
            )


def test_long_cpu_runs_are_explicit_and_still_bounded():
    with pytest.raises(ValueError):
        Plan("leduc", 480, Config(), maximum_seconds=7200)
    plan = Plan("leduc", 480, Config(), maximum_seconds=7200, execution="cpu-campaign")
    assert Plan.from_dict({**plan.__dict__, "training": Config().__dict__}) == plan
    with pytest.raises(ValueError):
        replace(plan, maximum_seconds=7201)
    for width in (0, -1, 513, True, 64.5):
        with pytest.raises(ValueError):
            Config(strategy_hidden=width)
