from dataclasses import asdict, replace

import pytest
import torch

from src.solver.neural.experiment import provenance
from src.solver.neural.network import deterministic_cpu
from src.solver.neural.solver import Config, DeepCFR
from src.solver.tree import GameTree


@pytest.mark.parametrize(
    "changes",
    [
        {"strategy_learning_rate_schedule": "linear"},
        {"strategy_final_learning_rate": 0.00001},
        *[
            {
                "strategy_learning_rate_schedule": "cosine",
                "strategy_final_learning_rate": end,
            }
            for end in (None, True, 0, -1, float("nan"), float("inf"), 0.01)
        ],
        {
            "strategy_learning_rate_schedule": "cosine",
            "strategy_final_learning_rate": 0.00001,
            "strategy_steps": 1,
        },
    ],
)
def test_schedule_rejects_ambiguous_or_invalid_configuration(changes):
    with pytest.raises(ValueError):
        Config(**changes)


def test_schedule_restarts_for_each_fit_and_records_actual_update_rates(monkeypatch):
    config = Config(
        hidden=8,
        traversals=8,
        advantage_steps=2,
        strategy_steps=3,
        batch_size=8,
        capacity=32,
        seed=131,
        strategy_learning_rate_schedule="cosine",
        strategy_final_learning_rate=0.00001,
    )
    rates = []
    original = torch.optim.Adam.step

    def step(optimizer, *args, **kwargs):
        rates.append(optimizer.param_groups[0]["lr"])
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    with deterministic_cpu():
        solver = DeepCFR(GameTree("kuhn"), config)
        solver.step()
        assert rates == [0.001] * 4
        rates.clear()
        before = torch.random.get_rng_state().clone()
        first_metrics = solver.fit_strategy()
        first = {k: v.clone() for k, v in solver.strategy.state_dict().items()}
        second_metrics = solver.fit_strategy()
        assert torch.equal(before, torch.random.get_rng_state())
        assert all(
            torch.equal(v, solver.strategy.state_dict()[k]) for k, v in first.items()
        )
        assert first_metrics == second_metrics
        assert rates == pytest.approx([0.001, 0.000505, 0.00001] * 2)


def test_schedule_changes_plan_provenance():
    constant = Config()
    decay = replace(
        constant,
        strategy_learning_rate_schedule="cosine",
        strategy_final_learning_rate=0.00001,
    )
    assert Config(**asdict(decay)) == decay
    a, b = provenance(asdict(constant)), provenance(asdict(decay))
    assert a["plan_sha256"] != b["plan_sha256"]
    assert b["plan"]["strategy_final_learning_rate"] == 0.00001
