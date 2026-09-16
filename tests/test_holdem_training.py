from dataclasses import replace

import pytest
import torch

from src.game.types import TableSeat
from src.holdem import training
from src.holdem.collection import CollectionLimitExceeded
from src.holdem.fitting import FitConfig
from src.holdem.policy import FrozenProfile
from src.holdem.training import HoldemTrainer, TrainConfig
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import table


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(183)
        yield


def config(**kwargs):
    return TrainConfig(
        seed=7, capacity=16, fit=FitConfig(width=8, steps=3, batch_size=4), **kwargs
    )


def state(trainer):
    return (
        trainer.iteration,
        trainer.current_profile().fingerprint,
        trainer.reports,
        tuple((m.items, m.seen, m._random.getstate()) for m in trainer.memories),
    )


@pytest.mark.parametrize("n", [4, 5, 6])
def test_two_collect_fit_cycles_keep_roles_separate_and_frozen(n, monkeypatch):
    trainer = HoldemTrainer(table(n, (10,) * n), config())
    initial = trainer.current_profile()
    assert initial.fingerprint == FrozenProfile([None] * n).fingerprint
    original_fit = training.fit_role
    used_profiles = []

    def checked(memory, *args, **kwargs):
        # No earlier fit in this iteration has been published.
        used_profiles.append(trainer.current_profile().fingerprint)
        assert trainer.iteration == kwargs["iteration"] - 1
        return original_fit(memory, *args, **kwargs)

    monkeypatch.setattr(training, "fit_role", checked)
    first = trainer.step()
    assert first.collection_profile == initial.fingerprint
    assert set(used_profiles) == {initial.fingerprint}
    assert (
        first.fitted_profile
        == trainer.current_profile().fingerprint
        != initial.fingerprint
    )
    used_profiles.clear()
    second = trainer.step()
    assert second.collection_profile == first.fitted_profile
    assert set(used_profiles) == {first.fitted_profile}
    assert trainer.iteration == 2 and trainer.reports == (first, second)
    initial.assert_unchanged()
    for role, memory in enumerate(trainer.memories):
        assert all(s.role == role for s in memory.items)
        assert (
            memory.seen
            == first.roles[role].new_samples + second.roles[role].new_samples
        )
        assert len(memory) <= 16
        assert {s.profile for s in memory.items} <= {
            first.collection_profile,
            second.collection_profile,
        }
        assert {s.iteration for s in memory.items} <= {1, 2}


def test_later_role_failure_rolls_back_models_memories_and_iteration(monkeypatch):
    trainer = HoldemTrainer(table(4, (10,) * 4), config())
    trainer.step()
    baseline = state(trainer)
    original_fit = training.fit_role
    completed = []

    def fail_later(memory, *args, **kwargs):
        if completed:
            raise RuntimeError("injected second-role failure")
        result = original_fit(memory, *args, **kwargs)
        completed.append(memory.role)
        return result

    monkeypatch.setattr(training, "fit_role", fail_later)
    with pytest.raises(RuntimeError, match="second-role"):
        trainer.step()
    assert completed and state(trainer) == baseline
    monkeypatch.setattr(training, "fit_role", original_fit)
    trainer.step()
    fresh = HoldemTrainer(table(4, (10,) * 4), config())
    fresh.step()
    fresh.step()
    assert state(trainer) == state(fresh)


def test_collection_failure_never_admits_partial_replay():
    trainer = HoldemTrainer(table(4, (10,) * 4), config(max_nodes=1))
    before = state(trainer)
    with pytest.raises(CollectionLimitExceeded):
        trainer.step()
    assert state(trainer) == before


def test_trainer_rejects_a_batch_from_another_generation(monkeypatch):
    trainer = HoldemTrainer(table(4, (10,) * 4), config())
    before = state(trainer)
    original = training.collect_phase

    def stale(*args, **kwargs):
        return replace(original(*args, **kwargs), profile="b" * 64)

    monkeypatch.setattr(training, "collect_phase", stale)
    with pytest.raises(ValueError, match="iteration"):
        trainer.step()
    assert state(trainer) == before


def test_empty_memories_retain_uniform_without_resampling_or_fitting(monkeypatch):
    # Both heads-up players are already all-in from the blinds.
    trainer = HoldemTrainer(table(2, (1, 1)), config())
    initial = trainer.current_profile().fingerprint

    def no_fit(*args, **kwargs):
        pytest.fail("An empty role must not be fitted")

    monkeypatch.setattr(training, "fit_role", no_fit)
    for _ in range(2):
        report = trainer.step()
        assert report.nodes == 2
        assert report.fitted_profile == report.collection_profile == initial
        assert all(
            r.new_samples == r.seen == r.stored == 0 and r.fit is None
            for r in report.roles
        )
    assert trainer.iteration == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"capacity": 0},
        {"traversals_per_player": True},
        {"max_nodes": 0},
        {"max_seconds": float("nan")},
    ],
)
def test_train_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_sparse_table_updates_only_its_physical_roles():
    seats = (0, 2, 3, 5)
    sparse = replace(
        table(4, (10,) * 4),
        capacity=6,
        seat_numbers=seats,
        table_seats=tuple(
            TableSeat(s, f"player-{i}", 10, "playing") for i, s in enumerate(seats)
        ),
    )
    trainer = HoldemTrainer(sparse, config())
    report = trainer.step()
    for role, memory in enumerate(trainer.memories):
        if role not in seats:
            assert not memory and report.roles[role].fit is None
        else:
            assert all(s.role == role for s in memory.items)
            assert report.roles[role].new_samples > 0


def test_training_rotates_button_without_changing_physical_roles(monkeypatch):
    trainer = HoldemTrainer(table(4, (10,) * 4), config())
    seen = []
    original = training.collect_phase

    def collect(current, *args, **kwargs):
        seen.append((current.button, current.seat_numbers))
        return original(current, *args, **kwargs)

    monkeypatch.setattr(training, "collect_phase", collect)
    trainer.step()
    trainer.step()
    assert seen == [(0, (0, 1, 2, 3)), (1, (0, 1, 2, 3))]
    assert trainer.table.button == 0
