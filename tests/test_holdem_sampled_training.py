import json
import subprocess
import sys
from dataclasses import asdict, replace
from itertools import combinations, product

import pytest
import torch

from src.game.hand import Hand
from src.game.types import TableSeat
from src.holdem import training
from src.holdem.betting import betting_loss
from src.holdem.checkpoint import (
    SAMPLED_TRAINING,
    TRAINING,
    _load,
    _save,
    load_policy,
    load_training,
    save_policy,
    save_training,
)
from src.holdem.collection import CollectionLimitExceeded
from src.holdem.experiment import Experiment
from src.holdem.fitting import FitConfig
from src.holdem.policy import FrozenProfile
from src.holdem.replay import ReplaySample, RoleReservoir, SampledReplaySample
from src.holdem.sampled_collection import (
    collect_sampled_phase,
    split_sampled_collection,
)
from src.holdem.sampled_loss import sampled_replay_loss
from src.holdem.targets import CandidateTargets
from src.holdem.training import HoldemTrainer, SampledTrainConfig
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import table
from tests.test_holdem_sampled_loss import predictions
from tests.test_holdem_training import state


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        yield


def config(**kwargs):
    return SampledTrainConfig(
        seed=31, capacity=2, fit=FitConfig(width=8, steps=2, batch_size=3), **kwargs
    )


@pytest.mark.parametrize("sampler", ["first-decision", "second-decision"])
@pytest.mark.parametrize("players", [4, 5, 6])
def test_sampled_resume_and_export(tmp_path, players, sampler):
    trainer = HoldemTrainer(table(players, (200,) * players), config(sampler=sampler))
    report = trainer.step()
    assert report.roots == (1,) * players
    assert report.terminals >= players
    assert all(
        isinstance(s, SampledReplaySample) for m in trainer.memories for s in m.items
    )
    assert any(m.seen > len(m) for m in trainer.memories)
    path = tmp_path / "paused.pt"
    digest = save_training(trainer, path, manifest={})
    restored = load_training(path, digest, manifest={})
    assert state(restored) == state(trainer)
    trainer.step()
    restored.step()
    assert state(restored) == state(trainer)
    assert save_training(trainer, tmp_path / "a.pt", manifest={}) == save_training(
        restored, tmp_path / "b.pt", manifest={}
    )
    export = tmp_path / "average.pt"
    policy, _ = load_policy(export, save_policy(trainer, export, manifest={}))
    hand = Hand.start(trainer.table, hand_id="held-out", seed=5)
    assert policy.distribution(
        hand.observe(hand.actor)
    ) == trainer.average_policy().distribution(hand.observe(hand.actor))
    assert all(
        r.fit.clipped_steps <= r.fit.steps for r in trainer.reports[-1].roles if r.fit
    )


@pytest.mark.parametrize("sampler", ["first-decision", "second-decision"])
def test_resume_matches_fresh_process_bytes(tmp_path, sampler):
    trainer = HoldemTrainer(table(4, (20,) * 4), config(sampler=sampler))
    trainer.step()
    paused = tmp_path / "paused.pt"
    digest = save_training(trainer, paused, manifest={})
    trainer.step()
    expected = save_training(trainer, tmp_path / "direct.pt", manifest={})
    code = """
import sys
from pathlib import Path
from src.holdem.checkpoint import load_training, save_training
trainer = load_training(Path(sys.argv[1]), sys.argv[2], manifest={})
trainer.step()
print(save_training(trainer, Path(sys.argv[3]), manifest={}))
"""
    actual = subprocess.check_output(
        [sys.executable, "-c", code, str(paused), digest, str(tmp_path / "resumed.pt")],
        text=True,
        timeout=90,
    )
    assert actual.strip() == expected


def test_later_role_failure_rolls_back_archive_replay_and_rng(monkeypatch):
    trainer = HoldemTrainer(table(4, (20,) * 4), config())
    trainer.step()
    before, archive = state(trainer), trainer.average_policy().fingerprints
    original = training.fit_role
    fitted = []

    def fail(memory, *args, **kwargs):
        if fitted:
            raise RuntimeError("second-role failure")
        fitted.append(memory.role)
        return original(memory, *args, **kwargs)

    monkeypatch.setattr(training, "fit_role", fail)
    with pytest.raises(RuntimeError, match="second-role"):
        trainer.step()
    assert state(trainer) == before
    assert trainer.average_policy().fingerprints == archive
    monkeypatch.setattr(training, "fit_role", original)
    trainer.step()
    fresh = HoldemTrainer(trainer.table, config())
    fresh.step()
    fresh.step()
    assert state(trainer) == state(fresh)


def test_empty_roots_and_sparse_roles_survive_recovery(tmp_path):
    seats = (0, 3)
    sparse = replace(
        table(2, (1, 1)),
        capacity=6,
        seat_numbers=seats,
        table_seats=tuple(
            TableSeat(s, f"player-{i}", 1, "playing") for i, s in enumerate(seats)
        ),
    )
    trainer = HoldemTrainer(sparse, config(traversals_per_player=3))
    report = trainer.step()
    assert report.roots == (3, 0, 0, 3, 0, 0)
    assert report.terminals == 6
    assert all(not m and m.seen == 0 for m in trainer.memories)
    path = tmp_path / "empty.pt"
    restored = load_training(
        path, save_training(trainer, path, manifest={}), manifest={}
    )
    assert state(restored) == state(trainer)
    assert restored.step().roots == report.roots


def test_partial_phase_and_wrong_generation_never_publish(monkeypatch):
    trainer = HoldemTrainer(table(4, (200,) * 4), config(max_nodes=1))
    before = state(trainer)
    with pytest.raises(CollectionLimitExceeded):
        trainer.step()
    assert state(trainer) == before
    trainer = HoldemTrainer(trainer.table, config())
    before = state(trainer)
    original = training.collect_sampled_phase
    monkeypatch.setattr(
        training,
        "collect_sampled_phase",
        lambda *a, **k: replace(original(*a, **k), profile="0" * 64),
    )
    with pytest.raises(ValueError, match="iteration"):
        trainer.step()
    assert state(trainer) == before


@pytest.mark.parametrize("damage", ["format", "roots", "sample_roots", "reach", "kind"])
def test_sampled_checkpoint_rejects_corrupt_normalization(tmp_path, damage):
    trainer = HoldemTrainer(table(4, (20,) * 4), config())
    trainer.step()
    path = tmp_path / "state.pt"
    data = _load(path, save_training(trainer, path, manifest={}), SAMPLED_TRAINING)
    if damage == "format":
        data["format"] = TRAINING
    elif damage == "roots":
        data["reports"][0]["fields"]["roots"] = (2,) * 4
    else:
        item = next(m for m in data["memories"] if m["items"])["items"][0]
        if damage == "sample_roots":
            item["fields"]["roots"] = 7
        elif damage == "reach":
            item["fields"]["target"]["fields"]["own_sample_reach"] = 0
        else:
            item["record"] = "ReplaySample"
    path.unlink()
    digest = _save(path, data)
    with pytest.raises((ValueError, TypeError)):
        load_training(path, digest, manifest={})


@pytest.fixture(params=["first-decision", "second-decision"])
def phase(request):
    return collect_sampled_phase(
        table(4, (20,) * 4),
        FrozenProfile([None] * 4),
        iteration=1,
        seed=31,
        traversals_per_player=2,
        sampler=request.param,
    )


def test_replay_types_cannot_mix_and_phase_is_reproducible(phase):
    assert phase == collect_sampled_phase(
        phase.table,
        FrozenProfile([None] * 4),
        iteration=1,
        seed=31,
        traversals_per_player=2,
        sampler=phase.sampler,
    )
    item = split_sampled_collection(phase)[0][0]
    memory = RoleReservoir(0, 2, 0)
    memory.extend([item])
    d = item.target
    external = ReplaySample(
        item.role,
        item.iteration,
        item.profile,
        item.action_seed,
        item.target_index,
        CandidateTargets(d.candidates, d.policy, d.values_bb, d.regrets_bb),
    )
    fingerprint = memory.fingerprint()
    with pytest.raises(ValueError, match="mix"):
        memory.extend([external])
    assert memory.fingerprint() == fingerprint


@pytest.mark.parametrize(
    "field", ["own_sample_reach", "inclusion_probabilities", "sampled_action"]
)
def test_phase_rejects_inconsistent_sampling_provenance(phase, field):
    traversal = phase.traversals[0]
    decision = traversal.decisions[-1]
    value = {
        "own_sample_reach": 0.5,
        "inclusion_probabilities": (0.5,) * len(decision.policy),
        "sampled_action": 0,
    }[field]
    traversal = replace(
        traversal,
        decisions=(*traversal.decisions[:-1], replace(decision, **{field: value})),
    )
    with pytest.raises(ValueError):
        split_sampled_collection(
            replace(phase, traversals=(traversal, *phase.traversals[1:]))
        )


def test_mixed_iterations_and_uniform_reservoir_match_direct_gradient(phase):
    first = split_sampled_collection(phase)[0][0]
    records = (
        first,
        replace(first, iteration=2, roots=7),
        replace(
            first,
            iteration=2,
            roots=7,
            target=replace(
                first.target, values_bb=tuple(v + 3 for v in first.target.values_bb)
            ),
        ),
    )

    def gradient(samples, population):
        parameter = torch.tensor(
            [0.7, -0.3, 1.1], dtype=torch.float64, requires_grad=True
        )
        loss = sampled_replay_loss(
            [predictions(s.target.candidates, parameter) for s in samples],
            samples,
            iteration=3,
            population=population,
        )
        return torch.autograd.grad(loss, parameter)[0]

    parameter = torch.tensor([0.7, -0.3, 1.1], dtype=torch.float64, requires_grad=True)
    # Iteration 3 has only empty roots; its weight still belongs in the denominator.
    exact = sum(
        s.iteration
        / s.roots
        / 6
        / s.target.own_sample_reach
        * betting_loss(
            [predictions(s.target.candidates, parameter)],
            [
                CandidateTargets(
                    s.target.candidates,
                    s.target.policy,
                    s.target.values_bb,
                    s.target.regrets_bb,
                )
            ],
        )
        for s in records
    )
    expected = torch.autograd.grad(exact, parameter)[0]
    estimates = [
        gradient(batch, len(records))
        for reservoir in combinations(records, 2)
        for batch in product(reservoir, repeat=2)
    ]
    torch.testing.assert_close(
        torch.stack(estimates).mean(0), expected, atol=1e-12, rtol=0
    )
    assert not torch.allclose(gradient(records, 2), expected)


@pytest.mark.parametrize("sampler", ["first-decision", "second-decision"])
def test_plan_preserves_sampler_on_roundtrip(sampler):
    from pathlib import Path

    plan = json.loads(Path("configs/holdem/baseline-check.json").read_text())
    plan["training"].update(sampler=sampler, exploration=0.5)
    parsed = Experiment.from_dict(plan)
    assert isinstance(parsed.training, SampledTrainConfig)
    assert Experiment.from_dict(asdict(parsed)) == parsed


@pytest.mark.parametrize("sampler", ["first-decision", "second-decision"])
def test_checkpoint_rejects_relabelled_expansion(tmp_path, sampler):
    trainer = HoldemTrainer(
        table(4, (20,) * 4), replace(config(sampler=sampler), capacity=128)
    )
    trainer.step()
    coverage = trainer.last_timing["collection_coverage"]
    assert all(
        set(c["records_by_street"]) == {"preflop", "flop", "turn", "river"}
        for c in coverage
    )
    assert sum(c["roots"] for c in coverage) == 4
    assert sum(sum(c["records_by_street"].values()) for c in coverage) == sum(
        r.new_samples for r in trainer.reports[0].roles
    )
    assert {c["position_from_button"] for c in coverage} == set(range(4))
    path = tmp_path / "state.pt"
    data = _load(path, save_training(trainer, path, manifest={}), SAMPLED_TRAINING)
    data["config"]["fields"]["sampler"] = (
        "second-decision" if sampler == "first-decision" else "first-decision"
    )
    path.unlink()
    with pytest.raises(ValueError, match="Replay expansion"):
        load_training(path, _save(path, data), manifest={})
