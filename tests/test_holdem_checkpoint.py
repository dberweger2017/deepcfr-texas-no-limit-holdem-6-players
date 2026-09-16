from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from src.game.hand import Hand
from src.holdem.checkpoint import (
    TRAINING,
    _load,
    _save,
    load_policy,
    load_training,
    save_policy,
    save_training,
)
from src.holdem.training import HoldemTrainer
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import table
from tests.test_holdem_training import config, state


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        yield


@pytest.mark.parametrize("players", [4, 5, 6])
def test_complete_resume_equals_uninterrupted_with_reservoir_replacement(
    tmp_path, players
):
    trainer = HoldemTrainer(
        table(players, (10,) * players), replace(config(), capacity=2)
    )
    trainer.step()
    path = tmp_path / "training.pt"
    digest = save_training(trainer, path, manifest={"source": "test"})
    restored = load_training(path, digest, manifest={"source": "test"})
    assert state(restored) == state(trainer)
    trainer.step()
    restored.step()
    assert state(restored) == state(trainer)
    assert (
        restored.average_policy().fingerprints == trainer.average_policy().fingerprints
    )
    assert any(m.seen > len(m) for m in restored.memories)
    first, second = tmp_path / "one.pt", tmp_path / "two.pt"
    assert save_training(trainer, first, manifest={}) == save_training(
        restored, second, manifest={}
    )


def test_bootstrap_checkpoint_and_inference_separation(tmp_path):
    trainer = HoldemTrainer(table(4, (10,) * 4), config())
    path = tmp_path / "initial.pt"
    digest = save_training(trainer, path, manifest={})
    restored = load_training(path, digest, manifest={})
    assert state(trainer) == state(restored)
    with pytest.raises(ValueError, match="average"):
        save_policy(trainer, tmp_path / "empty.pt", manifest={})
    trainer.step()
    export = tmp_path / "average.pt"
    exported = save_policy(trainer, export, manifest={"source": "test"})
    average, provenance = load_policy(export, exported)
    assert provenance["training_seed"] == config().seed
    hand = Hand.start(trainer.table, hand_id="export", seed=1)
    view = hand.observe(hand.actor)
    assert average.distribution(view) == trainer.average_policy().distribution(view)
    assert average.player(41).choose_action(view) == trainer.average_policy().player(
        41
    ).choose_action(view)
    with pytest.raises(ValueError, match="format"):
        load_training(export, exported, manifest={})
    with pytest.raises(ValueError, match="format"):
        load_policy(path, digest)


def test_atomic_no_clobber_hash_and_manifest_checks(tmp_path):
    trainer = HoldemTrainer(table(4, (10,) * 4), config())
    path = tmp_path / "checkpoint.pt"
    digest = save_training(trainer, path, manifest={"source": "one"})
    with pytest.raises(FileExistsError):
        save_training(trainer, path, manifest={"source": "two"})
    assert sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="provenance"):
        load_training(path, digest, manifest={"source": "two"})
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="hash"):
        load_training(path, digest, manifest={"source": "one"})
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    "damage", ["archive", "seen", "role", "weights", "report", "random", "record"]
)
def test_malformed_checkpoints_are_rejected(tmp_path, damage):
    trainer = HoldemTrainer(table(4, (10,) * 4), config())
    trainer.step()
    path = tmp_path / "checkpoint.pt"
    checksum = save_training(trainer, path, manifest={})
    data = _load(path, checksum, TRAINING)
    if damage == "archive":
        data["archive"] = []
    elif damage == "seen":
        data["memories"][0]["seen"] += 1
    elif damage == "role":
        data["memories"][0]["role"] = 1
    elif damage == "weights":
        model = next(m for m in data["current"]["models"] if m is not None)
        next(iter(model["weights"].values())).fill_(float("nan"))
    elif damage == "report":
        data["reports"][0]["fields"]["collection_profile"] = "f" * 64
    elif damage == "random":
        data["memories"][0]["random"] = (1, (), None)
    else:
        data["table"]["record"] = "not-a-record"
    path.unlink()
    _save(path, data)
    with pytest.raises((ValueError, TypeError)):
        load_training(path, sha256(path.read_bytes()).hexdigest(), manifest={})


def test_load_does_not_consume_global_torch_randomness(tmp_path):
    trainer = HoldemTrainer(table(4, (10,) * 4), config())
    trainer.step()
    path = tmp_path / "checkpoint.pt"
    digest = save_training(trainer, path, manifest={})
    before = torch.get_rng_state().clone()
    load_training(path, digest, manifest={})
    assert torch.equal(before, torch.get_rng_state())
