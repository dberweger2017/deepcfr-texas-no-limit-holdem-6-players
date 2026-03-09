import random
import sys
import types

import numpy as np
import pytest
import torch


class DummyWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class DummyNotifier:
    def __init__(self, *args, **kwargs):
        pass

    def send_message(self, *args, **kwargs):
        pass

    def send_training_progress(self, *args, **kwargs):
        pass

    def alert_state_error(self, *args, **kwargs):
        pass

    def alert_zero_reward_games(self, *args, **kwargs):
        pass


tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
tensorboard_stub.SummaryWriter = DummyWriter
sys.modules.setdefault("tensorboard", types.ModuleType("tensorboard"))
sys.modules["torch.utils.tensorboard"] = tensorboard_stub

telegram_stub = types.ModuleType("scripts.telegram_notifier")
telegram_stub.TelegramNotifier = DummyNotifier
sys.modules["scripts.telegram_notifier"] = telegram_stub

import src.training.train_opponent_modeling as train_om_mod
from src.core.deep_cfr import DeepCFRAgent
from src.opponent_modeling.deep_cfr_with_opponent_modeling import (
    DeepCFRAgentWithOpponentModeling,
)


def write_om_checkpoint(path):
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=6, device="cpu")
    agent.iteration_count = 1
    agent.save_model(path)


def write_standard_checkpoint(path):
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent.iteration_count = 1
    torch.save(
        {
            "iteration": agent.iteration_count,
            "advantage_net": agent.advantage_net.state_dict(),
            "strategy_net": agent.strategy_net.state_dict(),
            "min_bet_size": agent.min_bet_size,
            "max_bet_size": agent.max_bet_size,
        },
        path,
    )


def patch_opponent_model_training_side_effects(monkeypatch):
    tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
    tensorboard_stub.SummaryWriter = DummyWriter
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tensorboard_stub)

    telegram_stub = types.ModuleType("scripts.telegram_notifier")
    telegram_stub.TelegramNotifier = DummyNotifier
    monkeypatch.setitem(sys.modules, "scripts.telegram_notifier", telegram_stub)

    monkeypatch.setattr(train_om_mod, "evaluate_against_random", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(train_om_mod, "evaluate_against_opponents", lambda *args, **kwargs: 0.0)


def test_opponent_model_self_play_training_smoke(tmp_path, monkeypatch):
    patch_opponent_model_training_side_effects(monkeypatch)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    checkpoint_path = tmp_path / "phase1" / "checkpoint_iter_1.pt"
    checkpoint_path.parent.mkdir()
    write_om_checkpoint(checkpoint_path)

    (
        agent,
        advantage_losses,
        strategy_losses,
        opponent_model_losses,
        profits,
        profits_vs_checkpoints,
    ) = train_om_mod.train_against_checkpoint_with_opponent_modeling(
        checkpoint_path=str(checkpoint_path),
        additional_iterations=1,
        traversals_per_iteration=1,
        save_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        verbose=False,
    )

    assert agent.iteration_count == 2
    assert len(agent.advantage_memory) > 0
    assert len(agent.strategy_memory) > 0
    assert advantage_losses == [0]
    assert strategy_losses == [0]
    assert len(opponent_model_losses) == 1
    assert isinstance(float(opponent_model_losses[0]), float)
    assert profits == [0.0, 0.0]
    assert profits_vs_checkpoints == [0.0, 0.0]
    assert (tmp_path / "models" / "selfplay_checkpoint_iter_2.pt").exists()


def test_opponent_model_self_play_rejects_standard_checkpoint(tmp_path, monkeypatch):
    patch_opponent_model_training_side_effects(monkeypatch)

    checkpoint_path = tmp_path / "phase1" / "checkpoint_iter_1.pt"
    checkpoint_path.parent.mkdir()
    write_standard_checkpoint(checkpoint_path)

    with pytest.raises(ValueError, match="Opponent-model self-play requires an opponent-model checkpoint"):
        train_om_mod.train_against_checkpoint_with_opponent_modeling(
            checkpoint_path=str(checkpoint_path),
            additional_iterations=1,
            traversals_per_iteration=1,
            save_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs"),
            verbose=False,
        )


def test_train_opponent_modeling_main_dispatches_modes(monkeypatch):
    calls = []

    def fake_random(**kwargs):
        calls.append(("random", kwargs))
        return object(), [0], [0], [0.0], [0.0]

    def fake_self_play(**kwargs):
        calls.append(("self_play", kwargs))
        return object(), [0], [0], [0.0], [0.0], [0.0]

    def fake_mixed(**kwargs):
        calls.append(("mixed", kwargs))
        return object(), [0], [0], [0.0], [0.0], [0.0]

    monkeypatch.setattr(train_om_mod, "train_deep_cfr_with_opponent_modeling", fake_random)
    monkeypatch.setattr(train_om_mod, "train_against_checkpoint_with_opponent_modeling", fake_self_play)
    monkeypatch.setattr(train_om_mod, "train_mixed_with_opponent_modeling", fake_mixed)

    train_om_mod.main(["--iterations", "1"])
    train_om_mod.main(["--iterations", "1", "--checkpoint", "model.pt", "--self-play"])
    train_om_mod.main(["--iterations", "1", "--mixed", "--checkpoint-dir", "models"])

    assert [mode for mode, _ in calls] == ["random", "self_play", "mixed"]
