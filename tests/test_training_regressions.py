import random
import sys
import types

import numpy as np
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


tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
tensorboard_stub.SummaryWriter = DummyWriter
sys.modules.setdefault("tensorboard", types.ModuleType("tensorboard"))
sys.modules["torch.utils.tensorboard"] = tensorboard_stub

import src.training.train as train_mod
from src.core.deep_cfr import DeepCFRAgent
from src.opponent_modeling.deep_cfr_with_opponent_modeling import (
    DeepCFRAgentWithOpponentModeling,
)


def write_checkpoint(path):
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


def patch_training_side_effects(monkeypatch):
    tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
    tensorboard_stub.SummaryWriter = DummyWriter
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tensorboard_stub)
    monkeypatch.setattr(train_mod, "evaluate_against_random", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        train_mod,
        "evaluate_against_checkpoint_agents",
        lambda *args, **kwargs: 0.0,
    )


def assert_replay_memory_shapes(agent):
    assert len(agent.advantage_memory) > 0
    assert len(agent.strategy_memory) > 0

    state, opponent_features, action_type, bet_size, regret = agent.advantage_memory.buffer[0]
    assert len(state) > 0
    assert opponent_features.shape == (20,)
    assert action_type in (0, 1, 2)
    assert isinstance(float(bet_size), float)
    assert isinstance(float(regret), float)

    state, opponent_features, strategy, bet_size, iteration = agent.strategy_memory[0]
    assert len(state) > 0
    assert opponent_features.shape == (20,)
    assert strategy.shape == (agent.num_actions,)
    assert np.isclose(strategy.sum(), 1.0)
    assert isinstance(float(bet_size), float)
    assert iteration == 1


def test_self_play_training_smoke(tmp_path, monkeypatch):
    patch_training_side_effects(monkeypatch)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    write_checkpoint(checkpoint_path)

    agent, losses, profits = train_mod.train_against_checkpoint(
        checkpoint_path=str(checkpoint_path),
        additional_iterations=1,
        traversals_per_iteration=1,
        save_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        verbose=False,
    )

    assert losses == [0]
    assert profits == [0.0, 0.0]
    assert_replay_memory_shapes(agent)


def test_mixed_training_smoke(tmp_path, monkeypatch):
    patch_training_side_effects(monkeypatch)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    checkpoint_dir = tmp_path / "pool"
    checkpoint_dir.mkdir()
    write_checkpoint(checkpoint_dir / "t_seed.pt")

    agent, losses, profits, profits_vs_checkpoints = train_mod.train_with_mixed_checkpoints(
        checkpoint_dir=str(checkpoint_dir),
        training_model_prefix="t_",
        additional_iterations=1,
        traversals_per_iteration=1,
        save_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        refresh_interval=1000,
        num_opponents=1,
        verbose=False,
    )

    assert losses == [0]
    assert profits == [0.0, 0.0]
    assert profits_vs_checkpoints == [0.0, 0.0]
    assert_replay_memory_shapes(agent)


def test_save_model_respects_explicit_pt_path(tmp_path):
    model_path = tmp_path / "model.pt"
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent.iteration_count = 7
    agent.save_model(model_path)
    assert model_path.exists()
    assert not (tmp_path / "model.pt_iteration_7.pt").exists()

    om_model_path = tmp_path / "om_model.pt"
    om_agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=6, device="cpu")
    om_agent.iteration_count = 3
    om_agent.save_model(om_model_path)
    assert om_model_path.exists()
    assert not (tmp_path / "om_model.pt_iteration_3.pt").exists()
