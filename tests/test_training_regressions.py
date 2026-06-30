import random
import sys
import types

import numpy as np
import pokers as pkrs
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


tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
tensorboard_stub.SummaryWriter = DummyWriter
sys.modules.setdefault("tensorboard", types.ModuleType("tensorboard"))
sys.modules["torch.utils.tensorboard"] = tensorboard_stub

import src.training.train as train_mod
import src.core.deep_cfr as deep_cfr_mod
from src.agents.random_agent import RandomAgent
from src.core.deep_cfr import (
    DeepCFRAgent,
    linear_cfr_weights,
    traverse_agent_turn,
    weighted_sample_loss,
    weighted_strategy_cross_entropy,
)
from src.opponent_modeling.deep_cfr_with_opponent_modeling import (
    DeepCFRAgentWithOpponentModeling,
)
from src.utils.traversal_diagnostics import TraversalFailure


def write_checkpoint(path, iteration=1):
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent.iteration_count = iteration
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


def assert_replay_memory_shapes(agent, expected_iteration=1):
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
    assert iteration == expected_iteration


def test_self_play_training_smoke(tmp_path, monkeypatch):
    patch_training_side_effects(monkeypatch)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    checkpoint_path = tmp_path / "phase1" / "checkpoint_iter_1.pt"
    checkpoint_path.parent.mkdir()
    write_checkpoint(checkpoint_path)

    agent, losses, profits = train_mod.train_against_checkpoint(
        checkpoint_path=str(checkpoint_path),
        additional_iterations=1,
        traversals_per_iteration=1,
        save_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        verbose=False,
    )

    assert agent.iteration_count == 2
    assert losses == [0]
    assert profits == [0.0, 0.0]
    assert_replay_memory_shapes(agent, expected_iteration=1)
    assert (tmp_path / "models" / "selfplay_checkpoint_iter_2.pt").exists()


def test_mixed_training_smoke(tmp_path, monkeypatch):
    patch_training_side_effects(monkeypatch)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    checkpoint_dir = tmp_path / "pool"
    (checkpoint_dir / "phase1").mkdir(parents=True)
    write_checkpoint(checkpoint_dir / "phase1" / "checkpoint_iter_1.pt")

    agent, losses, profits, profits_vs_checkpoints = train_mod.train_with_mixed_checkpoints(
        checkpoint_dir=str(checkpoint_dir),
        training_model_prefix="*checkpoint_iter_",
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


def test_mixed_training_can_continue_from_checkpoint(tmp_path, monkeypatch):
    patch_training_side_effects(monkeypatch)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    checkpoint_dir = tmp_path / "pool"
    (checkpoint_dir / "phase1").mkdir(parents=True)
    write_checkpoint(checkpoint_dir / "phase1" / "checkpoint_iter_1.pt")

    resume_checkpoint = tmp_path / "resume" / "selfplay_checkpoint_iter_1.pt"
    resume_checkpoint.parent.mkdir()
    write_checkpoint(resume_checkpoint)

    agent, losses, profits, profits_vs_checkpoints = train_mod.train_with_mixed_checkpoints(
        checkpoint_dir=str(checkpoint_dir),
        training_model_prefix="*checkpoint_iter_",
        additional_iterations=1,
        traversals_per_iteration=1,
        save_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        refresh_interval=1000,
        num_opponents=1,
        verbose=False,
        checkpoint_path=str(resume_checkpoint),
    )

    assert agent.iteration_count == 2
    assert losses == [0]
    assert profits[0] == 0.0
    assert profits_vs_checkpoints[0] == 0.0
    assert_replay_memory_shapes(agent, expected_iteration=1)
    assert (tmp_path / "models" / "mixed_checkpoint_iter_2.pt").exists()


def test_strategy_weighting_is_1d_and_does_not_broadcast():
    strategies = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    predicted = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.30, 0.50],
        ],
        dtype=torch.float32,
    )
    weights = linear_cfr_weights(torch.tensor([1.0, 3.0]))

    assert weights.shape == (2,)
    loss = weighted_strategy_cross_entropy(strategies, predicted, weights)
    per_sample = -torch.sum(strategies * torch.log(predicted + 1e-8), dim=1)
    expected = torch.sum(weights * per_sample)
    broadcast_bug = torch.sum(weights.unsqueeze(1) * per_sample)

    assert torch.isclose(loss, expected)
    assert not torch.isclose(loss, broadcast_bug)

    raise_loss = weighted_sample_loss(torch.tensor([0.2, 0.4]), weights)
    assert torch.isclose(raise_loss, torch.tensor(0.35))


def test_traversal_invalid_agent_action_raises_instead_of_zero(monkeypatch):
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    state = pkrs.State.from_seed(
        n_players=6,
        button=3,
        sb=1,
        bb=2,
        stake=200.0,
        seed=0,
    )
    random_agents = [RandomAgent(i) for i in range(6)]

    def invalid_transition(*args, **kwargs):
        return None, "bad_transition.log", "InvalidTestStatus"

    monkeypatch.setattr(deep_cfr_mod, "apply_action_with_logging", invalid_transition)

    with pytest.raises(TraversalFailure) as exc_info:
        agent.cfr_traverse(state, iteration=1, random_agents=random_agents)

    snapshot = agent.traversal_diagnostics.snapshot()
    assert exc_info.value.reason == "agent_invalid_action"
    assert snapshot["failure_agent_invalid_action"] == 1
    assert "converted to value 0" in str(exc_info.value)


def test_traversal_records_action_value_diagnostics(monkeypatch):
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    state = pkrs.State.from_seed(
        n_players=6,
        button=3,
        sb=1,
        bb=2,
        stake=200.0,
        seed=0,
    )

    def ok_transition(state, action, **kwargs):
        return state, None, pkrs.StateStatus.Ok

    monkeypatch.setattr(deep_cfr_mod, "apply_action_with_logging", ok_transition)

    ev = traverse_agent_turn(
        agent,
        state,
        iteration=1,
        recurse_fn=lambda new_state, next_depth: 3.5,
    )
    snapshot = agent.traversal_diagnostics.snapshot()

    assert ev == pytest.approx(3.5)
    assert snapshot["agent_turns"] == 1
    assert snapshot["action_value_count"] == len(agent.get_legal_action_types(state))
    assert snapshot["action_value_zero_count"] == 0
    assert snapshot["action_value_mean"] == pytest.approx(3.5)
    assert snapshot["ev_mean"] == pytest.approx(3.5)


def test_opponent_modeling_missing_opponent_raises_traversal_failure():
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=3, device="cpu")
    state = pkrs.State.from_seed(
        n_players=3,
        button=1,
        sb=1,
        bb=2,
        stake=20.0,
        seed=0,
    )

    with pytest.raises(TraversalFailure) as exc_info:
        agent.cfr_traverse(state, iteration=1, opponents=[None, None, None])

    assert exc_info.value.reason == "opponent_missing"


def test_continue_training_uses_local_replay_iteration_after_high_resume(tmp_path, monkeypatch):
    patch_training_side_effects(monkeypatch)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    checkpoint_path = tmp_path / "resume" / "checkpoint_iter_5000.pt"
    checkpoint_path.parent.mkdir()
    write_checkpoint(checkpoint_path, iteration=5000)

    agent, losses, profits = train_mod.continue_training(
        checkpoint_path=str(checkpoint_path),
        additional_iterations=1,
        traversals_per_iteration=1,
        save_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        verbose=False,
        checkpoint_interval=1,
    )

    assert agent.iteration_count == 5001
    assert losses == [0]
    assert profits[0] == 0.0
    assert_replay_memory_shapes(agent, expected_iteration=1)

    saved_checkpoint = torch.load(
        tmp_path / "models" / "checkpoint_iter_5001.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert saved_checkpoint["iteration"] == 5001
    assert saved_checkpoint["metadata"]["iteration"] == 5001


def test_mixed_resume_keeps_absolute_checkpoint_metadata_but_local_replay(tmp_path, monkeypatch):
    patch_training_side_effects(monkeypatch)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    checkpoint_dir = tmp_path / "pool"
    (checkpoint_dir / "phase1").mkdir(parents=True)
    write_checkpoint(checkpoint_dir / "phase1" / "checkpoint_iter_1.pt")

    resume_checkpoint = tmp_path / "resume" / "checkpoint_iter_5000.pt"
    resume_checkpoint.parent.mkdir()
    write_checkpoint(resume_checkpoint, iteration=5000)

    agent, losses, profits, profits_vs_checkpoints = train_mod.train_with_mixed_checkpoints(
        checkpoint_dir=str(checkpoint_dir),
        training_model_prefix="*checkpoint_iter_",
        additional_iterations=1,
        traversals_per_iteration=1,
        save_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        refresh_interval=1000,
        num_opponents=1,
        verbose=False,
        checkpoint_path=str(resume_checkpoint),
        checkpoint_interval=1,
    )

    assert agent.iteration_count == 5001
    assert losses == [0]
    assert profits[0] == 0.0
    assert profits_vs_checkpoints[0] == 0.0
    assert_replay_memory_shapes(agent, expected_iteration=1)

    saved_checkpoint = torch.load(
        tmp_path / "models" / "mixed_checkpoint_iter_5001.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert saved_checkpoint["iteration"] == 5001
    assert saved_checkpoint["metadata"]["iteration"] == 5001


def test_mixed_opponent_selection_honors_five_checkpoint_opponents(tmp_path):
    random.seed(0)
    checkpoint_dir = tmp_path / "pool"
    checkpoint_dir.mkdir()
    for index in range(5):
        write_checkpoint(checkpoint_dir / f"checkpoint_iter_{index + 1}.pt")

    opponents, selected_files = train_mod._select_mixed_checkpoint_opponents(
        checkpoint_dir=str(checkpoint_dir),
        opponent_model_pattern="*checkpoint_iter_",
        num_opponents=5,
        device="cpu",
        player_id=0,
        num_players=6,
    )

    checkpoint_opponents = [
        opponent
        for opponent in opponents
        if isinstance(opponent, train_mod.CheckpointAgent)
    ]
    assert len(selected_files) == 5
    assert len(checkpoint_opponents) == 5
    assert {opponent.player_id for opponent in checkpoint_opponents} == {1, 2, 3, 4, 5}


def test_mixed_opponent_selection_rejects_empty_pool_without_opt_in(tmp_path):
    checkpoint_dir = tmp_path / "empty_pool"
    checkpoint_dir.mkdir()

    with pytest.raises(ValueError, match="No checkpoint files found"):
        train_mod._select_mixed_checkpoint_opponents(
            checkpoint_dir=str(checkpoint_dir),
            opponent_model_pattern="*checkpoint_iter_",
            num_opponents=5,
            device="cpu",
            player_id=0,
            num_players=6,
        )

    opponents, selected_files = train_mod._select_mixed_checkpoint_opponents(
        checkpoint_dir=str(checkpoint_dir),
        opponent_model_pattern="*checkpoint_iter_",
        num_opponents=5,
        device="cpu",
        player_id=0,
        num_players=6,
        allow_random_fallback=True,
    )

    assert selected_files == []
    assert sum(isinstance(opponent, train_mod.RandomAgent) for opponent in opponents) == 5


def test_mixed_training_does_not_write_to_pool_without_opt_in(tmp_path, monkeypatch):
    patch_training_side_effects(monkeypatch)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    checkpoint_dir = tmp_path / "pool"
    checkpoint_dir.mkdir()
    write_checkpoint(checkpoint_dir / "checkpoint_iter_1.pt")

    train_mod.train_with_mixed_checkpoints(
        checkpoint_dir=str(checkpoint_dir),
        training_model_prefix="*checkpoint_iter_",
        additional_iterations=1,
        traversals_per_iteration=1,
        save_dir=str(tmp_path / "models"),
        log_dir=str(tmp_path / "logs"),
        refresh_interval=1000,
        num_opponents=1,
        verbose=False,
        checkpoint_interval=1,
    )

    assert sorted(path.name for path in checkpoint_dir.glob("*.pt")) == ["checkpoint_iter_1.pt"]


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
