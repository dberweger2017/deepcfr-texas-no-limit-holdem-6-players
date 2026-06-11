import pytest
import torch

from src.utils.checkpoints import (
    AGENT_TYPE_OPPONENT_MODELING,
    AGENT_TYPE_STANDARD,
    CHECKPOINT_SCHEMA_VERSION,
    attach_checkpoint_metadata,
    checkpoint_uses_opponent_modeling_state,
    find_checkpoint_dirs,
    infer_agent_type,
    load_play_agent,
    checkpoint_path_uses_opponent_modeling,
    select_random_checkpoints_in_dir,
    validate_checkpoint_compatibility,
)


class DummyAgent:
    player_id = 2
    num_players = 6
    iteration_count = 17


def test_attach_checkpoint_metadata_sets_top_level_and_nested_fields():
    checkpoint = attach_checkpoint_metadata(
        {"iteration": 17, "weights": object()},
        DummyAgent(),
        AGENT_TYPE_STANDARD,
    )

    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["agent_type"] == AGENT_TYPE_STANDARD
    assert checkpoint["num_players"] == 6
    assert checkpoint["player_id"] == 2
    assert checkpoint["metadata"]["iteration"] == 17
    assert checkpoint["metadata"]["agent_type"] == AGENT_TYPE_STANDARD


def test_infer_agent_type_supports_legacy_opponent_model_checkpoints():
    checkpoint = {
        "iteration": 1,
        "history_encoder": object(),
        "opponent_model": object(),
    }

    assert infer_agent_type(checkpoint) == AGENT_TYPE_OPPONENT_MODELING
    assert checkpoint_uses_opponent_modeling_state(checkpoint)


def test_infer_agent_type_defaults_legacy_standard_checkpoints():
    assert infer_agent_type({"iteration": 1}) == AGENT_TYPE_STANDARD


def test_validate_checkpoint_compatibility_rejects_wrong_agent_type():
    checkpoint = {"agent_type": AGENT_TYPE_OPPONENT_MODELING, "num_players": 6}

    with pytest.raises(ValueError, match="agent_type"):
        validate_checkpoint_compatibility(
            checkpoint,
            expected_agent_type=AGENT_TYPE_STANDARD,
            expected_num_players=6,
        )


def test_validate_checkpoint_compatibility_rejects_wrong_player_count():
    checkpoint = {"agent_type": AGENT_TYPE_STANDARD, "num_players": 3}

    with pytest.raises(ValueError, match="num_players"):
        validate_checkpoint_compatibility(
            checkpoint,
            expected_agent_type=AGENT_TYPE_STANDARD,
            expected_num_players=6,
        )


def test_find_checkpoint_dirs_returns_relative_paths(tmp_path):
    models = tmp_path / "models" / "standard" / "phase1"
    models.mkdir(parents=True)
    (models / "checkpoint_iter_100.pt").write_bytes(b"not-a-checkpoint")

    dirs = find_checkpoint_dirs(tmp_path / "models", relative_to=tmp_path)

    assert dirs == ["models/standard/phase1"]


def test_select_random_checkpoints_in_dir_respects_limit(tmp_path):
    for index in range(8):
        (tmp_path / f"checkpoint_iter_{index}.pt").write_text("x")

    selected = select_random_checkpoints_in_dir(tmp_path, num_models=5)

    assert len(selected) == 5
    assert all(path.parent == tmp_path for path in selected)


def test_checkpoint_path_uses_opponent_modeling(tmp_path):
    om_path = tmp_path / "om.pt"
    torch.save(
        {"iteration": 1, "history_encoder": {}, "opponent_model": {}},
        om_path,
    )
    std_path = tmp_path / "std.pt"
    torch.save({"iteration": 1}, std_path)

    assert checkpoint_path_uses_opponent_modeling(om_path) is True
    assert checkpoint_path_uses_opponent_modeling(std_path) is False


def test_load_play_agent_falls_back_with_warning(tmp_path):
    bad_path = tmp_path / "broken.pt"
    bad_path.write_text("not a torch checkpoint")

    agent, warning = load_play_agent(bad_path, seat=2, device="cpu")

    assert warning is not None
    assert warning["seat"] == 2
    assert "broken.pt" in warning["checkpoint"]
    assert "Random (load failed" in agent.model_name
