import pytest

from src.utils.checkpoints import (
    AGENT_TYPE_OPPONENT_MODELING,
    AGENT_TYPE_STANDARD,
    CHECKPOINT_SCHEMA_VERSION,
    attach_checkpoint_metadata,
    checkpoint_uses_opponent_modeling_state,
    infer_agent_type,
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
