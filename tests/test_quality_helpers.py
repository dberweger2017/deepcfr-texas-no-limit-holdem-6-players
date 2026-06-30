import pokers as pkrs
import pytest
import torch

from src.agents import random_agent as random_agent_mod
from src.agents.random_agent import RandomAgent
from src.core.deep_cfr import DeepCFRAgent
from src.opponent_modeling.deep_cfr_with_opponent_modeling import (
    DeepCFRAgentWithOpponentModeling,
)
from src.utils.agents import (
    CheckpointAgent,
    checkpoint_uses_opponent_modeling,
    create_agent_for_checkpoint,
)
from src.utils.evaluation import evaluate_agent_matchup
from src.utils.settings import is_strict_checking, set_strict_checking
from src.utils.actions import (
    ACTION_TYPE_RAISE,
    ActionMappingFailure,
    action_type_to_pokers_action,
    build_raise_action,
    sanitize_action,
)


class _InvalidApplyResult:
    status = pkrs.StateStatus.HighBet


class _StrictProbeState:
    legal_actions = [pkrs.ActionEnum.Raise, pkrs.ActionEnum.Call]

    def apply_action(self, action):
        return _InvalidApplyResult()


class _CallOnlyState:
    legal_actions = [pkrs.ActionEnum.Call]
    pot = 10.0


class _CheckAgent:
    def __init__(self, player_id=0):
        self.player_id = player_id

    def choose_action(self, state):
        return pkrs.Action(pkrs.ActionEnum.Check)


def test_strict_checking_is_read_after_setter(monkeypatch):
    set_strict_checking(False)
    assert not is_strict_checking()

    monkeypatch.setattr(random_agent_mod, "log_game_error", lambda *args, **kwargs: "log.txt")
    monkeypatch.setattr(
        random_agent_mod,
        "preset_raise_action",
        lambda state, preset: pkrs.Action(pkrs.ActionEnum.Raise, 10.0),
    )
    monkeypatch.setattr(
        random_agent_mod.random,
        "choice",
        lambda values: pkrs.ActionEnum.Raise if pkrs.ActionEnum.Raise in values else values[0],
    )

    agent = RandomAgent(0)
    assert agent.choose_action(_StrictProbeState()).action == pkrs.ActionEnum.Raise

    set_strict_checking(True)
    try:
        with pytest.raises(ActionMappingFailure):
            agent.choose_action(_StrictProbeState())
    finally:
        set_strict_checking(False)


def test_strict_action_mapping_raises_instead_of_fallback(monkeypatch):
    with pytest.raises(ActionMappingFailure):
        action_type_to_pokers_action(ACTION_TYPE_RAISE, _CallOnlyState(), strict=True)

    state = pkrs.State.from_seed(
        n_players=6,
        button=3,
        sb=1,
        bb=2,
        stake=200.0,
        seed=0,
    )
    monkeypatch.setattr("src.utils.actions._engine_accepts_action", lambda state, action: False)
    with pytest.raises(ActionMappingFailure):
        build_raise_action(state, 10.0, strict=True)


def test_non_strict_sanitization_records_fallback():
    events = []
    action = sanitize_action(
        _CallOnlyState(),
        pkrs.Action(pkrs.ActionEnum.Fold),
        fallback_recorder=lambda **event: events.append(event),
    )

    assert action.action == pkrs.ActionEnum.Call
    assert len(events) == 1
    assert events[0]["reason"] == "Action ActionEnum.Fold is not legal"


def test_shared_evaluation_metrics_shape():
    set_strict_checking(False)
    agent = RandomAgent(0)
    opponents = [RandomAgent(i) for i in range(6)]

    metrics = evaluate_agent_matchup(
        agent,
        opponents,
        num_games=1,
        seed_start=0,
        strict=False,
        label="test metrics shape",
    )

    expected_keys = {
        "avg_profit",
        "completed_games",
        "invalid_state_games",
        "invalid_state_count",
        "non_zero_sum_games",
        "setup_errors",
        "requested_games",
    }
    assert expected_keys.issubset(metrics)
    assert metrics["requested_games"] == 1


def test_strict_evaluation_raises_on_illegal_action():
    agent = _CheckAgent(player_id=0)
    opponents = [RandomAgent(i) for i in range(6)]

    with pytest.raises(ActionMappingFailure):
        evaluate_agent_matchup(
            agent,
            opponents,
            num_games=1,
            button_start=3,
            seed_start=0,
            strict=True,
            label="strict illegal action",
        )


def test_non_strict_evaluation_reports_sanitized_actions():
    agent = _CheckAgent(player_id=0)
    opponents = [RandomAgent(i) for i in range(6)]

    metrics = evaluate_agent_matchup(
        agent,
        opponents,
        num_games=1,
        button_start=3,
        seed_start=0,
        strict=False,
        label="non-strict illegal action",
    )

    assert metrics["sanitized_actions"] > 0
    assert metrics["agent_sanitized_actions"] > 0


def test_checkpoint_agent_loader_uses_metadata_not_filename(tmp_path):
    standard_path = tmp_path / "standard_with_om_in_name.pt"
    standard_agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    standard_agent.iteration_count = 3
    standard_agent.save_model(standard_path)

    om_path = tmp_path / "plain_checkpoint.pt"
    om_agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=6, device="cpu")
    om_agent.iteration_count = 4
    om_agent.save_model(om_path)

    loaded_standard = create_agent_for_checkpoint(standard_path, player_id=2, device="cpu")
    loaded_om = create_agent_for_checkpoint(om_path, player_id=3, device="cpu")

    assert isinstance(loaded_standard, DeepCFRAgent)
    assert loaded_standard.player_id == 2
    assert isinstance(loaded_om, DeepCFRAgentWithOpponentModeling)
    assert loaded_om.player_id == 3

    wrapped_standard = CheckpointAgent(player_id=1, model_path=standard_path, device="cpu")
    wrapped_om = CheckpointAgent(player_id=1, model_path=om_path, device="cpu")
    assert not wrapped_standard.with_opponent_modeling
    assert wrapped_om.with_opponent_modeling


def test_legacy_checkpoint_type_detection_uses_contents(tmp_path):
    checkpoint_path = tmp_path / "not_named_om.pt"
    torch.save(
        {
            "iteration": 1,
            "history_encoder": {},
            "opponent_model": {},
        },
        checkpoint_path,
    )

    assert checkpoint_uses_opponent_modeling(checkpoint_path)
