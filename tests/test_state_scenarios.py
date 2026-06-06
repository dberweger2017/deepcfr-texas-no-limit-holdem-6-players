"""
Deterministic scenario coverage for core game-state edge cases.

This complements the smoke tests by replaying specific action sequences that
have broken in the past.
"""

from pathlib import Path

import pokers as pkrs
import pytest

from src.utils.logging import apply_action_with_logging


SCENARIO_CASES = [
    pytest.param(
        {
            "name": "forced-checkdown-goes-to-showdown",
            "setup": {
                "n_players": 3,
                "button": 0,
                "sb": 1.0,
                "bb": 2.0,
                "stake": 4.0,
                "seed": 0,
            },
            "actions": [
                (pkrs.ActionEnum.Raise, 2.0),
                (pkrs.ActionEnum.Call, 0.0),
                (pkrs.ActionEnum.Call, 0.0),
            ],
            "expected_stage": pkrs.Stage.Showdown,
            "expected_status": pkrs.StateStatus.Ok,
            "expected_final_state": True,
            "expected_legal_actions": [],
        },
        id="forced-checkdown-goes-to-showdown",
    ),
    pytest.param(
        {
            "name": "short-stack-can-call-but-cannot-raise",
            "setup": {
                "n_players": 3,
                "button": 0,
                "sb": 1.0,
                "bb": 2.0,
                "stake": 4.0,
                "seed": 0,
            },
            "actions": [
                (pkrs.ActionEnum.Raise, 2.0),
                (pkrs.ActionEnum.Call, 0.0),
            ],
            "expected_stage": pkrs.Stage.Preflop,
            "expected_status": pkrs.StateStatus.Ok,
            "expected_final_state": False,
            "expected_legal_actions": [pkrs.ActionEnum.Fold, pkrs.ActionEnum.Call],
        },
        id="short-stack-can-call-but-cannot-raise",
    ),
]


def replay_actions(case):
    state = pkrs.State.from_seed(**case["setup"])
    for action_enum, amount in case["actions"]:
        state = state.apply_action(pkrs.Action(action_enum, amount=amount))
    return state


@pytest.mark.parametrize("case", SCENARIO_CASES)
def test_deterministic_state_scenarios(case):
    state = replay_actions(case)

    assert state.stage == case["expected_stage"]
    assert state.status == case["expected_status"]
    assert state.final_state == case["expected_final_state"]
    assert state.legal_actions == case["expected_legal_actions"]


def test_high_bet_scenario_is_logged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = pkrs.State.from_seed(
        n_players=6,
        button=0,
        sb=1,
        bb=2,
        stake=200.0,
        seed=0,
    )

    new_state, log_file, status = apply_action_with_logging(
        state,
        pkrs.Action(pkrs.ActionEnum.Raise, 1000.0),
        strict=False,
    )

    assert new_state is None
    assert status == pkrs.StateStatus.HighBet
    assert log_file is not None
    assert Path(log_file).exists()


def test_high_bet_strict_mode_raises_and_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = pkrs.State.from_seed(
        n_players=6,
        button=0,
        sb=1,
        bb=2,
        stake=200.0,
        seed=0,
    )

    with pytest.raises(ValueError, match=r"State status not OK \(StateStatus.HighBet\)"):
        apply_action_with_logging(
            state,
            pkrs.Action(pkrs.ActionEnum.Raise, 1000.0),
            strict=True,
        )

    log_files = sorted((tmp_path / "logs").glob("poker_error_*.txt"))
    assert log_files
