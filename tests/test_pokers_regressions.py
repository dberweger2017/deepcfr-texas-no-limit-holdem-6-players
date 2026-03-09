"""
Regression tests for `pokers` behavior relied on by this project.

Run with:
`python3 -m pytest tests/test_pokers_regressions.py -q`

Add future fixes as new entries in `REGRESSION_CASES`.
"""

import pokers as pkrs
import pytest


REGRESSION_CASES = [
    pytest.param(
        {
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
            "expected": {
                "stage": pkrs.Stage.Showdown,
                "status": pkrs.StateStatus.Ok,
                "final_state": True,
                "legal_actions": [],
                "zero_sum": True,
            },
            "milestones": {
                1: {"final_state": False},
                2: {"final_state": False},
            },
        },
        id="all-in-runout-goes-straight-to-showdown",
    ),
]


def replay_case(case):
    """Return the full state trace for a regression case."""
    state = pkrs.State.from_seed(**case["setup"])
    trace = [state]

    for action_enum, amount in case["actions"]:
        state = state.apply_action(pkrs.Action(action_enum, amount=amount))
        trace.append(state)

    return trace


@pytest.mark.parametrize("case", REGRESSION_CASES)
def test_pokers_regressions(case):
    trace = replay_case(case)

    for action_index, checks in case.get("milestones", {}).items():
        state = trace[action_index]
        for attr_name, expected_value in checks.items():
            assert getattr(state, attr_name) == expected_value

    final_state = trace[-1]
    expected = case["expected"]

    assert final_state.stage == expected["stage"]
    assert final_state.status == expected["status"]
    assert final_state.final_state == expected["final_state"]
    assert final_state.legal_actions == expected["legal_actions"]

    if expected.get("zero_sum"):
        assert abs(sum(player.reward for player in final_state.players_state)) < 1e-9
