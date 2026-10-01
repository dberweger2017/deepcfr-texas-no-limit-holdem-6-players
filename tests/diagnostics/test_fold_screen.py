"""Fold screens use public wager accounting and preserve descriptive denominators."""
from copy import deepcopy

import pytest

from scripts.analyze_hu20_fold_screen import accumulate, fold_measure


def action(call=100, pot=300, own_bet=0, stack=1000, rival_bet=100, kind="fold"):
    return {"logical_player": 0, "street": "flop", "kind": kind,
            "observation": {"call_amount": call, "pot": pot, "street_bet": own_bet,
                "stack": stack, "position": "button", "visits": 1000,
                "menu": [{"kind": "fold"}, {"kind": "call"}], "probabilities": [.75, .25],
                "public_context": {"players": [{"street_bet": own_bet}, {"street_bet": rival_bet}]}}}


def row(block, actions, chips=-100):
    return {"panel": "lbr", "seed": 1, "version": "v1", "block": block,
            "rotation": 0, "status": "complete", "native_replay_verified": True,
            "actions": actions, "target_chips": chips}


def test_arithmetic_and_ending_partitions():
    result = accumulate([row(0, [action()]), row(1, [action(kind="call")], 200)])["lbr"]["v1"]
    cell = result["facing_wagers"]["flop"]["all"]["all"]
    assert cell["decisions"] == 2
    assert cell["mean_fold_probability"] == .75
    assert cell["actual_fold_fraction"] == .5
    assert cell["mean_call_pot_screen"] == pytest.approx(1 / 3)
    assert cell["mean_gap"] == pytest.approx(.75 - 1 / 3)
    endings = result["last_betting_street_and_ending"]["flop"]
    assert endings == {"target_fold": {"hands": 1, "chips": -100}, "showdown": {"hands": 1, "chips": 200}}


def test_first_bets_raises_and_stack_caps_are_separate():
    assert fold_measure(action())["spot"] == "first_bet"
    raised = action(own_bet=100, rival_bet=200, pot=500)
    assert fold_measure(raised)["spot"] == "raise_response"
    assert fold_measure(raised)["call_pot_screen"] == .2
    assert fold_measure(action(call=50, stack=50))["spot"] == "first_bet_capped_call"
    # A raise's paid increment may exceed the outstanding call: do not call
    # the latter divided by pot a universal MDF threshold.
    assert fold_measure(action(call=50, stack=50, own_bet=100, rival_bet=200))["spot"] == "raise_response_capped_call"


def test_no_private_cards_or_outcomes_enter_decision_screen():
    original = action()
    changed = deepcopy(original)
    changed["observation"].update(hole_cards=["As", "Ah"], hidden_cards=["Ks", "Kh"], future_deck=["2c"])
    assert fold_measure(original) == fold_measure(changed)


def test_reject_duplicate_incomplete_and_inconsistent_input():
    original = row(0, [action()])
    with pytest.raises(ValueError, match="Duplicate"):
        accumulate([original, original])
    with pytest.raises(ValueError, match="Incomplete"):
        accumulate([{**original, "native_replay_verified": False}])
    with pytest.raises(ValueError, match="public commitments"):
        fold_measure(action(call=90))
    invalid = action()
    invalid["observation"]["probabilities"] = [.5, .4]
    with pytest.raises(ValueError, match="Unnormalized"):
        fold_measure(invalid)
