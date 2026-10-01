"""Whole-hand partitions remain disjoint despite repeated target decisions."""
import pytest

from scripts.analyze_hu20_card_v2_lbr import accumulate


def row(block, chips, decisions, last):
    actions = [{"logical_player": 0, "street": street,
        "observation": {"visits": visits}} for street, visits in decisions]
    actions.append({"logical_player": 1, "street": last,
        "observation": {"visits": None}})
    return {"panel": "lbr", "seed": 1, "version": "v1", "block": block,
        "rotation": 0, "target_chips": chips, "actions": actions,
        "status": "complete", "native_replay_verified": True}


def test_returns_count_once_and_late_groups_are_disjoint():
    rows = [row(0, -2000, [("preflop", 1000)], "preflop"),
        row(1, 500, [("flop", 100), ("turn", 9), ("river", 0)], "river"),
        row(2, -100, [("turn", 100), ("river", 10)], "river"),
        row(3, 2000, [("turn", 100), ("river", 200)], "river")]
    actual = accumulate(rows)["v1"]
    assert actual["chips"] == 400 and actual["hands"] == 4
    assert actual["last_betting_street"] == {
        "preflop": {"hands": 1, "chips": -2000}, "river": {"hands": 3, "chips": 2400}}
    assert actual["minimum_target_late_visits"] == {
        "no_target_late_decision": {"hands": 1, "chips": -2000},
        "under_10": {"hands": 1, "chips": 500},
        "10-99": {"hands": 1, "chips": -100}, "100+": {"hands": 1, "chips": 2000}}
    assert actual["decision_visits"]["river"] == {"0": 1, "10-99": 1, "100+": 1}
    for partition in ("last_betting_street", "minimum_target_late_visits"):
        assert sum(group["chips"] for group in actual[partition].values()) == actual["chips"]


def test_reject_duplicate_or_incomplete_evidence():
    original = row(0, 50, [("preflop", 20)], "preflop")
    with pytest.raises(ValueError, match="Duplicate"):
        accumulate([original, original])
    with pytest.raises(ValueError, match="Incomplete"):
        accumulate([{**original, "native_replay_verified": False}])
