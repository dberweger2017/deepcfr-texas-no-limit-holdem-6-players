from dataclasses import replace

import pytest

from scripts.report_holdem_branching import (
    decision_counts,
    family_interval,
    paired_rates,
)
from src.arena.schedule import Plan, Scenario, digest


def outcomes(plan, changes=None):
    rows = []
    for block in range(plan.blocks):
        for rotation in range(6):
            for arm in ("candidate", "baseline"):
                row = {
                    "scenario": "six",
                    "block": block,
                    "rotation": rotation,
                    "arm": arm,
                    "hand": 0,
                    "status": "completed",
                    "candidate_chips": (changes[block] if changes else 0)
                    if arm == "candidate"
                    else -1,
                    "big_blind": 2,
                    "participants": ["player-0"],
                    "opponents": ["random"],
                }
                row["outcome_sha256"] = digest(row)
                rows.append(row)
    return rows


def test_paired_interval_counts_deals_not_rotations():
    plan = Plan(
        (Scenario("six", stacks=(200,) * 6, small_blind=1, big_blind=2),), blocks=32
    )
    first = outcomes(plan)
    second = outcomes(plan, list(range(32)))
    result = paired_rates(plan, first, second)
    assert result["difference"]["blocks"] == 32
    assert result["difference"]["bb_per_100"] == 50 * 15.5
    assert result["block_differences_bb100"] == [50 * n for n in range(32)]
    family = family_interval(result["block_differences_bb100"])
    assert family[0] < result["difference"]["ci95"][0]
    assert family[1] > result["difference"]["ci95"][1]
    assert (
        paired_rates(plan, second[::-1], first)["difference"]["bb_per_100"]
        == -50 * 15.5
    )


def test_missing_rotations_and_altered_controls_are_rejected():
    plan = Plan((Scenario("six"),), blocks=32)
    rows = outcomes(plan)
    with pytest.raises(ValueError, match="Incomplete"):
        paired_rates(plan, rows, rows[:-1])
    changed = outcomes(plan)
    changed[1]["candidate_chips"] += 1
    with pytest.raises(ValueError, match="altered"):
        paired_rates(plan, rows, changed)
    changed[1]["outcome_sha256"] = digest(
        {k: v for k, v in changed[1].items() if k != "outcome_sha256"}
    )
    with pytest.raises(ValueError, match="controls"):
        paired_rates(plan, rows, changed)
    with pytest.raises(ValueError, match="Incomplete"):
        paired_rates(replace(plan, blocks=33), rows, rows)


def test_small_or_constant_samples_do_not_claim_certainty():
    assert family_interval([1] * 256) is None
    assert family_interval(list(range(8))) is None


def test_actions_track_rotated_hero_and_include_all_in_calls():
    rows = [
        {
            "arm": "candidate",
            "events": [
                {
                    "event": "HandStarted",
                    "seat_numbers": [2, 4],
                    "player_ids": ["opponent", "player-0"],
                    "stacks": [200, 200],
                },
                {"event": "BlindPosted", "seat": 4, "amount": 2},
                {
                    "event": "ActionTaken",
                    "seat": 2,
                    "street": "preflop",
                    "paid": 200,
                    "action": {"kind": "raise"},
                },
                {
                    "event": "ActionTaken",
                    "seat": 4,
                    "street": "flop",
                    "paid": 198,
                    "action": {"kind": "call"},
                },
            ],
        }
    ]
    result = decision_counts(rows)
    assert result == {
        "decisions_by_street": {"flop": 1},
        "actions": {"call": 1},
        "all_ins_by_street": {"flop": 1},
        "hands_with_postflop_decision": 1,
    }
