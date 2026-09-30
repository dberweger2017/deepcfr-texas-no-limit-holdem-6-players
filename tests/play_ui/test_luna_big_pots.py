import csv
import json
from pathlib import Path

import pytest

from scripts.analyse_luna_big_pots import analyse


def test_recorded_decomposition_distinguishes_full_stacks_and_removed_hand_count():
    root = Path(__file__).resolve().parents[2] / "docs/reports/luna-browser/primary"
    for threshold in (5, 8, 12):
        result = analyse(root, threshold)
        groups = [result[key] for key in ("large_call_folded", "large_call_continued", "other_hands")]
        assert sum(group["hands"] for group in groups) == 400
        assert sum(group["luna_bb"] for group in groups) == 107
        assert result["full_stacks_won"] == 12 and result["full_stacks_lost"] == 0
        assert result["near_full_stack_wins_19_to_under20"] == 2
        trimmed = result["without_top10_wins"]
        assert trimmed["removed_hands"] == 10 and trimmed["remaining_hands"] == 390
        assert trimmed["remaining_bb"] == -93
        assert trimmed["bb_per_100"] == pytest.approx(-23.8461538462)
    result = analyse(root, 8)
    assert result["large_call_continued"]["hands"] == 15
    assert result["large_call_continued"]["luna_bb"] == 276
    assert result["large_call_continued"]["ties"] == 1
    assert sum(bool(h["legitimately_shown_bot_cards"]) for h in result["continued_public_hands"]) == 11
    assert not any(key in result for key in ("overall_interval", "illustrative_tail_probabilities"))


def test_first_threshold_action_and_empty_groups_use_public_disclosures_only(tmp_path):
    with (tmp_path / "hands.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["handId", "handOrdinal", "humanChips"])
        writer.writeheader()
        writer.writerow({"handId": "a", "handOrdinal": 1, "humanChips": -100})
        writer.writerow({"handId": "b", "handOrdinal": 2, "humanChips": 1900})
    with (tmp_path / "decisions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["handId", "legalButtonLabels", "acceptedKind", "street", "humanVisibleCards", "visibleBoard"])
        writer.writeheader()
        for hand_id, label, kind in (("a", "Call 7.99 BB", "call"), ("a", "Call 8 BB", "fold"), ("b", "Call 8 BB", "call")):
            writer.writerow({"handId": hand_id, "legalButtonLabels": json.dumps([label]), "acceptedKind": kind,
                             "street": "river", "humanVisibleCards": '["As", "Ks"]', "visibleBoard": '[]'})
    result = analyse(tmp_path, 8)
    assert result["large_call_folded"]["hands"] == result["large_call_continued"]["hands"] == 1
    assert result["other_hands"]["bb_per_100"] is None
    assert not result["public_history_available"]
    assert result["continued_public_hands"][0]["legitimately_shown_bot_cards"] == []
    assert result["full_stacks_won"] == 0 and result["near_full_stack_wins_19_to_under20"] == 1
    assert result["without_top10_wins"]["remaining_hands"] == 1
    assert result["without_top10_wins"]["bb_per_100"] == -100
