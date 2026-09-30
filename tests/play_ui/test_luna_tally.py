import json

import pytest

from scripts.tally_luna_public import tally


def result(hand, bb, metadata_hand=None):
    text = f'Hand {hand + 1} / 500\nThis hand: {bb} BB\n' + json.dumps(
        {"type": "luna_observed", "handOrdinal": hand if metadata_hand is None else metadata_hand})
    return {"type": "response_item", "payload": {"type": "function_call_output",
            "output": [{"type": "input_text", "text": text}]}}


def test_public_tally_deduplicates_and_ignores_reasoning():
    rows = [result(1, "+2"), result(1, "+2"), result(2, "−1.50"),
            {"type": "response_item", "payload": {"type": "reasoning", "text": "This hand: +999 BB"}}]
    report = tally(rows)
    assert report["throughHand"] == 2 and report["netChips"] == 50
    assert report["wins"] == report["losses"] == 1


def test_refuses_missing_or_conflicting_public_results():
    with pytest.raises(ValueError, match="Missing"):
        tally([result(1, "+2"), result(3, "+1")])
    with pytest.raises(ValueError, match="Conflicting"):
        tally([result(1, "+2"), result(1, "+1")])


def test_uses_rendered_progress_when_player_metadata_lags_auto_fold():
    report = tally([result(1, "+1"), result(2, "+0.50", metadata_hand=1)])
    assert report["throughHand"] == 2 and report["netChips"] == 150


def test_final_completed_progress_and_chip_precision():
    row = result(1, "+1")
    row["payload"]["output"][0]["text"] = '1 / 1 completed\nThis hand: +1 BB'
    assert tally([row])["netChips"] == 100
    with pytest.raises(ValueError, match="Invalid"):
        tally([result(1, "+0.001")])


def test_calibration_current_hand_progress_is_explicit():
    row = result(1, "+2")
    row["payload"]["output"][0]["text"] = 'Hand 1 / 100\nThis hand: +2 BB'
    assert tally([row], progress_mode="current-hand")["netChips"] == 200
    with pytest.raises(ValueError, match="Invalid"):
        tally([row])
