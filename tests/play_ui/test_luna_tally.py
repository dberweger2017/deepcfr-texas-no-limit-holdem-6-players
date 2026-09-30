import json

import pytest

from scripts.tally_luna_public import tally


def result(hand, bb):
    text = f'This hand: {bb} BB\n' + json.dumps({"type": "luna_observed", "handOrdinal": hand})
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
