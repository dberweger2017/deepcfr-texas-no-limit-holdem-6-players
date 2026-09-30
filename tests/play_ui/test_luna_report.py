import pytest

from scripts.report_luna_browser import reconcile
from tests.play_ui.test_service import FixturePolicy, benchmark_create, hand_start, human_action
from src.play_api.service import PlayService


def completed_fixture(tmp_path):
    service = PlayService(tmp_path / "journal.sqlite", FixturePolicy())
    public = hand_start(service, benchmark_create(service, target=1, mode="restricted"))
    human_action(service, public, "fold")
    state = service._load(public["sessionId"])
    service.close()
    metadata = [
        {"type": "luna_attempt", "handOrdinal": 1, "decisionOrdinal": 1,
         "attemptedButtonLabel": "Deal hand 1 / 1"},
        {"type": "luna_attempt", "handOrdinal": 1, "decisionOrdinal": 2,
         "attemptedButtonLabel": "Fold", "legalButtonLabels": ["Fold", "Call 0.50 BB"],
         "attemptedAtMs": 2000, "lastRenderedAtMs": 1000},
        {"type": "luna_observed", "handOrdinal": 1, "decisionOrdinal": 2,
         "observedAtMs": 2300, "visibleError": None},
    ]
    return state, metadata


def test_reconciles_deal_numbered_as_browser_decision_without_publishing_private_state(tmp_path):
    state, metadata = completed_fixture(tmp_path)
    rows, hands = reconcile(state, metadata)
    assert len(rows) == len(hands) == 1
    assert rows[0]["decisionOrdinal"] == 1 and rows[0]["browserDecisionOrdinal"] == 2
    assert rows[0]["acceptedKind"] == "fold"
    assert rows[0]["observedDecisionMs"] == 1000 and rows[0]["uiConfirmationMs"] == 300
    assert not {"dealSeed", "botRng", "dealRng", "holeCards", "lookup"} & rows[0].keys()


def test_refuses_active_incomplete_duplicate_and_tampered_results(tmp_path):
    state, metadata = completed_fixture(tmp_path)
    with pytest.raises(ValueError, match="count mismatch"):
        reconcile(state, metadata + [metadata[1]])
    state["history"][0]["humanChips"] += 1
    with pytest.raises(AssertionError):
        reconcile(state, metadata)
    state["benchmark"]["status"] = "ACTIVE"
    with pytest.raises(ValueError, match="completed"):
        reconcile(state, metadata)


def test_retains_misclick_result_instead_of_discarding_the_hand(tmp_path):
    state, metadata = completed_fixture(tmp_path)
    metadata[1]["attemptedButtonLabel"] = "Call 0.50 BB"
    rows, hands = reconcile(state, metadata)
    assert not rows[0]["attemptMatchesAccepted"]
    assert rows[0]["acceptedKind"] == "fold" and hands[0]["humanChips"] == -50
