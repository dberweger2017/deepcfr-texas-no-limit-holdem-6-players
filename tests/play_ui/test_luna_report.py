import pytest

from scripts.report_luna_browser import reconcile
from tests.play_ui.test_service import FixturePolicy, benchmark_create, hand_start, human_action, bot_action
from src.game.types import ActionKind
from src.play_api.service import PlayService


def completed_fixture(tmp_path, target=1):
    service = PlayService(tmp_path / "journal.sqlite", FixturePolicy())
    public = hand_start(service, benchmark_create(service, target=target, mode="restricted"))
    public = human_action(service, public, "fold")
    if target > 1:
        service.end_benchmark(public["sessionId"], "report-end-benchmark-001",
                              {"revision": public["revision"], "handId": public["hand"]["id"], "confirm": True})
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


def test_accepts_logged_visible_menu_field_without_changing_its_content(tmp_path):
    state, metadata = completed_fixture(tmp_path)
    metadata[1]["visibleLegalButtonLabels"] = metadata[1].pop("legalButtonLabels")
    rows, _ = reconcile(state, metadata)
    assert rows[0]["reportedMenuContainsAttempt"]
    assert rows[0]["legalButtonLabels"] == '["Fold", "Call 0.50 BB"]'
    metadata[1].pop("visibleLegalButtonLabels")
    with pytest.raises(ValueError, match="legal button labels"):
        reconcile(state, metadata)


def test_ended_early_requires_explicit_authorization_and_exact_completed_boundary(tmp_path):
    state, metadata = completed_fixture(tmp_path, target=2)
    assert state["benchmark"]["status"] == "ABORTED"
    with pytest.raises(ValueError, match="Only completed"):
        reconcile(state, metadata)
    with pytest.raises(ValueError, match="explicit completed-hand count"):
        reconcile(state, metadata, allow_aborted=True)
    with pytest.raises(ValueError, match="declared analysis count"):
        reconcile(state, metadata, allow_aborted=True, expected_completed=2)
    rows, hands = reconcile(state, metadata, allow_aborted=True, expected_completed=1)
    assert len(rows) == len(hands) == 1
    assert state["benchmark"]["targetHands"] == 2 and state["benchmark"]["status"] == "ABORTED"
    state["benchmark"]["abortedHandId"] = "unfinished-hand"
    with pytest.raises(ValueError, match="completed-hand boundary"):
        reconcile(state, metadata, allow_aborted=True, expected_completed=1)


def test_retains_explicit_retry_and_missing_observation_without_inventing_confirmation(tmp_path):
    state, metadata = completed_fixture(tmp_path)
    retry = dict(metadata[1], attemptedAtMs=2200, toolRetry=True)
    rows, _ = reconcile(state, [metadata[0], metadata[1], retry])
    assert rows[0]["attemptCount"] == 2 and rows[0]["toolRetry"] is True
    assert not rows[0]["observationRecorded"] and rows[0]["uiConfirmationMs"] is None
    assert '2200' in rows[0]["attemptRecords"] and '2000' in rows[0]["attemptRecords"]
    with pytest.raises(ValueError, match="count mismatch"):
        reconcile(state, [metadata[0], metadata[1], dict(retry, attemptedButtonLabel="Call 0.50 BB")])


def test_reset_stacks_do_not_reset_accumulated_profit(tmp_path):
    class MinRaisePolicy(FixturePolicy):
        def distribution(self, view):
            menu, _, _ = super().distribution(view)
            selected = next(i for i, item in enumerate(menu) if item.action.kind == ActionKind.RAISE)
            return menu, tuple(float(i == selected) for i in range(len(menu))), True

    service = PlayService(tmp_path / "reset.sqlite", MinRaisePolicy())
    try:
        state = hand_start(service, benchmark_create(service, target=2, mode="restricted"))
        state = human_action(service, state, "fold")
        assert state["hand"]["players"][0]["stack"] == 1950
        assert state["hand"]["result"]["humanChips"] == -50
        state = service.new_hand(state["sessionId"], "reset-second-deal-001", {"revision": state["revision"]})
        # Every seat receives a fresh 2,000, then pays its new blind.
        assert [p["stack"] for p in state["hand"]["players"]] == [1900, 1950]
        assert service._load(state["sessionId"])["totalChips"] == -50
        state = bot_action(service, state)
        state = human_action(service, state, "fold", key="reset-second-fold-001")
        assert state["hand"]["players"][0]["stack"] == 1900
        assert state["hand"]["result"]["humanChips"] == -100
        assert state["benchmarkResult"]["netChips"] == -150
        assert state["benchmarkResult"]["netBB"] == -1.5
        assert service.verify_replay(state["sessionId"]) == 2
    finally:
        service.close()
