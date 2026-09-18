import pytest

from scripts.analyze_multistreet_decision_errors import (
    _enrich_decision,
    paired_best_action_gap,
    reconstruct_cost,
)


def test_reconstruct_cost_is_zero_for_a_best_action_distribution():
    result = reconstruct_cost([0.0, 1.0], [-2.0, 3.0])
    assert result["cost_bb"] == pytest.approx(0.0)
    assert result["contributions_bb"] == pytest.approx([0.0, 0.0])
    assert result["policy_value_bb"] == pytest.approx(3.0)


def test_reconstruct_cost_allocates_each_action_loss():
    result = reconstruct_cost([0.25, 0.75], [0.0, 4.0])
    assert result["cost_bb"] == pytest.approx(1.0)
    assert result["contributions_bb"] == pytest.approx([1.0, 0.0])
    assert result["policy_value_bb"] == pytest.approx(3.0)


def test_paired_best_action_gap_uses_world_pairings_for_se():
    result = paired_best_action_gap([[3.0, 1.0], [5.0, 1.0], [1.0, 1.0]])
    assert result["best_action_index"] == 0
    assert result["second_action_index"] == 1
    assert result["gap_bb"] == pytest.approx(2.0)
    assert result["gap_se_bb"] == pytest.approx(2.0 / 3**0.5)


def test_paired_best_action_gap_has_no_false_precision_for_one_world():
    result = paired_best_action_gap([[3.0, 1.0]])
    assert result["gap_bb"] == pytest.approx(2.0)
    assert result["gap_se_bb"] is None


def test_enrich_decision_uses_saved_action_labels_for_kind_buckets():
    context = {
        "context": "flop-0",
        "split": "train",
        "street": "flop",
        "situation": "open",
        "family": 0,
        "board": ["Ah", "9h", "Qd"],
        "holding": ["Ac", "Ad"],
        "visible_hand_category": "pair",
        "board_paired": False,
        "board_flush_texture": "two_same_suit",
        "world_action_values_bb": [[1.0, 3.0], [1.0, 3.0]],
        "world_count": 2,
    }
    cache = {
        "target": {
            "actions": [
                "Action(kind=<ActionKind.FOLD: 'fold'>, raise_to=None)",
                "Action(kind=<ActionKind.CALL: 'call'>, raise_to=None)",
            ]
        }
    }
    decision = {
        "context": "flop-0",
        "probabilities": [0.5, 0.5],
        "predicted_regrets_bb": [1.0, 1.0],
        "reference_values_bb": [1.0, 3.0],
        "decision_cost_bb": 1.0,
    }
    result = _enrich_decision(decision, context, cache)
    assert result["actions"][0]["kind"] == "fold"
    assert result["missed_value_folds_bb"] == pytest.approx(1.0)
    assert result["costly_calls_or_raises_bb"] == pytest.approx(0.0)


def test_reconstruct_cost_rejects_non_normalized_policy():
    with pytest.raises(ValueError, match="sum to one"):
        reconstruct_cost([0.4, 0.4], [0.0, 1.0])


def test_enrich_decision_rejects_reference_values_not_equal_to_world_means():
    context = {
        "context": "flop-0", "split": "train", "street": "flop", "situation": "open",
        "family": 0, "board": ["Ah", "9h", "Qd"], "holding": ["Ac", "Ad"],
        "visible_hand_category": "trips", "board_paired": False,
        "board_flush_texture": "two_same_suit", "world_action_values_bb": [[1.0, 3.0]],
        "world_count": 1,
    }
    cache = {"target": {"actions": [
        "Action(kind=<ActionKind.FOLD: 'fold'>, raise_to=None)",
        "Action(kind=<ActionKind.CALL: 'call'>, raise_to=None)",
    ]}}
    decision = {
        "context": "flop-0", "probabilities": [0.0, 1.0], "predicted_regrets_bb": [0.0, 1.0],
        "reference_values_bb": [1.0, 2.0], "decision_cost_bb": 0.0,
    }
    with pytest.raises(ValueError, match="world-Q means"):
        _enrich_decision(decision, context, cache)
