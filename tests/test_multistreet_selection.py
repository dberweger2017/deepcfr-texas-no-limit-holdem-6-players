import pytest

from src.holdem.multistreet_selection import choose_durations, qualify


@pytest.fixture
def plan():
    return {
        "variants": ["scaled", "cards", "features"],
        "seeds": [11, 13, 17],
        "durations": [1024, 2048, 4096],
        "minimum_absolute_cost_gain_bb": 0.02,
        "minimum_relative_cost_gain": 0.1,
        "allowed_relative_rmse_increase": 0.02,
        "reference_se_multiplier": 2.0,
    }


def tuning_rows(plan):
    costs = {1024: [0.1, 0.8, 0.8], 2048: [0.5, 0.5, 0.5], 4096: [0.2, 0.8, 0.8]}
    return [
        {"variant": v, "seed": s, "duration": d, "weighted_decision_cost_bb": costs[d][i]}
        for v in plan["variants"] for i, s in enumerate(plan["seeds"]) for d in plan["durations"]
    ]


def validation_rows(plan):
    return [
        {
            "variant": v, "seed": s, "duration": 2048,
            "weighted_decision_cost_bb": 0.5 if v == "scaled" else 0.3,
            "relative_rmse": 0.6,
            "paired_gain_standard_error_bb": 0.01,
            "train_relative_rmse": 0.9,
        }
        for v in plan["variants"] for s in plan["seeds"]
    ]


def test_shared_duration_uses_all_seeds_not_best_seed_or_last_checkpoint(plan):
    chosen = choose_durations(tuning_rows(plan), plan)
    assert chosen["durations"] == dict.fromkeys(plan["variants"], 2048)


def test_duration_ties_choose_earliest(plan):
    rows = tuning_rows(plan)
    for row in rows:
        row["weighted_decision_cost_bb"] = 0.3
    assert set(choose_durations(rows, plan)["durations"].values()) == {1024}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "validation", "nan"])
def test_selection_rejects_incomplete_or_contaminated_tuning(plan, mutation):
    rows = tuning_rows(plan)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "validation":
        rows[0]["validation"] = {"cost": 0.0}
    else:
        rows[0]["weighted_decision_cost_bb"] = float("nan")
    with pytest.raises(ValueError):
        choose_durations(rows, plan)


def test_all_qualifiers_proceed_without_a_training_fit_gate(plan):
    result = qualify(validation_rows(plan), dict.fromkeys(plan["variants"], 2048), plan)
    assert result["eligible"] == ["cards", "features"]


def test_uncertain_gain_is_not_confirmation(plan):
    rows = validation_rows(plan)
    for row in rows:
        if row["variant"] == "features":
            row["paired_gain_standard_error_bb"] = 0.2
    result = qualify(rows, dict.fromkeys(plan["variants"], 2048), plan)
    assert result["eligible"] == ["cards"]


def test_validation_cannot_change_selected_duration(plan):
    rows = validation_rows(plan)
    rows[-1]["duration"] = 4096
    with pytest.raises(ValueError, match="selected duration"):
        qualify(rows, dict.fromkeys(plan["variants"], 2048), plan)


def test_paired_noise_excludes_between_context_variation(monkeypatch):
    import torch
    from scripts import check_multistreet_representation as runner

    candidate, baseline = object(), object()
    monkeypatch.setattr(
        runner, "model_policy",
        lambda model, target: torch.tensor(
            [1.0, 0.0] if model is candidate else [0.0, 1.0], dtype=torch.float64
        ),
    )
    rows = [
        {"target": None, "street": "flop", "group": "a", "world_action_values_bb": values}
        for values in ([[1, 0], [1, 0]], [[10, 0], [10, 0]])
    ]
    assert runner._paired_gain_se(candidate, baseline, rows) == 0


def test_paired_noise_weights_contexts_equally_despite_different_sample_counts(monkeypatch):
    import torch
    from scripts import check_multistreet_representation as runner

    candidate, baseline = object(), object()
    monkeypatch.setattr(
        runner, "model_policy",
        lambda model, target: torch.tensor(
            [1.0, 0.0] if model is candidate else [0.0, 1.0], dtype=torch.float64
        ),
    )
    rows = [
        {"target": None, "street": "flop", "group": "a", "world_action_values_bb": values}
        for values in ([[0, 0], [2, 0]], [[0, 0], [2, 0], [0, 0], [2, 0]])
    ]
    assert runner._paired_gain_se(candidate, baseline, rows) == pytest.approx((1 / 3) ** 0.5)
