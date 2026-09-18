import copy
import hashlib
import json

import numpy as np
import pytest

from src.holdem.multistreet_reference_stability import (
    FROZEN_WORLD_BUDGETS,
    _decision_metrics,
    audit_cache_entries,
    run_audit,
)


def _hash_entry(entry):
    identity = {key: value for key, value in entry.items() if key != "sha256"}
    entry["sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return entry


def _entry(
    context,
    street,
    situation,
    profile,
    batch,
    offset=0.0,
    world_count=None,
    tail_offset=0.0,
):
    stratum = f"{street}:{situation}"
    budget = FROZEN_WORLD_BUDGETS[stratum]
    worlds = world_count or (128 if batch == "calibration" else budget)
    rows = [
        [
            1.0 + offset + (world % 3) + (tail_offset if world >= budget else 0),
            2.0 + offset - (world % 2) + (tail_offset if world >= budget else 0),
            3.0 + offset + (tail_offset if world >= budget else 0),
        ]
        for world in range(worlds)
    ]
    values = np.asarray(rows).mean(axis=0)
    weights = np.array([1.0, 2.0, 3.0] if profile == "increasing" else [1.0, 1.0, 1.0])
    policy = weights / weights.sum()
    regrets = values - policy @ values
    entry = {
        "format": "multistreet-reference-cache-v1",
        "source_sha256": "same-source-for-synthetic-fixture",
        "plan_sha256": "plan-fingerprint",
        "stream_namespace": batch,
        "stream_seed": 11 if batch == "calibration" else 22,
        "stream_key": "",
        "selected_n": 128 if batch == "calibration" else dict(FROZEN_WORLD_BUDGETS),
        "context": context,
        "split": "train",
        "street": street,
        "situation": situation,
        "profile": profile,
        "world_seeds": list(range(worlds)),
        "worlds": worlds,
        "world_action_values_bb": rows,
        "action_standard_error_bb": [0.0, 0.0, 0.0],
        "uncertainty_status": "estimated",
        "nodes": 1,
        "seconds": 0.01,
        "target": {
            "actions": ["a", "b", "c"],
            "policy": policy.tolist(),
            "values_bb": values.tolist(),
            "regrets_bb": regrets.tolist(),
        },
        "world_fingerprint": {
            "assignments_sha256": f"assignments-{context}",
            "visible_observation_sha256": f"observation-{context}",
            "world_decks_sha256": f"decks-{context}",
            "world_seeds": list(range(worlds)),
            "worlds": worlds,
        },
    }
    entry["stream_key"] = hashlib.sha256(
        f"{entry['stream_namespace']}|{entry['stream_seed']}|"
        f"{entry['plan_sha256']}|{entry['selected_n']}".encode()
    ).hexdigest()
    return _hash_entry(entry)


def _fixture():
    calibration, production = [], []
    for index, stratum in enumerate(FROZEN_WORLD_BUDGETS):
        street, situation = stratum.split(":")
        context = f"{street}-{situation}-{index}"
        for profile in ("uniform", "increasing"):
            calibration.append(_entry(context, street, situation, profile, "calibration"))
            production.append(_entry(context, street, situation, profile, "production", offset=0.25))
    return calibration, production


def test_metrics_use_ties_and_null_correlation_explicitly():
    metrics = _decision_metrics(
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 2.0, 0.0]),
        np.array([0.0, 1.0, -1.0]),
    )
    assert metrics["pairwise_sign_agreement"] == pytest.approx(2 / 3)
    assert metrics["top_action_agreement"] == 0.0
    assert metrics["top_action_overlap"] == pytest.approx(1 / 2)
    constant = _decision_metrics(
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 0.0]),
        np.array([-1.0, 0.0, -2.0]),
    )
    assert constant["q_correlation"] is None


def test_cross_and_within_costs_use_same_q_with_explicit_signs():
    metrics = _decision_metrics(
        np.array([4.0, 1.0]),
        np.array([2.0, 1.0]),  # calibration policy: (2/3, 1/3)
        np.array([1.0, 3.0]),
        np.array([1.0, 3.0]),  # production policy: (1/4, 3/4)
    )
    assert metrics["within_cost_left_bb"] == pytest.approx(1.0)
    assert metrics["within_cost_right_bb"] == pytest.approx(0.5)
    assert metrics["cross_cost_left_policy_on_right_bb"] == pytest.approx(4 / 3)
    assert metrics["cross_cost_right_policy_on_left_bb"] == pytest.approx(9 / 4)
    assert metrics["paired_policy_value_difference_on_left_bb"] == pytest.approx(5 / 4)
    assert metrics["paired_policy_value_difference_on_right_bb"] == pytest.approx(-5 / 6)


def test_audit_recomputes_frozen_prefix_and_combines_profiles():
    calibration, production = _fixture()
    result = audit_cache_entries(calibration, production)
    assert result["overlap"] == {"contexts": 6, "entries": 12, "training_only": True}
    assert set(result["strata"]) == set(FROZEN_WORLD_BUDGETS)
    assert set(result["strata"]["river:open"]["profiles"]) == {
        "uniform",
        "increasing",
        "combined_1_to_2",
    }
    assert result["strata"]["river:open"]["world_budget"] == 32
    assert result["strata"]["flop:open"]["profiles"]["uniform"]["q_rmse_bb"] == pytest.approx(0.25)
    assert result["strata"]["flop:open"]["profiles"]["uniform"]["pooled_q_rmse_bb"] == pytest.approx(0.25)


def test_river_calibration_tail_is_ignored_in_frozen_prefix():
    calibration, production = [], []
    for profile in ("uniform", "increasing"):
        calibration.append(
            _entry(
                "river-open-tail",
                "river",
                "open",
                profile,
                "calibration",
                world_count=128,
                tail_offset=100.0,
            )
        )
        production.append(
            _entry(
                "river-open-tail",
                "river",
                "open",
                profile,
                "production",
                world_count=32,
            )
        )
    result = audit_cache_entries(calibration, production)
    metrics = result["strata"]["river:open"]["profiles"]["uniform"]
    assert metrics["q_rmse_bb"] == pytest.approx(0.0)
    assert metrics["pooled_q_rmse_bb"] == pytest.approx(0.0)


def test_stratum_reports_pooled_rmse_separately_from_context_mean():
    calibration, production = [], []
    for index, offset in enumerate((0.25, 1.0)):
        context = f"flop-open-heterogeneous-{index}"
        for profile in ("uniform", "increasing"):
            calibration.append(_entry(context, "flop", "open", profile, "calibration"))
            production.append(
                _entry(context, "flop", "open", profile, "production", offset=offset)
            )
    metrics = audit_cache_entries(calibration, production)["strata"]["flop:open"]["profiles"]["uniform"]
    assert metrics["q_rmse_bb"] == pytest.approx((0.25 + 1.0) / 2)
    assert metrics["pooled_q_rmse_bb"] == pytest.approx(np.sqrt((0.25**2 + 1.0**2) / 2))
    assert metrics["pooled_q_rmse_bb"] != metrics["q_rmse_bb"]


@pytest.mark.parametrize("field", ["profile", "stream_key", "target"])
def test_audit_rejects_mismatched_profile_actions_or_stream(field):
    calibration, production = _fixture()
    changed = copy.deepcopy(production[0])
    if field == "profile":
        changed["profile"] = "uniform" if changed["profile"] == "increasing" else "increasing"
    elif field == "stream_key":
        changed["stream_key"] = "calibration-key"
    else:
        changed["target"]["actions"][0] = "different"
    production[0] = _hash_entry(changed)
    with pytest.raises(ValueError):
        audit_cache_entries(calibration, production)


def test_audit_rejects_wrong_frozen_production_budget():
    calibration, production = _fixture()
    changed = copy.deepcopy(production[0])
    changed["selected_n"]["flop:open"] = 64
    changed["stream_key"] = hashlib.sha256(
        f"{changed['stream_namespace']}|{changed['stream_seed']}|"
        f"{changed['plan_sha256']}|{changed['selected_n']}".encode()
    ).hexdigest()
    production[0] = _hash_entry(changed)
    with pytest.raises(ValueError, match="frozen production budget"):
        audit_cache_entries(calibration, production)


def test_audit_rejects_mixed_source_fingerprints():
    calibration, production = _fixture()
    for entry in production:
        entry["source_sha256"] = "different-source"
        _hash_entry(entry)
    with pytest.raises(ValueError, match="source fingerprints"):
        audit_cache_entries(calibration, production)


def test_campaign_runner_rejects_a_dropped_context(tmp_path):
    calibration, production = _fixture()
    calibration = calibration[2:]
    calibration_dir = tmp_path / "calibration"
    production_dir = tmp_path / "production"
    calibration_dir.mkdir()
    production_dir.mkdir()
    for index, entry in enumerate(calibration):
        (calibration_dir / f"calibration-{index}.json").write_text(json.dumps(entry))
    for index, entry in enumerate(production):
        (production_dir / f"production-{index}.json").write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="72/12 context roster"):
        run_audit(calibration_dir, production_dir, tmp_path / "audit.json")
