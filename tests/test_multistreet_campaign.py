import json
from pathlib import Path

import pytest

from src.holdem.multistreet_campaign import (
    NESTED_WORLD_COUNTS,
    ReferenceCache,
    calibrate_world_counts,
    campaign_plan,
    paired_action_difference_se,
)
from src.holdem.multistreet_reference import build_context
from src.holdem.representation_reference import range_support
from src.holdem.targets import CandidateTargets
from src.holdem.actions import bet_candidates
from src.holdem.multistreet_reference import MultiStreetReference


def _strata(matrix):
    return {
        f"{street}:{situation}": [matrix]
        for street in ("flop", "turn", "river")
        for situation in ("open", "facing")
    }


def test_campaign_expands_exact_stratified_roster():
    plan = json.loads(Path("configs/holdem/multistreet-campaign.json").read_text())
    expanded = campaign_plan(plan)
    assert len(expanded["contexts"]) == 1152
    assert {
        split: sum(row["split"] == split for row in expanded["contexts"])
        for split in ("train", "tuning", "validation", "test")
    } == {"train": 576, "tuning": 192, "validation": 192, "test": 192}


def test_paired_action_se_uses_within_world_action_differences():
    values = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    assert paired_action_difference_se(values, 4) == pytest.approx(0.0)
    values[-1][1] = 6.0
    assert paired_action_difference_se(values, 4) > 0


def test_calibration_retains_eight_trace_and_freezes_at_minimum():
    values = [[0.0, 0.0] for _ in range(128)]
    result = calibrate_world_counts(_strata(values), precision_bb=0.1)
    assert result["minimum_n"] == 16
    assert set(row["n"] for row in result["decisions"]["flop:open"]["trace"]) == set(NESTED_WORLD_COUNTS)
    assert result["decisions"]["flop:open"]["n"] == 16
    assert result["unresolved"] == []


def test_calibration_marks_unresolved_at_128_without_aborting():
    values = [[0.0, 1.0] if index % 2 else [1.0, 0.0] for index in range(128)]
    result = calibrate_world_counts(_strata(values), precision_bb=0.0001)
    assert all(row["n"] == 128 for row in result["decisions"].values())
    assert set(result["unresolved"]) == set(result["decisions"])
    assert all(row["status"] == "unresolved_at_maximum" for row in result["decisions"].values())


def test_calibration_rejects_missing_stratum():
    values = [[0.0, 0.0] for _ in range(128)]
    strata = _strata(values)
    strata.pop("river:facing")
    with pytest.raises(ValueError, match="six street"):
        calibrate_world_counts(strata)


def test_calibration_requires_current_and_larger_prefixes_to_pass():
    # The 16-world prefix is quiet, the 32-world prefix is noisy, and the
    # later prefixes are quiet again.  A lucky early prefix must not freeze.
    values = [[0.0, 0.0] for _ in range(128)]
    for index in range(16, 32):
        values[index] = [float(index % 2), float((index + 1) % 2)]
    result = calibrate_world_counts(_strata(values), precision_bb=0.1)
    assert result["decisions"]["flop:open"]["n"] == 64


def test_reference_cache_rejects_world_or_action_identity_changes(tmp_path, monkeypatch):
    support = range_support((("Ac", "Ad", "Kh", "Ks", "Qc", "Jh", "9d", "8s", "5c", "4d"),))
    context = build_context(
        name="cache",
        split="train",
        street="river",
        board=("2h", "7d", "9s", "Tc", "Qh"),
        holding=("3c", "4c"),
        support=support,
        samples=2,
        deals_per_sample=1,
        seed=91,
    )
    candidates = bet_candidates(context.worlds[0].observe(context.hero_seat))
    target = CandidateTargets(
        candidates,
        tuple(1 / len(candidates.actions) for _ in candidates.actions),
        tuple(0.0 for _ in candidates.actions),
        tuple(0.0 for _ in candidates.actions),
    )
    fake = MultiStreetReference(
        target,
        tuple(tuple(0.0 for _ in candidates.actions) for _ in context.worlds),
        tuple(0.0 for _ in candidates.actions),
        "estimated",
        1,
        len(context.worlds),
        0.01,
    )
    monkeypatch.setattr("src.holdem.multistreet_campaign.enumerate_reference", lambda *args, **kwargs: fake)
    cache = ReferenceCache(
        tmp_path,
        source_sha256="source",
        plan_sha256="plan",
        stream_namespace="calibration",
        stream_seed=11,
        selected_n=16,
    )
    cache.get_or_compute(context, "uniform", max_nodes=10, deadline=float("inf"))
    assert cache.load(context, "uniform")["selected_n"] == 16
    path = cache.path(context, "uniform")
    payload = json.loads(path.read_text())
    payload["stream_key"] = "wrong"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stream mismatch"):
        cache.load(context, "uniform")
