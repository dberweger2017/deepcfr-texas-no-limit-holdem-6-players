"""Tiny process, cache, and fit-resume checks for the campaign orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts import check_multistreet_representation as pilot
from src.arena.schedule import digest
from src.holdem.actions import bet_candidates
from src.holdem.multistreet_campaign import ReferenceCache
from src.holdem.multistreet_collection import collect_rows
from src.holdem.multistreet_fitting import fit_all
from src.holdem.multistreet_reference import build_context
from src.holdem.representation_reference import range_support
from src.holdem.targets import CandidateTargets

SUPPORT = range_support((("Ac", "Ad", "Kh", "Ks", "Qc", "Jh", "9d", "8s", "5c", "4d"),))


def _collection_plan():
    return {
        "context_seed": 7,
        "stream_namespace": "tiny-test",
        "range_templates": [list(SUPPORT[0])],
        "world_samples": 1,
        "deals_per_sample": 1,
        "max_tree_nodes": 100_000,
        "contexts": [
            {
                "split": "train",
                "street": "river",
                "board": ["3s", "8c", "As", "4c", "5c"],
                "holding": ["2c", "3c"],
                "facing": False,
            },
            {
                "split": "train",
                "street": "river",
                "board": ["Jd", "Tc", "Ks", "4c", "5c"],
                "holding": ["2c", "3c"],
                "facing": True,
            },
        ],
    }


def _fit_fixture():
    context = build_context(
        name="tiny-fit",
        split="train",
        street="river",
        board=("3s", "8c", "As", "4c", "5c"),
        holding=("2c", "3c"),
        support=SUPPORT,
        samples=1,
        deals_per_sample=1,
        seed=7,
    )
    candidates = bet_candidates(context.worlds[0].observe(context.hero_seat))
    count = len(candidates.actions)
    target = CandidateTargets(
        candidates,
        tuple(1 / count for _ in candidates.actions),
        tuple(float(i) for i in range(count)),
        tuple(i - (count - 1) / 2 for i in range(count)),
    )
    rows = [
        {
            "name": split,
            "split": split,
            "street": "river",
            "situation": "open",
            "group": "tiny",
            "target": target,
            "world_count": 2,
            "uncertainty": ["estimated", "estimated"],
            "action_se": [0.0] * count,
            "world_action_values_bb": [
                tuple(float(i) for i in range(count)),
                tuple(float(i) for i in range(count)),
            ],
        }
        for split in ("train", "tuning", "validation", "test")
    ]
    plan = {
        "format": "tiny-multistreet-fit-v1",
        "durations": [1, 2],
        "variants": ["scaled_baseline"],
        "seeds": [1, 2],
        "batch_size": 2,
        "learning_rate": 0.001,
        "gradient_clip": 1.0,
        "max_reference_seconds": 20,
        "max_fit_seconds": 20,
        "minimum_relative_cost_gain": 0.1,
        "minimum_absolute_cost_gain_bb": 0.02,
        "allowed_relative_rmse_increase": 0.02,
        "reference_se_multiplier": 2.0,
    }
    return plan, rows


def test_serial_and_spawn_reference_rows_match(tmp_path: Path):
    plan = _collection_plan()
    serial_cache = ReferenceCache(
        tmp_path / "serial",
        source_sha256="tiny-source",
        plan_sha256=digest(plan),
        stream_namespace="tiny-test",
        stream_seed=7,
        selected_n=1,
    )
    parallel_cache = ReferenceCache(
        tmp_path / "parallel",
        source_sha256="tiny-source",
        plan_sha256=digest(plan),
        stream_namespace="tiny-test",
        stream_seed=7,
        selected_n=1,
    )
    serial = collect_rows(plan, serial_cache, deadline=10**9, reference_workers=1)
    parallel = collect_rows(plan, parallel_cache, deadline=10**9, reference_workers=2)
    assert [row["name"] for row in serial] == [row["name"] for row in parallel]
    for left, right in zip(serial, parallel, strict=True):
        assert left["world_action_values_bb"] == right["world_action_values_bb"]
        assert left["target"].policy == right["target"].policy
        assert left["target"].values_bb == right["target"].values_bb
        assert left["target"].regrets_bb == right["target"].regrets_bb
        assert left["reference_nodes"] == right["reference_nodes"]


def test_fit_process_seed_equivalence_and_completed_cache_resume(tmp_path: Path):
    plan, rows = _fit_fixture()
    targets = {
        split: [row["target"] for row in rows if row["split"] == split]
        for split in ("train", "tuning")
    }
    serial = fit_all(
        targets,
        plan,
        deadline=10**9,
        cache_dir=tmp_path / "serial-fit",
        identity="tiny-fit",
        workers=1,
    )
    parallel = fit_all(
        targets,
        plan,
        deadline=10**9,
        cache_dir=tmp_path / "parallel-fit",
        identity="tiny-fit",
        workers=2,
    )
    assert [(row[0], row[1]) for row in serial] == [
        (row[0], row[1]) for row in parallel
    ]
    for left, right in zip(serial, parallel, strict=True):
        assert left[2] == right[2]
        for step in left[3]:
            for name in left[3][step]:
                assert torch.equal(left[3][step][name], right[3][step][name])

    resumed = fit_all(
        targets,
        plan,
        deadline=0,
        cache_dir=tmp_path / "parallel-fit",
        identity="tiny-fit",
        workers=2,
    )
    assert resumed[0][2] == parallel[0][2]


def test_completed_fit_resume_rejects_changed_plan_or_source(
    tmp_path: Path, monkeypatch
):
    plan, rows = _fit_fixture()
    out = tmp_path / "fit"
    assert pilot.run(plan, out, reference_rows=rows)["status"] == "completed"
    changed_plan = dict(plan, batch_size=3)
    with pytest.raises(ValueError, match="changed fitting plan or source"):
        pilot.run(changed_plan, out, reference_rows=rows, resume=True)
    monkeypatch.setattr(pilot, "source_fingerprint", lambda: "changed-source")
    with pytest.raises(ValueError, match="changed fitting plan or source"):
        pilot.run(plan, out, reference_rows=rows, resume=True)
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "completed"
