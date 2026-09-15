import json
from dataclasses import replace
from hashlib import sha256

import pytest

from scripts.run_strategy_study import execute
from src.solver.neural.campaign import REFERENCE, Campaign
from src.solver.neural.experiment import Plan, run
from src.solver.neural.solver import Config
from src.solver.neural.study import Study, Variant, explore_seed, select


def small_study():
    plan = Plan(
        "leduc",
        2,
        Config(
            hidden=8,
            strategy_hidden=8,
            traversals=8,
            advantage_steps=2,
            strategy_steps=3,
            batch_size=8,
            capacity=32,
        ),
        evaluation_interval=1,
        maximum_seconds=30,
    )
    campaign = Campaign(
        plan, (11, 13, 17), 10, 10, sha256(REFERENCE.read_bytes()).hexdigest()
    )
    return Study(
        campaign,
        (71, 73, 79),
        replace(campaign, seeds=(71, 73, 79), training=replace(plan, game="kuhn")),
        (1, 2),
        (Variant("baseline", 8, 3), Variant("wider", 16, 4)),
        worker_seconds=60,
        maximum_seconds=180,
    )


def test_shared_collection_matches_fresh_training_with_the_selected_capacity(tmp_path):
    study = small_study()
    report = explore_seed(study, 11, tmp_path / "study")
    plan = study.exploration.for_seed(11)
    plan = replace(
        plan, training=replace(plan.training, strategy_hidden=16, strategy_steps=4)
    )
    fresh = run(plan, tmp_path / "fresh")
    rows = [row for row in report["fits"] if row["variant"] == "wider"]
    for row, expected in zip(rows, fresh["evaluations"]):
        assert row["evaluation"] == expected["neural_average"]
        assert row["fit"] == expected["strategy_fit"]
        assert row["policy_sha256"] == expected["strategy_sha256"]


def fake_reports(study):
    return [
        {
            "status": "completed",
            "seed": seed,
            "fits": [
                {
                    "iteration": t,
                    "variant": v.name,
                    "value_error": 0,
                    "evaluation": {
                        "exploitability": 0.2 if v.name == "baseline" else 0.1
                    },
                }
                for t in study.checkpoints
                for v in study.variants
            ],
        }
        for seed in study.exploration.seeds
    ]


def test_selection_uses_every_seed_and_only_the_predeclared_final_iteration():
    study = replace(
        small_study(),
        exploration=replace(small_study().exploration, maximum_exploitability=0.15),
    )
    reports = fake_reports(study)
    assert select(study, reports)["selected"] == "wider"
    reports[-1]["fits"][-1]["evaluation"]["exploitability"] = 0.16
    assert select(study, reports)["status"] == "no_candidate"
    with pytest.raises(ValueError, match="every exploration seed"):
        select(study, reports[:-1])
    with pytest.raises(ValueError, match="every exploration seed"):
        select(study, reports + reports[:1])
    reports[0]["fits"].pop()
    with pytest.raises(ValueError, match="every planned fit"):
        select(study, reports)
    with pytest.raises(ValueError, match="distinct"):
        replace(study, confirmation_seeds=study.exploration.seeds)


def test_full_study_pipeline_freezes_selection_and_retains_both_game_gates(tmp_path):
    study = small_study()
    exploration = execute(study, "explore", tmp_path / "explore", 1)
    assert exploration["status"] == "completed"
    report = execute(study, "confirm", tmp_path / "confirm", 2, tmp_path / "explore")
    assert report["gate_passed"]
    assert set(report["gates"]) == {"kuhn", "leduc"}
    for gate in report["gates"].values():
        assert [row["seed"] for row in gate["runs"]] == [71, 73, 79]
    selection = json.loads((tmp_path / "confirm/selection.json").read_text())
    assert len(selection["exploration_report_hashes"]) == 3
    path = tmp_path / "explore/selection.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="frozen selection"):
        execute(study, "confirm", tmp_path / "tampered", 1, tmp_path / "explore")
    assert not (tmp_path / "tampered/leduc").exists()
