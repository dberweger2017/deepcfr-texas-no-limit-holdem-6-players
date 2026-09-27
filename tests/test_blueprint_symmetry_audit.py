"""The seven-arm audit reuses one schedule and retains every attempt."""

import json

from scripts.evaluate_blueprint_symmetry import ARMS, run
from src.blueprint.artifact import save_training
from src.blueprint.lookup import BUTTON_ZERO_CHECKPOINTS
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Table


def test_resource_preflight_couples_seven_arms_without_policy_outcomes(tmp_path,
                                                                       monkeypatch):
    trainer = BlueprintTrainer(Table(tuple(f"p{i}" for i in range(6)), (10_000,) * 6), PilotConfig())
    checkpoint = tmp_path / "tiny-checkpoint.json.gz"
    digest = save_training(trainer, checkpoint)
    monkeypatch.setitem(BUTTON_ZERO_CHECKPOINTS, digest, 0)
    plan = {
        "schema": "blueprint-symmetry-preflight-v1",
        "checkpoint_sha256": digest, "arms": list(ARMS),
        "resource_only": True,
        "suites": {"styles": {"blocks": 1, "root_seed": 2026092801,
                              "opponents": ["tight_passive", "loose_aggressive", "pot_pressure"],
                              "max_decisions": 1000}},
        "limits": {"max_wall_seconds": 120, "max_rss_gib": 10.5,
                   "min_free_gib": 0.01},
    }
    out = tmp_path / "result"
    result = run(plan, checkpoint, out)
    assert result["status"] == "complete" and result["attempts"] == 42
    rows = [json.loads(line) for line in (out / "hands.jsonl").read_text().splitlines()]
    assert all(row["status"] == "completed" and row["candidate_chips"] is None
               for row in rows)
    for rotation in range(6):
        group = [row for row in rows if row["rotation"] == rotation]
        assert [row["arm"] for row in group] == list(ARMS)
        assert len({row["deal_seed"] for row in group}) == 1
        assert len({tuple(row["opponents"]) for row in group}) == 1
    reached = json.loads((out / "reached-decisions.json").read_text())
    assert reached["decisions"] > 0
    assert all(sum(row["decision_weighted_visit_histogram"].values()) ==
               row["decision_weighted_found"] for row in reached["coverage_rows"])
    assert all(row["selected"] == 0 for row in reached["free_fold_counts"]
               if row["arm"] in ("U_safe", "B_legacy_safe", "B_canonical_safe"))
    assert all(row["selected"] <= row["eligible"] for row in
               reached["free_fold_counts"])
    assert any(row["category"] in ("both", "neither")
               for row in reached["same_decision_counts"])
