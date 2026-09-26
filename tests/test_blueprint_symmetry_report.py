"""Seven-arm inference uses one complete rotation block as one sample."""

from scripts.report_blueprint_symmetry import analyze
from src.arena.schedule import digest


def test_report_pairs_six_rotations_within_each_block():
    plan = {
        "checkpoint_sha256": "a" * 64,
        "resource_only": False,
        "suites": {"styles": {"blocks": 30}},
        "limits": {"max_rss_gib": 10.5, "max_wall_seconds": 1000},
    }
    manifest = {"source_dirty": False, "plan": plan,
                "checkpoint_sha256": plan["checkpoint_sha256"],
                "plan_sha256": digest(plan), "swap_before": None,
                "memory_pressure_before": None}
    result = {"status": "complete", "attempts": 30 * 6 * 7,
              "peak_process_rss_bytes": 1000, "elapsed_seconds": 1,
              "stop_reason": None, "wrapper_interventions": {},
              "swap_after": None, "memory_pressure_after": None}
    rows = []
    arms = ("U", "U_safe", "B_legacy", "B_legacy_safe",
            "B_canonical", "B_canonical_safe", "TAG")
    for block in range(30):
        for rotation in range(6):
            for arm in arms:
                chips = 10 if arm == "B_canonical_safe" else 0
                net = [0] * 6
                net[rotation] = chips
                net[(rotation + 1) % 6] = -chips
                row = {"suite": "styles", "block": block,
                       "rotation": rotation, "arm": arm, "status": "completed",
                       "candidate_chips": chips, "net_chips": net,
                       "deal_seed": block, "opponents": ["tight_passive"] * 5}
                row["outcome_sha256"] = digest(row)
                rows.append(row)
    coverage = {"coverage_rows": [], "same_decision_counts": [],
                "action_counts": [], "decisions": 0}
    summary = analyze(plan, manifest, result, rows, coverage)
    assert summary["status"] == "complete"
    primary = summary["suites"]["styles"]["contrasts"]["B_canonical_safe - U_safe"]
    assert primary["blocks"] == 30
    assert primary["bb_per_100"] == 10
    assert primary["confidence"] == 0.975
    assert primary["interval"] == [10, 10]
    rows.pop()
    assert analyze(plan, manifest, result, rows, coverage)["status"] == "incomplete"
