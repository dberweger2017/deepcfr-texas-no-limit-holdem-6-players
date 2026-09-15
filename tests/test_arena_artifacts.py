import json

import pytest

from scripts.run_arena import main
from src.arena.artifacts import validate_manifest
from src.arena.policies import POLICIES
from src.arena.run import reproduce, run
from src.arena.schedule import Plan, Scenario
from src.game.types import Action, ActionKind


def plan():
    return Plan(
        (
            Scenario("four", (2000,) * 4),
            Scenario("bankroll", (2000,) * 4, mode="session", hands_per_rotation=3),
        ),
        blocks=1,
    )


def test_bundle_reproduces_exact_outcomes_and_records_environment(tmp_path):
    original, replay = tmp_path / "original", tmp_path / "replay"
    report = run(plan(), original)
    repeated = reproduce(original, replay)
    assert report["status"] == repeated["status"] == "valid"
    assert report["outcomes_sha256"] == repeated["outcomes_sha256"]
    assert (original / "hands.jsonl").read_bytes() == (
        replay / "hands.jsonl"
    ).read_bytes()
    value = json.loads((original / "manifest.json").read_text())
    assert value["environment"]["packages"]["pokers"] == "0.2.0"
    assert (
        value["environment"]["engine"]["commit"]
        == "5db20e3d5d6862b32a7402035c1340b622d3b005"
    )
    assert value["environment"]["engine"]["binaries"]
    assert len(value["source_sha256"]) == 64
    assert len(value["revision"]) == 40
    assert value["policies"]["fold"]["weights_sha256"] is None
    assert value["rules"]["observation_schema"] == 2
    with pytest.raises(FileExistsError):
        run(plan(), original)
    assert (original / "report.md").read_text().startswith("# Evaluation report")


@pytest.mark.parametrize(
    "field",
    [
        "source_sha256",
        "schedule_sha256",
        "rules",
        "policies",
        "environment",
        "protocol",
    ],
)
def test_manifest_mismatch_is_rejected(tmp_path, field):
    run(plan(), tmp_path / "run")
    value = json.loads((tmp_path / "run/manifest.json").read_text())
    value[field] = None
    with pytest.raises(ValueError, match=field):
        validate_manifest(value)


def test_failed_action_is_written_and_report_has_no_strength_estimate(
    tmp_path, monkeypatch
):
    class Illegal:
        def choose_action(self, view):
            return Action(ActionKind.RAISE, 10**10)

    monkeypatch.setitem(POLICIES, "check_call", lambda seed: Illegal())
    report = run(plan(), tmp_path / "failed")
    assert report["status"] == "invalid"
    assert report["invalid_actions"] == 1
    assert report["completed_hands"] == 0
    assert report["unattempted_hands"] == report["requested_hands"] - 1
    assert all(s["comparison"] is None for s in report["scenarios"].values())
    row = json.loads((tmp_path / "failed/hands.jsonl").read_text())
    assert row["events"] and "outside" in row["error"]
    assert "No strength estimate" in (tmp_path / "failed/report.md").read_text()


def test_interruption_leaves_an_invalid_report(tmp_path, monkeypatch):
    class Interrupted:
        def choose_action(self, view):
            raise KeyboardInterrupt()

    monkeypatch.setitem(POLICIES, "check_call", lambda seed: Interrupted())
    with pytest.raises(KeyboardInterrupt):
        run(plan(), tmp_path / "interrupted")
    report = json.loads((tmp_path / "interrupted/report.json").read_text())
    assert report["status"] == "invalid"
    assert report["completed_hands"] == 0


def test_cli_runs_and_reproduces_a_saved_plan(tmp_path):
    from dataclasses import asdict

    config = tmp_path / "plan.json"
    config.write_text(json.dumps(asdict(plan())))
    assert main(["--plan", str(config), "--out", str(tmp_path / "one")]) == 0
    assert (
        main(["--reproduce", str(tmp_path / "one"), "--out", str(tmp_path / "two")])
        == 0
    )


def test_tampered_schedule_is_rejected_before_running(tmp_path):
    run(plan(), tmp_path / "one")
    (tmp_path / "one/schedule.json").write_text("{}")
    with pytest.raises(ValueError, match="schedule"):
        reproduce(tmp_path / "one", tmp_path / "two")
    assert not (tmp_path / "two").exists()


def test_tampered_report_is_detected_by_reproduction(tmp_path):
    run(plan(), tmp_path / "one")
    path = tmp_path / "one/report.json"
    report = json.loads(path.read_text())
    report["completed_hands"] -= 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="report differs"):
        reproduce(tmp_path / "one", tmp_path / "two")
    assert (tmp_path / "two/report.json").exists()
