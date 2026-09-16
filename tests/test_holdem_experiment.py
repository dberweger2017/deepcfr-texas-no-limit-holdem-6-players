import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from src.arena.schedule import Scenario
from src.holdem.experiment import Experiment, run
from src.holdem.fitting import FitConfig
from src.holdem.training import TrainConfig

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return Experiment(
        seeds=(7,),
        scenarios=(
            Scenario("four", (10,) * 4, small_blind=1, big_blind=2, chip_unit="1"),
        ),
        training=TrainConfig(
            capacity=2, max_seconds=30, fit=FitConfig(width=8, steps=2, batch_size=2)
        ),
        iterations=2,
        save_every=1,
        evaluate_every=2,
        blocks=2,
        evaluation_seed=31,
        opponents=("tight_passive", "loose_aggressive"),
        max_seconds=120,
    )


def invoke(*args):
    subprocess.run(
        [sys.executable, "-m", "scripts.train_holdem", *map(str, args)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_fresh_process_training_resume_and_reproduction_match(tmp_path):
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(asdict(plan())))
    full, paused, resumed, reproduced = [
        tmp_path / n for n in ("full", "paused", "resumed", "reproduced")
    ]
    invoke("--plan", config, "--out", full)
    invoke("--plan", config, "--stop-after", 1, "--out", paused)
    assert not json.loads((paused / "result.json").read_text())["complete"]
    invoke("--resume", paused, "--out", resumed)
    invoke("--reproduce", full, "--out", reproduced)
    assert (
        (full / "result.json").read_bytes()
        == (resumed / "result.json").read_bytes()
        == (reproduced / "result.json").read_bytes()
    )
    job = "scenario-0-seed-7"
    assert (full / job / "training-2.pt").read_bytes() == (
        resumed / job / "training-2.pt"
    ).read_bytes()
    assert (full / job / "average-2.pt").read_bytes() == (
        resumed / job / "average-2.pt"
    ).read_bytes()
    assert (full / job / "outcomes-2.json").read_bytes() == (
        resumed / job / "outcomes-2.json"
    ).read_bytes()


def test_schedule_independence_and_completed_resume(tmp_path):
    original = run(plan(), tmp_path / "full")
    restored = run(plan(), tmp_path / "resumed", resume=tmp_path / "full")
    assert original == restored
    alternate = replace(plan(), save_every=2, evaluate_every=1)
    different = run(alternate, tmp_path / "alternate")
    first, second = original["jobs"][0], different["jobs"][0]
    for key in ("current_profile", "archive_profiles", "replay", "reports"):
        assert first[key] == second[key]
    assert (
        first["final_evaluation"]["outcomes_sha256"]
        == second["final_evaluation"]["outcomes_sha256"]
    )


def test_resume_checks_provenance_and_preserves_existing_outputs(tmp_path):
    out = tmp_path / "run"
    run(plan(), out, stop_after=1)
    with pytest.raises(FileExistsError):
        run(plan(), out)
    with pytest.raises(ValueError, match="mismatch"):
        run(replace(plan(), blocks=3), tmp_path / "wrong", resume=out)


def test_failed_iteration_can_resume_from_last_committed_boundary(
    tmp_path, monkeypatch
):
    from src.holdem.training import HoldemTrainer

    original = HoldemTrainer.step

    def fail(self):
        if self.iteration == 1:
            raise RuntimeError("interrupted iteration")
        return original(self)

    monkeypatch.setattr(HoldemTrainer, "step", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(plan(), tmp_path / "interrupted")
    failure = json.loads((tmp_path / "interrupted" / "failure.json").read_text())
    assert failure["error_type"] == "RuntimeError"
    assert failure["unfinished_jobs"] == ["scenario-0-seed-7"]
    assert failure["completed_jobs"] == []
    monkeypatch.setattr(HoldemTrainer, "step", original)
    resumed = run(plan(), tmp_path / "resumed", resume=tmp_path / "interrupted")
    full = run(plan(), tmp_path / "full")
    assert resumed == full


@pytest.mark.parametrize(
    "change",
    [
        {"seeds": (1, 1)},
        {"iterations": 0},
        {"evaluate_every": 0},
        {"max_seconds": float("nan")},
    ],
)
def test_invalid_experiments_are_rejected(change):
    with pytest.raises(ValueError):
        replace(plan(), **change)
