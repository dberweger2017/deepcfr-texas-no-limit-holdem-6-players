import json
from dataclasses import replace

import pytest

from src.arena.catalog import Checkpoint
from src.arena.registry import PolicyRegistry
from src.arena.run import reproduce
from src.arena.run import run as run_arena
from src.arena.schedule import Plan, Scenario
from src.holdem.checkpoint import save_policy
from src.holdem.experiment import run
from src.holdem.training import HoldemTrainer
from tests.test_hand_observations import table
from tests.test_holdem_experiment import plan
from tests.test_holdem_sampled_training import config


@pytest.fixture
def reference(tmp_path):
    trainer = HoldemTrainer(table(4, (10,) * 4), config())
    trainer.step()
    trainer.step()
    path = tmp_path / "release-05.pt"
    checksum = save_policy(trainer, path, manifest={"revision": "test"})
    return Checkpoint("release05", str(path), checksum, "holdem-average-v1")


def test_archived_average_runs_and_reproduces_without_source_file(tmp_path, reference):
    from pathlib import Path

    scenario = Scenario("four", (10,) * 4, small_blind=1, big_blind=2, chip_unit="1")
    selected = Plan(
        (scenario,),
        candidate="release05",
        baseline="random",
        opponents=("release05", "random"),
        models=(reference,),
        blocks=2,
    )
    out = tmp_path / "match"
    report = run_arena(selected, out)
    assert report["status"] == "valid"
    assert report["policies"]["release05"]["training_seed"] == 31
    assert report["policies"]["release05"]["iteration"] == 2
    Path(reference.path).unlink()
    replayed = reproduce(out, tmp_path / "replayed")
    assert replayed["outcomes_sha256"] == report["outcomes_sha256"]


def test_same_model_aliases_produce_identical_paired_outcomes(tmp_path, reference):
    scenario = Scenario("four", (10,) * 4, small_blind=1, big_blind=2, chip_unit="1")
    other = replace(reference, name="release06")
    selected = Plan(
        (scenario,),
        candidate=other.name,
        baseline=reference.name,
        opponents=(reference.name,),
        models=(reference, other),
        blocks=2,
    )
    registry = PolicyRegistry(selected)
    assert registry.make_policy(other.name, 7) is not registry.make_policy(
        other.name, 7
    )
    report = run_arena(selected, tmp_path / "same", registry=registry)
    comparison = report["scenarios"]["four"]["comparison"]
    assert comparison["paired_difference"]["bb_per_100"] == 0


def test_training_evaluates_random_previous_and_crossplay_and_keeps_artifacts(
    tmp_path, reference
):
    from pathlib import Path

    selected = replace(
        plan(), benchmarks=("random", "previous", "crossplay"), reference=reference
    )
    out = tmp_path / "full"
    full = run(selected, out)
    evaluation = full["jobs"][0]["final_evaluation"]
    assert set(evaluation["benchmarks"]) == {"random", "previous", "crossplay"}
    assert evaluation["benchmarks"]["random"]["opponent_pool"] == ["random"]
    assert evaluation["benchmarks"]["previous"]["baseline_policy"] == reference.name
    assert evaluation["benchmarks"]["crossplay"]["opponent_pool"] == [reference.name]
    assert (
        evaluation["benchmarks"]["previous"]["reference"]["weights_sha256"]
        == reference.sha256
    )
    directory = out / "scenario-0-seed-7"
    timings = [
        json.loads(row)
        for row in (directory / "training-timing.jsonl").read_text().splitlines()
    ]
    assert [t["iteration"] for t in timings] == [1, 2]
    assert all(
        t["status"] == "complete" and t["total_seconds"] >= t["collection_seconds"] > 0
        for t in timings
    )
    assert all(
        t["fitting_seconds"] > 0 and t["peak_process_rss_bytes"] > 0 for t in timings
    )
    artifacts = [
        json.loads(row)
        for row in (directory / "artifacts.jsonl").read_text().splitlines()
    ]
    assert {a["path"] for a in artifacts} == {
        "training-1.pt",
        "training-2.pt",
        "average-2.pt",
    }
    Path(reference.path).unlink()
    resumed = run(selected, tmp_path / "resumed", resume=out)
    assert resumed == full
    reproduced = run(selected, tmp_path / "reproduced", reproduce=out)
    assert reproduced == full


def test_bad_reference_hash_fails_before_training(tmp_path, reference):
    selected = replace(
        plan(), benchmarks=("previous",), reference=replace(reference, sha256="0" * 64)
    )
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="hash"):
        run(selected, out)
    assert (out / "failure.json").exists()
    assert not list(out.glob("*/training-*.pt"))


def test_snapshot_table_size_mismatch_is_rejected(reference):
    selected = Plan((Scenario("six"),), candidate=reference.name, models=(reference,))
    with pytest.raises(ValueError, match="4-player"):
        PolicyRegistry(selected)


@pytest.mark.parametrize(
    "change",
    [
        {"benchmarks": ("random", "random")},
        {"benchmarks": ("unknown",)},
        {"benchmarks": ("previous",)},
        {"benchmarks": ("crossplay",)},
    ],
)
def test_invalid_benchmark_configuration(change):
    with pytest.raises(ValueError):
        replace(plan(), **change)


def test_failed_collection_retains_timing_without_publishing():
    from src.holdem.collection import CollectionLimitExceeded

    trainer = HoldemTrainer(table(4, (10,) * 4), config(max_nodes=1))
    with pytest.raises(CollectionLimitExceeded):
        trainer.step()
    assert trainer.iteration == 0
    assert trainer.last_timing["status"] == "failed"
    assert trainer.last_timing["collection_seconds"] > 0
    assert (
        trainer.last_timing["total_seconds"]
        >= trainer.last_timing["collection_seconds"]
    )


def test_resumed_learning_curve_keeps_earlier_evaluations(tmp_path):
    selected = replace(plan(), iterations=4, benchmarks=("random",))
    full = tmp_path / "full"
    paused = tmp_path / "paused"
    resumed = tmp_path / "resumed"
    run(selected, full)
    run(selected, paused, stop_after=3)
    run(selected, resumed, resume=paused)
    job = "scenario-0-seed-7"
    assert (resumed / job / "learning-curve.json").read_bytes() == (
        full / job / "learning-curve.json"
    ).read_bytes()
    curve = json.loads((resumed / job / "learning-curve.json").read_text())
    assert {r["iteration"] for r in curve} == {2, 4}
    assert {r["benchmark"] for r in curve} == {"styles", "random"}


def test_cli_selects_only_a_declared_seed(tmp_path):
    import subprocess
    import sys
    from dataclasses import asdict

    selected = replace(plan(), seeds=(7, 11))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(asdict(selected)))
    out = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_holdem",
            "--plan",
            str(path),
            "--seed",
            "11",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert json.loads((out / "manifest.json").read_text())["plan"]["seeds"] == [11]
    assert (out / "scenario-0-seed-11/result.json").exists()
    assert not (out / "scenario-0-seed-7").exists()
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_holdem",
            "--plan",
            str(path),
            "--seed",
            "13",
            "--out",
            str(tmp_path / "bad"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert failed.returncode != 0 and not (tmp_path / "bad").exists()
