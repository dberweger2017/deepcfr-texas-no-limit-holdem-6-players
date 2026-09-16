import json
import subprocess
import sys
from dataclasses import asdict
from hashlib import sha256

import numpy as np
import pytest
import torch

from src.solver.neural.average import (
    StrategyArchive,
    load_archive,
    record_iteration,
    save_archive,
    tabulate,
)
from src.solver.neural.experiment import Plan
from src.solver.neural.network import deterministic_cpu
from src.solver.neural.snapshot_training import load_training, save_training
from src.solver.neural.solver import Config, DeepCFR
from src.solver.tree import GameTree
from tests.test_snapshot_average import archive_for, table_for


def config():
    return Config(
        hidden=4,
        traversals=8,
        advantage_steps=2,
        strategy_steps=2,
        batch_size=8,
        capacity=16,
    )


def manifest():
    return {
        key: {}
        for key in (
            "version",
            "plan_sha256",
            "source_sha256",
            "environment",
            "protocol",
        )
    }


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_batched_export_matches_public_queries_and_recorded_play(game):
    with deterministic_cpu():
        tree = GameTree(game)
        archive = archive_for(game)
        assert np.allclose(
            tabulate(archive, tree), table_for(tree, archive), atol=1e-14
        )
        solver = DeepCFR(tree, config())
        archive = StrategyArchive(game, 4)
        for _ in range(3):
            record_iteration(solver, archive)
        assert np.allclose(tabulate(archive, tree), solver.played_average(), atol=1e-14)


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_complete_archive_resume_matches_uninterrupted(tmp_path, game):
    with deterministic_cpu():
        solver = DeepCFR(GameTree(game), config())
        archive = StrategyArchive(game, 4)
        record_iteration(solver, archive)
        path = tmp_path / "training.pt"
        digest = save_training(
            solver, archive, path, manifest=manifest(), progress={"boundary": 1}
        )
        recovered, saved, progress = load_training(path, digest, manifest=manifest())
        assert progress == {"boundary": 1}
        record_iteration(solver, archive)
        record_iteration(recovered, saved)
        assert solver.fits == recovered.fits
        assert (
            solver.traversal_random.getstate() == recovered.traversal_random.getstate()
        )
        for a, b in zip(
            solver.advantage_memories + [solver.strategy_memory],
            recovered.advantage_memories + [recovered.strategy_memory],
        ):
            assert a.seen == b.seen and a.random.getstate() == b.random.getstate()
            for name in ("infos", "targets", "iterations"):
                assert np.array_equal(
                    getattr(a, name)[: a.size], getattr(b, name)[: b.size]
                )
        assert np.array_equal(
            tabulate(archive, solver.tree), tabulate(saved, recovered.tree)
        )
        assert save_archive(archive, tmp_path / "first.pt") == save_archive(
            saved, tmp_path / "second.pt"
        )


def test_inference_archive_cannot_be_promoted_to_training_state(tmp_path):
    with deterministic_cpu():
        solver = DeepCFR(GameTree("kuhn"), config())
        archive = StrategyArchive("kuhn", 4)
        record_iteration(solver, archive)
        path = tmp_path / "average.pt"
        digest = save_archive(archive, path)
        inference = load_archive(path, digest)
        with pytest.raises(ValueError, match="attached"):
            save_training(
                solver, inference, tmp_path / "bad.pt", manifest=manifest(), progress={}
            )
        with pytest.raises(ValueError, match="inference"):
            load_training(path, digest, manifest=manifest())


def test_snapshot_training_rejects_wrong_generation_and_hash(tmp_path):
    with deterministic_cpu():
        solver = DeepCFR(GameTree("kuhn"), config())
        archive = StrategyArchive("kuhn", 4)
        record_iteration(solver, archive)
        path = tmp_path / "training.pt"
        digest = save_training(solver, archive, path, manifest=manifest(), progress={})
        with pytest.raises(ValueError, match="hash"):
            load_training(path, "0" * 64, manifest=manifest())
        changed = {**manifest(), "protocol": {"different": True}}
        with pytest.raises(ValueError, match="protocol"):
            load_training(path, digest, manifest=changed)
        with torch.no_grad():
            next(solver.advantages[0].parameters()).add_(1)
        with pytest.raises(ValueError, match="current"):
            save_training(
                solver, archive, tmp_path / "bad.pt", manifest=manifest(), progress={}
            )


def test_snapshot_cli_fresh_process_resume_and_reproduction(tmp_path):
    plan = Plan(
        "leduc",
        3,
        config(),
        evaluation_interval=3,
        maximum_seconds=60,
        average="snapshots",
    )
    config_file = tmp_path / "plan.json"
    config_file.write_text(json.dumps(asdict(plan)))

    def invoke(*args):
        subprocess.run(
            [sys.executable, "-m", "scripts.check_deep_cfr", *map(str, args)],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )

    full, paused, resumed, reproduced = [
        tmp_path / n for n in ("full", "paused", "resumed", "reproduced")
    ]
    invoke("--plan", config_file, "--out", full)
    invoke("--plan", config_file, "--stop-after", 1, "--out", paused)
    invoke("--resume", paused, "--out", resumed)
    invoke("--reproduce", full, "--out", reproduced)
    assert (
        (full / "policy.pt").read_bytes()
        == (resumed / "policy.pt").read_bytes()
        == (reproduced / "policy.pt").read_bytes()
    )
    for path in (full, resumed, reproduced):
        report = json.loads((path / "report.json").read_text())
        assert report["average_kind"] == "snapshots"
        assert report["evaluations"][-1]["strategy_fit"] is None
        assert (
            report["policy_file_sha256"]
            == sha256((path / "policy.pt").read_bytes()).hexdigest()
        )


def test_periodic_recovery_does_not_require_an_evaluation(tmp_path):
    from src.solver.neural.experiment import run

    plan = Plan(
        "kuhn",
        3,
        config(),
        evaluation_interval=3,
        checkpoint_interval=1,
        maximum_seconds=60,
        average="snapshots",
    )
    output = tmp_path / "full"
    report = run(plan, output)
    assert report["status"] == "completed"
    assert [e["iteration"] for e in report["evaluations"]] == [3]
    assert sorted(p.name for p in output.glob("iteration-*.pt")) == [
        "iteration-000001.pt",
        "iteration-000002.pt",
        "iteration-000003.pt",
    ]
    saved_manifest = json.loads((output / "manifest.json").read_text())
    path = output / "iteration-000001.pt"
    _, archive, progress = load_training(
        path, sha256(path.read_bytes()).hexdigest(), manifest=saved_manifest
    )
    assert archive.iterations == 1
    assert progress["report"]["evaluations"] == []


def test_failed_archive_publication_cannot_be_saved_or_continued(tmp_path, monkeypatch):
    with deterministic_cpu():
        solver = DeepCFR(GameTree("kuhn"), config())
        archive = StrategyArchive("kuhn", 4)
        record_iteration(solver, archive)
        path = tmp_path / "safe.pt"
        digest = save_training(solver, archive, path, manifest=manifest(), progress={})

        def fail(*args):
            raise MemoryError("archive allocation failed")

        monkeypatch.setattr(archive, "append", fail)
        with pytest.raises(MemoryError):
            record_iteration(solver, archive)
        assert solver.failed
        with pytest.raises(ValueError, match="completed"):
            save_training(
                solver, archive, tmp_path / "bad.pt", manifest=manifest(), progress={}
            )
        recovered, saved, _ = load_training(path, digest, manifest=manifest())
        record_iteration(recovered, saved)
        assert recovered.iterations == saved.iterations == 2
