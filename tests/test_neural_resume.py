import json
import subprocess
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from io import BytesIO

import numpy as np
import pytest
import torch

from src.solver.experiment import canonical
from src.solver.neural.checkpoint import atomic_write, load_training, save_training
from src.solver.neural.experiment import Plan, provenance, run
from src.solver.neural.network import deterministic_cpu
from src.solver.neural.solver import Config, DeepCFR
from src.solver.tree import GameTree


def plan(game="kuhn"):
    return Plan(
        game,
        4,
        Config(
            hidden=8,
            strategy_hidden=16,
            traversals=12,
            advantage_steps=4,
            strategy_steps=6,
            batch_size=8,
            capacity=16,
            seed=89,
        ),
        evaluation_interval=2,
    )


def deterministic_report(report):
    return canonical({k: v for k, v in report.items() if k != "wall_seconds"})


def test_cli_resume_in_a_new_process_matches_uninterrupted_training(tmp_path):
    spec = plan()
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(asdict(spec)))
    whole = run(spec, tmp_path / "whole")
    for args in (
        ["--plan", str(config), "--stop-after", "3", "--out", str(tmp_path / "paused")],
        ["--resume", str(tmp_path / "paused"), "--out", str(tmp_path / "resumed")],
    ):
        subprocess.run(
            [sys.executable, "-m", "scripts.check_deep_cfr", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    paused = json.loads((tmp_path / "paused/report.json").read_text())
    resumed = json.loads((tmp_path / "resumed/report.json").read_text())
    assert paused["status"] == "paused"
    assert not (tmp_path / "paused/policy.pt").exists()
    assert deterministic_report(whole) == deterministic_report(resumed)
    assert (tmp_path / "whole/policy.pt").read_bytes() == (
        tmp_path / "resumed/policy.pt"
    ).read_bytes()
    assert all(m["seen"] > m["stored"] == 16 for m in resumed["memory_counts"].values())


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_snapshot_preserves_replay_generators_and_frozen_networks(game, tmp_path):
    spec = plan(game)
    manifest = provenance(asdict(spec))
    with deterministic_cpu():
        original = DeepCFR(GameTree(game), spec.training)
        original.step()
        original.fit_strategy()
        digest = save_training(
            original, tmp_path / "checkpoint.pt", manifest=manifest, progress={}
        )
        before = torch.random.get_rng_state().clone()
        restored, _ = load_training(
            tmp_path / "checkpoint.pt", digest, manifest=manifest
        )
        assert torch.equal(before, torch.random.get_rng_state())
        for solver in (original, restored):
            solver.step()
            solver.fit_strategy()
        assert np.array_equal(original.average_policy(), restored.average_policy())
        assert np.array_equal(
            original.played_strategy_sum, restored.played_strategy_sum
        )
        assert original.fits == restored.fits
        assert (
            original.traversal_random.getstate() == restored.traversal_random.getstate()
        )
        for a, b in zip(
            original.advantage_memories + [original.strategy_memory],
            restored.advantage_memories + [restored.strategy_memory],
        ):
            assert a.size == b.size and a.seen == b.seen
            assert a.random.getstate() == b.random.getstate()
            for field in ("infos", "iterations", "targets"):
                assert np.array_equal(
                    getattr(a, field)[: a.size], getattr(b, field)[: b.size]
                )
        assert all(
            not p.requires_grad
            for net in restored.advantages + [restored.strategy]
            for p in net.parameters()
        )


def test_failed_iteration_resumes_from_last_complete_snapshot(tmp_path, monkeypatch):
    spec = replace(plan(), evaluation_interval=1)
    expected = run(spec, tmp_path / "whole")
    original = DeepCFR.step

    def interrupted(self, deadline):
        original(self, deadline)
        if self.iterations == 2:
            self.failed = True
            raise RuntimeError("simulated interruption after an update")

    monkeypatch.setattr(DeepCFR, "step", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(spec, tmp_path / "interrupted")
    descriptor = json.loads((tmp_path / "interrupted/checkpoint.json").read_text())
    assert descriptor["iteration"] == 1
    monkeypatch.setattr(DeepCFR, "step", original)
    actual = run(spec, tmp_path / "resumed", resume=tmp_path / "interrupted")
    assert deterministic_report(expected) == deterministic_report(actual)


def test_checkpoint_rejects_corruption_and_changed_contracts(tmp_path):
    spec = plan()
    run(spec, tmp_path / "paused", stop_after=2)
    descriptor = json.loads((tmp_path / "paused/checkpoint.json").read_text())
    path = tmp_path / "paused" / descriptor["file"]
    manifest = provenance(asdict(spec))
    with pytest.raises(ValueError, match="hash"):
        load_training(path, "0" * 64, manifest=manifest)
    for field in ("source_sha256", "plan_sha256", "environment", "protocol"):
        with pytest.raises(ValueError, match=field):
            load_training(
                path, descriptor["sha256"], manifest={**manifest, field: "changed"}
            )
    payload = torch.load(path, weights_only=True)
    payload["solver"]["memories"][0]["iterations"][0] = 99
    buffer = BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="iterations"):
        load_training(path, sha256(path.read_bytes()).hexdigest(), manifest=manifest)


def test_resume_does_not_reset_elapsed_budget(tmp_path):
    spec = plan()
    run(spec, tmp_path / "paused", stop_after=2)
    descriptor_path = tmp_path / "paused/checkpoint.json"
    descriptor = json.loads(descriptor_path.read_text())
    path = tmp_path / "paused" / descriptor["file"]
    payload = torch.load(path, weights_only=True)
    payload["progress"]["elapsed_seconds"] = spec.maximum_seconds
    buffer = BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(buffer.getvalue())
    descriptor["sha256"] = sha256(path.read_bytes()).hexdigest()
    descriptor_path.write_text(json.dumps(descriptor))
    result = run(spec, tmp_path / "resumed", resume=tmp_path / "paused")
    assert result["status"] == "timed_out"
    assert result["completed_iterations"] == 2
    assert not (tmp_path / "resumed/policy.pt").exists()


def test_atomic_snapshot_never_overwrites_or_publishes_partial_bytes(
    tmp_path, monkeypatch
):
    import src.solver.neural.checkpoint as module

    path = tmp_path / "snapshot.pt"
    atomic_write(path, b"complete")
    with pytest.raises(FileExistsError):
        atomic_write(path, b"replacement")
    assert path.read_bytes() == b"complete"

    def failed_link(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(module.os, "link", failed_link)
    with pytest.raises(OSError):
        atomic_write(tmp_path / "new.pt", b"unpublished")
    assert list(tmp_path.iterdir()) == [path]
