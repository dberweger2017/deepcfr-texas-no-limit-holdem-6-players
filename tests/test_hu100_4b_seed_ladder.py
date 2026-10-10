"""Campaign gates reject changed evidence before dependent poker play."""
import json
from pathlib import Path

import pytest

from scripts import run_hu100_4b_seed_ladder as run
from scripts import archive_hu100_4b_seed_ladder as archive


def test_reference_caps_and_recipe(tmp_path):
    assert run.cap_for(2026100601, 1_000_000_000) == 57_658_644
    assert run.cap_for(2026100601, 2_000_000_000) == 67_419_934
    assert run.cap_for(2026100601, 4_000_000_000) == 61_801_606
    assert run.cap_for(2026100901, 2_000_000_000) == 61_801_606
    command = list(map(str, run.train_command(2026100902, 2_000_000_000, tmp_path)))
    assert command[command.index("--average-rule")+1] == "opponent-sampled"
    assert command[command.index("--stack-bb")+1] == "100"
    assert "--resume" not in command
    assert "--regret-floor" not in command


def test_checkpoint_mismatch_latches_exactness_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(run, "BINARY", tmp_path/"binary")
    run.BINARY.write_bytes(b"fixture")
    dest = run.folder(2026100901, 1_000_000_000)
    dest.mkdir(parents=True)
    checkpoint = dest/"checkpoint.gz"
    checkpoint.write_bytes(b"changed checkpoint")
    row = {"checkpoint_sha256": run.file_hash(checkpoint), "binary_sha256": run.file_hash(run.BINARY)}
    (dest/"telemetry.jsonl").write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="exact checkpoint gate mismatch"):
        run.check_gate(2026100901, 1_000_000_000)
    assert run.read(dest/"exactness-mismatch.json")["actual"] == run.file_hash(checkpoint)
    assert not (dest/"gate.json").exists()


def test_clean_capacity_terminal_and_unexplained_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(run, "BINARY", tmp_path/"binary")
    run.BINARY.write_bytes(b"fixture")
    dest = run.folder(2026100601, 4_000_000_000)
    dest.mkdir(parents=True)
    checkpoint = dest/"checkpoint.gz"
    checkpoint.write_bytes(b"fixture checkpoint")
    row = {"checkpoint_sha256": run.file_hash(checkpoint), "binary_sha256": run.file_hash(run.BINARY),
        "diagnostics": {"entries": run.CAP+3}, "completed_nodes": 2_750_001_000,
        "stop_requested": False, "status": "incomplete-target"}
    (dest/"telemetry.jsonl").write_text(json.dumps(row)+"\n")
    run.check_gate(2026100601, 4_000_000_000)
    assert run.read(dest/"gate.json")["terminal_capacity_stop"]
    (dest/"gate.json").unlink()
    row["diagnostics"]["entries"] = 10
    (dest/"telemetry.jsonl").write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="unexplained incomplete"):
        run.check_gate(2026100601, 4_000_000_000)


def test_full_archive_readback_and_exclusive_output(tmp_path):
    source = tmp_path/"sample.gz"
    source.write_bytes(b"preserved evidence"*1000)
    target = tmp_path/"bundle.zip"
    receipt = archive.seal([(source, "research/sample.gz")], target, {"pr": 226})
    assert receipt["all_member_sizes_and_hashes_readback"]
    assert receipt["sha256"] == run.file_hash(target)
    assert receipt["cloud_upload_accepted"] is False
    with pytest.raises(FileExistsError):
        archive.seal([(source, "research/sample.gz")], target, {"pr": 226})


def test_controller_is_counted_but_never_owned_for_cleanup(monkeypatch):
    import os
    import psutil
    from scripts import overnight_research_guard as guard
    from scripts import run_native_hu100_growth_1b as resources
    parent = psutil.Process(os.getpid())
    monkeypatch.setenv("HU100_PR226_CONTROLLER_PID", str(parent.pid))
    monkeypatch.setenv("HU100_PR226_CONTROLLER_CREATED", str(parent.create_time()))
    monkeypatch.setattr(run.sys, "argv", ["entry", "guard"])
    monkeypatch.setattr(guard, "owned_processes", lambda known: [])
    # Register restoration for globals that guard_main replaces.
    monkeypatch.setattr(resources, "host", resources.host)
    monkeypatch.setattr(resources, "BINARY", resources.BINARY)
    def inspect():
        known = {}
        members = guard.owned_processes(known)
        assert [p.pid for p in members] == [parent.pid]
        assert known == {}
    monkeypatch.setattr(guard, "main", inspect)
    run.guard_main()
