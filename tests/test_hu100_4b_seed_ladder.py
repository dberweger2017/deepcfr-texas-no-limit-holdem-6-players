"""Campaign gates reject changed evidence before dependent poker play."""
import json
from pathlib import Path

import pytest

from scripts import run_hu100_4b_seed_ladder as run
from scripts import archive_hu100_4b_seed_ladder as archive


def test_reference_caps_and_recipe(tmp_path):
    assert run.cap_for(2026100601, 1_000_000_000) == 57_658_644
    assert run.cap_for(2026100601, 2_000_000_000) == 67_419_934
    assert run.cap_for(2026100601, 4_000_000_000) == 72_437_552
    assert run.cap_for(2026100901, 2_000_000_000) == 72_437_552
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


def test_current_and_average_hash_gates_before_model_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    dest = run.folder(2026100901, 1_000_000_000)
    dest.mkdir(parents=True)
    pins = {}
    for name in ("checkpoint.gz", "current.gz", "average.gz"):
        path = dest/name
        path.write_bytes(name.encode())
        pins[name] = {"kind": name, "seed": 2026100901, "endpoint": "terminal",
            "sha256": run.file_hash(path), "bytes": path.stat().st_size}
    monkeypatch.setattr(run, "read", lambda p: {"assets": list(pins.values())})
    (dest/"current.gz").write_bytes(b"tampered current export")
    with pytest.raises(ValueError, match="exact indexed checkpoint/current/average mismatch"):
        run.model_spec(2026100901, 1_000_000_000)
    assert (dest/"export-exactness-mismatch.json").exists()
    assert not (dest/"spec.json").exists()


def test_executable_match_specs_bound_to_frozen_pairs(tmp_path, monkeypatch):
    from scripts import evaluate_hu100_4b_seed_ladder as evaluation
    monkeypatch.setattr(evaluation, "OUT", tmp_path)
    pairs = {"primary": [{"name": "fixed-candidate"}, {"name": "fixed-baseline"}]}
    spec = tmp_path/"specs/primary.json"
    spec.parent.mkdir()
    spec.write_text(json.dumps(pairs["primary"]))
    pins = evaluation.checked_specs(pairs)
    spec.write_text(json.dumps(pairs["primary"], indent=2))
    with pytest.raises(ValueError, match="frozen executable match spec bytes changed"):
        evaluation.checked_specs(pairs, pins)
    spec.write_text(json.dumps([{"name": "selected-other-candidate"}, {"name": "fixed-baseline"}]))
    with pytest.raises(ValueError, match="differ from declared pair"):
        evaluation.checked_specs(pairs)


def test_freeze_preserves_each_primary_memory_admitted_sample(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import evaluate_hu100_4b_seed_ladder as evaluation
    monkeypatch.setattr(evaluation, "OUT", tmp_path/"out")
    monkeypatch.setattr(evaluation, "ROOT", tmp_path/"repo")
    monkeypatch.setattr(evaluation.campaign, "source", lambda: "fixed-source")
    monkeypatch.setattr(evaluation.shutil, "disk_usage", lambda p: SimpleNamespace(free=120*run.GIB))
    monkeypatch.setattr(evaluation.subprocess, "check_output", lambda *a, **k: "100\n")
    captured = []
    monkeypatch.setattr(evaluation, "freshness", lambda blocks, label: captured.append((blocks.copy(), label)))
    pairs = {r: [{"name": r+"-a"}, {"name": r+"-b"}] for r in ("p0", "p1", "p2", "d0", "d1")}
    evaluation.put(evaluation.OUT/"pairs.json", {"pairs": pairs, "primary": ["p0", "p1", "p2"], "descriptive": ["d0", "d1"]})
    for rung, pair in pairs.items():
        evaluation.put(evaluation.OUT/"specs"/(rung+".json"), pair)
        arm = {"load_or_validation_seconds": 3, "files": {"trace.gz": {"bytes": 32000}}}
        evaluation.put(evaluation.OUT/"direct-pilot"/rung/"costs.json",
            {"outcomes_inspected_for_quote": False, "blocks": 32, "seconds": 8, "primary": arm, "repeat": arm})
        peak = int(5.5*run.GIB) if rung == "p0" else 4*run.GIB
        evaluation.put(evaluation.OUT/"guards/operations"/("direct-pilot-"+rung)/"receipt.json",
            {"peak_family_rss_bytes": peak, "seconds": 8})
    evaluation.put(evaluation.ROOT/"docs/reports/hu100-3b-ladder-artifacts/model-input-index.json", {"models": []})
    evaluation.put(evaluation.ROOT/"docs/reports/hu100-independent-stages-artifacts/model-index.json", {"assets": []})
    evaluation.put(evaluation.OUT/"archive-pilot-cost.json",
        {"seconds_including_source_hash_pack_readback_whole_hash": 1, "logical_member_bytes": 1000000})
    evaluation.freeze()
    frozen = evaluation.read(evaluation.OUT/"frozen-final.json")
    assert frozen["blocks_per_contrast"]["p0"] == 262144
    assert frozen["blocks_per_contrast"]["p1"] == 524288
    assert frozen["blocks_per_contrast"]["p2"] == 524288
    assert frozen["primary_confidence"] == 1-.05/3
    assert captured == [(frozen["blocks_per_contrast"], "final")]
    assert frozen["outcomes_used_for_budget"] is False
