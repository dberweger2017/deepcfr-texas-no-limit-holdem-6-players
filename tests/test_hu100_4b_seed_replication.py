"""The new seed gate fails before continuation and preserves clean terminals."""
import json
import pytest
from scripts import run_hu100_4b_seed_replication as run


def fixture(tmp_path, monkeypatch, row):
    monkeypatch.setattr(run, "BINARY", tmp_path / "trainer")
    run.BINARY.write_bytes(b"frozen binary")
    (tmp_path / "checkpoint.gz").write_bytes(b"checkpoint fixture")
    row.update(binary_sha256=run.file_hash(run.BINARY),
               checkpoint_sha256=run.file_hash(tmp_path / "checkpoint.gz"))
    (tmp_path / "telemetry.jsonl").write_text(json.dumps(row)+"\n")
    return row


def test_independent_2b_gate_refuses_changed_checkpoint(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch, {
        "diagnostics": {"entries": 54_579_477}, "completed_nodes": 2_000_001_279,
        "stop_requested": False, "status": "saved"})
    monkeypatch.setattr(run, "reference", lambda *args: {
        "checkpoint.gz": {"sha256": "0"*64, "bytes": 18}})
    with pytest.raises(ValueError, match="exact checkpoint mismatch"):
        run.checkpoint_gate(2026100901, 2_000_000_000, tmp_path)
    assert (tmp_path / "exactness-mismatch.json").exists()
    assert not (tmp_path / "gate.json").exists()


def test_clean_terminal_records_actual_nodes_and_unexplained_stop_fails(tmp_path, monkeypatch):
    row = fixture(tmp_path, monkeypatch, {
        "diagnostics": {"entries": 70_001}, "completed_nodes": 3_900_000_123,
        "stop_requested": False, "status": "incomplete-target"})
    monkeypatch.setattr(run, "reference", lambda *args: {})
    monkeypatch.setattr(run, "cap_for", lambda *args: 70_000)
    run.checkpoint_gate(2026100902, 4_000_000_000, tmp_path)
    gate = run.read(tmp_path / "gate.json")
    assert gate["terminal_capacity_stop"]
    assert gate["actual_nodes"] == 3_900_000_123
    (tmp_path / "gate.json").unlink()
    row["diagnostics"]["entries"] = 69_999
    (tmp_path / "telemetry.jsonl").write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="unexplained incomplete"):
        run.checkpoint_gate(2026100902, 4_000_000_000, tmp_path)


def test_failed_campaign_blocks_new_science(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    (tmp_path / "campaign-stopped.json").write_text("{}")
    with pytest.raises(ValueError, match="no failed science readmission"):
        run.source()
