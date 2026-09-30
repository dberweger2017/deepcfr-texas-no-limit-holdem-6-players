import gzip
import json

from scripts.hu20_platform_pilot import compare, fingerprint


def artifacts(directory, payload, *, mtime=0):
    directory.mkdir()
    for name in ("final.json.gz", "current.json.gz", "next.json.gz"):
        (directory / name).write_bytes(gzip.compress(payload, mtime=mtime))


def test_transport_difference_does_not_hide_identical_state(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    state = b'{"iteration":2}\n["key",["call"],[0.125],[2.5],3]\n'
    artifacts(left, state)
    artifacts(right, state, mtime=1)
    result = compare(left, right, tmp_path / "comparison.json")
    assert result["equal"]
    assert all(not row["byte_equal"] and row["semantic_equal"]
               for row in result["files"].values())
    assert fingerprint(left / "final.json.gz")["sha256"] != fingerprint(right / "final.json.gz")["sha256"]


def test_single_regret_change_fails_exact_parity(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    artifacts(left, b'{"iteration":2}\n["key",["call"],[0.125],[2.5],3]\n')
    artifacts(right, b'{"iteration":2}\n["key",["call"],[0.12500000000000003],[2.5],3]\n')
    result = compare(left, right, tmp_path / "comparison.json")
    assert not result["equal"]
    assert result["files"]["final.json.gz"]["first_different_record"] == 1


def test_missing_visit_record_fails_parity(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    artifacts(left, b'{"iteration":2}\n["key",["call"],[1],[2],3]\n')
    artifacts(right, b'{"iteration":2}\n')
    result = compare(left, right, tmp_path / "comparison.json")
    assert not result["equal"]
    row = result["files"]["next.json.gz"]
    assert row["right_record"] is None
    assert json.loads((tmp_path / "comparison.json").read_text())["equal"] is False
