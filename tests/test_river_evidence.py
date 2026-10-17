import gzip
import json

import pytest

from src.blueprint import river_evidence as evidence


def prepared(tmp_path, status="completed"):
    request = dict(board=[1, 2, 3, 4, 5], ranges=[[1], [2]], seconds=100,
                   dump_path="/tmp/one", mode="play", profile_format="f64")
    (tmp_path / "request.json").write_text(json.dumps(request) + "\n")
    (tmp_path / "profile.jsonl").write_text('{"format":"fixture"}\n')
    (tmp_path / "profile.f64").write_bytes(bytes(range(128)))
    files = {p.name: evidence.file_pin(p) for p in tmp_path.iterdir()}
    return request, dict(status=status, request_sha256=files["request.json"]["sha256"]), files


def test_sample_is_stable_across_paths_timing_and_transport_but_not_scientific_inputs():
    request = dict(board=[1, 2], iterations=50, dump_path="a", seconds=2)
    key = evidence.sample_key(request)
    assert key == evidence.sample_key(dict(request, dump_path="b", seconds=99,
                                         profile_format="different", mode="quality"))
    assert key != evidence.sample_key(dict(request, iterations=51))
    assert evidence.selected_body("0" * 64)
    assert not evidence.selected_body("f" * 64)


@pytest.mark.parametrize("sampled", [False, True])
def test_success_seals_all_body_hashes_and_lossless_request_before_removing_scratch(tmp_path, monkeypatch, sampled):
    request, record, files = prepared(tmp_path)
    original = (tmp_path / "request.json").read_bytes()
    monkeypatch.setattr(evidence, "selected_body", lambda _: sampled)
    kept = evidence.seal_evidence(tmp_path, request, record, files)
    stored = json.loads((tmp_path / "receipt.json").read_text())
    assert stored["profile_retention"]["generated_files"] == {
        name: files[name] for name in ("profile.jsonl", "profile.f64")}
    assert stored["profile_retention"]["body_retained"] == sampled
    assert gzip.decompress((tmp_path / "request.json.gz").read_bytes()) == original
    assert evidence.read_request(tmp_path) == request
    assert "request.json" not in kept and not (tmp_path / "request.json").exists()
    assert not (tmp_path / "profile.f64").exists()
    assert not (tmp_path / "profile.jsonl").exists()
    assert (tmp_path / "profile.f64.gz").exists() == sampled
    assert (tmp_path / "profile.jsonl.gz").exists()
    if sampled:
        assert gzip.decompress((tmp_path / "profile.f64.gz").read_bytes()) == bytes(range(128))
    assert kept["request.json.gz"] == evidence.file_pin(tmp_path / "request.json.gz")


def test_every_failure_keeps_both_native_bodies_and_original_request(tmp_path, monkeypatch):
    request, record, files = prepared(tmp_path, status="failure")
    (tmp_path / "profile.jsonl").write_bytes(b"{truncated")
    files["profile.jsonl"] = evidence.file_pin(tmp_path / "profile.jsonl")
    monkeypatch.setattr(evidence, "selected_body", lambda _: False)
    kept = evidence.seal_evidence(tmp_path, request, record, files)
    for name,pin in files.items():
        body = gzip.decompress((tmp_path / (name + ".gz")).read_bytes())
        assert len(body) == pin['bytes']
        from hashlib import sha256
        assert sha256(body).hexdigest() == pin['sha256']
        assert name + '.gz' in kept
    assert record["profile_retention"]["failure_kept"]
    assert record["profile_retention"]["body_retained"]


def test_compression_failure_never_removes_original_evidence(tmp_path, monkeypatch):
    request, record, files = prepared(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("fixture disk error")
    monkeypatch.setattr(evidence.gzip, "GzipFile", fail)
    with pytest.raises(OSError, match="fixture disk error"):
        evidence.seal_evidence(tmp_path, request, record, files)
    assert all((tmp_path / name).is_file() for name in files)
    assert not (tmp_path / "receipt.json").exists()


def test_header_semantics_exclude_only_runtime_rss(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.write_text(json.dumps(dict(player=0, actions=[1], solver_peak_rss_bytes=100))+"\n")
    two.write_text(json.dumps(dict(player=0, actions=[1], solver_peak_rss_bytes=900))+"\n")
    assert evidence.header_semantic_sha256(one) == evidence.header_semantic_sha256(two)
    two.write_text(json.dumps(dict(player=1, actions=[1], solver_peak_rss_bytes=100))+"\n")
    assert evidence.header_semantic_sha256(one) != evidence.header_semantic_sha256(two)
