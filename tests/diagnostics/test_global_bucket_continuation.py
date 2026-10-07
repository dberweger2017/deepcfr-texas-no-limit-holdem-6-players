import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import continue_global_bucket_validation as continuation
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash


def test_member_verification_detects_changed_payload_even_with_matching_zip_hash(tmp_path):
    path = tmp_path/"native.zip"
    original = b"frozen response\n"
    digest = hashlib.sha256(original).hexdigest()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("response.jsonl", b"different bytes\n")
        archive.writestr("ARCHIVE-MANIFEST.json", json.dumps(dict(
            binary_sha256=continuation.BINARY_HASH,
            members=[dict(path="response.jsonl", bytes=len(original), sha256=digest)])))
    with pytest.raises(ValueError, match="member differs"):
        continuation.verify_payload(dict(zip_path=str(path), zip_sha256=file_hash(path),
                                         raw_sha256=digest, raw_bytes=len(original)))


def test_reuse_rejects_a_hole_in_completed_frozen_order(tmp_path, monkeypatch):
    jobs = [dict(job=str(i), request_sha256="request") for i in range(3)]
    for job in (jobs[0], jobs[2]):
        atomic_json(tmp_path/"run/collect"/job["job"]/"result.json",
                    dict(job=job, reference_gate=dict(passed=True), binary_sha256=continuation.BINARY_HASH,
                         request_sha256="request", rows=[]))
    monkeypatch.setattr(continuation, "compare_collection", lambda *_: None)
    with pytest.raises(ValueError, match="not a frozen prefix"):
        continuation.verified_prefix(tmp_path, jobs)


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    root = tmp_path/"worktree/planning/global-bucket-validation"
    root.mkdir(parents=True)
    jobs = [dict(job=str(i), lineage=i) for i in range(3)]
    prefix = [dict(job="0", result_sha256="digest")]
    atomic_json(root/"prepared/manifest.json", dict(jobs=jobs))
    atomic_json(root/"budget.json", {})
    proof = dict(passed=True, attempt=continuation.ATTEMPT, created_epoch=100,
                 budget_sha256="digest", prepared_manifest_sha256="digest", stop_receipt_sha256="digest",
                 source_sha256={p: "digest" for p in continuation.SOURCES}, retained_prefix=prefix,
                 incremental_disk_bytes=continuation.GIB, conservative_remaining_seconds=1000,
                 original_deadline_epoch=100000)
    atomic_json(root/"continuation-admission.json", proof)
    interrupted = root/"run/collect/1"
    interrupted.mkdir(parents=True)
    (interrupted/"response.jsonl").write_bytes(b"partial unchanged\n")
    monkeypatch.setattr(continuation.time, "time", lambda: 200)
    monkeypatch.setattr(continuation, "file_hash", lambda _: "digest")
    monkeypatch.setattr(continuation, "verified_prefix", lambda *_: prefix)
    monkeypatch.setattr(continuation, "guard", lambda *_: dict(free_disk_bytes=20*continuation.GIB))
    calls = []
    def child(root, mode, **kwargs):
        calls.append((mode, kwargs))
        if mode == "fit":
            atomic_json(root/"run"/f"crossfit-{kwargs['fold']}-{kwargs['lineage']}.json.receipt.json", {})
    monkeypatch.setattr(continuation, "child", child)
    return root, calls


def test_one_shot_reuses_prefix_completes_all_phases_and_preserves_partial(admitted):
    root, calls = admitted
    continuation.run(root)
    collect = [kwargs["job"] for mode, kwargs in calls if kwargs.get("phase") == "collect"]
    locks = [kwargs["job"] for mode, kwargs in calls if kwargs.get("phase") == "relock"]
    fits = {(kwargs["lineage"], kwargs["fold"]) for mode, kwargs in calls if mode == "fit"}
    assert collect == ["1", "2"]
    assert locks == ["0", "1", "2"]
    assert fits == {(lineage, fold) for lineage in range(3) for fold in (0, 1)}
    assert (root/"attempts/initial-resource-stop/collect/1/response.jsonl").read_bytes() == b"partial unchanged\n"
    assert json.loads((root/"status.json").read_text())["phase"] == "complete"
    before = list(calls)
    with pytest.raises(FileExistsError):
        continuation.run(root)
    assert calls == before


def test_resource_failure_is_terminal_without_retry(admitted, monkeypatch):
    root, calls = admitted
    def failed(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("Swap growth ceiling")
    monkeypatch.setattr(continuation, "child", failed)
    with pytest.raises(RuntimeError, match="Swap growth ceiling"):
        continuation.run(root)
    assert len(calls) == 1
    assert json.loads((root/"status.json").read_text())["phase"] == "stopped"
    assert not json.loads((root/"continuation-failure.json").read_text())["automatic_restart"]
    assert (root/"attempts/initial-resource-stop/collect/1/response.jsonl").exists()


def test_changed_admission_source_blocks_dispatch(admitted, monkeypatch):
    root, calls = admitted
    monkeypatch.setattr(continuation, "file_hash", lambda _: "changed")
    with pytest.raises(ValueError, match="changed"):
        continuation.run(root)
    assert not calls
    assert not (root/"continuation-worker-start.lock").exists()
