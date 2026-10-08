"""Publication preserves original provenance and rejects inconsistent approval bindings."""

import gzip
import json

import pytest

from scripts import build_v042_bundle as builder
from scripts import verify_v042_bundle as verifier


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    model = dict(verifier.MODEL)
    header = {
        "format": model["format"], "kind": "diagnostic-inference",
        "extraction": model["extraction"], "source_checkpoint_sha256": model["checkpoint_sha256"],
        "checkpoint_header": {
            "config": {"seed": model["seed"], "game": model["game"], "raise_cap": None},
            "abstraction": model["schema"], "iteration": model["iteration"],
            "average_rule": "opponent-sampled", "identity": {"players": 2},
        },
    }
    source = tmp_path / "small-export.gz"
    source.write_bytes(gzip.compress((json.dumps(header) + "\n").encode(), mtime=0))
    model.update(bytes=source.stat().st_size, sha256=verifier.sha(source))
    monkeypatch.setattr(verifier, "MODEL", model)
    monkeypatch.setattr(builder, "MODEL", model)
    destination = tmp_path / "prepared"
    builder.build(source, destination, "a" * 40)
    return source, destination


def rehash(directory):
    (directory / "SHA256SUMS").write_text("".join(
        f"{verifier.sha(directory / name)}  {name}\n" for name in sorted(verifier.ASSETS)))


def test_exact_copy_source_binding_and_no_overwrite(prepared):
    source, directory = prepared
    assert (directory / verifier.ASSET_NAME).read_bytes() == source.read_bytes()
    assert verifier.verify_bundle(directory, "a" * 40)["status"] == "unpublished-owner-review"
    with pytest.raises(ValueError, match="source"):
        verifier.verify_bundle(directory, "b" * 40)
    with pytest.raises(ValueError, match="Preserve"):
        builder.build(source, directory, "a" * 40)


@pytest.mark.parametrize("field,value", [
    ("plan_sha256", "0" * 64), ("candidate", "v0.4.1"),
    ("owner_publication_approval", True), ("approved_release_source_commit", "a" * 40),
    ("status", "owner-approved-publication"),
])
def test_rehashed_manifest_cannot_change_identity_or_grant_publication(prepared, field, value):
    _, directory = prepared
    path = directory / "release-manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    rehash(directory)
    with pytest.raises(ValueError, match="Manifest|publication hold|Publication source"):
        verifier.verify_bundle(directory)


def test_rehashed_model_change_still_fails_pinned_identity(prepared):
    _, directory = prepared
    path = directory / verifier.ASSET_NAME
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    rehash(directory)
    with pytest.raises(ValueError, match="fixed first-seed"):
        verifier.verify_bundle(directory)


def test_checksum_coverage_and_regular_members(prepared):
    _, directory = prepared
    checksums = directory / "SHA256SUMS"
    original = checksums.read_text()
    checksums.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        verifier.verify_bundle(directory)
    checksums.write_text("\n".join(original.splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="Missing release checksum"):
        verifier.verify_bundle(directory)
    checksums.write_text(original)
    notes = directory / "RELEASE_NOTES.md"
    notes.write_text("corrupted")
    with pytest.raises(ValueError, match="Checksum"):
        verifier.verify_bundle(directory)
    notes.unlink()
    notes.symlink_to(directory / "MODEL_CARD.md")
    with pytest.raises(ValueError, match="regular"):
        verifier.verify_bundle(directory)


def test_publication_requires_explicit_approval_and_tagged_source(prepared, tmp_path):
    source, preparation = prepared
    with pytest.raises(ValueError, match="approval is required"):
        verifier.verify_bundle(preparation, "a" * 40, require_publication=True)
    with pytest.raises(ValueError, match="explicit boolean"):
        builder.build(source, tmp_path / "invalid", "b" * 40, publication_approved=1)
    publication = tmp_path / "publication"
    builder.build(source, publication, "b" * 40, publication_approved=True)
    result = verifier.verify_bundle(publication, "b" * 40, require_publication=True)
    assert result["status"] == "owner-approved-publication"
    manifest = json.loads((publication / "release-manifest.json").read_text())
    assert manifest["preparation_source_commit"] == verifier.PREPARATION_SOURCE
    assert manifest["preparation_provenance"] == verifier.PREPARATION_PROVENANCE
    assert manifest["approved_release_source_commit"] == manifest["package_source_commit"] == "b" * 40
    with pytest.raises(ValueError, match="Expected tagged source"):
        verifier.verify_bundle(publication, require_publication=True)
    with pytest.raises(ValueError, match="source commit differs"):
        verifier.verify_bundle(publication, "c" * 40, require_publication=True)
    for field, wrong in (("approved_release_source_commit", "c" * 40),
                         ("release_tag", "v0.4.1"), ("owner_publication_approval", 1),
                         ("preparation_source_commit", "d" * 40)):
        path = publication / "release-manifest.json"
        changed = dict(manifest, **{field: wrong})
        path.write_text(json.dumps(changed))
        rehash(publication)
        with pytest.raises(ValueError):
            verifier.verify_bundle(publication, "b" * 40, require_publication=True)


def test_catalog_bytes_are_fixed_and_must_match_runtime_pin():
    from src.play_api.versions import RELEASES
    from src.policies import v042
    from pathlib import Path
    import hashlib

    data = Path("configs/play/release-manifests/v0.4.2.json").read_bytes()
    release = next(r for r in RELEASES if r.version == "v0.4.2")
    assert data == verifier.catalog_bytes()
    assert hashlib.sha256(data).hexdigest() == release.manifest_sha256
    assert verifier.MODEL["sha256"] == v042.MODEL_SHA256
    assert verifier.MODEL["bytes"] == v042.MODEL_BYTES
    assert release.manifest_asset_name == "catalog-manifest.json"
