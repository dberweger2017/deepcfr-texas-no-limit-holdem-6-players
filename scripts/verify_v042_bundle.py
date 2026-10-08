"""Standalone Python 3.11 verifier for v0.4.2 provenance and publication binding."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ASSET_NAME = "O10B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz"
MODEL = {
    "file": ASSET_NAME,
    "bytes": 249_237_403,
    "sha256": "15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae",
    "checkpoint_sha256": "54553c008231126c94ec455e89dcbcdb162aa46736c1a40161cf0d63d666ab49",
    "seed": 2026100601,
    "iteration": 19_538_759,
    "training_budget_nodes": 10_000_000_000,
    "format": "holdem-hu20-stored-cfr-average-diagnostic-v1",
    "game": "hu20-native-reopening-20bb-52card-no-ante-rake-v1",
    "schema": "hu20-native-reopening-ordered-history-card-v1",
    "extraction": "normalize-lifetime-iteration-opponent-sampled-accumulator-v1",
    "players": 2,
    "raise_cap": None,
    "resumable": False,
}
CONFIRMATION = "https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/188"
PLAN_SHA256 = "8d9fd3ff6f17e602f495fdebaccf8eb156a6a02300312e0a102720c56c611d99"
PREPARATION_SOURCE = "1d80bd02c2e8acf2dc99b75f6e0349bb55a941d8"
PREPARATION_MANIFEST_SHA256 = "e92a0c859b0f4db2a8306c58da354d4bf79e1ad95db567db7489b0277d6e33e8"
PREPARATION_PROVENANCE = {
    "source_commit": PREPARATION_SOURCE,
    "archive_id": "1ismlfD-LKQfFqAAR3O5LN_6UVmebHChH",
    "archive_bytes": 1_982_165_922,
    "archive_sha256": "7a36e20f5e4560ec98d14ecaa4a6bb1fe77a7ecf1a5aa710cd1e03e586b45a80",
    "archive_manifest_sha256": "f5354027221c0750b086c3ff2553f6acdaded38b218a37963922f5e42544b702",
    "member": "research/package/release-manifest.json",
    "manifest_sha256": PREPARATION_MANIFEST_SHA256,
}
LEGACY_ASSETS = {ASSET_NAME, "MODEL_CARD.md", "RELEASE_NOTES.md", "release-manifest.json", "verify_v042_bundle.py"}
ASSETS = LEGACY_ASSETS | {"catalog-manifest.json"}


def catalog_manifest() -> dict:
    # Independent of its containing Git commit: the publication manifest binds
    # the eventual tagged source, while runtime pins these fixed identity bytes.
    return {"release": "v0.4.2", "kind": "holdem-hu20-release-catalog-v1",
            "model": MODEL, "confirmation": CONFIRMATION,
            "preparation_source_commit": PREPARATION_SOURCE,
            "preparation_manifest_sha256": PREPARATION_MANIFEST_SHA256,
            "publication_manifest_file": "release-manifest.json"}


def catalog_bytes() -> bytes:
    return (json.dumps(catalog_manifest(), sort_keys=True, indent=2) + "\n").encode()


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def full_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def verify_model(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Model must be a regular file")
    if path.stat().st_size != MODEL["bytes"] or sha(path) != MODEL["sha256"]:
        raise ValueError("Model bytes/SHA256 differ from the fixed first-seed export")
    with gzip.open(path, "rt") as stream:
        header = json.loads(stream.readline())
    checkpoint = header["checkpoint_header"]
    if (header["format"] != MODEL["format"]
            or header["kind"] != "diagnostic-inference"
            or header["extraction"] != MODEL["extraction"]
            or header["source_checkpoint_sha256"] != MODEL["checkpoint_sha256"]
            or checkpoint["config"]["seed"] != MODEL["seed"]
            or checkpoint["config"]["game"] != MODEL["game"]
            or checkpoint["config"]["raise_cap"] is not None
            or checkpoint["abstraction"] != MODEL["schema"]
            or checkpoint["iteration"] != MODEL["iteration"]
            or checkpoint["average_rule"] != "opponent-sampled"
            or checkpoint["identity"]["players"] != 2):
        raise ValueError("Model lineage, game or extraction differs")


def verify_bundle(directory: Path, expected_source=None, *, require_publication=False) -> dict:
    members = set(p.name for p in directory.iterdir())
    legacy = members == LEGACY_ASSETS | {"SHA256SUMS"}
    assets = LEGACY_ASSETS if legacy else ASSETS
    if members != assets | {"SHA256SUMS"}:
        raise ValueError("Bundle members differ from the declared inference assets")
    for name in assets | {"SHA256SUMS"}:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Bundle assets must be regular files")
    hashes = {}
    for row in (directory / "SHA256SUMS").read_text().splitlines():
        fields = row.split("  ")
        if len(fields) != 2 or fields[1] not in assets or fields[1] in hashes:
            raise ValueError("Invalid, duplicate or undeclared checksum member")
        digest, name = fields
        if sha(directory / name) != digest:
            raise ValueError(f"Checksum differs: {name}")
        hashes[name] = digest
    if set(hashes) != assets:
        raise ValueError("Missing release checksum")
    verify_model(directory / ASSET_NAME)
    manifest = json.loads((directory / "release-manifest.json").read_text())
    if (manifest["model"] != MODEL or manifest["candidate"] != "v0.4.2"
            or manifest["confirmation"] != CONFIRMATION or manifest["plan_sha256"] != PLAN_SHA256):
        raise ValueError("Manifest does not identify the confirmed model and plan")
    if manifest["preparation_source_commit"] != PREPARATION_SOURCE:
        raise ValueError("Original preparation source differs")
    if legacy:
        if sha(directory / "release-manifest.json") != PREPARATION_MANIFEST_SHA256:
            raise ValueError("Legacy preparation manifest differs")
    else:
        if manifest["preparation_provenance"] != PREPARATION_PROVENANCE:
            raise ValueError("Original preparation provenance differs")
        catalog_sha = hashlib.sha256(catalog_bytes()).hexdigest()
        if (sha(directory / "catalog-manifest.json") != catalog_sha
                or manifest["catalog_manifest_sha256"] != catalog_sha):
            raise ValueError("Pinned catalog manifest differs")
    source = manifest["preparation_source_commit"] if legacy else manifest["package_source_commit"]
    if not full_sha(source) or expected_source is not None and source != expected_source:
        raise ValueError("Bundle source commit differs")
    if manifest["owner_publication_approval"] is True:
        if (legacy or manifest["approved_release_source_commit"] != source
                or manifest["status"] != "owner-approved-publication"
                or manifest["release_tag"] != "v0.4.2"):
            raise ValueError("Publication source/tag/approval binding differs")
    elif (manifest["owner_publication_approval"] is not False
            or manifest["approved_release_source_commit"] is not None
            or manifest["status"] != "unpublished-owner-review"
            or not legacy and manifest["release_tag"] is not None):
        raise ValueError("Preparation must preserve the explicit publication hold")
    if require_publication:
        if expected_source is None or not full_sha(expected_source):
            raise ValueError("Expected tagged source is required for publication verification")
        if manifest["owner_publication_approval"] is not True:
            raise ValueError("Publication approval is required")
    return {"source_commit": source, "model_sha256": MODEL["sha256"],
            "assets_verified": len(hashes), "status": manifest["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expect-source")
    parser.add_argument("--require-publication", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(verify_bundle(args.directory, args.expect_source,
                                       require_publication=args.require_publication), sort_keys=True))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
