"""Verify every staged/downloaded release asset, its identity and source binding."""

import argparse
import hashlib
import json
from pathlib import Path

from src.policies.v041 import ASSET_NAME, CHECKPOINT_SHA256, ITERATION, MODEL_BYTES, MODEL_SHA256, SEED, verify

ASSETS = {ASSET_NAME, "MODEL_CARD.md", "RELEASE_NOTES.md", "release-manifest.json"}


def full_sha(value):
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def verify_bundle(directory: Path, expected_source=None):
    if set(p.name for p in directory.iterdir()) != ASSETS | {"SHA256SUMS"}:
        raise ValueError("Bundle members differ from the declared inference-only assets")
    for name in ASSETS | {"SHA256SUMS"}:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Bundle assets must be regular files")
    rows = (directory / "SHA256SUMS").read_text().splitlines()
    hashes = {}
    for row in rows:
        fields = row.split("  ")
        if len(fields) != 2 or fields[1] not in ASSETS or fields[1] in hashes:
            raise ValueError("Invalid, duplicate or undeclared checksum member")
        digest, name = fields
        with (directory / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Checksum differs: {name}")
        hashes[name] = digest
    if set(hashes) != ASSETS:
        raise ValueError("Missing release checksum")
    verify(directory / ASSET_NAME)
    manifest = json.loads((directory / "release-manifest.json").read_text())
    model = manifest["model"]
    pinned = {"file": ASSET_NAME, "bytes": MODEL_BYTES, "sha256": MODEL_SHA256,
              "seed": SEED, "iteration": ITERATION, "checkpoint_sha256": CHECKPOINT_SHA256,
              "training_budget_nodes": 1_000_000_000, "resumable": False, "players": 2,
              "raise_cap": None, "extraction": "normalize-lifetime-iteration-opponent-sampled-accumulator-v1",
              "format": "holdem-hu20-stored-cfr-average-diagnostic-v1",
              "game": "hu20-native-reopening-20bb-52card-no-ante-rake-v1",
              "schema": "hu20-native-reopening-ordered-history-card-v1"}
    if model != pinned or manifest["candidate"] != "v0.4.1":
        raise ValueError("Manifest does not identify the confirmed exact O model")
    source = manifest["preparation_source_commit"]
    if not full_sha(source) or expected_source is not None and source != expected_source:
        raise ValueError("Bundle source commit differs")
    if manifest["owner_publication_approval"] is True:
        if manifest["approved_release_source_commit"] != source or manifest["status"] != "owner-approved-publication":
            raise ValueError("Publication source/gate binding differs")
    elif (manifest["owner_publication_approval"] is not False
          or manifest["approved_release_source_commit"] is not None
          or manifest["status"] != "unpublished-owner-review"):
        raise ValueError("Preparation must preserve the explicit publication hold")
    return {"source_commit": source, "model_sha256": MODEL_SHA256,
            "assets_verified": len(hashes), "status": manifest["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expect-source")
    args = parser.parse_args()
    try:
        print(json.dumps(verify_bundle(args.directory, args.expect_source), sort_keys=True))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
