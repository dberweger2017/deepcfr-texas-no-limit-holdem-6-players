"""Prepare the exact confirmed 10B export for owner review; never publish."""

import argparse
import json
import shutil
from pathlib import Path

from scripts.verify_v042_bundle import (
    ASSET_NAME, ASSETS, CONFIRMATION, MODEL, PLAN_SHA256, full_sha, sha, verify_bundle, verify_model,
)


def build(source: Path, destination: Path, source_sha: str) -> dict:
    verify_model(source)
    if not full_sha(source_sha):
        raise ValueError("Full lowercase preparation Git SHA required")
    if destination.is_symlink() or destination.exists() and any(destination.iterdir()):
        raise ValueError("Preserve the existing candidate bundle")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination / ASSET_NAME)
    repo = Path(__file__).resolve().parents[1]
    for name in ("MODEL_CARD.md", "RELEASE_NOTES.md"):
        shutil.copyfile(repo / "docs/releases/v0.4.2" / name, destination / name)
    shutil.copyfile(repo / "scripts/verify_v042_bundle.py", destination / "verify_v042_bundle.py")
    manifest = {
        "candidate": "v0.4.2",
        "status": "unpublished-owner-review",
        "preparation_source_commit": source_sha,
        "approved_release_source_commit": None,
        "owner_publication_approval": False,
        "confirmation": CONFIRMATION,
        "plan_sha256": PLAN_SHA256,
        "model": MODEL,
        "dependency": {"pokers_git_revision": "5db20e3d5d6862b32a7402035c1340b622d3b005", "python": "3.11"},
        "publication_gate": "Separate explicit owner chat go; bind approved merged source before any tag or release",
    }
    (destination / "release-manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (destination / "SHA256SUMS").write_text("".join(f"{sha(destination / name)}  {name}\n" for name in sorted(ASSETS)))
    return verify_bundle(destination, source_sha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.out, args.source_sha), sort_keys=True))


if __name__ == "__main__":
    main()
