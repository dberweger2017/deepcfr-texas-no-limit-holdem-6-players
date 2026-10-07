"""Stage unchanged, verified B100M inference bytes and small release assets.

Run only on a trusted host with the authoritative export. Output is ignored
under results/ and must be checked again before upload; this does not publish.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from src.policies.v040 import EXPECTED_BYTES, EXPECTED_NAME, EXPECTED_SHA256, verify

ENGINE_REVISION = "5db20e3d5d6862b32a7402035c1340b622d3b005"
GAME = "hu20-native-reopening-20bb-52card-no-ante-rake-v1"
SCHEMA = "hu20-native-reopening-ordered-history-card-v1"
FORMAT = "holdem-hu20-native-reopening-blueprint-v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source: Path, destination: Path, source_sha: str) -> None:
    verify(source)
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("Full lowercase Git source SHA is required")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output directory must be empty to prevent overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    model = destination / EXPECTED_NAME
    shutil.copyfile(source, model)
    verify(model)
    card = Path("docs/releases/v0.4.0/MODEL_CARD.md")
    shutil.copyfile(card, destination / card.name)
    manifest = {
        "release": "v0.4.0",
        "source_commit": source_sha,
        "artifact_kind": "inference-export-not-resumable-training-checkpoint",
        "source_record": "training/B-2026093001/current-100000000.json.gz",
        "training_lineage": "fixed-first-seed-2026093001; 100000029 completed traversal nodes",
        "model": {"file": EXPECTED_NAME, "bytes": EXPECTED_BYTES, "sha256": EXPECTED_SHA256,
                  "format": FORMAT, "game": GAME, "schema": SCHEMA,
                  "players": 2, "raise_cap": None, "extraction": "current"},
        "dependency": {"pokers_git_revision": ENGINE_REVISION, "python": "3.11"},
        "validation_scope": "Hash and metadata pin; see release-readiness PR for candidate tests and real-model smoke. No strength qualification.",
        "training_artifacts": "Full checkpoints and raw reports are separately retained on M4; see docs/reports/hu20-scaling-m4-recovery.md and its sealed manifest.",
    }
    manifest_path = destination / "release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    with (destination / "SHA256SUMS").open("w") as sums:
        for name in (EXPECTED_NAME, "MODEL_CARD.md", "release-manifest.json"):
            sums.write(f"{sha(destination / name)}  {name}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    build(args.source, args.out, args.source_sha)
    print(f"Staged verified release assets in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
