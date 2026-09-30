"""Verify the v0.4 B100M inference bytes before loading them."""

import argparse
import hashlib
from pathlib import Path

EXPECTED_NAME = "B100M-HU20-current-seed-2026093001.json.gz"
EXPECTED_BYTES = 40_144_034
EXPECTED_SHA256 = "4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf"


def verify(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Model must be a regular file")
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError("Model byte count does not match v0.4")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    if value != EXPECTED_SHA256:
        raise ValueError("Model SHA-256 does not match v0.4")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    try:
        value = verify(args.model)
    except ValueError as error:
        parser.error(str(error))
    print(f"Verified B100M HU20 inference export: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
