"""Verify the v0.4 B100M inference bytes before loading them."""

import argparse
from pathlib import Path

from src.policies.v040 import verify

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
