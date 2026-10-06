"""Verify unchanged, prospectively fixed O candidate bytes without publishing."""

import argparse
from pathlib import Path

from src.play_api.o_candidate import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    try:
        value = verify(args.model)
    except ValueError as error:
        parser.error(str(error))
    print(f"Verified O1B HU20 opponent-sampled average: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
