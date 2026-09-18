#!/usr/bin/env python3
"""Run the bounded independent-reference stability audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.holdem.multistreet_reference_stability import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-cache",
        type=Path,
        default=Path(
            "results/multistreet-retrieved/poker/results/"
            "multistreet-campaign-cache"
        ),
    )
    parser.add_argument(
        "--production-cache",
        type=Path,
        default=Path(
            "results/multistreet-retrieved/poker/results/"
            "multistreet-campaign-cache"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/multistreet-reference-stability.json"),
    )
    args = parser.parse_args()
    result = run_audit(args.calibration_cache, args.production_cache, args.out)
    print(
        f"audited {result['overlap']['contexts']} training contexts "
        f"across {len(result['strata'])} strata; wrote {args.out}"
    )


if __name__ == "__main__":
    main()
