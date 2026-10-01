"""Run one file-boundary solver request with the M4 watchdog."""

import argparse
from pathlib import Path

from src.diagnostics.flop_check_runtime import run_tool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    result = run_tool(args.binary, args.request, args.out,
                      memory_bytes=int(args.memory_gib * 1024**3),
                      threads=args.threads, seconds=args.seconds)
    if result["status"] != "completed":
        raise SystemExit(result["failure"])


if __name__ == "__main__":
    main()
