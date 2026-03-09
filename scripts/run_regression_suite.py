"""Run the project's deterministic regression suite."""

from pathlib import Path
import sys

import pytest


REGRESSION_TESTS = [
    "tests/test_state_scenarios.py",
    "tests/test_logging_regressions.py",
    "tests/test_pokers_regressions.py",
    "tests/test_training_regressions.py",
]


def main():
    repo_root = Path(__file__).resolve().parent.parent
    args = [str(repo_root / test_file) for test_file in REGRESSION_TESTS]
    args.append("-q")
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
