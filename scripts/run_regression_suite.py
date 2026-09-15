"""Run the project's deterministic regression suite."""

from pathlib import Path
import sys

import pytest


REGRESSION_TESTS = [
    "tests/test_neural_primitives.py",
    "tests/test_deep_cfr.py",
    "tests/test_neural_experiments.py",
    "tests/test_reference_games.py",
    "tests/test_tabular_cfr.py",
    "tests/test_solver_experiments.py",
    "tests/test_evaluation_cli.py",
    "tests/test_arena_schedule.py",
    "tests/test_arena_runner.py",
    "tests/test_arena_reports.py",
    "tests/test_arena_artifacts.py",
    "tests/test_arena_opponents.py",
    "tests/test_frozen_policies.py",
    "tests/test_training_opponent_modeling_regressions.py",
    "tests/test_state_scenarios.py",
    "tests/test_logging_regressions.py",
    "tests/test_pokers_regressions.py",
    "tests/test_engine_integration.py",
    "tests/test_observations.py",
    "tests/test_sessions.py",
    "tests/test_opponent_modeling_features.py",
    "tests/test_hand_observations.py",
    "tests/test_policy_boundary.py",
    "tests/test_observed_cli.py",
    "tests/test_observed_gui.py",
    "tests/test_training_regressions.py",
]


def main():
    repo_root = Path(__file__).resolve().parent.parent
    args = [str(repo_root / test_file) for test_file in REGRESSION_TESTS]
    args.append("-q")
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
