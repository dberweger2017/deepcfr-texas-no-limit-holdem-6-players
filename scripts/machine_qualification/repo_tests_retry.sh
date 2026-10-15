#!/bin/bash
# Repair only the diagnosed missing pip module, preserving the first attempt.
set -euo pipefail
cd "$QUALIFICATION_ROOT"
python3 watchdog-overnight.py --output pip-repair --seconds 120 \
  --min-disk-gib 40 --deadline-utc 2026-10-15T04:50:00Z -- \
  "$HOME/.local/bin/uv" pip install --python repo-main/.venv/bin/python pip
repo-main/.venv/bin/python -m pip --version > pip-repair/version.txt
"$HOME/.local/bin/uv" pip freeze --python repo-main/.venv/bin/python > requirements-installed-after-pip.txt
exec python3 watchdog-overnight.py --output repo-tests-retry --seconds 1800 \
  --max-rss-gib 16 --min-disk-gib 40 --deadline-utc 2026-10-15T04:50:00Z -- \
  bash repo_tests.sh
