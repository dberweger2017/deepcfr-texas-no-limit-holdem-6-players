#!/bin/bash
# Ordered admission checks; never start training after a failed identity check.
set -euo pipefail
cd "$QUALIFICATION_ROOT"
python3 watchdog-overnight.py --output build-identity --seconds 120 \
  --min-disk-gib 40 --deadline-utc 2026-10-15T04:50:00Z -- \
  python3 build_comparison.py
exec python3 watchdog-overnight.py --output recovery-gates --seconds 1200 \
  --max-rss-gib 16 --min-disk-gib 40 --deadline-utc 2026-10-15T04:50:00Z -- \
  python3 trainer_gates.py
