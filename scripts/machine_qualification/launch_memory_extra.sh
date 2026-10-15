#!/bin/bash
# The selected-pattern screen cannot extend beyond the owner's compute cutoff.
set -euo pipefail
cd "$QUALIFICATION_ROOT"
remaining_s=$(python3 -c 'import datetime,time; print(int(datetime.datetime.fromisoformat("2026-10-15T04:50:00+00:00").timestamp()-time.time()))')
test "$remaining_s" -ge 300
export MEMTESTER_TEST_MASK=0x1ec00
exec python3 watchdog-overnight.py --output memory-extra --seconds "$remaining_s" \
  --min-disk-gib 40 --deadline-utc 2026-10-15T04:50:00Z -- python3 memory_extra.py
