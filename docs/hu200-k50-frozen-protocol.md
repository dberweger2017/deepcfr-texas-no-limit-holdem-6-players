# HU200 K50 frozen continuation

Owner-authorized October 13, 2026. Test whether #250's capped model is limited
by visits to its retained keys. Tonight is training and audited exports only;
playing evaluation is a separate owner task. This cannot establish playing
strength or convergence in the unrestricted game.

## Admission and exact reproduction

PR260 must be merged and its M4 campaign processes exited. Confirmed merged
at 17:05:44 UTC; read-only M4 process checks found no campaign workers.
Build current main `c53f7935ebbc45180e179e5a3aea74af48bff783` on M4 using
`python -m scripts.build_native_trainer`, retaining the build receipt.
Use a separate unsynced worktree and ignored results directory. Copy #250's
table files read-only into this campaign; verify #163's three full SHA256s.
Do not retrieve #250's model from Drive.

Reproduce #250's fresh 20M and 10M-to-20M recovery gate and exact fresh main
command, including its 500M/1B/1.5B/2B/2.5B save schedule and 3B target.
HU200, K50 cards, menu v1, seed 2026101302, one root per seat,
opponent-sampled averaging and linear CFR remain fixed.
The cap is 85,373,862 entries; omit `--freeze-at-cap`.

- 20M: 20,002,695 nodes, 8,728 iterations, 7,883,678 entries;
  270,143,830 bytes; SHA256
  `4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`.
- Cap terminal: 752,593,175 nodes, 493,496 iterations, 85,373,907 entries;
  3,442,723,146 bytes; SHA256
  `4b25e8029ed0ebfbeeb42ac02dee3179303e7f2300c23c21d32d27382f390da4`.

Any byte or count mismatch is terminal and prohibits continuation. Checkpoint
filenames name requested targets; report actual counters separately.

## Frozen continuation and time bound

Resume only the reproduced terminal, hash-pinned, with `--freeze-at-cap` and
`--max-entries 85373907`, as [the merged mode specifies](reports/frozen-table.md).
Target 20B cumulative nodes; recoverable saves at 2B, 5B, 10B and 20B.
Use telemetry and a fresh stop file. No additional training-recipe change.

The nine-hour clock starts with this campaign's first detached build.
Reserve two hours for both current/average exports and full audits, plus
15 minutes for the terminal save; calculate `--max-seconds` from the remaining
fixed clock and loading allowance. A detached deadline guard requests a clean
training stop when the reserve starts. Native time starts after checkpoint
load, so the guard also enforces wall time. An incomplete requested milestone
is recorded as incomplete, retaining actual counters and the last complete
milestone. Never relabel a time-stopped run as 20B.

Measure the first frozen segment from native timing minus write time, with
five-second observations of the first save's temporary file to locate the
2B crossing within the first ten minutes. State any load/sort/timing uncertainty.
Keep the milestones and recipe even if the mature rate cannot reach 20B.

Keep the stage-1 terminal, 5B and final frozen checkpoints. Remove earlier
frozen checkpoints only after their successors pass saved-byte, count and
sorted-key verification, inside this campaign's own unsynced results root.
Export 5B and final current/average with `--zero-mass uniform`; audit every row
using `src.diagnostics.cfr_average.audit`. Hash the sorted serialized key list
and check it equals the stage-1 list for each frozen save. Retain all receipts.

## Detached guards and evidence

All M4 jobs, including the five-second sampler and stopping rules, run detached
on M4. No open M1 session is required. Keep every reading and one campaign swap
baseline. Stop only after 60 continuous observed seconds above 13 GiB family
RSS, critical pressure, free memory below 8%, or swap growth above 4 GB.
Warning pressure is permitted. Disk below 16 GiB immediately requests the
fresh stop file, allowing native serialization to complete. Verify process
creation times before signaling owned processes; never signal other jobs.
Broken accounting, observation or command failures preserve the attempt and
block dependent training. No scientific retry, budget extension or paid compute.

Per save report actual nodes, iterations, fixed entry count, mean traverser
visits/key, zero-visit share, discarded visits/updates, traverser and opponent
missing-decision shares by street, RSS and write seconds. Preserve all sampled
resource readings, exact commands, build/input receipts and incomplete attempts.
Archive member-hashed ZIPs with complete local readback under this PR's existing
Research-Cloud folder, including a separate 5B/final-average-only ZIP. Index
Drive locators, archive/member hashes, restoration commands and separate native
and independent cloud acceptance. Never delete or evict synced files.
No evaluation, external poker calls, release, tag or default change.
