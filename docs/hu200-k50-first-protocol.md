# First HU200 K50 lineage

Owner request October 13: sequential stages C, A, B, each a separate PR. This PR
is C only. A stage stop preserves partials and ends that stage; the next stage
remains authorized. No paid hosts or M1 research compute. PR246 roots and uploads
are protected; only separately restored PR163 tables enter this stage.

Fresh seed 2026101302, 200 BB, K50 schema
`hu200-native-reopening-ordered-history-equity-k50-v1`; opponent-sampled average,
linear iteration weighting, one root per seat, existing menu. Build current-main
native source (including PR247) with `python -m scripts.build_native_trainer` and
retain the receipt. Verify full table SHA256s against the indexed PR163 archive
and `EQUITY_K50_SHA256`, then require byte-identical fresh20M and 10M-resumed20M
recoverable checkpoints before the fresh main lineage.

The node ceiling is 3,000,000,000. Complete-iteration recoverable saves at 500M,
1B, 1.5B, 2B, 2.5B and terminal. Entry cap:
`floor((11,000,000,000 - 1,400,000,000 - 500,000,000)/(96.9 * 1.1))`.
The resulting cap is **85,373,862 entries**. The first complete iteration reaching the cap is terminal, including its small
entry overshoot. Entry projections do not stop training. Retain every native
telemetry receipt, actual nodes, iterations, entries, total traverser visits,
zero-visit fraction, speed including saves, write time and sampled family RSS.

Export current and average at 1B and terminal with `--zero-mass uniform`; check
every row using the existing native export accumulator auditor. No matches,
release, default change or strength claim. Slumbot evaluation is separate.

## Observed resource guards

Five-second samples enumerate processes with psutil, tolerate AccessDenied for
unowned helpers, retain PID creation identities and stop only owned descendants.
Each stage records its starting swap baseline; transient readings are logged.
Stop after 60 seconds continuously above 13 GiB whole-family RSS, critical macOS
pressure, below 8% free memory, or above 4,000,000,000 bytes swap growth from that
baseline. Warning pressure is allowed. Free disk below 16 GiB stops immediately.
Native training receives a stop file with at most ten seconds of grace; disk
floor stops terminate immediately. Preserve prior milestones and interrupted
write prefixes if the save cannot finish. Guards
persist their breach clocks across operations. Broken accounting or failed
commands preserve a failure receipt and admit no scientific rerun.

## Evidence and delivery

Report `docs/reports/hu200-k50-first.md`; full evidence ZIP and separate small ZIP
containing terminal/1B averages only, each with an embedded member manifest and
local readback. Destination `~/Local/Research-Cloud/PR-<n>-hu200-k50-first/`.
RESULTS_INDEX records archive/member hashes, native upload status, independent
Drive ID/name/size/parent, retrieval commands, all failures and partials. Do not
claim remote-byte verification unless downloaded. Retain originals pending
accepted upload/dependency review; no synced deletion or forced offloading.
Stage intended text/code files and run `python -m scripts.check_repository_artifacts`
before every commit. Review diff and relevant checks; merge only after green
final-head CI, then update Current position for the landed outcome.
