# HU100 v0.5.3 candidate: three training seeds

Owner-authorized October 15, 2026. Research only; the owner decides merge and
release. No tag, release, default change, Slumbot or credential work.

## Recipe and clock

Build current-main native source with `python -m scripts.build_native_trainer`,
retaining its receipt. A single 14-hour maximum starts no later than the first
build. Train **2026101501, then 2026101502**, never concurrently. Each uses
HU100, two 10,000-chip stacks, 50/100 blinds, no ante/rake/raise cap, #163's
hash-pinned K50 tables, `hu100-native-reopening-ordered-history-equity-k50-menu-v2`,
ordered history, linear CFR, opponent-sampled average and one root per seat.

Stage 1 copies #260: fixed cap **97,290,497** (never re-estimated), milestone
500M and target 1B. Stop at the first complete iteration past capacity; record
actual nodes, iteration and entries separately from target filenames. Stage 2
hash-pins that terminal and resumes with `--freeze-at-cap --max-entries ACTUAL`
and `--nodes 5000000000`. The final sorted-key SHA256 and count must equal
stage 1's. Export final current/average with `--zero-mass uniform`; every row
passes `src.diagnostics.cfr_average.audit`. Retain native telemetry for stored
visits/key, zero visits/mass, discarded updates and both players' frozen
missing decisions by street. No scientific retry or budget extension.

## Evaluation before readout

Use #272's shared-deal duplicate harness and nominal paired Student-t 95%
intervals, conditional on each fixed policy pair. For each new seed: frozen v2
translation off versus released v0.5.2 translation on (512 states/128 events),
all seven opponents, **16,384 independent duplicate blocks each**. Include the
original #272 panel beside both replications. Report #272's comparison rule
(no primary upper bound below −20 BB/100 and pot-pressure lower >−20) as a
historical comparison only. An improvement release has no pass/fail gate.

Fresh direct matches for **all three seeds**, including #272's 2026100601,
use v0.5.2 translation on (512/128), v2 off, **131,072 blocks each**, and
reached-key coverage for both sides. Direct results are reported, never gated.

Reserve distinct roots against main and every open PR, and use physically
separate pilot/final output directories for every contrast. Pilots use 1,024
direct/64 panel blocks, expose only costs and dispersion, and never join final
samples. Post and read back a canonical PR-description plan naming every root,
input, count and cost before the first final hand. Separate one-off loading,
cache creation and hash checks from per-block play, trace hashes, independent
replay and exact reproduction. Scale only per-block work. Include 25% future
work margin and at least 30 minutes for closeout. If full counts cannot fit
14 hours from first build, reduce once, before outcomes, to 8,192 panel and
65,536 direct; stop if those cannot fit. No outcome-driven extension or failed
hand reruns. Every final hand across the whole campaign independently replays
and exactly reproduces before any result readout.

## Machine and storage

Apple M4 /16 GB only; detached tmux over SSH. Hold both
`~/Local/.hu20-m4-research.lock` and `.hu100-m4-research.lock` throughout, refuse
another campaign, retain every five-second resource reading. Stop after 60
continuous observed seconds of family RSS >13 GiB, critical pressure (4), free
memory <8%, or fixed-baseline swap growth >4 GB. Disk <16 GiB stops immediately.
Verify PID creation times before signals. Admission requires at least 30 GiB.

Work only in this campaign's unsynced root. Retain each stage-1 terminal,
final checkpoint and final exports; remove superseded own checkpoints only
after successors verify. No deletion inside synced folders or forced offload.
At preparation the M4 had 43.9 GiB free: two #272-sized retained sets would
consume about 26.3 GiB before caches/traces/ZIPs. The owner confirmed PR278
merged and authorized starting while the owner cleans storage. This does not
waive disk guards or authorize removing another PR's files. Check available
space before every materialization and preserve a partial if admission fails.

## Evidence

One feature PR, left for owner merge. Report actual counters, all audits,
three panels, three direct matches, coverage, fixed/per-block costs, all guards
and every failure. Archive raw evidence and a separate average-only ZIP per
new seed in this PR's existing Research-Cloud folder, embedded manifests and
complete local member readback. The average export lives in its separate ZIP,
referenced by the raw-evidence manifest, avoiding duplicate archive storage.
Reference accepted #272/#163/release inputs.
Record actual Drive IDs, whole/member hashes and restore commands in
RESULTS_INDEX, and update ROADMAP Current position. Local integrity, native
upload status and independent cloud metadata acceptance are separate claims.
