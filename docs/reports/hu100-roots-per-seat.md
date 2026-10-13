# HU100 roots per seat at fixed nodes

Owner-authorized Stage B follows the merged [HU200 K50 run](hu200-k50-first.md) and [pot-pressure diagnosis](hu100-pot-pressure-diagnosis.md). Fresh R=4 and R=8 runs reach 1B nodes on the M4, seed 2026100601, with the production HU100 v1 recipe otherwise unchanged. No release or adoption change.

## Training and CPU

Native was rebuilt from main `7b0090242767bfe93e24a7501a4a87664bef7f9d` with `python -m scripts.build_native_trainer`. Its native tree matches that main source and includes [#247’s two-core traversals](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/247), which preserve the table snapshot and ordered delta merge; #247 records byte-identical HU100 20M checkpoints against the serial implementation. The [build receipt](hu100-roots-per-seat-artifacts/build-receipt.json) pins SHA256 `10a6fce897637cfd76a7dcc494469a5c39e3c65e10abeab00db850326e3b5fd7`. Game stacks are 100 BB, existing uncapped menu, linear weights, opponent-sampled average; only roots per seat differ. The entry limit is far above the run, and observed guards control resource stops.

| Recipe | Actual nodes | Iterations | Entries | Mean traverser visits/key | Zero-visit fraction |
|---|---:|---:|---:|---:|
| Published R=1 /v0.5.0 | 1,000,002,065 | 885,307 | 41,010,014 | 4.8470 | 38.70% |
| R=4 | 1,000,002,794 | 221,223 | 40,858,142 | 4.8602 | 38.59% |
| R=8 | 1,000,009,152 | 111,532 | 41,049,696 | 4.8402 | 38.64% |

| Recipe | Train excluding write (s) | Checkpoint write (s) | Process wall (s) | User /system CPU (s) | Mean training CPU cores* | Peak family RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| R=4 | 235.19 | 140.99 | 382.62 | 884.30 /233.50 | 4.12 | 3.634 |
| R=8 | 167.61 | 142.10 | 316.18 | 924.81 /159.16 | 5.56 | 3.653 |

*Mean native CPU core equivalents are observed five-second CPU deltas before checkpoint serialization, over the explicitly recorded window. `/usr/bin/time -l` gives actual process user/system/wall time and kernel RSS (bytes on macOS), including save, checkpoint hashing and diagnostics. Native telemetry train time excludes checkpoint write; its elapsed counter precedes the final hash/diagnostics, explaining the remaining wall-time difference. All original timing logs and samples are archived.

## Speedup

Matched cost-only 100M runs at R=1, 4 and 8 use the same current binary, seed, machine and one-save schedule. No poker payoffs were inspected. These measure root parallelism at 100M; they do not establish a matched 1B R=1 speedup.

| Roots | Train (s) | Save (s) | Process wall (s) | Training speedup over R=1 | Wall speedup | Mean CPU cores including save |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 37.88 | 41.09 | 80.55 | 1.000× | 1.000× | 1.12 |
| 4 | 19.70 | 41.04 | 62.35 | 1.923× | 1.292× | 2.09 |
| 8 | 14.16 | 41.61 | 57.37 | 2.676× | 1.404× | 2.22 |

Published R=1’s historical 1B run took 922.77 seconds including four saves, about 562.27 seconds after subtracting those writes. The raw historical/current R=4 and R=8 wall ratios are 2.412× and 2.919×; training-only ratios are 2.391× and 3.355×. Those comparisons combine roots with the older serial binary and save schedule, so they are not causal root-parallelism estimates. At 1B on the current binary, R=8 is 1.403× faster in training than R=4.

## Fixed direct matches

v0.5.0 was downloaded from its GitHub release and verified with its bundled `--require-publication` verifier and expected source `b9c9bd161422e87371a2641ae4c60304c66eed79`. All full input hashes are in [inputs](hu100-roots-per-seat-artifacts/inputs.json). Both new checkpoints were exported to current and average with `--zero-mass uniform`; the existing auditor verified every row against current regrets and accumulated average mass before play.

Fresh pilot root `20261013200101` and final root `20261013200102` were checked against tracked current main and every other current open PR head’s configs/reports/plans/scripts before outcomes. The distinct `r4-r1` and `r8-r1` scenario labels yield disjoint physical schedules, also disjoint from pilot deals. Each timing-only pilot has 128 duplicate blocks and full replay/reproduction without payoff inspection. [Canonical plan](hu100-roots-per-seat-artifacts/canonical-plan.json), posted and exactly read back at [this PR comment](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/252#issuecomment-6138185202), freezes **131,072 duplicate blocks /262,144 hands per contrast**, model pins, roots, translation off and analysis before final hands. Scientific source stays `f2115d717a96f5c876a5cb3b7c963d6094abd38c` through final verification.

All **524,288 final hands** independently replayed and deterministically reproduced before readout. No failed hand was rerun and no count was extended after outcomes. The unchanged HU100 direct harness uses 50/100-chip blinds, 100 BB resets, no rake, exact native menus and private action streams. Nominal paired duplicate-block Student-t 95% intervals condition on these fixed models and this one training seed, with no multiple-comparison adjustment.

Loss is the negative of direct gain; a negative upper loss bound supports a gain.

| Contrast | Gain BB/100 [nominal 95%] | Loss BB/100 [nominal 95%] | Upper bound on loss |
|---|---:|---:|---:|
| R=4 vs R=1 | -0.49 [-6.26, +5.29] | [-5.29, +6.26] | +6.26 |
| R=8 vs R=1 | -2.09 [-7.78, +3.61] | [-3.61, +7.78] | +7.78 |

**R=4:** There is no detectable loss at nominal 95%, with an upper bound of 6.26 BB/100 on loss; this is not equivalence or proof of learning at least as well. These results do not generalize over training seeds or certify performance against other opponents.

**R=8:** There is no detectable loss at nominal 95%, with an upper bound of 7.78 BB/100 on loss; this is not equivalence or proof of learning at least as well. These results do not generalize over training seeds or certify performance against other opponents.

## Safety and evidence

Before archival, 1,308 retained five-second readings show peak whole-family RSS 4.479 GiB, minimum free memory 77%, maximum swap growth 0.116 GB from B’s baseline, minimum free disk 23.85 GiB and pressure levels [1]. Every reading, CPU receipt and any stop/failure record is retained. Guards require independent continuous 60-second breaches for >13 GiB RSS, critical pressure, <8% free memory or >4 GB swap growth; warning is allowed. <16 GiB disk is immediate. psutil enumeration handles AccessDenied and signals only owned processes with checked creation/parent identity.

A’s currently merged, accepted archive permitted removal of its three unchanged local model copies before B training: 5,729,368,451 bytes reclaimed. Exact paths, size/mtime checks, live-dependency review and restoration members are in the [cleanup receipt](hu100-roots-per-seat-artifacts/a-cleanup-receipt.json) and [wording clarification](hu100-roots-per-seat-artifacts/a-cleanup-clarification.json). PR246 roots/uploads, synced files, shared Git and other agents’ dependencies remain untouched.

C’s unchanged local terminal-average copy was subsequently removed after current merged-owner/upload/stat/dependency checks, reclaiming 2,221,370,011 bytes. The [receipt](hu100-roots-per-seat-artifacts/c-average-cleanup-receipt.json) restores the exact 752.6M terminal from C’s accepted small averages ZIP; C tables remain.

The stage ZIP and embedded all-member manifest preserve native checkpoints/exports, verified release input, full timing/final traces/reproductions, replay audits, plan/comment readback and raw resource logs. Final archive/native/cloud acceptance and model restoration hashes/commands are indexed in [RESULTS_INDEX](../../RESULTS_INDEX.md). No research binaries enter Git. The final closeout resource summary includes archival work; local integrity and upload acceptance are distinct from remote byte verification.
