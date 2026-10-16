# Frozen HU200 three-seed blend: fixed M4 readout

The full owner-authorized continuation of #297 completes on the base Apple M4
with 16 GiB physical RAM. All three original averages freshly reproduce their
pinned SHA256s before the single raw blend is built and every row audited.
All fixed final counts run; every stored hand independently replays and
reproduces, and all repeated blend fingerprints match before readout.

**No direct gain is established:** the blend-minus-seed estimates are
+3.61, +0.37 and +1.85 BB/100, with all three nominal 95% intervals crossing zero.
All 21 paired scripted difference intervals also include zero, both nominally
and with the descriptive family adjustment. The blend still loses against
1.5× pot pressure: **−29.39 [−55.24, −3.53] BB/100**. That adverse finding is
recorded as failed research evidence under the owner's instruction; it is kept
without rerun, tuning, extra hands or an added pass/fail gate.

**Preservation is in progress.** The PR remains draft while the main and
administrative archives await full readback/upload acceptance and the requested
unpacked 2 TB copies. The already accepted seed 1421 input bundle has its
verified unpacked copy. The owner decides merge and any future Slumbot use.

## Frozen study and verification

[Protocol](../hu200-v055-blend-protocol.md), [original #297 report](hu200-v055-blend.md).
Per union key/action, the accumulator is `fsum(raw Node.average_s / W_s)` over
present seeds, with absent keys contributing zero. Regrets and visits are zero;
the existing frozen exporter supplies uniform zero-mass play. W is the pinned
per-seat scheduled `T*(T+1)/2`, with no root-mass reweighting.

All construction, export, cache, play, replay, reproduction and statistical
readout run on M4. M1 performs Git, report and archival administration; bb128
is used only for the owner-requested storage copy. M4 scientific source remains
`13ccb498482c228776128e70ad9e733fb0f5621a`; no later main changes enter execution.
Both exclusive research locks and the unchanged five-second guards apply:
13 GiB family RSS, 8% free memory, critical pressure and 4 GB swap growth
sustained for 60 observed seconds, and immediate 16 GiB disk floor. Admission
also retains the explicit 2 GiB margin.

Final direct counts are **131,072 duplicate blocks per seed**. Each of seven
scripts has **16,384 paired blocks**, with all four models on common deals and
logical-role streams. Translation is **512 states /128 events on all models**.
Disjoint cost pilots use 128 direct and 32 scripted blocks per seed. Fixed roots
are reserved against main and every other open head before pilots/finals.
[Final plan](hu200-v055-blend-run-artifacts/final-plan.json),
[fresh root reservation](hu200-v055-blend-run-artifacts/root-reservation-authorized-finals.json).

The global barrier verifies **2,162,688 stored primary records**, representing
**1,703,936 unique primary hands**. Every stored hand independently replays and
completely reproduces; the repeated blend hand and deterministic decision
fingerprints match across all three pairs for each script. Repeated blend arms
add no independent samples. Verification returns 0 before any readout.
[Global barrier](hu200-v055-blend-run-artifacts/all-final-verified.json),
[verified M4 readout](hu200-v055-blend-run-artifacts/summary.json).

Intervals are paired Student-t nominal 95% intervals over duplicate blocks.
The 21-contrast Bonferroni scripted adjustment is descriptive. These intervals
refer to the selected fixed models, not the training-seed population. No
Slumbot, server, release, tag or default change is made.

## Direct results

| Opposing seed | Blend minus seed, BB/100 [nominal 95% CI] |
|---|---:|
| 2026101302 | +3.61 [-1.15, +8.37] |
| 2026101421 | +0.37 [-4.30, +5.05] |
| 2026101422 | +1.85 [-2.83, +6.52] |

## Scripted panel

Each cell is BB/100 versus the script, with its nominal 95% interval.

| Script | Blend | Seed 1302 | Seed 1421 | Seed 1422 |
|---|---:|---:|---:|---:|
| random | +186.17 [+111.34, +261.01] | +170.52 [+95.63, +245.41] | +173.04 [+97.88, +248.20] | +184.44 [+109.68, +259.19] |
| check_call | +160.50 [+148.05, +172.96] | +162.98 [+150.38, +175.58] | +165.85 [+153.44, +178.26] | +162.82 [+150.33, +175.31] |
| tight_aggressive | +68.10 [+56.60, +79.60] | +68.12 [+56.95, +79.29] | +72.35 [+60.22, +84.49] | +65.65 [+54.66, +76.64] |
| loose_aggressive | +120.78 [+97.44, +144.12] | +111.76 [+88.71, +134.81] | +113.15 [+89.93, +136.37] | +112.52 [+89.52, +135.53] |
| pot_pressure | -29.39 [-55.24, -3.53] | -42.96 [-69.24, -16.68] | -39.38 [-64.96, -13.80] | -37.72 [-63.81, -11.64] |
| half_pot_pressure | +62.89 [+47.24, +78.53] | +68.01 [+52.68, +83.34] | +62.15 [+46.90, +77.41] | +62.18 [+47.01, +77.35] |
| matched_pot_pressure | +64.95 [+49.60, +80.30] | +66.35 [+51.16, +81.54] | +65.10 [+50.10, +80.10] | +66.42 [+50.23, +82.60] |

## Paired scripted differences

Blend minus seed. The 21-contrast Bonferroni intervals are descriptive; there is no pass/fail gate.

| Script | Seed | Nominal 95% CI with mean | Descriptive 21-contrast interval |
|---|---|---:|---:|
| random | 2026101302 | +15.65 [-3.12, +34.42] | [-13.44, +44.75] |
| random | 2026101421 | +13.13 [-6.15, +32.41] | [-16.76, +43.02] |
| random | 2026101422 | +1.73 [-18.47, +21.94] | [-29.59, +33.06] |
| check_call | 2026101302 | -2.48 [-9.96, +5.01] | [-14.08, +9.12] |
| check_call | 2026101421 | -5.35 [-13.25, +2.56] | [-17.59, +6.90] |
| check_call | 2026101422 | -2.32 [-10.17, +5.53] | [-14.49, +9.85] |
| tight_aggressive | 2026101302 | -0.02 [-8.32, +8.27] | [-12.89, +12.84] |
| tight_aggressive | 2026101421 | -4.26 [-12.55, +4.04] | [-17.12, +8.60] |
| tight_aggressive | 2026101422 | +2.45 [-4.94, +9.84] | [-9.00, +13.90] |
| loose_aggressive | 2026101302 | +9.02 [-7.14, +25.17] | [-16.02, +34.06] |
| loose_aggressive | 2026101421 | +7.63 [-8.92, +24.18] | [-18.03, +33.29] |
| loose_aggressive | 2026101422 | +8.26 [-9.17, +25.69] | [-18.76, +35.28] |
| pot_pressure | 2026101302 | +13.57 [-6.58, +33.72] | [-17.67, +44.81] |
| pot_pressure | 2026101421 | +9.99 [-11.25, +31.24] | [-22.94, +42.93] |
| pot_pressure | 2026101422 | +8.34 [-12.18, +28.85] | [-23.47, +40.14] |
| half_pot_pressure | 2026101302 | -5.12 [-17.23, +6.99] | [-23.89, +13.65] |
| half_pot_pressure | 2026101421 | +0.74 [-11.72, +13.19] | [-18.58, +20.05] |
| half_pot_pressure | 2026101422 | +0.71 [-11.65, +13.06] | [-18.45, +19.86] |
| matched_pot_pressure | 2026101302 | -1.40 [-13.03, +10.24] | [-19.43, +16.64] |
| matched_pot_pressure | 2026101421 | -0.15 [-10.96, +10.66] | [-16.91, +16.61] |
| matched_pot_pressure | 2026101422 | -1.46 [-13.95, +11.02] | [-20.82, +17.89] |

## Union, zero mass and root diagnostics

The union contains **156,208,331 keys /482,975,338 action slots**. Keys present
in exactly one, two and three seeds number **88,880,705 /34,741,832 /32,585,794**.
Each single seed contains about 85.374 million keys. All seven nonempty
membership-mask counts, per-menu counts and root-class masses are retained in
the [full union scan](hu200-v055-blend-run-artifacts/union-scan.json).

The unchanged every-row auditor verifies all 156,208,331 rows, including
**124,272,760 positive-mass** and **31,935,571 zero-mass uniform** rows.
The raw synthetic checkpoint is 6,042,193,732 bytes; its average inference
export is 6,282,968,150 bytes. Synthetic identity is
`hu200-v055-sequence-blend-2026101529701`, with format-required iteration
marker 1 and no training or resume claim. The 2,960,585,129-byte current export
is retained for the auditor, not selected as another candidate.
[Full model hashes](hu200-v055-blend-run-artifacts/blend-model-manifest.json),
[every-row audit](hu200-v055-blend-run-artifacts/blend-every-row-audit.json).

| Seed | T | Scheduled W | Root mass | Mass − W | Missing /classes |
|---|---:|---:|---:|---:|---:|
| 2026101302 | 16,670,844 | 138,958,528,171,590 | 138958528171589.98 | -0.015625 | 0 /169 |
| 2026101421 | 16,745,107 | 140,199,312,593,278 | 140199312593278.0 | +0.000000 | 0 /169 |
| 2026101422 | 16,827,728 | 141,586,223,234,856 | 141586223234856.06 | +0.062500 | 0 /169 |

Root masses are diagnostic floating-point totals over all 169 small-blind
preflop classes; the displayed differences never change the pinned W values.
The three original average hashes reproduce exactly from their pinned raw
checkpoints before blending: [1302](hu200-v055-blend-run-artifacts/seed1302-input-verified.json),
[1421](hu200-v055-blend-run-artifacts/local-average-reproduced.json),
[1422](hu200-v055-blend-run-artifacts/seed1422-input-verified.json).

## Played coverage

Exact absence is measured before translation; unresolved absence means no
selected key after translation. Zero-mass known rows are separate and retain
uniform play. The [136-row coverage table](hu200-v055-blend-run-artifacts/played-coverage.csv)
contains counts and both shares for every direct model/street and every
script/model/street, with each identical repeated blend panel arm counted once.
[Raw panel coverage summary](hu200-v055-blend-run-artifacts/panel-played-coverage.json).

| Direct opponent | Model | Street | Decisions | Exact absent % | Unresolved % |
|---|---|---|---:|---:|---:|
| 2026101302 | 2026101302 | flop | 213,536 | 0.000 | 0.000 |
| 2026101302 | 2026101302 | preflop | 283,553 | 0.000 | 0.000 |
| 2026101302 | 2026101302 | river | 100,141 | 0.060 | 0.060 |
| 2026101302 | 2026101302 | turn | 141,393 | 0.016 | 0.016 |
| 2026101302 | blend | flop | 214,943 | 0.000 | 0.000 |
| 2026101302 | blend | preflop | 284,300 | 0.000 | 0.000 |
| 2026101302 | blend | river | 100,226 | 0.016 | 0.016 |
| 2026101302 | blend | turn | 141,373 | 0.001 | 0.001 |
| 2026101421 | 2026101421 | flop | 216,041 | 0.000 | 0.000 |
| 2026101421 | 2026101421 | preflop | 283,363 | 0.000 | 0.000 |
| 2026101421 | 2026101421 | river | 102,247 | 0.054 | 0.054 |
| 2026101421 | 2026101421 | turn | 143,612 | 0.018 | 0.018 |
| 2026101421 | blend | flop | 215,258 | 0.000 | 0.000 |
| 2026101421 | blend | preflop | 283,479 | 0.000 | 0.000 |
| 2026101421 | blend | river | 102,190 | 0.010 | 0.010 |
| 2026101421 | blend | turn | 143,606 | 0.003 | 0.003 |
| 2026101422 | 2026101422 | flop | 214,872 | 0.000 | 0.000 |
| 2026101422 | 2026101422 | preflop | 283,307 | 0.000 | 0.000 |
| 2026101422 | 2026101422 | river | 101,908 | 0.049 | 0.049 |
| 2026101422 | 2026101422 | turn | 143,417 | 0.017 | 0.017 |
| 2026101422 | blend | flop | 214,240 | 0.000 | 0.000 |
| 2026101422 | blend | preflop | 282,928 | 0.000 | 0.000 |
| 2026101422 | blend | river | 102,212 | 0.010 | 0.010 |
| 2026101422 | blend | turn | 143,129 | 0.001 | 0.001 |

## Resources, timing and failures

Hardware is Apple M4 Mac16,1, 17,179,869,184 physical memory bytes (16 GiB),
10 logical CPUs, macOS 26.6. Through readout, 6,243 guard observations measure
a **12,748,324,864-byte peak simultaneous process-family RSS (12.75 GB)**.
The kernel's largest per-command process maximum is **12,734,185,472 bytes**;
these separate command maxima are not summed. Minimum observed free memory
is 67%; minimum free disk is 47,236,538,368 bytes. Maximum swap growth from
the campaign baseline is 1,213,139,517 bytes. No observed resource threshold
breach or resource stop occurs. Closed preservation-job resources will be
added at archival closeout.
[Hardware](hu200-v055-blend-run-artifacts/machine.json),
[sampled families](hu200-v055-blend-run-artifacts/resource-summary.json),
[kernel maxima](hu200-v055-blend-run-artifacts/kernel-rss-summary.json).

| Work | Observed wall time |
|---|---:|
| Full raw merge | 28m53s |
| Frozen current/average export | 28m44s |
| Full every-row audit | 49m12s |
| Final group 1302, including replay/reproduction | 40m24s |
| Final group 1421, including replay/reproduction | 41m20s |
| Final group 1422, including replay/reproduction | 41m10s |
| Global verification | 75.47s |
| Readout and full panel coverage | 70.44s |

The frozen cost-only formula quotes 10h16m remaining science with its required
1.5 multiplier and reserves. It fits disk but exceeds the initial science
deadline. Before any final hand or strength readout, the owner explicitly
authorizes a prospective extension to **02:30 science /04:30 preservation,
Madrid October 17**, preserving the two-hour closeout reserve. Original clock
and time-stop quote remain evidence; guard baseline/held-state hashes are
unchanged by the extension. Actual finals complete substantially faster than
the conservative pilot projection; no formula, count or scientific rule is
changed. [Authorization](hu200-v055-blend-run-artifacts/owner-budget-extension.json),
[original quote](hu200-v055-blend-run-artifacts/final-cost-admission.json),
[admitted same-cost quote](hu200-v055-blend-run-artifacts/final-cost-admission-owner-extended.json).

All unfavorable findings remain in the tables. The losing 1.5× pot-pressure
result is recorded as failed research evidence; direct gains remain unestablished.
No original-average mismatch, row-audit failure, hand-replay/reproduction
mismatch or global-fingerprint failure occurs.

Plumbing history is retained: #303's missing 19.408 GB of inputs and provisional
disk shortage; the old partial seed 1421 full-cloud ZIP, resolved for this run
by a new accepted checkpoint-only archive; the missing manifest control field
after a correct whole-ZIP hash; the refused recovery launch and preserved STOP;
and the prospective time-budget hold. Original baseline/held state, failures
and source versions remain. Administrative Python-version, fetch-ref and tool
syntax errors are also retained. [Complete failure ledger](hu200-v055-blend-run-artifacts/failures.json),
[dated execution history](hu200-v055-blend-run-artifacts/execution-history.md).

Only inactive derived caches retire, after full comparison verification, fresh
owning-PR status, canonical-source pins, exact inventory/hashes and no handles.
The final blend cache additionally retires after global verification and readout.
Every research original and active dependency remains. Exact cache rebuild
commands and retirement receipts are retained.

## Evidence and preservation

Scientific outputs are retained on M4 under
`~/Local/hu200-v055-blend-run/results/hu200-v055-blend/`. The designated
[Drive folder](https://drive.google.com/drive/folders/1FZiDigTX-_Do4pPihTF4pdZTDaxNRxPK)
is `~/Local/Research-Cloud/PR-306-HU200-v055-blend-run/`. The final main bundle
and administrative supplement are pending creation/readback/upload acceptance.
The owner-requested unpacked copies will be retained under
`bb128:/mnt/data2tb/research/PR-306-HU200-v055-blend-run/`, with matching manifests,
every-member size/SHA256 verification and independent copied-file readback.
No final mirror or upload acceptance is claimed prematurely.

The separately accepted seed 1421 checkpoint-only input bundle already has its
verified unpacked copy. [Current archive/retrieval index](../../RESULTS_INDEX.md#pr306--frozen-hu200-three-seed-blend-run).
