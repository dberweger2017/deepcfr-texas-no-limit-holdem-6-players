# HU20 v0.4.1 frozen arena

October 5, 2026. [PR #165](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/165).

The frozen O−R release rule is **not met**. All 165,888 scheduled hands independently replay and the raw-chip arithmetic matches the production reporter. Shipping v0.4.1 remains a separate owner decision.

## Design and integrity

The [owner-approved design](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/165#issuecomment-5994292073) and [freeze record](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/165#issuecomment-5995009340) are unchanged: four arms, three lineages, both positions and thirteen panels. R is the three B100M current-policy exports, including the v0.4.0 shipped seed-1 export; O is the new 1B opponent-sampled average, C the same O runs’ current policy, and T the corresponding 1B traverser-reach average. O−R is primary; C−R, O−C and O−T are exploratory.

Root `202610052001`; 2,048 paired blocks each for bounded LBR and native pressure, 256 for every other panel; 165,888 hands. The outcome-blind timing pilot uses the separate root `202610051901` and is excluded from poker estimates. Frozen source `376763af4f52a5829f7aac463750e3c501c20cff`. Canonical plan SHA256 `5d096197840097b93e56aca3ccbb480ac3335e9ef546c7faac358233ac18626f`; formatted-file SHA256 `3b98e9c467371b653a47d2ed4b7348525669a4ce0cc085a49d7779fe481ac053`. The plan starts at 13:03:14.568 UTC with an eight-hour deadline at 21:03:14.568 UTC.

The independent audit checks exact coordinate coverage, unique coordinates, model/source/plan identity, deal seeds, action legality, acting seat and observed street/cards/chips, target information keys, coverage counters, terminal stacks, zero-sum settlement and public-event hashes. It replays all native actions from each recorded deal without loading policies or calling the production reporter. Every recorded native-replay flag also passes.

At 100 chips/BB, mean `target_chips` per hand equals BB/100. For each panel/block/arm the arithmetic averages both positions and all three lineages; paired differences use those same blocks. Independent estimates use `math.fsum`, a separately computed sample variance and Student-t 95% intervals, and agree with every aggregate absolute estimate and all 52 panel contrasts within 1e−9. Rotations and lineages are not counted as independent deal samples. Intervals are unadjusted, conditional on these three lineages and declared opponents; they do not establish full-game exploitability or multiplayer strength.

## Predeclared release rule

The release rule is **not met** because the native-pressure lower bound is −18.47 BB/100, below the required −10. Native pressure is **inconclusive, not a measured regression**: its point estimate is approximately zero (+0.44) and its 95% interval [−18.47, +19.34] is too wide to establish the safeguard. Bounded LBR improves by +30.83 [14.03, 47.63] BB/100; the no-severe-scenario-regression check passes. The failed precision safeguard is retained exactly as predeclared; no follow-up arena is proposed or run.

| Check | Observed O−R (BB/100, 95% CI) | Result |
| --- | --- | --- |
| Bounded LBR lower bound > 0 | +30.83 [+14.03, +47.63] | True |
| Native-pressure lower bound > −10 | +0.44 [-18.47, +19.34] | False |
| Every other panel upper bound ≥ −20 | All eleven panels shown below | True |

## Every aggregate contrast

BB/100 with paired 95% Student-t intervals. All thirteen panels are retained.

| Panel | Blocks | O−R (primary) | C−R | O−C | O−T |
| --- | --- | --- | --- | --- | --- |
| uniform | 256 | -19.99 [-61.06, +21.09] | -33.63 [-75.91, +8.65] | +13.64 [-17.94, +45.22] | -6.41 [-22.60, +9.77] |
| passive | 256 | +28.97 [-6.12, +64.06] | +7.45 [-29.25, +44.16] | +21.52 [-7.80, +50.84] | -4.13 [-11.70, +3.43] |
| minraise-cap2 | 256 | +16.28 [-28.61, +61.16] | +45.21 [+0.74, +89.69] | -28.94 [-66.90, +9.02] | +9.80 [-8.60, +28.19] |
| pressure-cap2 | 256 | +29.82 [-7.24, +66.87] | +20.67 [-18.17, +59.51] | +9.15 [-20.98, +39.27] | +11.10 [-0.33, +22.53] |
| tight_passive | 256 | +3.84 [-4.02, +11.70] | +7.26 [-1.41, +15.93] | -3.42 [-12.64, +5.80] | +0.36 [-0.09, +0.81] |
| loose_passive | 256 | +13.48 [-8.73, +35.69] | +9.67 [-15.76, +35.09] | +3.81 [-14.25, +21.87] | +0.49 [-3.63, +4.61] |
| tight_aggressive | 256 | -2.08 [-14.36, +10.19] | -0.55 [-16.14, +15.03] | -1.53 [-13.32, +10.26] | -1.60 [-3.88, +0.69] |
| loose_aggressive | 256 | -2.08 [-29.81, +25.64] | -10.97 [-40.37, +18.43] | +8.89 [-10.81, +28.58] | +4.33 [-5.34, +14.00] |
| pot_pressure | 256 | +4.10 [-14.49, +22.69] | +22.62 [+1.27, +43.98] | -18.52 [-35.01, -2.03] | -0.16 [-1.40, +1.08] |
| train_pressure | 256 | -18.29 [-40.47, +3.88] | -12.92 [-37.10, +11.26] | -5.37 [-25.58, +14.84] | -0.81 [-5.18, +3.55] |
| native-pressure | 2048 | +0.44 [-18.47, +19.34] | +10.34 [-8.03, +28.70] | -9.90 [-24.95, +5.15] | +23.95 [+15.57, +32.32] |
| selective-stackoff | 256 | +8.59 [-2.90, +20.09] | +6.28 [-7.68, +20.25] | +2.31 [-10.21, +14.84] | +0.68 [-1.38, +2.75] |
| lbr | 2048 | +30.83 [+14.03, +47.63] | -13.61 [-30.21, +2.98] | +44.45 [+31.00, +57.89] | +5.18 [-2.28, +12.64] |

## Absolute returns

All arms against each panel, using the same three-lineage paired-block means. Both O and R still lose to bounded LBR: O −30.15 [−46.71, −13.59], R −60.98 [−76.00, −45.96] BB/100. The relative improvement does not establish positive profit against this probe.

| Panel | R | O | C | T |
| --- | --- | --- | --- | --- |
| uniform | +111.36 [+65.36, +157.36] | +91.37 [+43.92, +138.83] | +77.73 [+31.33, +124.14] | +97.79 [+48.84, +146.73] |
| passive | +91.63 [+54.28, +128.99] | +120.61 [+76.78, +164.44] | +99.09 [+52.44, +145.73] | +124.74 [+80.94, +168.54] |
| minraise-cap2 | +112.01 [+51.56, +172.47] | +128.29 [+67.04, +189.54] | +157.23 [+97.78, +216.67] | +118.49 [+57.98, +179.00] |
| pressure-cap2 | +56.67 [+12.89, +100.46] | +86.49 [+42.64, +130.34] | +77.34 [+36.20, +118.49] | +75.39 [+33.02, +117.76] |
| tight_passive | +47.85 [+38.18, +57.53] | +51.69 [+43.51, +59.88] | +55.11 [+44.57, +65.65] | +51.33 [+43.04, +59.63] |
| loose_passive | +14.97 [-13.89, +43.83] | +28.45 [-1.75, +58.65] | +24.64 [-6.66, +55.94] | +27.96 [-2.65, +58.58] |
| tight_aggressive | +50.20 [+36.96, +63.43] | +48.11 [+32.08, +64.15] | +49.64 [+33.67, +65.62] | +49.71 [+33.02, +66.39] |
| loose_aggressive | +37.04 [+2.37, +71.72] | +34.96 [-3.48, +73.40] | +26.07 [-10.92, +63.07] | +30.63 [-8.64, +69.90] |
| pot_pressure | +16.50 [-17.82, +50.82] | +20.61 [-13.72, +54.94] | +39.13 [+7.43, +70.82] | +20.77 [-13.56, +55.10] |
| train_pressure | +36.10 [+7.52, +64.68] | +17.81 [-10.78, +46.39] | +23.18 [-4.20, +50.55] | +18.62 [-9.19, +46.43] |
| native-pressure | +127.62 [+109.15, +146.08] | +128.05 [+107.41, +148.70] | +137.95 [+118.13, +157.77] | +104.11 [+83.70, +124.52] |
| selective-stackoff | +42.97 [+31.26, +54.68] | +51.56 [+39.47, +63.65] | +49.25 [+34.33, +64.17] | +50.88 [+38.60, +63.16] |
| lbr | -60.98 [-76.00, -45.96] | -30.15 [-46.71, -13.59] | -74.60 [-90.22, -58.98] | -35.33 [-52.04, -18.62] |

## Street coverage and bounded-LBR limits

Missing keys and stored zero-average-mass keys both play uniformly, and are counted separately. Current-policy stored keys have no average-mass classification. Counts below include all thirteen panels; the appendix reports every panel/arm/lineage/street cell, including explicit zero counts. LBR incomplete means a sampled comparison stopped at its declared soft budget; over-budget counts record completed batches crossing that soft limit. Neither is an action substitution.

| Arm | Street | Target decisions | Missing/uniform | Zero mass/uniform | Positive average | Stored current | LBR decisions | LBR incomplete | LBR over soft budget |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R | preflop | 51472 | 173 | 0 | 0 | 51299 | 13499 | 0 | 0 |
| R | flop | 28829 | 163 | 0 | 0 | 28666 | 7866 | 0 | 0 |
| R | turn | 15072 | 211 | 0 | 0 | 14861 | 3219 | 0 | 0 |
| R | river | 8456 | 175 | 0 | 0 | 8281 | 1239 | 0 | 0 |
| O | preflop | 51827 | 155 | 0 | 51672 | 0 | 13346 | 0 | 0 |
| O | flop | 31590 | 122 | 0 | 31468 | 0 | 8216 | 0 | 0 |
| O | turn | 17520 | 78 | 2 | 17440 | 0 | 3793 | 0 | 0 |
| O | river | 10598 | 56 | 6 | 10536 | 0 | 1918 | 0 | 0 |
| C | preflop | 52070 | 153 | 0 | 0 | 51917 | 13398 | 0 | 0 |
| C | flop | 32054 | 134 | 0 | 0 | 31920 | 7818 | 0 | 0 |
| C | turn | 18184 | 84 | 0 | 0 | 18100 | 3447 | 0 | 0 |
| C | river | 10701 | 57 | 0 | 0 | 10644 | 1499 | 0 | 0 |
| T | preflop | 51793 | 162 | 15 | 51616 | 0 | 13351 | 0 | 0 |
| T | flop | 31662 | 133 | 59 | 31470 | 0 | 8225 | 0 | 0 |
| T | turn | 17627 | 110 | 208 | 17309 | 0 | 3797 | 0 | 0 |
| T | river | 10618 | 80 | 379 | 10159 | 0 | 1876 | 0 | 0 |

## Resources and retained evidence

All twelve model result files are complete, with 13,824 hands each; zero failed, missing or duplicate scheduled coordinates. M4’s eight result/raw-hand files were copied into M1’s run folder and independently verified by size and SHA256; all twelve policy sizes and hashes pass on both hosts. M1 finished at 14:08:11 UTC and M4 at 13:45:31 UTC, both within the original cap. Originals remain on both hosts, including training checkpoints, all policy exports, pilot, timing, launch scripts and logs. The [independent audit](hu20-v041-arena-artifacts/audit.json), [M4 transfer proof](hu20-v041-arena-artifacts/m4-run-transfer.json), [archive receipt](hu20-v041-arena-artifacts/campaign-archive.json), [member manifest](hu20-v041-arena-artifacts/campaign-archive-manifest.json) and [validation receipt](hu20-v041-arena-artifacts/validation.json) preserve full precision, identities and reproducibility. Lossless archive restoration and upload state are indexed in [RESULTS_INDEX.md](../../RESULTS_INDEX.md). Both whole archives are confirmed uploaded with native uploaded/not-uploading status and exact cloud name/size/parent readback; the [upload receipt](hu20-v041-arena-artifacts/drive-upload-confirmed.json) preserves the proof. No evidence is deleted, no paid host is used, and this report does not tag or release a model.

| Model | Hands | Wall seconds | Load seconds | Peak GiB |
| --- | --- | --- | --- | --- |
| R-2026093001 | 13824 | 859.39 | 7.74 | 1.337 |
| R-2026093002 | 13824 | 831.22 | 5.99 | 1.622 |
| R-2026093003 | 13824 | 857.60 | 6.02 | 1.636 |
| O-2026100601 | 13824 | 1254.03 | 22.88 | 1.982 |
| O-2026100602 | 13824 | 1276.05 | 23.36 | 2.071 |
| O-2026100603 | 13824 | 1237.33 | 23.09 | 2.297 |
| C-2026100601 | 13824 | 967.00 | 27.26 | 2.525 |
| C-2026100602 | 13824 | 954.86 | 29.01 | 1.776 |
| C-2026100603 | 13824 | 874.50 | 28.89 | 1.777 |
| T-2026100601 | 13824 | 1257.41 | 15.77 | 2.616 |
| T-2026100602 | 13824 | 2075.37 | 34.63 | 1.614 |
| T-2026100603 | 13824 | 2017.18 | 34.28 | 1.675 |

Independent replay/audit: 850,106 actions checked in 95.90 seconds. [All lineage/position contrasts and panel street counts](hu20-v041-arena-details.md) accompany this report; the linked `audit.json` preserves full precision and raw member identities. All six focused arena/average-evaluation tests pass; the audit uses no production report aggregation or estimate helper.
