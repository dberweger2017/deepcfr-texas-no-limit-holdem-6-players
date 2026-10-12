# HU20 K50 5B first pass

**K50 stronger at 5B: +10.31 BB/100 [+7.66, +12.96].** Recommend continuing to the longer protocol. Primary and secondary results below use the frozen, equally weighted three-seed/two-seat duplicate-block schedule. All 393,216 final hands independently replay before numerical readout. This answers the equal-node first-pass question; the owner decides next.

[Protocol](../hu20-k50-5b-first-pass.md) · [canonical plan posted before final play](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/238#issuecomment-6125369610) · [exact plan](hu20-k50-5b-first-pass-artifacts/canonical-final-plan.json).

## Direct results

Each contrast contains 32,768 duplicate blocks per seed, three fixed seeds, two rotated seats, 196,608 hands. Both seats and the three seeds receive equal weight within each shared deal block. Intervals are nominal paired Student-t 95%, conditional on these trained policies; they do not include population uncertainty over future training seeds. Raw chips equal BB/100 because the big blind is 100 chips. Secondary and splits are descriptive; only the primary aggregate selects the predeclared recommendation.

### PRIMARY: K50 5B − matched-seed v1 5B

| Split | BB/100 | Nominal 95% interval |
| --- | ---: | --- |
| Aggregate | +10.31 | [+7.66, +12.96] |
| Seed 2026100601 | +11.10 | [+7.57, +14.62] |
| Seed 2026100602 | +9.40 | [+5.87, +12.92] |
| Seed 2026100603 | +10.45 | [+6.88, +14.01] |
| Seat big_blind | -3.54 | [-10.43, +3.36] |
| Seat button | +24.16 | [+17.28, +31.04] |
| Seed 2026100601 / big_blind | -1.25 | [-8.90, +6.41] |
| Seed 2026100601 / button | +23.44 | [+15.78, +31.09] |
| Seed 2026100602 / big_blind | -4.18 | [-11.82, +3.46] |
| Seed 2026100602 / button | +22.97 | [+15.31, +30.63] |
| Seed 2026100603 / big_blind | -5.18 | [-12.83, +2.46] |
| Seed 2026100603 / button | +26.08 | [+18.39, +33.76] |

### SECONDARY: K50 5B versus v0.4.2 O10B

| Split | BB/100 | Nominal 95% interval |
| --- | ---: | --- |
| Aggregate | +9.97 | [+7.09, +12.85] |
| Seed 2026100601 | +12.05 | [+8.52, +15.58] |
| Seed 2026100602 | +8.15 | [+4.63, +11.67] |
| Seed 2026100603 | +9.72 | [+6.18, +13.26] |
| Seat big_blind | -0.59 | [-7.83, +6.65] |
| Seat button | +20.54 | [+13.49, +27.58] |
| Seed 2026100601 / big_blind | +2.43 | [-5.34, +10.19] |
| Seed 2026100601 / button | +21.67 | [+14.01, +29.33] |
| Seed 2026100602 / big_blind | -2.44 | [-10.23, +5.34] |
| Seed 2026100602 / button | +18.75 | [+11.09, +26.41] |
| Seed 2026100603 / big_blind | -1.76 | [-9.50, +5.97] |
| Seed 2026100603 / button | +21.20 | [+13.58, +28.81] |

The primary rule was frozen before final hands: lower bound > 0 means stronger/recommend the longer protocol; upper bound < 0 means weaker/recommend stopping full-game K50; otherwise inconclusive/no extension. No LBR, pressure/scripted panel, seed selection, failed-hand rerun, outcome-driven extension, adoption or release.

## Exact whole-table training density

| Policy / seed | Actual nodes | Iterations | Keys | Mean traverser visits/key | Zero-visit fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 / 2026100601 | 5,000,000,043 | 9,971,555 | 6,315,718 | 153.89 | 17.5539% |
| v1 / 2026100602 | 5,000,000,208 | 9,809,166 | 6,414,988 | 151.50 | 17.4333% |
| v1 / 2026100603 | 5,000,000,544 | 10,008,052 | 6,374,215 | 152.30 | 17.5901% |
| k50 / 2026100601 | 5,000,000,783 | 10,536,954 | 8,303,507 | 117.12 | 16.2143% |
| k50 / 2026100602 | 5,000,000,182 | 10,472,679 | 8,290,014 | 117.43 | 16.1377% |
| k50 / 2026100603 | 5,000,000,137 | 10,444,021 | 8,308,763 | 117.04 | 16.4316% |

K50 has 1.29–1.31 times the matched v1 keys, below the anticipated ~1.4. Mean visits/key are ~117 versus 151–154 at the same requested 5B node budget. Training work was not adjusted. Complete-iteration overshoots are shown.

[All current/average full-row audits and model hashes](hu20-k50-5b-first-pass-artifacts/terminal-audits.json). Checkpoint keys are irreversible hashes without a street map. The owner accepted reached-key quantiles with disclosed coverage; no complete-table street distribution is inferred.

## Street coverage and reached-key visit quantiles

The following covers distinct stored keys reached in the **primary frozen direct matches**. Each stored key counts once per policy. The secondary coverage is retained separately in the full readout receipt. Missing reached keys are separate from stored zero-visit keys. Unclassified stored keys remain outside these street quantiles.

| Policy | Classified stored keys | Full-table coverage | Unclassified stored keys | Missing reached keys |
| --- | ---: | ---: | ---: | ---: |
| k50-2026100601-5b | 24,617 | 0.2965% | 8,278,890 | 0 |
| k50-2026100602-5b | 24,913 | 0.3005% | 8,265,101 | 0 |
| k50-2026100603-5b | 24,699 | 0.2973% | 8,284,064 | 0 |
| v1-2026100601-5b | 20,292 | 0.3213% | 6,295,426 | 0 |
| v1-2026100602-5b | 20,509 | 0.3197% | 6,394,479 | 0 |
| v1-2026100603-5b | 20,392 | 0.3199% | 6,353,823 | 1 |

### k50-2026100601-5b

| Street | Stored reached keys | Mean visits | Min | p10 | p25 | Median | p75 | p90 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| preflop | 1,831 | 15,587.77 | 23.00 | 674.00 | 1,722.50 | 6,708.00 | 19,922.00 | 41,775.00 | 95,610.10 | 95,974.00 |
| flop | 3,406 | 17,633.68 | 29.00 | 986.50 | 2,551.50 | 6,908.50 | 18,942.50 | 46,449.00 | 134,010.70 | 192,597.00 |
| turn | 8,494 | 9,973.27 | 9.00 | 549.00 | 1,604.25 | 4,435.50 | 11,183.75 | 25,689.30 | 84,877.82 | 142,250.00 |
| river | 10,886 | 6,584.57 | 2.00 | 353.50 | 1,042.00 | 3,214.00 | 8,587.50 | 17,493.00 | 41,006.65 | 69,679.00 |

### k50-2026100602-5b

| Street | Stored reached keys | Mean visits | Min | p10 | p25 | Median | p75 | p90 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| preflop | 1,831 | 15,590.61 | 42.00 | 634.00 | 1,911.00 | 6,808.00 | 20,078.00 | 41,647.00 | 95,025.70 | 95,477.00 |
| flop | 3,490 | 17,109.58 | 16.00 | 895.90 | 2,470.25 | 6,520.50 | 18,344.25 | 44,649.90 | 131,766.76 | 187,665.00 |
| turn | 8,589 | 9,818.29 | 2.00 | 535.00 | 1,559.00 | 4,347.00 | 10,784.00 | 25,445.40 | 85,586.84 | 137,143.00 |
| river | 11,003 | 6,560.98 | 0.00 | 343.00 | 1,026.00 | 3,224.00 | 8,479.00 | 17,292.80 | 40,817.70 | 68,973.00 |

### k50-2026100603-5b

| Street | Stored reached keys | Mean visits | Min | p10 | p25 | Median | p75 | p90 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| preflop | 1,782 | 15,798.99 | 44.00 | 764.30 | 2,080.25 | 6,767.00 | 19,634.25 | 40,834.80 | 94,787.66 | 95,449.00 |
| flop | 3,481 | 17,252.52 | 45.00 | 1,008.00 | 2,632.00 | 6,719.00 | 18,260.00 | 45,579.00 | 134,148.80 | 192,855.00 |
| turn | 8,530 | 9,917.09 | 13.00 | 593.60 | 1,617.00 | 4,446.50 | 11,232.50 | 24,929.20 | 84,711.08 | 139,090.00 |
| river | 10,906 | 6,518.32 | 1.00 | 342.00 | 1,053.00 | 3,134.50 | 8,464.75 | 17,374.00 | 40,227.55 | 73,913.00 |

### v1-2026100601-5b

| Street | Stored reached keys | Mean visits | Min | p10 | p25 | Median | p75 | p90 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| preflop | 1,837 | 14,257.58 | 30.00 | 603.20 | 1,770.00 | 5,998.00 | 17,572.00 | 39,996.20 | 90,444.64 | 90,874.00 |
| flop | 2,640 | 26,208.16 | 4.00 | 421.70 | 1,466.75 | 5,515.00 | 18,579.25 | 53,320.20 | 303,498.52 | 1,451,837.00 |
| turn | 7,936 | 11,821.88 | 4.00 | 329.00 | 1,065.50 | 3,576.00 | 10,852.50 | 28,148.00 | 131,877.90 | 510,154.00 |
| river | 7,879 | 13,417.67 | 1.00 | 526.80 | 1,628.00 | 4,908.00 | 14,102.50 | 32,830.40 | 133,095.54 | 394,937.00 |

### v1-2026100602-5b

| Street | Stored reached keys | Mean visits | Min | p10 | p25 | Median | p75 | p90 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| preflop | 1,793 | 14,295.19 | 58.00 | 734.20 | 1,913.00 | 5,957.00 | 17,307.00 | 38,420.80 | 89,002.08 | 89,449.00 |
| flop | 2,721 | 25,268.75 | 4.00 | 364.00 | 1,343.00 | 5,196.00 | 18,052.00 | 51,581.00 | 302,520.40 | 1,449,407.00 |
| turn | 7,967 | 11,569.08 | 2.00 | 307.20 | 999.50 | 3,421.00 | 10,395.00 | 27,419.80 | 132,817.10 | 503,173.00 |
| river | 8,028 | 13,136.00 | 0.00 | 515.00 | 1,563.75 | 4,870.00 | 13,564.50 | 32,420.00 | 130,048.09 | 423,427.00 |

### v1-2026100603-5b

| Street | Stored reached keys | Mean visits | Min | p10 | p25 | Median | p75 | p90 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| preflop | 1,787 | 14,541.98 | 5.00 | 737.40 | 2,005.50 | 5,950.00 | 17,352.50 | 39,811.00 | 90,833.96 | 91,422.00 |
| flop | 2,663 | 25,991.11 | 2.00 | 434.20 | 1,455.50 | 5,350.00 | 18,923.50 | 52,313.60 | 311,264.92 | 1,477,116.00 |
| turn | 7,911 | 11,841.89 | 2.00 | 336.00 | 1,080.50 | 3,653.00 | 11,146.00 | 28,002.00 | 132,523.70 | 512,052.00 |
| river | 8,031 | 13,360.41 | 0.00 | 571.00 | 1,666.00 | 4,958.00 | 14,040.00 | 32,463.00 | 129,453.10 | 427,715.00 |

## Correctness, cost and resource evidence

Current-main native source 045c793a4abd2e371d4d77de663fd3dd2f00316b is unchanged. Binary SHA256 aa7d3a5ef197f113c82a3e861abdc3081af9f5262f96f991e093d6d5a7a0add6; 14 native tests pass. Fresh v1 seed 602 at 1B reproduces archived bytes (a0ef695d6807aa4554ba01b32d27f64c39ccdc72d03be5a21c99cf67e52ad716), including max_entries 1B. Fresh K50 4M versus its own 2M resume has 785,675 exact rows, including IEEE signed zero and all three table hashes. Each of the six terminal checkpoint/current/average sets passes the full existing export auditor.

The cost-only 2048-block/seed/contrast pilot inspected costs and paired SD only. Primary SD 253.03 BB/100 projected half-width 2.7398 BB/100; total 14992.71 s including 25% play/replay headroom and 1,800 s archive/closeout. Both limits passed. Counts/roots/input hashes/decision were frozen on PR before any final hand. [Pilot quote](hu20-k50-5b-first-pass-artifacts/pilot-quote.json).

Independent replay rekeys each acting policy with its own schema/tables and own observation, checks menus, public history/legal bounds, probability normalization, selected actions, deal coordinates/seeds, public event digest and exact zero-sum settlement for every final hand. Independent math.fsum/Student-t arithmetic agrees with the existing production reporter for aggregate, seed and seat splits. Additional seed-by-seat estimates preserve aggregate weighting.

The original fixed six-hour budget started 10:20:26 UTC /12:20:26 Madrid and ends 16:20:26 UTC /18:20:26 Madrid. No reset. Continuous whole-family RSS < 7 GiB, normal pressure, free memory >= 15%, fixed-baseline swap growth <= 512 MiB, disk >= 16 GiB and AC remained binding. Actual closed-stage resource receipt follows archival closeout.

## Admission, provenance and preservation

PR236 merged 04:29:05 UTC; the 06:30 Madrid admission check found no campaign processes or denied metadata. The owner superseded the 02:02 ETA with 06:00 Madrid then 30-minute checks. Only this thread timer was modified. M4 is the scientific machine; M1 only coordinates.

Initial 21.87 GiB disk prompted a preventive 38 GiB forecast hold, not a guard failure. The owner authorized individually reserved stages while provisioning; guarded admission then had 40.50 GiB. No other PR root/process/schedule, synced deletion/offload or paid compute was used. [Initial storage formula](hu20-k50-5b-first-pass-artifacts/storage-admission.json).

Setup failures are retained: remote PATH, HTTPS keychain, quoting/missing test paths, unused wrapper indentation, canonical-post local time-module error (existing exact comment read back; no final hands started), and an older direct test double lacking the newly logged key contract. The corrected fixture preserves runtime bytes and 11 focused direct checks pass; two archive inventory/hash-refusal checks pass. These were source/setup checks, not failed scientific attempts or hand reruns.

Archive local/member/whole-hash verification, separate native and independent Drive acceptance, RESULTS_INDEX restoration records and final exact-head checks remain pending. Originals are preserved; the PR remains draft until closeout is complete.

### Measured guarded stages (before archive)

| Stage | Seconds | Peak family GiB |
| --- | ---: | ---: |
| build-release | 8.37 | 1.046 |
| cost-only-direct-pilot | 458.07 | 1.728 |
| final-direct-play | 967.68 | 2.054 |
| final-numerical-readout | 27.80 | 0.202 |
| freeze-canonical-final-plan | 1.94 | 0.111 |
| independent-final-replay | 339.07 | 1.198 |
| native-tests | 8.22 | 0.594 |
| recipe-and-recovery-gates | 496.02 | 1.404 |
| report-final-readout | 0.31 | 0.019 |
| restore-export-audit-controls | 848.35 | 0.115 |
| restore-reference | 47.86 | 0.064 |
| three-sequential-k50-lineages | 9112.42 | 2.085 |

Across 52,107 samples through reporting: peak family 2.085 GiB; minimum free memory 72%; maximum swap growth 59.82 MiB; minimum disk 32.65 GiB. All sampled pressure levels are normal and all samples are on AC. No campaign failure. Archive-stage receipts will supplement this record.

## Archive acceptance and supplemental closeout stop

[Main ZIP](https://drive.google.com/file/d/1FdnBZP2pRu-JB0JtHgtTuijNlXhxNmh-/view?usp=drivesdk) is accepted: 4,697,444,347 bytes, SHA256 2b7e5f8bf41ebdd8f8a9f8dcd5ed0d1d174e865ab62c49962bea4089a6241f27;245 locally size/hash-verified members, manifestSHA256 528f23792f27488882a37f76ff6f2ac453af2dcf69fafead3ae6b41b406b3fc0. Native uploaded/not-uploading/no-conflicts and independent cloud ID/name/size/parent agree. Remote bytes were not downloaded. [Acceptance](hu20-k50-5b-first-pass-artifacts/archive-acceptance.json), [model members](hu20-k50-5b-first-pass-artifacts/model-index.json) and RESULTS_INDEX retain full provenance.

A supplemental closeout worker failed before packaging with ModuleNotFoundError: scripts; its planning-directory launch omitted the repository import path. Main science and ZIP verification are unchanged. No sidecar was created, no scientific stage/hand retried, failure latch preserved. Bootstrap repair prepared, not run; the original no-retry rule requires an owner exception before archival repair. [Failure/log](hu20-k50-5b-first-pass-artifacts/closeout-startup-failure.json). Closed archive guard and later failure stay local. PR remains draft, no merge or cleanup.
