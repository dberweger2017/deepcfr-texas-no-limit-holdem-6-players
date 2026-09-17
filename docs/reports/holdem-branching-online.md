# Fresh six-player branching comparison

**All six jobs completed 512 iterations. Expanding a second own decision did not meet the predeclared consistent-improvement criterion.** Seed 733 improves relative to its control; seeds 719 and 727 remain inconclusive after the three-comparison adjustment. Both collectors still produce policies that lose heavily against the fixed style pool. First-decision collection remains the default; no model is promoted.

The [protocol](../holdem-branching-online.md) and [machine-readable results](holdem-branching-online.json) retain all seeds, checkpoints, paired deal-block differences, coverage and resource measurements. This is additional v0.5 research evidence, not a milestone-4 useful-learning pass or the v1.0 professional standard.

## What ran

Seeds 719, 727 and 733 each trained with first- and second-decision expansion from fresh uniform bootstrap. The only scientific setting changed between arms was the collector. Each job trained all six roles in six-player, 100 BB, no-ante/no-rake Hold’em using the existing finite legal betting menu and player-visible observations. Width 32, 32 collection roots per role, exploration 0.5, historical Q baselines, 4,096 replay records per role and 64 fresh Adam steps per role stayed fixed.

The executed checkout was `d19ddc14a7ddd7b1f4529f8208be362c622035b1`, with CPU PyTorch 2.5.1 and the pinned Rust engine. Checkpoints were saved every 64 iterations; snapshot-average exports and style/random evaluations ran at 128, 256, 384 and 512. Training started September 16 at 23:26:17 UTC and all final evaluations finished September 17 at 01:11:38 UTC: **105 minutes 21 seconds**. There were no training retries, omitted seeds or outcome-dependent recipe changes.

## Playing results

The campaign completed **147,456 scheduled table hands** across 48 suite evaluations. Every suite has 256 deal blocks, six seat rotations and two arena arms: trained average and a uniform-candidate control. Comparisons below pair the two trained policies on the shared schedule; uniform-control outcomes must reproduce exactly before comparison. Rotations, repeated checkpoints and shared seeds are not independent training runs.

### Primary endpoint: iteration 512 against styles

Values are BB/100, second-decision minus first-decision. Intervals have Bonferroni family coverage 95% across the three final seed comparisons.

| Seed | First decision | Second decision | Difference | Adjusted interval |
| --- | ---: | ---: | ---: | --- |
| 719 | -1,148.47 | -1,339.88 | -191.41 | [-515.04, 132.23] |
| 727 | -1,286.75 | -1,176.40 | 110.35 | [-185.87, 406.57] |
| 733 | -1,488.44 | -1,021.29 | 467.15 | [110.95, 823.36] |

The equally weighted mean seed gain is **+128.70 BB/100**, above the 25 BB/100 materiality threshold. However, only seed 733 has an adjusted interval entirely above zero; the required three-seed consistency condition fails. There is no confidence interval for a population of training seeds from these three runs. The shared arena deals also correlate the seed estimates.

The large relative gain for seed 733 is partly a deteriorating control: first-decision play falls from −939.68 at iteration 128 to −1,488.44 at 512, while second-decision play moves from −1,044.95 to −1,021.29. That is not evidence of a steadily strengthening second-decision policy. Absolute performance remains poor in every final style-pool evaluation.

### Final random-opponent results

These are descriptive, with pointwise paired 95% intervals; they are not alternative promotion endpoints.

| Seed | First decision | Second decision | Difference | Pointwise interval |
| --- | ---: | ---: | ---: | --- |
| 719 | -208.40 | -173.21 | 35.19 | [-396.01, 466.39] |
| 727 | -429.04 | 48.40 | 477.44 | [51.14, 903.74] |
| 733 | -271.32 | -156.54 | 114.78 | [-339.43, 568.99] |

Seed 727 has a positive relative interval against random opponents, but this does not establish broad strength. No stronger historical model, professional opponent or independent final-test pool was part of this experiment.

### Complete style-pool curves

All scheduled checkpoints are shown; random curves, absolute intervals and raw paired block differences are in the JSON report.

| Seed | Iteration | First decision | Second decision | Difference |
| --- | ---: | ---: | ---: | ---: |
| 719 | 128 | -1,213.44 | -1,276.01 | -62.57 |
| 719 | 256 | -1,163.61 | -1,275.65 | -112.04 |
| 719 | 384 | -1,266.86 | -1,535.71 | -268.85 |
| 719 | 512 | -1,148.47 | -1,339.88 | -191.41 |
| 727 | 128 | -1,149.28 | -1,354.17 | -204.88 |
| 727 | 256 | -1,013.77 | -1,067.61 | -53.84 |
| 727 | 384 | -1,218.52 | -1,130.24 | 88.28 |
| 727 | 512 | -1,286.75 | -1,176.40 | 110.35 |
| 733 | 128 | -939.68 | -1,044.95 | -105.27 |
| 733 | 256 | -1,014.65 | -926.53 | 88.12 |
| 733 | 384 | -1,184.57 | -1,027.47 | 157.10 |
| 733 | 512 | -1,488.44 | -1,021.29 | 467.15 |

## Coverage and costly behavior

Each job collected 98,304 scheduled roots. These counts describe the entire collection history, not the final retained replay distribution. A root can generate several correlated decision records.

| Collector / seed | Roots with postflop decisions | Preflop records | Flop | Turn | River |
| --- | ---: | ---: | ---: | ---: | ---: |
| first-719 | 3.31% | 303,132 | 5,633 | 574 | 126 |
| first-727 | 4.62% | 316,221 | 8,310 | 855 | 177 |
| first-733 | 4.57% | 318,869 | 8,032 | 714 | 89 |
| second-719 | 3.92% | 302,456 | 11,297 | 1,185 | 159 |
| second-727 | 5.39% | 305,167 | 16,443 | 3,000 | 731 |
| second-733 | 4.66% | 311,950 | 12,570 | 1,590 | 313 |

Additional branching increases postflop record counts, especially for seed 727, but only **3.31–5.39% of collection roots** contain any postflop decision. In the final style arena, the trained hero actually makes a postflop decision in roughly **20–25% of its 1,536 hands**. These distributions differ in opponents, policy averaging and exploration, so this is a coverage warning rather than an unbiased estimate of missing training mass.

Final style evaluations contain 708–795 preflop hero all-ins for the control and 809–828 for second-decision play, out of 1,536 hero hands per seed. Counts include all-in calls as well as raises; they are reconstructed from public chip payments, not from showdown boards. More branching did not remove the costly preflop behavior.

## Fitting and estimator tails

All **1,179,648 fitting steps** across six jobs clipped at norm 1. This repeats a known diagnostic; the earlier fitting ablation did not establish clipping as the primary cause of weak play.

| Collector / seed | Maximum inverse own sampling reach | Largest absolute regret update (BB) |
| --- | ---: | ---: |
| first-719 | 219,902.33 | 42,069,932.17 |
| first-727 | 802,816.00 | 199,521,101.53 |
| first-733 | 47,040.00 | 16,605,171.08 |
| second-719 | 10,623.36 | 708,449.05 |
| second-727 | 8,738,133.33 | 13,798,051,373.96 |
| second-733 | 5,076.20 | 546,995.29 |

The second-decision seed-727 run still produces a rare corrected target near **13.8 billion BB**. These are importance-corrected estimator values, not physical pot winnings. Lower conditional variance in the frozen probes does not guarantee smaller extremes throughout changing self-play. The measurements support retaining target quality and coverage as open problems; they do not isolate either as the sole cause.

## Measured cost

Host: Threadripper 7960X with 16 allocated vCPUs and a 32 GB cgroup memory limit. Six independent single-threaded workers shared the host. Stage durations below accumulate within each worker and overlap across workers; summing them does not give elapsed campaign time.

| Collector / seed | Training minutes | Collection seconds | Fitting seconds | Replay seconds | Checkpoint seconds | Evaluation seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| first-719 | 80.81 | 2,007.37 | 2,708.26 | 101.60 | 747.86 | 68.05 |
| first-727 | 84.43 | 2,179.34 | 2,747.22 | 105.71 | 725.99 | 75.21 |
| first-733 | 82.87 | 2,128.62 | 2,701.04 | 104.32 | 743.72 | 72.44 |
| second-719 | 88.06 | 2,247.42 | 2,888.61 | 108.62 | 699.06 | 73.85 |
| second-727 | 90.31 | 2,352.64 | 2,922.23 | 107.85 | 740.74 | 68.57 |
| second-733 | 89.63 | 2,343.78 | 2,887.92 | 110.92 | 707.49 | 63.65 |

Second-decision runs used 8.02% more summed training work. The learned policies and visited trees differ, so this is end-to-end cost rather than a fixed-workload speed comparison. Per-process peak RSS ranged from 5.11 to 5.34 GB. Total cgroup usage reached its 32 GB ceiling with reclaimable file cache, but there were no OOM kills or failed memory allocations; individual process high-water marks are not simultaneous usage.

The descriptive **1,800-second training-work ceiling** selects iteration 128 for both collectors in every seed. Differences are −62.57, −204.88 and −105.27 BB/100 for seeds 719, 727 and 733; all pointwise intervals include zero. This coarse checkpoint selection leaves unused time and excludes serialization, evaluation and setup. It is not an exact equal-cost experiment or a replacement for the primary endpoint.

The original host admission gate failed its conservative 90-minute projection: the slowest four-iteration pilot projected 107.25 minutes. Before any campaign seed, commit `2e9937f` revised only resource admission to 120 minutes plus a 30% growth allowance and 45 minutes for remaining work, within the unchanged four-hour/$3.50 cap. All six pilots completed, peak early cgroup memory was 4.88 GB of 32 GB, and 39 remote correctness tests passed. The failed original gate remains in the operations record.

## Recovery, integrity and artifact access

All six completed checkpoints were resumed in fresh processes on the original environment. **36 file comparisons matched byte for byte**, covering each final training checkpoint, average export, style/random evaluation and style/random raw outcomes. No new training iterations ran. Sequential verification took **1,401.05 seconds**. Identical duplicate recovery model files were omitted from the archive; every original model remains.

Local streaming verification checked **528 inventoried files and all 90 catalog artifacts**, including the campaign’s **48 training checkpoints and 24 average exports** and 18 host-pilot artifacts. All matched. The archive has 597 regular members; its remaining entries are operations/source records. Analysis rerun from the verified local outcomes agrees with the remote analysis: all nonfloating fields match, and six floating-point interval endpoints differ by at most 1.71e-13 BB/100 across platforms. This is distinct from the byte-identical recovery check on the original training environment.

The [integrity record](holdem-branching-online-integrity.json) contains model hashes, manifests, recovery comparisons, calibration/amendment records, memory checks and rental accounting. CI passed **637 tests** plus the repository’s end-to-end checks; the six focused analysis tests also pass locally.

The owner retains `results/branching-comparison.tar.gz`: **14,041,643,443 bytes**, containing all runs, original models, outcomes, TensorBoard events, recovery records, source and dependencies. It is a local archive, not a public download. A fresh clone needs an owner-provided copy; public artifact hosting remains release work. SHA-256:

```text
06dbb252d64aefeecba1f586b23a10faa262755eb8bf0ea60207078d57cf23a9
```

Extract the archive on a machine with at least 31 GB available for its contents. To reproduce the analysis from an extracted directory:

```bash
python -m scripts.report_holdem_branching \
  --results /path/to/extracted/results \
  --out /path/to/comparison.json
```

Full model recovery uses the recorded execution revision and dependencies. `branching-archive-operations/source.tar.gz`, `requirements-frozen.txt` and the manifests retain that provenance. The owner’s compact local extraction contains JSON and monitoring records; model bytes remain in the verified archive to avoid keeping a second large copy on the Mac.

The archive was ready at 01:42:25 UTC. Transfer ran from 01:43:00 to 02:10:08; local verification took 75.27 seconds. The provider accepted pod termination at **02:11:25 UTC**, after verification. The separate network volume was subsequently deleted, confirmed by **02:27:59 UTC**. No billable storage from this campaign remains. No local caches or previous experiment files needed deletion.

At the quoted CPU/container/network rates, the total is approximately **$1.77**; reserve **$1.80** conservatively rather than presenting a final invoice. Conservative cumulative CPU spending is **$5.67**, leaving **$4.33** of the original $10 authorization. No GPU budget was used.

The initial remote test command named a nonexistent test file and ran zero tests; the corrected invocation passed all 39. Both logs are retained. There were no failed training arms, numerical failures, invalid-play substitutions or recovery mismatches. The analysis has tests for block-level inference, altered controls, missing rotations, all-in calls and hero identity.

## Decision and next work

Keep first-decision collection as the default and second-decision collection available for controlled experiments. The frozen variance benefit is real in its measured setting, but this online comparison does not justify a default change, a stronger-model claim or another larger run of the same recipe.

The next PR should test **representation and generalization on reliable poker targets**: broaden the existing exact six-player river contexts across held-out boards and hole-card combinations, then compare the current encoder with a separate card branch and well-scaled numerical context. Freeze context-level splits and decision-cost measurements before fitting; hold sampling and fitting settings fixed. Start locally with a bounded run and retain both successes and failures.

This addresses the independently observed clean-target generalization failure without asking another long self-play run to diagnose itself. Coverage and extreme sampled targets remain separate unresolved problems. Any later bootstrap, opponent-distribution or replay intervention needs explicit sampling/weighting semantics and its own comparison. Further paid training should follow a measured mechanism improvement; none is started by this report.
