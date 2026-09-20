# Daytime replay and paper-architecture comparison: final report

September 19–20, 2026. **Complete. No model is promoted and no production
default changes.** Every arm trained its full budget; on the predeclared fresh
confirmation suite the three arms are statistically indistinguishable.

This report closes the batch described by the [protocol](../holdem-day-paper.md)
and launched in [PR #89](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/89).
It supersedes the two interim reports ([first](holdem-day-paper-interim.md),
[second](holdem-day-paper-interim-2.md)), which remain the record of what was
believed at 15:22 and 18:50 UTC. The machine-readable record is
[holdem-day-paper.json](holdem-day-paper.json).

## 1. Headline

Three fresh six-player, 100 BB, no-rake Deep CFR arms ran 1,024 iterations each
from the same seed, differing only in replay capacity and architecture:

| Arm | Architecture | Replay per role |
| --- | --- | ---: |
| `current-4k` | Existing width-32 GRU with action-conditioned regret/Q heads | 4,096 |
| `current-16k` | Identical model | 16,384 |
| `paper-16k` | Paper-inspired encoder package: card embeddings, residual card tower, width-64 history GRU, LayerNorm trunk (172,866 parameters per role) | 16,384 |

**On the predeclared fresh 2,048-block suite, all three arms are statistically
indistinguishable from one another.** Both primary comparisons are inconclusive:

| Comparison | Difference | Bonferroni 97.5% | Verdict |
| --- | ---: | --- | --- |
| `current-16k − current-4k` (replay capacity) | −4.20 | [−142.36, +133.96] | inconclusive |
| `paper-16k − current-16k` (architecture package) | +12.86 | [−117.00, +142.71] | inconclusive |
| `paper-16k − current-4k` | +8.65 | [−120.07, +137.38] | inconclusive |

All three arms do beat the uniform-candidate control by roughly +230 BB/100 on
fresh deals, with intervals excluding zero, and all three still lose roughly
950 BB/100 to the scripted style pool.

A single significant replay advantage appeared at iteration 256 on the reused
validation schedule (+197.57), decayed to +30.25 at 512 and +24.51 at 768, and is
−4.20 on the fresh suite. The batch therefore provides no evidence that
quadrupling replay capacity, or the paper-inspired architecture package, improves
playing strength at this scale — and it does provide evidence about how easily a
reused-deal intermediate checkpoint can look like a finding.

## 2. What ran, and how it is measured

- Launched 2026-09-19T11:25:18Z on the M4 from revision `2c23594`, fresh training
  seed **2026091902**, uniform bootstrap, first-decision sampling, exploration
  0.5, 128 traversal roots per role, 256 fresh fitting steps per role at batch
  32, Adam 0.001, norm-1 clipping. Each arm stopped at iteration 1,024.
- Validation evaluations at iterations 64, 128, 256, 512, 768 and 1,024 use 1,024
  deal blocks with six seat rotations per block; the random-opponent benchmark
  rides along at 256, 512 and 1,024.
- The predeclared decision point is one fresh 2,048-block scripted evaluation per
  arm at iteration 1,024, root seed `2026091920`, in a separate `final`
  directory, never run at an interim checkpoint.
- Every evaluation plays 12,288 hands (styles and random) or 24,576 hands (final):
  half with the arm's snapshot-average policy in the candidate seat, half with a
  uniform-random control in the same seat. Intervals cluster the six rotations of
  a deal block together, because rotations of one deal are not independent.
- **Provenance checks.** The schedule digest is identical across all three arms
  at every boundary (64 → 1,024), the uniform-control outcome digest is identical
  at every boundary of every arm, every report is `valid`, and **no evaluation
  recorded a single invalid action**.
- The two predeclared comparisons use Bonferroni-adjusted 97.5% individual
  intervals, as declared before the run.

## 3. A resource failure, and its recovery

The batch did not finish cleanly. At 22:29:45Z the `paper-16k` worker was killed
by its supervisor with `RuntimeError('disk_limit')` while running the
iteration-1,024 evaluation. The supervisor samples machine-wide free space every
five seconds and aborts when free space falls below 12 GiB; the last recorded
poll before the abort showed 12.68 GiB free with 11.19 GiB retained by that arm
alone.

The cause was a transient peak, not a shortage: at iteration 1,024 the arm
published `training-1024.pt` (4.13 GiB) and, inside the evaluation step,
`average-1024.pt` (4.01 GiB), while its previous export `average-768.pt`
(3.0 GiB) was still retained — retirement runs only after the evaluation
returns, and the worker was terminated first. With `keep=1`, publication
therefore still holds two generations of both artifacts simultaneously, and the
guard has no allowance for that headroom.

**What was lost:** the iteration-1,024 validation evaluation and the fresh
2,048-block final suite for that arm. **What was not lost:** all 1,024 training
iterations (1,024 timing records and 1,024 iteration reports), and both the
checkpoint and the frozen export at the declared analysis point.

**Recovery.** The missing measurements were reconstructed without any training.
The published checkpoint was reloaded and its SHA-256 verified against its
marker (`e9d255c8…`), the frozen export `average-1024.pt` was loaded and
confirmed to play a legal, normalized distribution, and the runner's own
completion path was then executed: the iteration-1,024 validation boundary and
the fresh 2,048-block final suite, both reusing the frozen export
(`reuse_export=True`) with the declared seeds and block counts. No checkpoint was
selected, no training resumed, and the source revision is unchanged. The
recovery record is `completion.json` in the run directory.

Afterwards the run's own retention was applied to the recovered directory:
the stale export was retired (logged in `retention.jsonl`) and the raw outcomes
were compressed losslessly (195 MB → 13.1 MB in `final/`, 98 MB → 6.6 MB and
95 MB → 6.4 MB at the boundary). The arm's directory is now 8.2 GiB and the host
has ~32 GiB free.

**Consequence for interpretation:** the `paper-16k` final numbers come from a
post-hoc evaluation of its frozen export rather than from its live worker. The
schedule, seeds, code revision and evaluation procedure are the declared ones,
and the two other arms were unaffected — but this is a reconstruction and is
labelled as such wherever those numbers appear. Its `status.json` deliberately
still reads `failed`; the failure record was not rewritten.

## 4. Scripted-pool results

Validation schedule, 1,024 reused deal blocks (monitoring, not confirmation):

| Iteration | `current-4k` | `current-16k` | `paper-16k` | Uniform control |
| ---: | ---: | ---: | ---: | ---: |
| 64 | −1,189.19 | −1,236.44 | −971.72 | −1,172.68 |
| 128 | −1,134.69 | −1,114.09 | −950.83 | −1,172.68 |
| 256 | −1,148.64 | −951.07 | −952.63 | −1,172.68 |
| 512 | −990.18 | −959.93 | −903.13 | −1,172.68 |
| 768 | −982.51 | −958.00 | −902.52 | −1,172.68 |
| 1,024 | −996.25 | −1,031.85 | −947.60 | −1,172.68 |

Predeclared paired comparisons on that reused schedule, Bonferroni 97.5%:

| Iteration | `current-16k − current-4k` | `paper-16k − current-16k` |
| ---: | --- | --- |
| 64 | −47.25 [−202.16, +107.66] | **+264.72 [+96.26, +433.18]** |
| 128 | +20.61 [−143.75, +184.96] | +163.26 [−16.23, +342.75] |
| 256 | **+197.57 [+41.57, +353.56]** | −1.55 [−162.19, +159.08] |
| 512 | +30.25 [−156.69, +217.19] | +56.80 [−127.89, +241.49] |
| 768 | +24.51 [−166.79, +215.81] | not evaluated |
| 1,024 | −35.60 [−231.72, +160.51] | +84.25 [−95.03, +263.53] |

Fresh 2,048-block confirmation suite at iteration 1,024 — the predeclared
decision point:

| Arm | BB/100 | Nominal 95% | vs uniform control |
| --- | ---: | --- | --- |
| `paper-16k` | −945.13 | [−1,052.34, −837.93] | +239.29 [+125.82, +352.75] candidate better |
| `current-4k` | −953.79 | [−1,065.15, −842.43] | +230.63 [+118.34, +342.92] candidate better |
| `current-16k` | −957.99 | [−1,069.58, −846.40] | +226.43 [+114.24, +338.62] candidate better |

**The reused schedule and the fresh suite disagree, and the fresh suite is the
declared one.** On reused deals at iteration 1,024 the paper arm leads by 48–84
BB/100; on fresh deals the same three arms span 12.86 BB/100 with an interval
four times wider than the difference. The reused-deal lead was an artifact of
evaluating on the deals the comparison was tuned by watching.

## 5. Random-opponent benchmark

| Arm | 256 | 512 | 1,024 |
| --- | ---: | ---: | ---: |
| `current-4k` | +130.31 | +145.63 (candidate better) | **+209.59 (+299.47 [+66.01, +532.93]) candidate better** |
| `current-16k` | +99.56 | −32.78 | −1.55 (+88.33 [−162.65, +339.31]) |
| `paper-16k` | +15.96 | +59.94 | +40.19 (+130.07 [−115.95, +376.09]) |

`current-4k` is the only arm with a significant random-opponent result, and it
improves monotonically. The two 16k arms remain inconclusive. A trained policy
beating uniform-random opponents is a basic sanity check, not evidence of
playing strength.

## 6. Observed play

On the fresh 2,048-block suite the three arms behave far more alike than they did
on the reused schedule mid-run:

| Measurement | `current-4k` | `current-16k` | `paper-16k` |
| --- | ---: | ---: | ---: |
| First-action preflop shove | 36.63% | 37.05% | 34.29% |
| Hands containing a preflop all-in | 39.87% | 39.93% | 36.68% |
| BB/100 in preflop-all-in hands | −1,808 | −1,758 | −1,860 |
| BB/100 in other hands | −388 | −426 | −415 |

At iteration 256 the same measurement had separated them sharply (`current-4k`
45.56% shoves against `current-16k` 30.73%); by the analysis point the three arms
had converged to nearly identical preflop commitment. Preflop all-in hands still
carry roughly 80% of the total loss in every arm. The uniform control, measured on
the same deals, shoves on 9.42% of its first actions and commits preflop in
16.46%.

These groups are selected by each policy's own choices, so the split is
descriptive and not a causal estimate of what banning all-ins would recover.

## 7. Collection, fitting and targets

| Measurement | `current-4k` | `current-16k` | `paper-16k` |
| --- | ---: | ---: | ---: |
| Completed iterations | 1,024 | 1,024 | 1,024 |
| Collection roots | 786,432 | 786,432 | 786,432 |
| Visited nodes | 24,253,932 | 25,386,653 | 26,577,472 |
| Roots reaching a postflop decision | 5.81% | 7.48% | 9.63% |
| Newly collected records that are postflop | 3.63% | 5.11% | 9.50% |
| Optimizer steps | 1,572,864 | 1,572,864 | 1,572,864 |
| Steps clipped by norm 1 | 100% | 100% | 100% |
| Largest sampled regret update, BB | 5.37e8 | 2.47e8 | **2.12e9** |

All three arms collected exactly the same number of roots, yet their postflop
coverage differs by almost 2× (5.81% → 9.63%). Coverage is therefore
**policy-dependent**: replay capacity and architecture change which situations
the learner visits, not only how much history it keeps. That is why "replay
capacity" was never a clean single-variable manipulation.

Every fitting step in every arm clipped at norm 1, and importance-corrected
regret updates still reached 10⁸–10⁹ BB while the policies lose ~950 BB/100.
Universal clipping does not establish that clipping causes weak play; the earlier
controlled fits found that removing it moves the objective by well under one
percent.

## 8. Cost and storage

| Measurement | `current-4k` | `current-16k` | `paper-16k` |
| --- | ---: | ---: | ---: |
| Training seconds (excludes checkpoints and evaluations) | 24,793 (6.9 h) | 26,347 (7.3 h) | 31,226 (8.7 h) |
| Last checkpoint cost (per 64 iterations) | 85 s | 419 s | 472 s |
| Checkpoint at 1,024 | 0.64 GiB | 0.73 GiB | 4.13 GiB |
| Export at 1,024 | 0.61 GiB | 0.61 GiB | 4.01 GiB |
| Unique bytes retained | 1.33 GiB | 1.42 GiB | 8.21 GiB |
| Peak RSS | 3.72 GiB | 5.75 GiB | 9.49 GiB |

The paper arm is the expensive one on every axis: 26% more training time, a
checkpoint archive 6× larger, and peak memory of 9.49 GiB on a 16 GB host — which
is why the machine spent the run swapping heavily. An iteration is not a
comparable unit of work between these arms, and equal iteration counts are not
equal compute.

For the record, the checkpoint sizes here are dramatically smaller than the
overnight PR #88 directories because retention keeps one checkpoint and one
export and the replay records are deflated (~10×: 1,281,410,158 → 124,590,398
bytes measured on a complete old 16K checkpoint).

## 9. What we learned

1. **Neither intervention is confirmed.** Quadrupling replay capacity per role
   produced no measurable improvement on fresh confirmation deals (−4.20
   BB/100), and the 6.8×-larger paper-inspired architecture package produced none
   either (+12.86). At this scale, on this game, with one training seed, these
   were not the binding constraints.
2. **Intermediate boundaries on reused deals are not findings.** The replay
   advantage was real at iteration 256 (+197.57, interval excluding zero) and
   gone by 512–768; the architecture advantage was real at iteration 64 (+264.72)
   and gone by 256. Both were recovered from reused validation deals. The
   predeclared fixed analysis point on fresh deals is what settles a comparison,
   and this batch is the clearest demonstration of that in the project so far.
3. **"Replay capacity" is a joint manipulation.** The 16k arms reached a postflop
   decision on 7.48–9.63% of roots against 5.81% for the 4k control at identical
   root counts. Capacity and coverage moved together, so nothing here separates
   them — which is exactly the gap a `paired-replay` arm was designed to close.
4. **Architecture packages are not the cheap win.** The paper package trained
   correctly end to end and reproduced its pilot behaviour, but it added 147,264
   parameters per role, 26% training time, 6× checkpoint size and 9.49 GiB peak
   memory to produce a difference the design cannot distinguish from zero.
5. **The failure mode is operational, and it is fixable.** The only thing that
   went wrong in 1,024 iterations × 3 arms was a free-disk guard tripping on a
   transient publication peak that a retain-one policy still requires, because
   retirement happens after publication. No invalid action, no corrupt artifact,
   no unrecoverable state — and the missing measurements were reconstructible
   from the retained export.
6. **The absolute level has not moved.** Every arm still loses ~950 BB/100 to the
   scripted style pool, which is the same order as the previous campaigns
   (−905.54 overnight baseline at 1,024, −1,056.11/−869.63 for the M4 campaign).
   The v0.5 requirement — reliable profit against that pool — is untouched by
   anything here.

## 10. Limits of this evidence

- **One training seed.** Nothing here measures between-seed variation; the
  project has seen same-recipe runs differ by ~190 BB/100 at the same iteration.
  With a single seed, these intervals answer "did this arm differ" and not "is
  this intervention better".
- **Inconclusive is not proof of no effect.** The fresh-suite intervals are about
  ±130 BB/100 wide. A real effect smaller than that would be invisible here, and
  the observed point estimates (+8.65, +12.86, −4.20) are an order of magnitude
  below the effects the project needs to close a 950 BB/100 gap.
- **Restricted to this recipe.** Width 32/64, first-decision sampling, 128 roots
  per role, 256 fitting steps per iteration, exploration 0.5. The paper's own
  regime is roughly 4,883× larger per fit; a null here is a null at this scale.
- **The paper arm's final suite is a reconstruction** of the runner's own
  completion path after a disk-guard failure (section 3).
- **No promotion.** No default model, checkpoint or production setting changes as
  a result of this batch.

## 11. Evidence and reproduction

- [Machine-readable record](holdem-day-paper.json): per-arm status and telemetry,
  every evaluation with its paired comparison, evaluation/schedule digests,
  collection, fitting and checkpoint totals, artifact sizes and hashes, the fresh
  final-suite estimates and comparisons, and observed-play statistics.
- [Protocol](../holdem-day-paper.md), [launch record](holdem-day-paper-launch.json),
  [pilot report](holdem-day-paper-pilot.json),
  [first interim](holdem-day-paper-interim.md),
  [second interim](holdem-day-paper-interim-2.md).
- Raw artifacts remain on the M4 under
  `~/Local/deepcfr-training/results/day-{current-4k,current-16k,paper-16k}-2026091902`:
  per-hand outcomes (`outcomes-*.json.gz`), evaluation reports, per-iteration
  timing and iteration reports, checkpoint timing and retention logs, the
  iteration-1,024 checkpoints and exports, and `completion.json` for the
  recovered arm.
- Recomputing any paired comparison needs only the saved outcomes: aggregate
  `candidate_chips / big_blind` by block for the candidate and control arms, then
  apply the block-clustered estimator used by `src/arena/report.py`. The schedule
  and control digests recorded in the JSON are what license the pairing.
