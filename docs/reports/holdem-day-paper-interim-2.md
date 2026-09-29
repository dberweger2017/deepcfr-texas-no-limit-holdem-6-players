# Daytime replay and paper-architecture comparison: second interim results

September 19, 2026, 18:50 UTC. **The batch is still training; this is a second
interim monitoring report, not the predeclared comparison.** The
[first interim report](holdem-day-paper-interim.md) covers the state at 15:22 UTC
and remains valid for that time. No model is promoted and no production default
changes.

This report covers the new boundaries since then: iteration 512 for all three
arms, iteration 768 for two of them, and the first arm reaching its iteration
limit.

## 1. Where the batch stands

| Arm | Iteration | Scripted evaluations | Random evaluations | State |
| --- | ---: | --- | --- | --- |
| `current-4k` | 1,021 | 64, 128, 256, 512, 768 | 256, 512 | about to stop; final suite pending |
| `current-16k` | 785 | 64, 128, 256, 512, 768 | 256, 512 | running |
| `paper-16k` | 683 | 64, 128, 256, 512 | 256, 512 | running |

Elapsed since launch (11:25:18Z): 7.4 hours. Every completed evaluation is
`valid`, with 12,288 completed hands and zero invalid actions. The schedule
digest and the uniform-control outcome digest are identical across all arms at
every shared boundary (64, 128, 256, 512, 768 and the 512 random suite), which is
what licenses the paired arithmetic below.

The control's own outcome digest is in fact identical at *every* boundary of
every arm (`79f468ba894cb7d3…`), so the uniform control is a fixed, deterministic
benchmark on this schedule: comparing a candidate with it is exactly paired, and
all movement between boundaries comes from the candidate side.

The design is unchanged from the [protocol](../holdem-day-paper.md): one fresh
training seed (2026091902), three arms differing only in replay capacity and
architecture, a fixed analysis point at iteration 1,024, and a fresh 2,048-block
scripted suite at that point as the only declared confirmation data.

## 2. The replay advantage present at iteration 256 has not persisted

This is the substantive change since the first interim report.

| Iteration | `current-4k` | `current-16k` | `current-16k − current-4k` | Verdict |
| ---: | ---: | ---: | ---: | --- |
| 64 | −1,189.19 | −1,236.44 | −47.25 [−202.16, +107.66] | inconclusive |
| 128 | −1,134.69 | −1,114.09 | +20.61 [−143.75, +184.96] | inconclusive |
| 256 | −1,148.64 | −951.07 | **+197.57 [+41.57, +353.56]** | **replay better** |
| 512 | −990.18 | −959.93 | +30.25 [−156.69, +217.19] | inconclusive |
| 768 | −982.51 | −958.00 | +24.51 [−166.79, +215.81] | inconclusive |

Intervals are Bonferroni-adjusted 97.5% individual intervals, clustered on 1,024
deal blocks.

The 4,096-record control improved from −1,148.64 at iteration 256 to −990.18 at
512 and −982.51 at 768, while the 16,384-record arm flattened at roughly −950 to
−960. The difference at the one boundary where it was significant (+197.57) has
fallen to +30.25 and then +24.51, with both later intervals comfortably spanning
zero.

The honest reading: **the replay-capacity advantage observed at iteration 256 was
transient on this seed.** It is not that the 16k arm got worse — it barely moved —
but that the 4k control caught up to it. A single significant intermediate
boundary was a checkpoint-level artifact of comparing two arms mid-convergence,
which is exactly what the protocol's fixed analysis point exists to prevent.

## 3. All three arms are now statistically indistinguishable

At iteration 512, the only boundary with all three arms:

| Arm | BB/100 | Nominal 95% |
| --- | ---: | --- |
| `paper-16k` | **−903.13** | [−1,061.29, −744.97] |
| `current-16k` | −959.93 | [−1,117.70, −802.15] |
| `current-4k` | −990.18 | [−1,160.28, −820.08] |
| uniform control | −1,172.68 | [−1,331.74, −1,013.63] |

Both predeclared primary comparisons are inconclusive at 512:

| Comparison | Difference | Bonferroni 97.5% | Verdict |
| --- | ---: | --- | --- |
| `current-16k − current-4k` | +30.25 | [−156.69, +217.19] | inconclusive |
| `paper-16k − current-16k` | +56.80 | [−127.89, +241.49] | inconclusive |
| `paper-16k − current-4k` | +87.04 | [−113.25, +287.33] | inconclusive |

The paper arm's trajectory over the whole run is a decay followed by a small
recovery, and none of it is significant at 512:

| Boundary | `paper-16k − current-16k` |
| ---: | ---: |
| 64 | +264.72 (paper better) |
| 128 | +163.26 (inconclusive) |
| 256 | −1.55 (inconclusive) |
| 512 | +56.80 (inconclusive) |

`paper-16k` is the best of the three in absolute terms at 512 by roughly 57–87
BB/100, but with this schedule that margin is not resolvable: at 1,024 deal
blocks the nominal interval half-width is ≈180 BB/100, so a difference of this
size cannot be distinguished from noise by a single-seed comparison.

## 4. Random-opponent benchmark at 512

| Arm | 256 | 512 |
| --- | ---: | ---: |
| `current-4k` | +130.31 (+220.19 [−19.01, +459.40]) | **+145.63 (+235.51 [+1.41, +469.60])** |
| `current-16k` | +99.56 (+189.44 [−51.84, +430.71]) | −32.78 (+57.10 [−185.18, +299.38]) |
| `paper-16k` | +15.96 (+105.83 [−141.88, +353.55]) | +59.94 (+149.81 [−92.36, +391.98]) |

Only `current-4k` at 512 clears zero, and it is also the arm that improves most on
the primary suite. The 16k arms' random-suite point estimates fall below the 4k
control's at 512, though all three intervals include zero.

## 5. Observed play at iteration 512

| Measurement | `current-4k` | `current-16k` | `paper-16k` |
| --- | ---: | ---: | ---: |
| First-action preflop shove | 40.09% | 32.26% | 36.57% |
| Hands containing a preflop all-in | 43.65% | 35.89% | 38.92% |
| First action: raise / fold / call | 3,809 / 1,095 / 862 | 3,493 / 1,259 / 979 | 3,131 / 1,512 / 1,063 |
| BB/100 in preflop-all-in hands | −1,833 | −2,059 | −1,859 |
| BB/100 in other hands | −337 | −345 | −294 |

Commitment changed only slightly since iteration 256 (first-action shove rates
45.56% → 40.09%, 30.73% → 32.26%, 39.62% → 36.57%), and losses in hands without a
preflop all-in moved from −520, −394 and −266 to −337, −345 and −294 BB/100 for
`current-4k`, `current-16k` and `paper-16k` respectively. Preflop all-in hands
still carry most of the loss. These groups are selected by each policy's own
actions, so the split is descriptive rather than causal.

## 6. Collection, fitting and targets

| Measurement | `current-4k` | `current-16k` | `paper-16k` |
| --- | ---: | ---: | ---: |
| Completed iterations | 1,021 | 785 | 683 |
| Collection roots | 784,128 | 602,880 | 524,544 |
| Visited nodes | 24,169,198 | 19,919,499 | 17,013,950 |
| Roots reaching a postflop decision | 5.82% | 8.29% | 8.58% |
| Newly collected records that are postflop | 3.64% | 5.73% | 9.04% |
| Optimizer steps | 1,568,256 | 1,205,760 | 1,049,088 |
| Steps clipped by norm 1 | 100% | 100% | 100% |
| Largest sampled regret update, BB | 5.37e8 | 6.00e7 | 4.36e8 |

Postflop coverage continues to rise for every arm as training proceeds, and the
gap between the 4k control (5.82% of roots) and the two 16k arms (8.29%, 8.58%)
has persisted across the run. Because all three collect the same 128 roots per
role, this remains a **policy-dependent** quantity: the replay increase changes
which situations the learner visits, not only how much history it retains.

Every fitting step still clips at norm 1 in every arm, and the largest sampled
regret updates remain 10⁷–10⁸ BB while all three policies lose ~900–1,000 BB/100
against the style pool.

## 7. Cost, storage and host pressure

| Measurement | `current-4k` | `current-16k` | `paper-16k` |
| --- | ---: | ---: | ---: |
| Checkpoint cost, latest 64 | 171 s | 524 s | 693 s |
| Checkpoint cost, first 64 | 65 s | 334 s | 334 s |
| Checkpoint size, latest | 618 MB | 591 MB | 2,687 MB |
| Peak RSS | 3.72 GiB | 5.15 GiB | 4.86 GiB |
| Unique bytes retained | 1.11 GiB | 1.08 GiB | 4.67 GiB |

Checkpoint cost grows with the snapshot archive for every arm: the control's cost
nearly tripled over the run (65 → 171 s per 64 iterations) and the paper arm now
spends about 11.5 minutes per 64 iterations serializing 2.7 GB. An iteration is
therefore not a comparable unit of compute between arms or even across boundaries
of the same arm, and elapsed-time comparisons must carry that.

Host pressure is the one operational concern. At 18:44 UTC the M4 had 32 GiB
free disk against the 12 GiB floor, but swap in use was **7.2 GB of 8.2 GB** —
up from 5.97 GB at 15:19 UTC and 3.78 GB three hours before that. The paper arm
alone retains 4.67 GiB and will approach 8 GB at iteration 1,024, when its
checkpoint and export are written together. No per-worker memory cap is set, per
the owner's standing instruction; RSS is measured instead.

## 8. What is still pending

- `current-4k` stopping at iteration 1,024 and running its fresh 2,048-block
  final evaluation (imminent).
- Iterations 768 and 1,024 for `paper-16k`, and 1,024 for `current-16k`.
- The predeclared iteration-1,024 comparison for all three arms, and the fresh
  final suite for the other two arms.
- From measured rates, remaining training is roughly 2 hours for `current-16k`
  and 4 hours for `paper-16k`, plus final evaluations.

## 9. What this changes

The first interim report said the measurable gain at iteration 256 tracked replay
capacity rather than the architecture. That statement was true of iteration 256
and is now misleading on its own: the replay gain has decayed to +30 at 512 and
+25 at 768, the control having converged upward to meet the 16k arm.

Consequently, on the evidence available now:

- **Nothing in this batch has yet separated any two arms at a predeclared
  boundary.** Both primary comparisons are inconclusive at 512, and the only
  significant result anywhere in the run was an intermediate checkpoint
  difference that did not persist.
- **The architecture comparison remains open.** `paper-16k` leads in absolute
  terms at 512, but by less than the schedule can resolve with one seed.
- **The replay interpretation must stay unsettled.** A capacity advantage that
  appears at one boundary and disappears at the next is consistent with
  convergence-rate differences rather than a lasting improvement, and it also
  cannot be separated from the coverage difference the same change induces.
- Any next-batch design that plans to replicate "replay capacity works" should
  first require that effect to survive to the fixed analysis point, on the fresh
  suite, in this batch.

The batch is doing what it was designed to do: the reused-deal intermediate
boundaries are monitoring signals, and the predeclared 1,024 comparison on fresh
deals is where a claim may be made. What has changed is that the monitoring
signal that looked like a finding at 256 no longer looks like one.

## 10. Evidence and reproduction

- [Machine-readable second interim record](holdem-day-paper-interim-2.json): the
  same structure as the first interim record, including per-arm status, every
  evaluation with its paired comparison, evaluation and control digests,
  collection/fitting totals, checkpoint timings, artifact hashes and the
  block-level statistics behind each paired result.
- [First interim record](holdem-day-paper-interim.json) at 15:22 UTC.
- Raw per-hand outcomes and per-iteration telemetry remain on the M4 beside each
  evaluation report (`outcomes-*.json.gz`, `evaluation-*.json`,
  `training-timing.jsonl`, `iteration-reports.jsonl`, `checkpoint-timing.jsonl`).
