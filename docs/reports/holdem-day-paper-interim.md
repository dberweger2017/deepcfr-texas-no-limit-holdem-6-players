# Daytime replay and paper-architecture comparison: interim results

September 19, 2026, 15:22 UTC. **The batch is still training; this is an interim
monitoring report, not the predeclared comparison.** No model is promoted, no
production default changes, and none of the numbers below is a v0.5
qualification result.

Three fresh six-player, 100 BB, no-rake Deep CFR arms are running on the M4 from
revision `2c23594`, launched 2026-09-19T11:25:18Z. This report records every
scheduled evaluation completed so far, the recomputed paired comparisons, the
observed play, the collection and fitting diagnostics, and the measured cost.
The [protocol](../holdem-day-paper.md), [launch record](holdem-day-paper-launch.json)
and [pilot report](holdem-day-paper-pilot.json) remain the authoritative
statement of the design.

## 1. What is running

All three arms start from uniform bootstrap on fresh training seed **2026091902**,
use first-decision sampling, exploration 0.5, 128 traversal roots per role, 256
fresh fitting steps per role at batch 32, Adam 0.001 and norm-1 clipping. They
differ only in the declared setting:

| Arm | Architecture | Replay records per role |
| --- | --- | ---: |
| `current-4k` | Existing width-32 GRU with action-conditioned regret/Q heads | 4,096 |
| `current-16k` | Identical model | 16,384 |
| `paper-16k` | Paper-inspired encoder (card embeddings, residual card tower, width-64 history GRU, LayerNorm trunk) | 16,384 |

Each arm stops at iteration 1,024. Validation runs at 64, 128, 256, 512, 768 and
1,024; the random-opponent benchmark rides along at 256, 512 and 1,024. At 1,024
each arm then freezes and runs one fresh 2,048-block scripted evaluation on root
seed `2026091920` in a separate `final` directory, and only that suite is
declared confirmation data.

The `paper-16k` model is a **package**, not a component: card encoder, width,
depth and normalization change together. It has 172,866 parameters per role
against 25,602 for the two controls, so any difference it shows belongs to the
package and cannot be attributed to a single architectural choice.

Progress at 15:22 UTC, 3.94 hours after launch:

| Arm | Iteration | Evaluations completed | State |
| --- | ---: | --- | --- |
| `current-4k` | 576 | 64, 128, 256, 512 styles; 256, 512 random | running |
| `current-16k` | 448 | 64, 128, 256 styles; 256 random | running |
| `paper-16k` | 420 | 64, 128, 256 styles; 256 random | running |

No arm has failed, no invalid action has been recorded, and every completed
evaluation is `valid` with 12,288 completed hands.

## 2. How the numbers are produced

Every validation evaluation plays 12,288 hands: 6,144 with the arm's
snapshot-average policy in the candidate seat and 6,144 with a uniform-random
control in the same seat, over 1,024 deal blocks with six seat rotations each.
Opponents are the fixed scripted style pool (tight-passive, loose-passive,
tight-aggressive, loose-aggressive, pot-pressure).

Results are reported as BB/100. Confidence intervals are computed on **deal
blocks**, with the six seat rotations of a block kept together, because rotations
of one deal are not independent samples. The two predeclared primary comparisons
are `current-16k − current-4k` and `paper-16k − current-16k`; their intervals here
are **Bonferroni-adjusted 97.5% individual intervals**, which is the rule the
protocol declared in advance.

Three properties of these numbers must be kept in view:

1. **One training seed.** Every arm-to-arm difference below is within a single
   seed. Training-seed variation is not measured, and the project has seen
   same-recipe runs differ by roughly 190 BB/100 across seeds at the same
   iteration.
2. **Reused validation deals.** The 1,024-block schedule is shared across
   checkpoints and arms. The intervals are therefore not independent test
   evidence; they are monitoring statistics.
3. **The decision point is iteration 1,024 on the fresh suite.** Nothing here
   selects a checkpoint, and the predeclared comparison is unaffected by which
   intermediate boundary looks best.

## 3. Scripted-pool results by boundary

Higher is better. The uniform control is fixed by the schedule and is identical
across arms, which is why it appears once.

| Iteration | `current-4k` | `current-16k` | `paper-16k` | Uniform control |
| ---: | ---: | ---: | ---: | ---: |
| 64 | −1,189.19 | −1,236.44 | −971.72 | −1,172.68 |
| 128 | −1,134.69 | −1,114.09 | −950.83 | −1,172.68 |
| 256 | −1,148.64 | −951.07 | −952.63 | −1,172.68 |
| 512 | **−990.18** | not reached | not reached | −1,172.68 |

Nominal 95% intervals are approximately ±180 BB/100 at this sample size, so
differences below roughly 180 BB/100 are not resolvable at a single boundary with
this schedule.

Paired differences against the uniform control (nominal 95%):

| Arm | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: |
| `current-4k` | −16.51 | +37.99 | +24.04 | **+182.50** (candidate better) |
| `current-16k` | −63.76 | +58.59 | **+221.61** (candidate better) | — |
| `paper-16k` | **+200.96** (candidate better) | **+221.85** (candidate better) | **+220.05** (candidate better) | — |

All three arms beat the uniform control at some boundary on these deals. That is
a low bar: the control plays the legal candidate set uniformly, so beating it
does not indicate competent poker, and every arm remains roughly 950–1,200
BB/100 below break-even against the style pool.

## 4. Predeclared paired comparisons

Differences are candidate minus reference, clustered on 1,024 deal blocks, with
Bonferroni-adjusted 97.5% individual intervals.

| Iteration | `current-16k − current-4k` | `paper-16k − current-16k` | `paper-16k − current-4k` |
| ---: | --- | --- | --- |
| 64 | −47.25 [−202.16, +107.66] inconclusive | **+264.72 [+96.26, +433.18] paper better** | **+217.47 [+72.61, +362.34] paper better** |
| 128 | +20.61 [−143.75, +184.96] inconclusive | +163.26 [−16.23, +342.75] inconclusive | **+183.86 [+13.96, +353.76] paper better** |
| 256 | **+197.57 [+41.57, +353.56] replay better** | −1.55 [−162.19, +159.08] inconclusive | **+196.01 [+16.81, +375.21] paper better** |

Random-opponent benchmark (secondary, nominal 95%):

| Arm | 256 | 512 |
| --- | ---: | ---: |
| `current-4k` | +130.31 (+220.19 [−19.01, +459.40]) | **+145.63 (+235.51 [+1.41, +469.60])** |
| `current-16k` | +99.56 (+189.44 [−51.84, +430.71]) | — |
| `paper-16k` | +15.96 (+105.83 [−141.88, +353.55]) | — |

All random-opponent results are inconclusive except `current-4k` at 512.

## 5. The paper arm's early lead has decayed

The `paper-16k` arm opened 264.72 BB/100 ahead of its replay-matched control at
iteration 64, still led by 163.26 at 128, and is level at 256:

| Boundary | `paper-16k − current-16k` |
| ---: | ---: |
| 64 | +264.72 (paper better) |
| 128 | +163.26 (inconclusive) |
| 256 | −1.55 (inconclusive) |

Meanwhile `current-16k` moved from −1,236.44 at 64 to −951.07 at 256 while
`paper-16k` moved from −971.72 to −952.63. The two 16k arms are now statistically
indistinguishable, and `paper-16k`'s remaining advantage over `current-4k`
(+196.01) is statistically indistinguishable from `current-16k`'s (+197.57).

The honest reading at this point: on this seed, the measurable gain over the
4,096-record control tracks the **replay increase**, and the 172,866-parameter
paper package adds nothing detectable on top of it. Two cautions keep this from
being a conclusion. First, one seed. Second, and more importantly,
`paper-16k` has the **worst** random-opponent result of the three arms at 256
(+15.96 against +130.31 and +99.56), so its scripted-pool parity is not matched
by broader competence.

## 6. Observed play at iteration 256

Measured from the saved public hand records of the same 6,144 candidate hands per
arm. A shove means an all-in raise on the hero's first preflop action; an all-in
call is counted only in the "preflop all-in" row.

| Measurement | `current-4k` | `current-16k` | `paper-16k` |
| --- | ---: | ---: | ---: |
| First-action preflop shove | 45.56% | 30.73% | 39.62% |
| Hands containing a preflop all-in | 49.25% | 34.88% | 42.07% |
| First action: raise / fold / call | 4,397 / 857 / 534 | 3,781 / 976 / 984 | 3,312 / 1,405 / 1,021 |
| BB/100 in preflop-all-in hands | −1,797 | −1,990 | −1,898 |
| BB/100 in other hands | −520 | −394 | −266 |

Two observations. Preflop all-in frequency is still very high in all three arms
(the uniform control, measured on the same hands, shoves on its first action in
9.42% of them and commits preflop in 16.46%), and those hands carry the bulk of
the losses. And the arms differ in temperament: `current-16k` shoves far
less and folds and calls more than `current-4k`, while `paper-16k` is the
tightest of the three and loses least in hands without a preflop all-in
(−266 BB/100 against −520 and −394).

This is an association between commitment and results, not a causal estimate.
Reducing shove frequency is not a success criterion, and the preflop-all-in
group is selected by the policy's own choices.

## 7. Collection, fitting and targets

Measured over every completed iteration of each arm (`training-timing.jsonl`,
`iteration-reports.jsonl`).

| Measurement | `current-4k` | `current-16k` | `paper-16k` |
| --- | ---: | ---: | ---: |
| Completed iterations | 576 | 448 | 420 |
| Collection roots | 442,368 | 344,064 | 322,560 |
| Visited nodes | 13,428,390 | 11,713,352 | 9,440,940 |
| Roots reaching a postflop decision | 5.06% | 9.57% | 5.02% |
| Newly collected records that are postflop | 3.43% | 6.89% | 4.89% |
| Optimizer steps | 884,736 | 688,128 | 645,120 |
| Steps clipped by norm 1 | 100% | 100% | 100% |
| Largest sampled regret update, BB | 5.37e8 | 6.00e7 | 2.57e7 |

Postflop coverage differs sharply between the arms even though all three collect
the same 128 roots per role: the 16k arms reach a postflop decision on 9.57% of
roots against 5.06% and 5.02% for the 4k control and the paper arm. Coverage is
therefore **partly a consequence of the learned policy**, which means the replay
increase is not a clean single-variable change in what the learner sees: it moves
both how much history is retained and which situations the policy then visits.

Every fitting step in every arm clips at norm 1, and sampled regret updates still
reach 10^7–10^8 BB while the policies lose ~1,000 BB/100. Clipping being
universal does not establish that clipping causes weak play; earlier controlled
fits found removing it changes the objective by well under one percent.

## 8. Cost, storage and host resources

| Measurement | `current-4k` | `current-16k` | `paper-16k` |
| --- | ---: | ---: | ---: |
| Mean iteration cost to date | 23.1 s | 25.5 s | 27.4 s |
| Checkpoint cost (per 64 iterations) | 65–106 s | 334–470 s | 334–513 s |
| Checkpoint size, first → latest | 67 → 343 MB | 147 → 359 MB | 360 → 1,654 MB |
| Peak RSS | 3.72 GiB | 4.74 GiB | 4.86 GiB |
| Unique bytes retained | 0.68 GiB | 0.53 GiB | 2.64 GiB |

Checkpointing is the dominant overhead for the 16k arms: roughly 5–8 minutes per
64 iterations, because the retained replay is serialized with the snapshot
archive. The three arms are therefore **not** equal-work runs — an iteration is
not a unit of comparable compute across them, and elapsed-time comparisons must
carry that. The paper arm's checkpoint grows fastest (about +266 MB per 64
iterations) because its per-iteration snapshot archive holds six role networks
of 172,866 parameters.

Host context at 15:19 UTC: per-worker RSS 1.23–2.84 GiB, swap in use 5.97 GB of
7.17 GB, free disk 30 GiB against the supervisor's 12 GiB floor and a 16 GiB
per-job output limit. No per-worker memory cap is configured, per the owner's
standing instruction; RSS is measured and recorded instead. The host is swapping,
which is part of why the two 16k arms advance at roughly 70% of the control's
iteration rate.

## 9. Still pending in this batch

- Iterations 512 and 768 for `current-16k` and `paper-16k`, and 768 for
  `current-4k`. The `current-16k − current-4k` comparison at 512 is the next
  informative boundary, because `current-4k` improved from −1,148.64 at 256 to
  −990.18 at 512 and may be converging toward the 16k arms.
- The predeclared iteration-1,024 comparison for all three arms.
- The fresh 2,048-block scripted suite at 1,024 — the only declared confirmation
  data — and the disjointness check against validation and training roots.
- Projected finish (training only, from measured rates): `current-4k` ≈18:25 UTC,
  `current-16k` ≈20:25 UTC, `paper-16k` ≈21:00 UTC, each followed by its final
  evaluation.

## 10. What may and may not be concluded

**May be concluded.** The three arms run correctly end to end: legal actions
only, complete evaluation schedules, valid reports, recoverable checkpoints,
retained artifacts, and measured costs. On this seed and these reused deals,
increasing replay from 4,096 to 16,384 records per role is associated with a
≈197 BB/100 improvement over the 4,096 control at iteration 256, and the
paper-inspired package shows no measurable advantage over the replay-matched
control at that boundary.

**May not be concluded.** That the replay increase is a genuine capacity effect
rather than a coverage effect, since it also raised postflop root share from 5.06%
to 9.57%. That the architecture comparison is settled, since the paper arm led
early, is level now, and has the weakest random-opponent result. That any arm is
close to playable: all remain ~950 BB/100 below break-even against the scripted
pool, and the v0.5 requirement is reliable profit against that pool. That these
intervals are independent evidence: the deals are reused, the seed count is one,
and the predeclared comparison is at iteration 1,024 on the fresh suite.

## 11. Evidence and reproduction

- [Machine-readable interim record](holdem-day-paper-interim.json): per-arm
  status, every evaluation with its paired comparison, collection/fitting totals,
  checkpoint timings, artifact sizes and SHA-256 hashes, and the block-level
  statistics behind each paired result.
- Raw per-hand outcomes are retained losslessly beside each evaluation report on
  the M4 as `outcomes-<iteration>.json.gz` (and `-random` for the benchmark),
  with `evaluation-<iteration>.json` holding the report digest and the schedule
  digest.
- Per-iteration telemetry: `training-timing.jsonl`, `iteration-reports.jsonl`,
  `checkpoint-timing.jsonl`, `retention.jsonl` and `status.json` in each arm's
  scenario directory.

Recomputing the paired comparisons requires only the saved outcomes:

```sh
# on the M4 host, from the day-training checkout
~/Local/deepcfr-training/.venv/bin/python - <<'PY'
# load outcomes-<iteration>.json.gz, aggregate candidate_chips/big_blind by block,
# then use src.arena.report.estimate / comparison on the per-block rates.
PY
```

The uniform-control outcome digests and the schedule digests match between arms
at every shared boundary, which is what licenses the paired arithmetic above; the
interim JSON records both digests per boundary.
