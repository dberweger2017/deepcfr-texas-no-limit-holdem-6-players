# M4 overnight training: completed and stopped

September 19, 2026. **All training is stopped. No model qualifies for v0.5.**
The larger replay buffer improved this seed's primary comparison; additional
branching regressed. Simply extending the baseline did not resolve its large
losses against scripted opponents.

This was actual six-player, 100 BB, no-rake Deep CFR self-play with width-32
networks, not the restricted multistreet supervised benchmark. All three arms
started fresh on seed 2026091901. Arm A changed only the sampler to
`second-decision`; arm B changed only replay capacity from 4,096 to 16,384 per
role. The [protocol](../holdem-overnight-comparisons.md) and original launch
records remain historical evidence.

## What happened overnight

| Arm | Training stopped at | Last complete scripted evaluation | End state |
| --- | ---: | ---: | --- |
| Baseline | 1,685 | 1,664 | Owner-requested stop; final checkpoint saved |
| Additional branching | 1,659 | 1,600 | Owner-requested stop; final checkpoint saved |
| Larger replay | 1,024 | 1,024 | Output-size guard stopped it during evaluation |

The first two workers ran for roughly 12 hours. On the owner's request, STOP
files were written; both finished their active iteration, published the final
checkpoint and exited. Their stopped timestamps are 09:47:23 and 09:47:30 UTC.
No extra training or final-policy evaluation was launched during closeout.

The replay run stopped at 06:24:53 UTC. Its **scripted evaluation at 1,024
completed and is valid**, with all 12,288 scheduled hands. It did not finish
the random-opponent suite, so its learning-curve file still ends at 960. The
report below uses the complete saved evaluation directly; it does not pretend
the entire evaluation phase succeeded. The failed status and logs are retained.

Across all arms, 133 complete arena evaluations retain **1,634,304 hands**, with
zero invalid actions. Half are uniform-control hands, and validation deals are
reused across checkpoints and arms. This is not 1.6 million independent test
hands. No fresh sealed or release-qualification evaluation was performed.

## Main comparison: iteration 1,024

Higher profit is better. Each policy has 6,144 candidate hands in 1,024 deal
blocks with six seat rotations. Confidence intervals cluster rotations within
a deal block. The candidate-minus-baseline comparisons below mean the trained
continuous baseline, not the uniform arena control.

| Arm | Scripted-pool BB/100 | Nominal 95% interval | Difference from trained baseline | Paired nominal 95% interval |
| --- | ---: | --- | ---: | --- |
| Baseline | −905.54 | [−1,064.27, −746.82] | — | — |
| Additional branching | −1,173.52 | [−1,357.40, −989.64] | −267.98 | [−424.15, −111.80] |
| Larger replay | −714.19 | [−867.96, −560.42] | +191.35 | [+25.08, +357.61] |

The scheduled deals, opponent lineups and uniform-control outcome digests match
between arms. Recomputed candidate estimates exactly match the saved reports.
The larger-replay result is encouraging **for this seed**, while branching
clearly loses ground on these deals. These nominal intervals are not adjusted
for all inspected comparisons and do not measure training-seed uncertainty.
Repeated validation and one training seed prevent a confirmation or promotion
claim. All absolute intervals remain far below zero.

Random-opponent results at 1,024 are +116.36 BB/100 for baseline (interval
[−104.43, +337.14]) and +216.84 for branching ([−1.08, +434.75]). Replay has no
completed random evaluation at that boundary; its last one, at 960, is +44.17
([−169.29, +257.62]). These do not establish reliable profitability.

## Did longer training help?

| Iteration | Baseline styles BB/100 | Branching | Larger replay |
| --- | ---: | ---: | ---: |
| 64 | −847.28 | −1,280.78 | −872.44 |
| 256 | −958.61 | −1,007.06 | −803.47 |
| 512 | −905.13 | −1,104.87 | −781.25 |
| 960 | −900.23 | −1,127.26 | −791.76 |
| 1,024 | −905.54 | −1,173.52 | −714.19 |
| 1,600 | −971.22 | −1,221.83 | Not reached |
| 1,664 | −1,029.82 | Not evaluated | Not reached |

The baseline does not show sustained improvement across this longer horizon.
Branching improves from its particularly poor early value, then deteriorates.
Replay performs better at several later checkpoints, but its intermediate paired
intervals at 256/512/960 still include zero. We keep the predeclared 1,024
comparison rather than selecting the most flattering checkpoint.

## The actual decisions remain problematic

| At iteration 1,024 | Baseline | Branching | Larger replay |
| --- | ---: | ---: | ---: |
| First-action preflop shove, fraction of all hands | 30.44% | 45.44% | 32.34% |
| Any preflop all-in, fraction of all hands | 34.75% | 49.33% | 35.47% |

A shove here means an all-in raise; an all-in call counts only in the second
row. Blinds and prior investments are subtracted from the remaining stack
before identifying an all-in. Walks remain in the denominator.

Branching's poor result comes with much more preflop commitment. That is an
association, not proof that its extra branching caused the losses through that
mechanism. Larger replay actually shoves first slightly more often than the
baseline at 1,024 despite losing less. **Reducing shove frequency alone would
be the wrong optimization target.** We need to evaluate which holdings and
situations receive those actions and their counterfactual values.

## Collection and fitting through iteration 1,024

| Measurement | Baseline | Branching | Larger replay |
| --- | ---: | ---: | ---: |
| Collection roots | 786,432 | 786,432 | 786,432 |
| Visited nodes | 28,206,083 | 27,104,554 | 25,423,420 |
| Roots reaching a postflop decision | 12.20% | 4.78% | 7.68% |
| Newly collected records that are postflop | 8.26% | 5.34% | 5.91% |
| Fitting steps | 1,572,864 | 1,572,864 | 1,572,864 |
| Fitting steps clipped | 100% | 100% | 100% |
| Largest sampled regret update, BB | 204,961,941 | 54,451,306 | 867,190,285 |
| Recorded training time, excluding checkpoint/evaluation | 7.18 h | 6.85 h | 6.73 h |

The arms learn different policies and therefore visit different trajectories.
Lower total nodes in the branching arm do not mean that expanding another own
decision is intrinsically cheaper on the same tree. Coverage fractions here
refer to collected roots and new records; they are **not measured final
reservoir street proportions**.

The target maxima are importance-corrected learning updates, not poker winnings
or stacks. Their persistence shows that extreme targets have not disappeared.
A smaller observed maximum is not a variance estimate, and clipping every
step does not establish clipping as the cause of poor play. Previous controlled
fitting experiments already caution against that inference.

### Elapsed-cost view

Using manifest creation and evaluation-timing file timestamps as approximate
worker-start/completion times, the last completed scripted evaluations within
2 hours were iteration 256 for all three arms. Within 4 hours they were
iteration 512 for baseline/branching and 448 for replay. Replay's iteration-448
result was −813.79 BB/100. The other values are in the table above.

These approximate budgets include intervening training, evaluations and storage
work. They are not exact process-start timestamps or an isolated hardware
benchmark: the baseline began earlier and workers shared the M4. Larger replay
made checkpointing materially more expensive: its first checkpoint took 314
seconds and the 1,024 checkpoint 481 seconds, versus 66 seconds for the
baseline's first checkpoint. Increasing replay is not free even when fitting
steps stay fixed.

## Storage failure and fix

The supervisor's output-size calculation summed every file path. Pinning a
checkpoint/export creates hard links, so the same inode appeared twice while
both the rolling and pinned paths existed. This overcounted storage.

The replay directory after shutdown contained 8,594,092,131 logical path bytes
but only 5,995,018,301 unique file bytes. Its 8 GiB output guard fired even
though unique retained data was about 5.58 GiB and the host still had ample
free space. The guard's last recorded pre-failure sample was slightly below
the limit; the next measurement triggered the stop. This was not a RAM cap,
learning exception, or exhausted physical disk.

`used_bytes` now counts `(device, inode)` once. A regression test checks pinned
links, retirement of the original path, and a distinct file with identical
contents. The 8 GiB output and 12 GiB free-space thresholds remain unchanged;
the owner's no-RAM-cap setting remains unchanged. This correction is for future
runs. **The failed run was not restarted, and its history was not rewritten.**

## Decision

Keep the current production recipe. Treat 16,384-record replay as a candidate
worth independently confirming, not a proven new default. Do not prioritize
another long second-decision run on this evidence.

Next, implement the bounded 100 BB first-to-act action-value/card-sensitivity
probe. Compare normal raises and all-ins under declared frozen continuations,
with uncertainty on action differences, using the retained models. This can
distinguish a failure to value the actions from a poor choice of which hands
receive them. If replay remains the preferred intervention, confirm it with
independent training seeds and fresh paired scripted-pool deals before any
promotion. No additional overnight run is automatically scheduled.

## Evidence and reproduction

- [Machine-readable analysis](holdem-continuous-m4-results.json): paired results,
  public-action counts, collection/fitting totals and stopped/failed states.
- [Elapsed-budget audit](holdem-continuous-m4-elapsed.json): approximate timing
  method and selected boundaries.
- [Complete file inventory](holdem-continuous-m4-inventory.json): 470 retained
  files with size and SHA-256; about 19.71 GB of unique run data.
- Full archive: `results/m4-overnight-20260919.tar.zst`, SHA-256
  `f53f112295c785e19209132499e177257e31669da6cb74b82ef88b4b150b8fbb`.
  Raw data and models are in the archive, not Git. The M4 originals remain intact.

`python -m scripts.analyze_continuous_holdem --root <retrieved-results> --out
<analysis.json>` reproduces the saved-outcome analysis. It verifies outcome
hashes, seat rosters, report estimates and paired controls; it does not run a
model or train. A failing random-suite phase is not silently reconstructed.

Closeout verification results are recorded in the accompanying PR and the
recovery/transfer record linked there. The focused storage/continuous/local
runner checks passed (15 tests), and all 744 repository tests passed before
adding the read-only report extractor. Required CI runs again on the final PR
head, including end-to-end reproduction and fresh-process resume checks.
