# Fixed-work postflop continuation replication on M4

**PR #110, September 27, 2026.** This is a completed, bounded development
comparison. The three K=4 continuations did not establish a scripted-pool
playing gain over K=1 at the same additional traversal-node budget. The
candidate remains opt-in; no checkpoint or player is promoted.

## Frozen comparison

All six runs independently loaded the immutable 5.83M-entry checkpoint,
SHA-256 `94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a`.
The executing source was `68f9e2c33c696764847e97cb0bb718a49ee7c928`.
The [frozen plan](../../configs/blueprint/postflop-replication-m4.json) has
SHA-256 `41e6883251cc9e663d01261c60b454dc32cc029a859fb421d72131b8c7470a9a`
under the arena's canonical JSON digest. Seeds `2026092701`–`03` each ran
the unchanged K=1 trainer and the optional K=4 first-flop continuation
replication. Both used 20 million **additional traversal nodes**, stopping
after the completed outer iteration that crossed the target. The maximum
observed overshoot was 14,853 nodes, below the declared 250,000-node cap.

Current regret-matched play, button-zero-compatible lookup and the same
no-free-fold wrapper were used for both trained arms. An independently
generated `U_safe` observation set used 128 scripted and 32 random blocks,
with seeds separate from the playing test. The fresh playing schedule used
4,096 six-rotation scripted-pool blocks and 1,024 random-opponent blocks.
The three saved continuation pairs were evaluated on the same deals,
positions and opponent assignments. Each aggregate seed contrast was
averaged **within a block** before calculating block-level uncertainty.
The two scripted primary comparisons have two-sided 97.5% intervals, a
Bonferroni adjustment across the two claims. Random results and other
comparisons are descriptive.

## Playing results

| Scripted-pool comparison | Paired BB/100 | 97.5% interval |
| --- | ---: | ---: |
| K=4 minus K=1, mean over three pairs | +6.38 | [−14.45, +27.21] |
| K=4 minus `U_safe` | +5.27 | [−34.88, +45.42] |

Neither primary interval clears zero. The three K=4-minus-K=1 seed-pair
effects were +32.82, −17.63 and +3.94 BB/100. Their signs differ; the
first pair's positive interval does not make the prespecified aggregate
positive. Mean absolute scripted profit was −508.12 BB/100 for K=4,
−514.50 for K=1, −513.39 for `U_safe`, −517.92 for the unchanged parent,
and +45.93 for the `tight_aggressive` hero. The trained players still lose
heavily to this scripted pool.

On the secondary random suite, K=4 minus K=1 was −39.38 BB/100 with a
95% interval [−83.17, +4.41]. K=4 minus `U_safe` was +106.34
[+24.10, +188.57]. The random suite does not reverse the inconclusive
scripted primary result; it gives no evidence that K=4 improves on its
compute-matched K=1 control. The complete per-seed effects and absolute
rates are in [analysis.json](postflop-replication-m4-artifacts/analysis.json).

## Effective work and coverage

| Mean additional work per run | K=1 | K=4 |
| --- | ---: | ---: |
| Traversal nodes | 20.001M | 20.010M |
| Completed outer iterations | 3,720 | 1,044 |
| New abstract entries | 2.387M | 2.520M |
| Revisited outer-iteration contributions | 0.633M | 0.437M |
| Raw traverser visits | 3.020M | 3.210M |
| Normalized update mass | 3.020M | 0.846M |
| K=4 first-flop prefixes / continuations | — | 59,478 / 237,912 |

`node.visits` counts raw traverser visits, so four conditional visits can
inflate that number without adding four independent policy-refinement
iterations. The normalized mass divides downstream K=4 increments by four;
the separate contribution sidecar counts additional outer iterations that
reached each key. At fixed node work, K=4 produced more new entries but
roughly 31% fewer revisited outer contributions and about 72% fewer
completed outer iterations. This is evidence of the work-allocation
tradeoff, not proof that every new K=4 entry is unhelpful.

On the fixed independent observation set, the mean number of flop keys
receiving at least one **additional outer-iteration update** was 128/414
for K=1 versus 83/414 for K=4. Mean trained flop-key coverage was
187/414 versus 179/414. Turn update coverage was 16/207 versus 19/207;
river was 4/138 versus 6/138. The latter counts are too sparse to establish
a late-street improvement. On each policy's actual scripted trajectories,
K=1 reached about 2,095 distinct trained flop keys on average, versus
1,939 for K=4; these trajectories differ by policy, so that comparison is
descriptive. The independent-set measurement is the cleaner density check.

The K=4 runs recorded about 59.5 thousand retained flop prefixes and
four independent conditional continuations per prefix. The mean observed
within-prefix return variance across the four draws was about 978 BB²;
the summed, iteration-normalized regret variance metric was about 78,896
BB² per prefix. K=1 did not record an equivalent within-prefix variance,
and these values do not demonstrate variance reduction **per unit compute**.
The finite nonuniform reference tests establish the fixed-profile
expectation contract; the Hold'em measurements show how the proposed
replication spends its fixed budget.

## Validity and resources

The stable-source repository suite passed **805 tests**. Outcome-free M4
preflights checked checkpoint load, K=1/K=4 throughput and memory growth;
a one-iteration K=4 checkpoint also passed descendant-lineage lookup and
48 legal resource-preflight hands. In the main campaign all six training
runs reached the work target with zero discarded iterations or recorded
resource stops. All nine evaluation arms completed 30,720 hands each:
**276,480 legal, chip-conserving hands** in total, zero failed attempts,
zero schedule-pairing mismatches and zero selected free folds. Every
saved checkpoint has its own output hash and validated lineage; the parent
hash was not substituted for a descendant.

Mean training time was 27.72 minutes for K=1 and 30.17 minutes for K=4
at the matched node budget. K=4 spent about 51.9 seconds per run rebuilding
sampled flop prefixes. The whole campaign, including setup, loads, saves,
evaluation and analysis, completed in about 3.18 hours, below the ten-hour
hard limit. Maximum process RSS was 6.54 GiB during training and 6.61 GiB
during evaluation, below 10.5 GiB. Reported system swap use stayed at
769.38 MB; memory pressure was recorded separately. The parent checkpoint,
six outputs and all raw attempts remain on the M4. No paid host was used.

## Interpretation and artifacts

At this fixed budget, K=4's extra conditional sampling did not translate
to a detectable scripted-pool gain and reduced the number of independent
outer updates to important flop keys. The result does not rule out a
different sampling design, work budget or checkpoint. The three seeds are
**continuations of one shared trained parent**, not independent from-scratch
training seeds. Intervals are conditional on these saved policies and the
declared opponent suites; no v0.5 or professional-strength claim follows.

The compact [artifact directory](postflop-replication-m4-artifacts) contains
the machine-readable analysis, frozen observation set, six lineage files,
work-milestone telemetry, run manifests, checksums, and a full
[130-file inventory](postflop-replication-m4-artifacts/inventory.json).
The [preflight subdirectory](postflop-replication-m4-artifacts/preflight)
also retains the short K=1/K=4 throughput results, the 96-step K=4 growth
check, the 48-hand evaluation checks and their separate 27-file inventory.
The 2.996 GB of raw files, including checkpoints, per-iteration rows,
per-key outer contributions and every hand row, remain at:

```text
ssh m4
/Users/dberweger/Local/postflop-replication-pr110/results/postflop-replication-m4-20260927
```

Each inventory entry has a relative path, byte count and SHA-256. The
61 compact files copied into this PR were checked against that inventory.
The remaining large files can be fetched by their listed paths using the
`m4` SSH alias. The M4 parent checkpoint was independently rehashed to
the frozen SHA-256 after the run.

**Next decision:** review #110 as an opt-in experiment. Do not promote a
model or extend K, node work or evaluation samples on this opened schedule.
If more blueprint training is pursued, first design a separate experiment
that preserves diverse-prefix and useful flop-key update coverage while
testing conditional variance reduction.
