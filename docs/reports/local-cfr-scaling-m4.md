# Conditional local CFR compute scaling on the M4

The [frozen protocol](../blueprint-local-cfr-scaling.md) extended draft PR #107's
48 eligible first-hero-decision three-player flop observations. Clean source
revision `68c4f604c65dc002ec87b978dcd79cdd2cbd3f12` loaded the same saved
12M-entry blueprint once. Each case reused one committed public-range sample
for five independent traversal seeds and both targeting modes. The solver's
regret updates, continuation styles, targeting heuristic and final-iteration
action policy were unchanged. These are selected conditional situations, not
independent poker hands or a BB/100 experiment.

## Completion and target work

All **480/480** planned attempts were retained. Every attempt completed at
least 1,933 full per-player cycles, exceeding the earlier 128-cycle budget.
No attempt hit the node or memory cap or failed before the minimum 32 cycles.
`target_unvisited` means that the cycles completed but no policy for the
hero's actual information set could be returned.

| Measure | Targeting off | Targeting on |
| --- | ---: | ---: |
| Actual decision policy returned | 123/240 | **240/240** |
| Cases with a policy in all five repetitions | 6/48 | **48/48** |
| Completed cycles, median (range) | 2,951 (2,166–3,060) | 2,450.5 (1,933–2,750) |
| Visits to actual decision, median | 1 | 1,394.5 |
| Distinct hero holdings visited at target public decision, median | 533 | 466 |
| Public root-range prior mass of those holdings, median | 46.60% | 40.86% |
| Total information sets, median | 20,565 | 16,766 |
| Sampled nodes, median | 383,778.5 | 381,435.5 |

The extra actual-hand traversal reliably reaches the decision but uses work
that ordinary traversals could spend across more hero holdings. Targeting off
explored somewhat more of the public range yet missed the actual decision in
**117/240** attempts, including 50/60 attempts in the two-check prefix
pattern. The public-prior-mass measure is a sum of marginal range weights;
it is neither collision-conditioned joint probability nor opponent-action
reach. It measures where local regrets were recorded, not equilibrium quality.

The completed-cycle time snapshots show how work accumulated within this
run. All 480 attempts have snapshots at 5, 15 and 30 seconds. The table gives
medians at the first completed cycle at or after each threshold.

| Time | Targeting | Cycles | Actual-decision visits | Hero holdings at target | Public prior mass |
| --- | --- | ---: | ---: | ---: | ---: |
| 5 s | Off | 257 | 0 | 64 | 5.57% |
| 15 s | Off | 759 | 0 | 177 | 15.48% |
| 30 s | Off | 1,496.5 | 0 | 317 | 27.68% |
| ~60 s | Off | 2,951 | 1 | 533 | 46.60% |
| 5 s | On | 203 | 150 | 50 | 4.51% |
| 15 s | On | 620 | 391.5 | 148.5 | 13.05% |
| 30 s | On | 1,231 | 737.5 | 271 | 24.06% |
| ~60 s | On | 2,450.5 | 1,394.5 | 466 | 40.86% |

The ~60-second rows use the retained last fully completed-cycle state for
every attempt. Only four cycles happened to finish at or after the exact
60-second threshold (three targeting on, one off); the deadline normally
interrupts the next incomplete cycle. No incomplete cycle is counted as work.

## Policy variability and continuations

For targeting on, all 48 cases yielded five final policies. Across the ten
within-case pairs of repetitions per case, mean action-policy L1 distance
(0–2 scale) was **1.055** at the end, versus **1.176 at 5 seconds**, **1.089
at 15 seconds** and **1.110 at 30 seconds**. The final mean is somewhat lower
than at five seconds, but not steadily decreasing, and remains substantial.
Within the same targeted solve, mean L1 movement was **1.005** from 5 to 15
seconds, **0.718** from 15 to 30 seconds, and **0.718** from 30 seconds to the
last completed cycle. At fixed cycle checkpoints, mean targeted movement was
0.932 from 128 to 256, 0.870 from 256 to 512, 0.680 from 512 to 1,024,
and 0.737 from 1,024 to 2,048 among the 226 solves reaching the last pair.
Time-snapshot policies are read after each completed regret publication; the
returned action policy precedes the final publication. Both definitions give
mean within-case final L1 **1.055** to three decimals. These are current
policies, not average strategies or convergence certificates.

Targeting off supplied enough policies for 155 within-case repetition pairs;
their mean final L1 distance was 1.220. This is a selected subset because
117 untargeted attempts had no actual-hand policy. Comparing its variability
directly with the complete targeted cohort would be misleading. The 123
repetitions with both modes returning a policy had mean on/off L1 1.275;
that describes different conditional search outputs, not which plays better.

Trained blueprint probabilities were found in **4,283,623 of 118,563,729**
trained-plus-untrained continuation lookup calls with targeting on (**3.61%**),
and **4,349,860 of 117,862,019** with targeting off (**3.69%**). These are
repeated lookup calls, not distinct states. Untrained continuations still
apply the four style biases, so this fraction alone does not establish that
the search has no effect. More cycles did not resolve the continuation-model
coverage question.

## Resources, provenance and limits

The sequential process took **28,863.8 seconds (8.02 hours)**, under the
10-hour wall limit. Peak process RSS was **7.54 GiB**, under 10.5 GiB; the
largest sampled-node count in one attempt was 417,325, under 2,000,000.
Attempt duration ranged from 60.000 to 60.011 seconds; the small overshoot is
deadline-check overhead at a node or rollout boundary. The run used the M4
and no paid host. The preflight is [reported separately](local-cfr-scaling-preflight-m4.md)
and is not pooled here.

The [manifest](local-cfr-scaling-m4-manifest.json) records the clean source
revision and SHA-256 of the checkpoint, 48-case manifest and
[frozen public ranges](../../configs/blueprint/local-cfr-scaling-ranges.json.gz).
The [result JSON](local-cfr-scaling-m4-result.json),
[all 480 attempt rows](local-cfr-scaling-m4-attempts.jsonl),
[checksums](local-cfr-scaling-m4-checksums.json) and
[TensorBoard events](local-cfr-scaling-m4-events.tfevents) are versioned in
PR #107. The attempt file SHA-256 is
`eb06038f9d455adcc7e02c430a28c6a15fdbd0ee8282b145c990b6a93572d6ca`.
All four M4 artifact checksums verified, including the TensorBoard file;
the original output remains at
`/Users/dberweger/Local/local-cfr-scaling-overnight-107` on the M4.

**Conclusion:** additional CPU time greatly increases recorded range work,
and targeting remains necessary to return the actual-hand policy reliably.
The selected final policies remain variable across seeds and the continuation
table is sparsely trained at looked-up states. The next decision should isolate
one of those strategic limitations before spending on a larger blueprint or
paid compute. This experiment does not establish a playing-strength gain,
whole-game feasibility rate, or model promotion.
