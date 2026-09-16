# Hold’em sampling comparison

## Decision

**Outcome sampling removes the observed branching bottleneck, but this raw estimator is not ready for neural training.** All eight sampled-path cells complete. The difficult unequal-stack root still defeats external sampling, while the sampled alternatives generate very large late-decision correction weights. Keep the training default unchanged.

The next proposed PR should compare variance reduction under an explicit replay/loss contract, starting with a frozen action-value baseline and retaining this zero-baseline estimator as the control. Check expected updates and expected regression gradients exactly, then measure variance per unit work on these same roots. A baseline can reduce continuation noise; it does not remove the own-prefix visitation correction. If that correction still dominates, compare limited early branching before integrating. Do not clip weights or quietly omit rare decisions. This is one bounded implementation decision before an end-to-end Hold’em pilot, not another small-game hyperparameter campaign.

Milestone 4's meaningful-learning exit remains open. No fitting, policy promotion, professional-strength evidence, rental, or GPU spending occurred.

## What ran

The [protocol](../holdem-sampling-comparison.md) was committed at `670dd53` before measurement. Implementation: `ee43c8b`. The [machine-readable report](holdem-sampling-comparison.json) records full revisions, source/engine/environment hashes, checkpoint manifests, cell limits and summaries. All cells use the same source fingerprint. The working tree is marked dirty because unrelated local agent directories and documentation edits existed; the measured source is identified separately.

Apple M1, CPU inference with one Torch thread, one fresh process per cell, sequential execution. Collection time excludes checkpoint loading, environment inspection and report serialization. Peak RSS includes setup and the loaded checkpoint. Requested work: 32 external traversals or 256 sampled paths per cell, under the same 50,000-node / 30-second ceiling. Total measured collection time: **41.714 seconds**. Setup makes total command time longer.

Each saved model has completed only one short training iteration and uses width 16. These are regression roots, not representative trained policies. Deals, button rotation, profile hashes and checkpoint bytes are fixed; the action random streams are independent across arms. This experiment does not compare identical DFS paths with the previous profiling experiment.

## Correctness and exact variance

Full enumeration of a heads-up fixed-deal river tree checks every updating-player decision against an independent exhaustive recurrence. Tests cover both roles, uniform/nonuniform policies, zero-own-policy-reach branches and exploration 0.5/1. Absent decisions count as zero update mass. Expected returns and counterfactual regret updates agree within an absolute tolerance of `1e-12`, with relative tolerance disabled.

For the uniform fixture, both exploration settings induce the same distribution. The table gives exact moments, not Monte Carlo confidence estimates:

| Traverser | Estimator | Expected visited nodes | Sum of per-action update variances (BB²) |
| --- | --- | ---: | ---: |
| 0 | External | 4.000 | 1.3843 |
| 0 | Outcome | 2.778 | 2.8148 |
| 1 | External | 6.667 | 1.7454 |
| 1 | Outcome | 2.778 | 10.5926 |

The maximum expectation discrepancy in this table is `1.67e-16`. The variance sum spans all traverser histories/actions, including decisions absent from a sample. Fewer nodes do not imply a more statistically efficient update: for traverser 1, outcome sampling uses about 42% as many nodes but has about 6.1 times the summed update variance. The proof identity in the protocol explains the correction; these finite tests do not prove a neural convergence theorem.

## Saved-root measurements

All fixed-stack roots use 100 BB, checkpoint seed 101 and traverser 0. The unequal-stack root uses seed 103, traverser 4 and stacks 20/40/60/100/150/200 BB. “Half” mixes the current policy and uniform exploration equally; “uniform” samples the traverser's actions uniformly. Opponents always use the frozen current policies.

| Root | Arm | Completed / requested | Nodes in completed samples | Collection seconds | Process peak MiB | Root-value sample SD / √N (BB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 players | External | 32 / 32 | 1,645 | 0.565 | 293.7 | 5.209 |
| 4 players | Half | 256 / 256 | 1,374 | 0.662 | 296.3 | 5.645 |
| 4 players | Uniform | 256 / 256 | 1,490 | 0.726 | 295.4 | 16.588 |
| 5 players | External | 32 / 32 | 1,514 | 0.636 | 286.8 | 6.975 |
| 5 players | Half | 256 / 256 | 1,896 | 1.062 | 286.1 | 10.357 |
| 5 players | Uniform | 256 / 256 | 1,988 | 1.080 | 288.2 | 28.429 |
| 6 players | External | 32 / 32 | 2,574 | 1.259 | 259.4 | 14.444 |
| 6 players | Half | 256 / 256 | 2,455 | 1.432 | 254.8 | 7.107 |
| 6 players | Uniform | 256 / 256 | 2,660 | 1.521 | 250.2 | 15.330 |
| Unequal stacks | External | **0 / 32** | 0* | **30.065, invalid** | 420.7 | — |
| Unequal stacks | Half | 256 / 256 | 2,092 | 1.166 | 280.3 | 6.078 |
| Unequal stacks | Uniform | 256 / 256 | 2,623 | 1.540 | 280.8 | 16.164 |

\* The failed first external traversal visited **33,945 nodes** before the deadline. They are retained in the error, not counted as completed samples. It yields no payoff/variance estimate. The earlier [109,901-node incomplete probe](holdem-collection-performance.md) remains unresolved. There is no full-root speedup ratio to report.

The final column describes the root return only. It is not a win rate, strength comparison, or guarantee of estimation accuracy. Rare unseen tails can make an empirical standard error optimistic. Half exploration has a lower observed root-value standard error than uniform in these cells, but this small diagnostic does not establish an optimal exploration setting. Root-action means, variances and coverage counts are in the JSON where the root actor is the traverser.

### The important failure mode: late decisions

| Unequal-stack arm | Largest observed inverse own-prefix sampling reach | Largest absolute regret update (BB) |
| --- | ---: | ---: |
| Half | 60,480 | 8,696,308 |
| Uniform | 345,744 | 164,574,144 |

These are importance-corrected **updates**, not terminal winnings. Actual terminal payoffs still obey chip accounting. The root-value estimates can look relatively stable while a rare later decision receives an enormous update. Directly putting these records into the existing uniform replay with its current loss would also change the objective: unbiased update mass does not by itself make conditional regression on visited decisions correct.

The new diagnostic uses a distinct record type and schema. `collect_phase`, the trainer, replay, fitting, average policy and checkpoint format are unchanged. No partial path produces a target; a failed cell retains its completed rows for diagnosis but has no aggregate estimate.

## Validation and artifacts

The 33 focused checks cover exact means/second moments, neural path reproduction, global RNG isolation, public policy inputs, immutable roots/profiles, actual legal actions, terminal payoffs, invalid settings, resource failure, saved-checkpoint reconstruction and failed-cell accounting. All **487 repository tests pass** in 117.08 seconds; Ruff and diff checks pass.

Raw cell reports retain every completed replicate's seed, node/decision counts, value, largest correction/update and root-action estimate. The compact JSON links their paths and hashes. Local archive: `results/sampling-comparison.tar.gz`, 73,982 bytes, SHA-256 `f2d0bbc7dea25ef1c2bb554405ee754126f2c7d89a98a1f47918b989b916720f`. It includes all twelve raw reports, the run log, exact moments and the small reproduction helpers. Saved input checkpoints remain under `results/holdem-baseline-v1`; hashes and the historical manifest are in the JSON. These ignored local artifacts must be copied separately when reproducing on another machine; no public artifact host was created.

No rental was started. Conservative CPU authorization remains **$7.33** of the original $10; no paid resources are retained by this experiment.
