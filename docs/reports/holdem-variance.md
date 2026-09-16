# Sampling variance and regression results

[PR #68](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/68)

## Decision

**Advance first-decision expansion with a frozen value baseline to a separate opt-in training/recovery pilot.** It passes the predeclared diagnostic screen on all four roots. The value baseline alone fails the difficult-root improvement requirement. The training default remains external sampling; no model was fitted or promoted here.

This is evidence for a practical sampler candidate, not for playing strength. The difficult-root improvement is highly sensitive to rare samples. Keep the pilot short, retain every seed/failure, and measure learning, update tails and reproducible recovery before considering a larger campaign.

## What changed

The diagnostic collector can now expand every action at the first traverser decision, then follow one sampled continuation per action. This bounds terminal paths by that decision's menu size without truncating hands. The continuations remain correlated and count as one replicate.

A frozen own-seat value head supplies an optional action baseline; only sampled residuals receive action-probability correction. Opponents' own value heads are never substituted for the traverser's payoff. Baselines receive the same public candidate inputs as the policy and cannot change during collection.

The new loss primitive keeps conditional value/regret estimates as labels and applies inverse own sampling reach to the **loss**. It normalizes by scheduled roots, including empty ones, and corrects uniform replay minibatches with the total stream size. Treating update mass as the regression label or normalizing by the observed weight sum would change the expected gradient. The primitive covers one collection phase; persistent mixed-iteration replay integration is the next task.

## Correctness

Full enumeration checks both players, uniform/nonuniform and zero-own-reach policies, zero/inaccurate/exact baselines, and single-path/expanded sampling. Expected returns, regret updates and autograd gradients match an independent full-tree recurrence to absolute `1e-12`, with relative tolerance disabled. Predictions share parameters across decisions, so the test covers how sampled targets affect shared model parameters.

A perfect baseline removes action-sampling variance in a terminal-decision fixture. That does not imply zero variance when opponent actions or decision visitation remain sampled. Uniform reservoir admission and replacement minibatches are separately enumerated against the full, root-normalized gradient. The tests explicitly distinguish weighting a loss from scaling its target and retain empty roots in normalization.

The original sampler at `e754059` reproduces the 1,024 saved control samples. The replacement's zero-baseline path matches their values, nodes, largest regret updates and reconstructed action hashes. Both single-path arms also share identical action traces and own-prefix weights. Frozen baseline predictions preserve physical-seat ownership and hidden-world/suit invariance.

**87 focused checks and all 534 repository tests pass** (full suite: 120.11 seconds). Ruff and diff checks pass. Training, replay persistence and inference formats are unchanged.

## Frozen experiment

[Protocol](../holdem-variance.md), committed at `034f9f8` before measurements. Implementation `d0929f2`; [full JSON](holdem-variance.json) records exact source/checkpoint/profile/engine hashes and environment. All cells used the same source fingerprint. The dirty flag records unrelated local agent directories; the source fingerprint is separate.

Apple M1, CPU, one Torch thread, sequential fresh processes. All twelve cells finish within the declared 50,000-node / 60-second ceiling. Total cell time: **21.765 seconds**, excluding checkpoint/environment setup and final serialization. No optimizer steps or paid resources.

Each scenario uses its retained one-iteration width-16 model and fixed deal. Four/five/six-player fixed-stack scenarios use 100 BB. The difficult six-player scenario uses 20/40/60/100/150/200 BB. These are diagnostic roots, not representative independent training runs or arena hands.

## Gradient variance per unit of work

The metric is the sum of parameter-gradient sample variances multiplied by mean collection-plus-gradient seconds per replicate. Lower is better. It accounts for expanded traversals costing more than single paths; it does not count their correlated continuations as independent data. Single arms use 256 replicates, expansion uses 64.

| Root | Frozen baseline alone / raw control | First-decision expansion + baseline / raw control | Reduction with expansion |
| --- | ---: | ---: | ---: |
| Four players, 100 BB | 1.028 | 0.164 | 83.6% |
| Five players, 100 BB | 1.131 | 0.125 | 87.5% |
| Six players, 100 BB | 1.010 | 0.404 | 59.6% |
| Six players, unequal stacks | 1.030 | 0.00746 | 99.25%* |

The screen required at least 20% improvement on the difficult root and no more than 25% regression on any fixed-stack root. Only expansion qualifies. The saved baseline modestly reduces raw variance, but its extra inference cost outweighs that benefit when used alone. This experiment does not establish the best exploration setting or isolate expansion with a zero baseline.

### Why the difficult-root percentage needs caution

| Measure | Single path, zero baseline | Single path, frozen baseline | First-decision expansion, frozen baseline |
| --- | ---: | ---: | ---: |
| Completed independent traversals | 256 | 256 | 64 |
| Terminal continuations | 256 | 256 | 576 |
| Visited nodes | 2,092 | 2,092 | 5,004 |
| Collection + gradient seconds | 1.652 | 1.790 | 2.949 |
| Largest inverse own-prefix reach | 60,480 | 60,480 | 280 |
| Largest absolute regret update, BB | 8,696,308 | 7,895,551 | 160,802 |
| Largest gradient norm | 2.913 billion | 2.839 billion | 24.785 million |
| Variance contribution removed with largest-gradient sample | 99.71% | 99.68% | 26.08% |

\* The original raw control is dominated by one rare sample. As a sensitivity calculation only, excluding the largest-gradient sample from **each** arm changes expansion's difficult-root cost ratio from `0.00746` to **1.90**, reversing the ranking. All original samples remain in the result and screen; none were discarded or clipped. This shows why the headline reduction cannot justify a broad training campaign. The fixed-stack improvement survives the same sensitivity calculation, with ratios 0.163, 0.125 and 0.434.

Large updates remain possible. A frozen continuation baseline cannot eliminate own-prefix visitation variance, and first-decision expansion removes only the first own-action sampling factor. The next pilot must investigate the actual training behavior rather than extrapolating from this small frozen-policy sample.

## Next implementation

Add an explicit opt-in sampler to the trainer, with typed replay records carrying conditional targets, own reach, scheduled root counts and collection provenance. Preserve linear iteration weighting and uniform-reservoir normalization across phases; checkpoint the corresponding counts and random states. Keep whole-iteration rollback and collection-aligned strategy snapshots. Validate uninterrupted versus fresh-process resumed training before a bounded multi-seed Hold’em pilot with a fixed evaluation schedule. Keep the current external sampler as a reference and retain failed jobs.

This is the next coherent PR. The collector comparison and the one-phase gradient identity do not complete that integration. Milestone 4's meaningful learning exit remains open.

## Artifacts and budget

The [JSON](holdem-variance.json) includes all cell summaries, parameter ordering, the screen and sensitivity calculations, paired-control checks, and raw report/mean-gradient hashes. Every completed replicate remains in the local reports. Mean vectors are NumPy arrays loadable with `allow_pickle=False`.

Local archive: `results/sampling-variance.tar.gz`, **450,685 bytes**, SHA-256 `068c17d27d5b6eee2a1cd58210b1a661005f6015c68ee34ac9d7c2e6d71f33c5`. It contains the twelve raw reports and mean-gradient arrays, the old collector source used for reproduction, action hashes, logs and reproduction/summary helpers. Input checkpoints remain under `results/holdem-baseline-v1`; the JSON pins their hashes and historical manifest. These ignored artifacts must be copied separately to another machine; no public artifact host was created.

No rental spending. Conservative CPU authorization remains **$7.33** of the original $10.
