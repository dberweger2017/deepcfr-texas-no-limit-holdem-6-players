# Sampled Hold'em training

## Decision

The [variance comparison](reports/holdem-variance.md) supports a bounded training pilot of first-decision expansion with frozen value baselines. Enable it explicitly with `training.sampler: "first-decision"` and `training.exploration: 0.5`. Plans without a sampler keep the external-sampling reference. Both paths retain actual collection profiles for snapshot-average play.

For each scheduled traverser root, expand every legal candidate at the first own decision; sample one continuation at subsequent own decisions with `(1 - exploration) * policy + exploration / action_count`. Opponents sample their frozen policies. Baseline values come from the traverser's frozen public-observation value head; the initial uniform profile uses zero. Whole roots and phases either complete or fail. Limits never become synthetic payoffs.

## Optional second-decision expansion

`training.sampler: "second-decision"` expands the first two own decisions on each path, then uses the same exploration mixture and frozen baseline. The [branching diagnostic](reports/holdem-collector-branching.md) supports testing this mode in fresh training; it does not establish stronger play. First-decision remains the default.

Replay admission checks each decision against the number of preceding own actions on that branch. Expanded edges have inclusion probability one. The objective below is unchanged: extra records enter the admitted-stream count, while the denominator remains scheduled roots, including empty roots. Checkpoints retain the sampler in the closed configuration record; loading checks the stored decisions' expansion depth against that setting. Both modes retain the same snapshot alignment and fresh-per-fit optimizer semantics.

Per-iteration timing records additionally contain `collection_coverage`: physical role, position relative to the button, scheduled roots, roots with at least one postflop decision, record counts by street and visited nodes. These counters are observations of collection, not independent samples or inputs to learning. More records from one root do not imply broader experience.

## Replay and objective

Sampled replay stores conditional value/regret estimates, own sampling reach, action inclusion probabilities, baseline values, collection provenance and the number of scheduled roots for that role and iteration. It cannot mix with complete-branch replay. Iteration reports preserve scheduled root counts even when roots produce no decisions; inactive physical seats have zero roots.

For training iteration `T`, full admitted stream size `N`, and a uniform replay minibatch of size `m`, minimize:

```text
N / m / (T * (T + 1) / 2)
    * sum_records [record.iteration / record.roots
                   * conditional_squared_error / record.own_sample_reach]
```

`conditional_squared_error` sums both heads' squared errors over candidate actions. `N` is the total admitted stream, not reservoir capacity. Empty-root iterations still count in the fixed linear-weight denominator. Weights multiply regression errors, never labels; there is no normalization by a random observed weight sum. This estimates the linearly weighted mean of per-iteration, root-normalized counterfactual regression gradients. The earlier exact-tree tests establish the conditional estimator identity; the mixed-iteration test enumerates reservoir subsets and minibatches against the direct gradient, including an empty third iteration.

Fresh fits still initialize a new network and Adam per role. The existing gradient norm limit of 1 remains; sampled fit reports now retain the pre-clipping maximum norm and clipped-step count. That optimizer transformation does not preserve the unbiased-gradient identity. Large targets and weights are not clipped or discarded. Raw noisy loss includes sampling variance and is not a direct measure of fitting error or poker strength.

## Recovery

Sampled checkpoints use `holdem-sampled-training-v1`; external checkpoints and inference exports retain their existing formats. The closed record registry explicitly admits the sampled configuration, reports and records. Loading checks format/config agreement, per-iteration roots, replay ownership/types, counters, RNG state, profile chain and hashes. Completed iterations publish atomically after every role fit succeeds. A failed later fit leaves replay, reservoir randomness, models and archive unchanged.

Tests cover 4/5/6-player recovery with actual reservoir replacement, empty roots and sparse roles, corrupted normalization, collection failure, later-role rollback, and byte-identical recovery in a fresh process. The CLI also compares uninterrupted, paused/resumed and reproduced sampled runs, including arena outcomes and inference bytes.

## Frozen pilot protocol

Freeze this document and [the executable plan](../configs/holdem/sampled-pilot.json) before measurements. This is an engineering/resource pilot for the longer v0.5 research experiment, with no minimum poker win rate.

- Three independent training seeds: **211, 223, 227**.
- Four scenarios: four/five/six players at 100 BB; six players at 20/40/60/100/150/200 BB. Same no-rake rules and bet menu as the reference.
- **6 iterations**, **4 roots per physical role per iteration**, exploration **0.5**. Separate replay capacity **256** per role; width **32**, **16** Adam steps, batch **16**, learning rate **0.001**, **32** fixed diagnostic records per fit.
- Phase limit **50,000 nodes**; whole collect/fit iteration **60 seconds**. Entire experiment **1,200 seconds** locally, no rental. Save and evaluate at iterations **3 and 6**.
- Evaluation uses **30 independent deal blocks** per checkpoint, paired against uniform-candidate play and rotating seats, with seed **91861** and the five existing style opponents. Both checkpoints are reported. They share an evaluation schedule and are not independent tests; there is no best-checkpoint selection.
- Retain every scheduled seed, failure and unattempted job. No retries or extensions based on results. A failure stops this bounded runner and leaves its complete recovery boundaries and failure manifest.
- Report all scenario/seed results, intervals, invalid actions, nodes, scheduled roots, retained/seen replay, maximum inverse reach and regret-update magnitude, gradient norms/clipping frequency, runtime and artifact size. Verify every saved checkpoint hash; re-export and re-evaluate every final model after loading.
- Verification has a separate **600-second** local ceiling and may not add optimizer steps. Stop and retain an incomplete verification if exhausted. Tests establish fresh-training recovery separately; this pilot's verification evaluates saved models only.

```bash
python -m scripts.train_holdem --plan configs/holdem/sampled-pilot.json \
  --out results/sampled-pilot
```

Proceed to a longer, pre-budgeted exploratory training run if the pipeline remains numerically valid, produces legal evaluations, recovery checks hold and observed costs are manageable. Weak or inconclusive win rates are data, not a veto. If numerical or resource failures occur, repair their concrete cause before expanding the run. The pilot does not select a network size, establish convergence, promote a default model or satisfy v1.0's professional-strength requirements.
