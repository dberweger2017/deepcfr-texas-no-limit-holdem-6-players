# Persistent value critic: local protocol

## Question and boundary

Can a separately trained, persistent own-information critic reduce the cost of estimating regrets beyond both the historical value head and a cheap accounting baseline? This is a diagnostic of frozen-policy sampling. It neither changes the production trainer nor trains a stronger playing policy.

The [plan](../configs/holdem/persistent-critic.json) is committed before measurements. Run locally, one CPU thread, at most 900 seconds for the complete experiment, no rental. Preserve failures and partial output; no outcome-driven retry or additional seeds. Implementation tests use other seeds and small fixtures.

## Value objective

Use a width-32 decision encoder and action-conditioned scalar value head, with explicit physical-seat conditioning. Inputs are the acting player's observation and legal candidates only. Each target is that player's terminal net chip change from the beginning of the hand, in BB. Opponents' values are never obtained by negating this value.

At each training root sample one legal candidate uniformly, then follow the frozen continuation profile to settlement. Retain the root and subsequent on-policy decisions with each actor's own return. This covers root actions with zero policy mass without changing the continuation policy. Sample compatible worlds uniformly for the restricted river's declared two-world prior. Full-game roots use fresh deals and forced call/check prefixes; these define synthetic starting distributions, not inferred ranges following natural learned-policy bets.

Keep network weights and Adam state across two fitting phases. Use a separate circular replay of 4,096 records, uniform minibatches of 32, 128 Adam steps per phase at learning rate 0.001 and gradient clipping 1. Clear replay when either continuation profile or starting-distribution identity changes; retain parameters and optimizer. Snapshot the critic before evaluation and never update a baseline during traversal. Save and verify full phase-boundary recovery: weights, Adam, replay, cursor, sampling RNG, profile identity and counters.

This first prototype uses Monte Carlo returns, not expected-SARSA bootstrapping. It is not a reproduction of [DREAM](https://arxiv.org/abs/2006.10410), whose persistent critic uses a different information boundary and target construction. Keeping the current encoder intentionally leaves its measured generalization limitations in the comparison.

## Training and evaluation

Use critic seeds 503, 509 and 521. These are three critic initializations/data streams, **not three independently trained full-game policies**.

- **Restricted river:** retain the previous 24 contexts, train only on boards 0 and 1, evaluate all contexts with board 2 held out. Fit 256 rollouts under uniform continuation, then clear replay and fit 256 under increasing-action continuation. Evaluate the final critic against exact increasing-profile references. Preserve per-context Q errors and the held-out Ac Kd facing-bet decision. A greedy choice from critic Q is only a fixed-continuation diagnostic, never a deployed CFR strategy.
- **Full 100 BB:** use the hash-pinned original iteration-256 seed-307 current profile. Build one independent critic per starting street (flop, turn, river) to avoid combining different synthetic range problems. Each has two batches of 64 fresh-deal training rollouts under the same profile. Evaluate four disjoint fresh roots per street. Subsequent on-policy decisions may reach later streets. Evaluate sampler variance conditional on each fixed hidden deal; without exact values this is not a public-information bias estimate. Full-game unseen-card uncertainty is not measured by these conditional replicates.
- **Sampling:** first-own-decision expansion, exploration 0.5, 32 paired replicates per root and seed, maximum 20,000 nodes per traversal. Baselines: zero; known chip deficit `(remaining stack - initial stack) / BB` for every candidate; the frozen historical Q head; and the learned critic. In the restricted reference, additionally retain the oracle comparison. All baseline variants use identical policies and randomness. Require identical execution-path hashes and node counts. Facing-all-in river estimates must be exactly unchanged across baselines.
- **Coverage:** separately measure 96 naturally dealt, rotating-button hands of seed-307 current-profile self-play collection and 96 current-policy hero hands against the fixed style pool. Count collector decision records and arena hero decisions by street. Report denominators and unique collector observations. These are different distributions and correlated record counts, not effective sample sizes. Baseline changes cannot directly change frozen-policy paths or coverage.

## Cost and decision rule

Report per-cell centered root-regret trace variance, mean nodes, variance × nodes, inference/traversal wall time, data collection and fitting time, replay street counts, and model/checkpoint sizes. For exact river roots also report the Monte Carlo mean's deviation from the reference; finite-sample deviation is not proof of estimator bias. Retain individual seeds, roots and regressions rather than only aggregate improvements.

The primary screen is full-game postflop variance × seconds, including training cost amortized over a declared 100,000 future traversals of that critic. Compute each street's mean of `variance * (mean traversal seconds + that critic's data-and-fit seconds / 100000)`, then equally average the three streets. Require the learned value to be at most 0.75 times **both** accounting and historical baselines for every critic seed, with no street above 1.25 times either comparator. Zero-denominator cases pass only if the learned value is also zero. Also report results without amortization and break-even traversal counts where defined; 100,000 is an assumption, not observed usage. Wall-time screens are exploratory on a shared local machine, not confidence intervals.

Passing supports proposing an online integration comparison, not production promotion. Failing closes this prototype without another fitting sweep: use the retained error and cost evidence to choose controlled additional branching or a focused representation probe. No claim about poker strength, Nash convergence, full-game variance reduction, or coverage improvement follows from a successful restricted reference alone.

Stop on non-finite values, illegal actions, broken accounting, changed frozen policies/baselines, recovery mismatch, resource exhaustion or leaked player information. Retain raw JSONL, checkpoints with hashes, configuration, source/environment provenance and a complete or explicitly failed report under an ignored results directory; commit compact reports.

## Running and auditing

From the repository root, with the retained hash-pinned training checkpoint and its original manifest available:

```bash
python -m scripts.check_persistent_critic --out results/persistent-critic
python -m scripts.check_persistent_critic --verify results/persistent-critic
```

The output directory must not exist. `fit-phases.jsonl` preserves each completed phase even if a later phase fails. `samples.jsonl` contains all replicate-level estimates and execution hashes; `coverage.jsonl` contains the separate collection/arena visitation records. Phase checkpoints retain the complete critic recovery state. The verifier checks artifact hashes, recomputes sampling moments and the cost screen, and reloads every checkpoint. The experiment additionally verifies one further Adam update from each original and recovered phase state; tests cover fresh-process continuation.

The full-stack probe roots are first-to-act postflop positions after a six-way limped/check-through prefix, with a 6 BB pot. Their buttons rotate over the four evaluation roots. This is a deliberately narrow starting distribution; it does not cover all natural raise histories or facing-bet roots. The separate coverage measurement rotates physical seats while keeping the hero on the button in both distributions. It is a button-position comparison, not a balanced sample of all relative positions.

Wall times include the implementation's frozen-state integrity checks and use a fixed baseline evaluation order. The historical-head wrapper checks its policy and value source separately even when both refer to the same profile. These overheads, local contention and cache order limit fine-grained cost conclusions; retain variance × nodes alongside timings. No claim of a production throughput advantage should be based on this screen alone.
