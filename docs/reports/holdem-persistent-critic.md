# Persistent critic comparison

**Completed locally; the learned critic fails the predeclared cost screen on all three seeds.** Keep the production trainer unchanged. The next collector experiment should test selective additional branching at measured cost and extend the coverage audit beyond the button.

Results of the [committed protocol](../holdem-persistent-critic.md); [complete compact measurements](holdem-persistent-critic.json).

## Outcome

- All **12 critics / 24 fitting phases**, **504 sampling cells / 16,128 sampled estimates**, 24 exact references and both 96-hand coverage checks completed in **731.28 seconds (12.19 minutes)**, below the 900-second cap. No failures, retries or rental spend.
- All paired execution hashes and node counts agree. All final-expanded facing-all-in estimates remain exactly unchanged across baselines. Each of the 24 checkpoints reproduces the next Adam update, model parameters and sampling RNG from its saved phase state.
- Learned-critic full-stack cost ratios against accounting are **1.166, 1.051 and 1.029**; lower is better. None meets the required ≤0.75 against both accounting and historical baselines. Seed 503 also violates the per-street regression limit on flop.
- Restricted-river variance decreases, but much of the oracle advantage remains unrealized. All three diagnostic critics still choose the costly held-out Ac Kd call.
- The current-profile collection and style-pool play distributions differ sharply in this button-position sample: **0 postflop collector records versus 51 postflop hero decisions**.

## Primary full-stack cost screen

Equal weighting of flop, turn and river; learned training cost is amortized over the declared 100,000 future traversals per critic. Ratios compare the means of variance × seconds, not the mean of individual ratios.

| Critic seed | Learned / accounting | Learned / historical head | Screen |
| --- | ---: | ---: | --- |
| 503 | 1.166 | 1.009 | Failed |
| 509 | 1.051 | 0.933 | Failed |
| 521 | 1.029 | 0.903 | Failed |

The variance-only direction is mixed too. Against accounting, seed 503’s flop variance × nodes is 76.3% worse, while seeds 509 and 521 improve there by 12.2% and 7.9%. Seed 521 improves river by 16.2%; the other two river comparisons are slightly worse. The retained per-root cells include every regression.

The 100,000-traversal reuse assumption is generous: the existing six-player recipe collects 32 roots per role, or 192 roots per profile. Break-even counts against each comparator are retained where the measured per-traversal savings are positive. Persistence alone does not establish that values remain accurate as profiles change.

## Retained river references

These are unopened-river means across the 12 contexts and three critic seeds, under the increasing-action profile. They use different sampling seeds and only one final profile, so they are not a rerun of the previous two-profile 84% figure.

| Baseline | Variance × nodes | Variance × seconds |
| --- | ---: | ---: |
| zero | 281.661 | 0.063592 |
| accounting | 296.955 | 0.066476 |
| historical | 280.455 | 0.112489 |
| learned | 218.327 | 0.069745 |
| oracle | 30.754 | 0.007274 |

The learned critic reduces variance × nodes by **22.5% versus zero**, compared with **89.1% for the exact oracle**. Its measured traversal cost erases that advantage over zero in this restricted suite. The simple accounting baseline does not explain the oracle result here. Oracle timings exclude exact-tree construction and lookup storage; the oracle is a diagnostic reference, not an available low-cost baseline.

Root Q prediction is already weak on the training boards; this is not merely a held-out transfer failure:

| Critic seed | Training-root Q MSE (BB²) | Held-out-root Q MSE (BB²) | Ac Kd facing-bet choice / cost |
| --- | ---: | ---: | --- |
| 503 | 10.565 | 16.935 | Call / 1 BB |
| 509 | 11.097 | 18.208 | Call / 1 BB |
| 521 | 10.435 | 19.508 | Call / 1 BB |

The Ac Kd reference is −1 BB for folding and −2 BB for calling. All three critics predict positive values for both actions and prefer calling. Training uses noisy returns across all acting roles and only 128 fitting steps per phase; these results do not isolate persistence, Monte Carlo targets, representation, capacity or fitting budget as the cause. They reject integrating this particular small prototype.

## Coverage remains a separate problem

| Street | Self-play collector records | Hero decisions against style pool |
| --- | ---: | ---: |
| preflop | 96 | 110 |
| flop | 0 | 22 |
| turn | 0 | 15 |
| river | 0 | 14 |

Each distribution contains 96 naturally dealt hands. Collector records cover 96 unique observations; arena play contains 161 hero decisions, including 51 postflop (31.7%). Hero remains on the button while its physical role rotates. Counts are not independent samples, and this does not establish the distribution for every seat or training iteration.

Changing the baseline cannot change these frozen-policy paths. Additional branching can improve estimates where further own decisions exist; it cannot manufacture postflop decisions after all-in continuations. Extend the audit across all relative positions and earlier snapshots before treating one intervention as a coverage fix.

## Resources and artifacts

The critics contain **25,761 parameters** each. Total data collection plus fitting was **73.72 seconds**; paired traversal calls took **341.10 seconds**. Phase recovery verification took **163.26 seconds**. Overall wall time also includes the historical checkpoint load, exact references, serialization, coverage and reporting.

Artifacts occupy approximately **514 MB** locally under `results/persistent-critic/`. They include raw sampled estimates, coverage actions, phase logs and 24 full critic checkpoints; hashes are in the compact JSON. They are not publicly hosted. The original seed-307 checkpoint is retained separately under `results/frozen-fitting/inputs/` and hash-pinned by the plan.

Measured source revision: `fd34844239c47f70ade5fd02e8ffc1aa00061b42`. Protocol was committed before this revision. Source fingerprint: `2f365a85a4f1284b9bdb114e81a43e4f729f9d159955991f10c68789343193d9`. The artifact verifier was added after the run; it does not change the measurements.

## Decision and next task

Keep this critic as a diagnostic control. Do not integrate it or fund a larger version from these results. Complete a controlled additional-branching comparison on the retained references and full-stack probes, reporting variance at equal cost, and audit coverage across all positions and selected earlier profiles. That should lead to one explicit collector change for fresh online confirmation. A coverage intervention must state which policy distribution and regret target it changes; simply collecting more of the same all-in hands is insufficient.

This result does not reject Deep CFR, a persistent expected-SARSA critic, or a training-only history-aware baseline. It shows that our inexpensive own-information Monte Carlo critic does not recover enough of the measured oracle headroom to justify integration.

## What this experiment measures

The playing policies remain frozen. A separate value network receives only the acting player’s observation and legal candidates, with physical-seat conditioning. It learns Monte Carlo terminal net payoffs, retaining weights and Adam across two fitting phases. Replay is cleared when the continuation profile changes. It is an own-information Monte Carlo prototype, not DREAM or a new self-play training recipe.

Three critic seeds face the same retained river contexts and the same original iteration-256 seed-307 current profile in 100 BB probes. They are not three independent full-game training seeds. The river study integrates its declared two-world prior and supplies exact conditional values. Full-stack variance is conditional on a fixed hidden deal at each root; it does not measure hidden-deal uncertainty or full-game bias.

Full-stack roots start first to act after six-way call/check prefixes with a 6 BB pot. Evaluation deals are disjoint from training deals. Baseline arms share policies, action seeds and hidden worlds; all execution hashes and node counts must agree. Baselines affect estimates, not visitation. The separate coverage comparison keeps the hero on the button while rotating physical seats.

## Cost interpretation

The predeclared screen includes training cost amortized over 100,000 future traversals per critic. That reuse count is an assumption. Variance × nodes and timings without amortization are retained alongside the screen. Timings include frozen-state integrity checks and fixed arm order; the historical wrapper checks the shared policy/value source twice. Cache order, these checks and local contention prevent treating small timing differences as production speedups.

No model promotion, production trainer change or rental is part of this experiment. The held-out Ac Kd check is a greedy choice from the diagnostic critic’s Q values under a fixed continuation policy; it does not replace or repair the regret network’s deployed policy.

## Validation

All **610 repository tests pass**, including 13 critic/study checks covering own-player payoff accounting, unchanged sampled paths, hidden-world invariance, replay replacement, profile changes, failure handling, cost accounting, artifact verification and recovery in a fresh Python process. The actual study verifier confirms all 504 cells, 3,456 paired replicate groups and 24 checkpoints; the recomputed cost screen remains failed. Lint and local documentation-link checks pass.
