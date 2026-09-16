# Sampling comparison

## Question and limits

Can a single sampled terminal path remove the collection bottleneck without giving up correct counterfactual updates? This is a diagnostic comparison, not a change to the trainer or a strength test. The existing external sampler remains the training default.

The [collection profile](reports/holdem-collection-performance.md) retains a six-player unequal-stack root that did not finish after 109,901 nodes. Do not discard it, truncate its payoff, or infer its full value from a completed prefix.

## Estimator

At an updating player's decision, sample action `a` with `q(a) = (1 - epsilon) * sigma(a) + epsilon / A`, where `0 < epsilon <= 1`. Sample opponents from their frozen policies. Finish one actual hand; no leaf evaluator or artificial horizon is used. One path bounds branching, not hand length. Node/time limits still invalidate an unfinished sample.

Working backward from terminal net payoff in BB, let `V` be the estimated continuation value after the sampled action. Set `Q(a) = V / q(a)` for that action and zero for the others; the policy value is `sum(sigma * Q)`. Pass that policy value to the preceding node. Opponent nodes pass the continuation value unchanged. These are random estimates, not assertions that unsampled actions have zero payoff.

At each updating-player decision also retain `s`, the product of that player's earlier sampling probabilities since the supplied root. Emit regret **update mass** `(Q - sum(sigma * Q)) / s`. Opponent reach is already represented by how often the decision is visited. Absent decisions contribute zero to an unconditional expectation. Dividing only by the current action's probability would miss the earlier own-action sampling correction.

For any fixed history, path probability times its emitted update cancels all own sampling probabilities; summing suffixes leaves opponent reach times the counterfactual action regret. Summing histories in the same information set and averaging fresh deals gives the corresponding counterfactual update. This identity also covers branches with zero own policy reach because `q` stays positive. It does not prove convergence of a fitted neural learner or equilibrium convergence in six-player poker.

The approach follows the outcome-sampling construction in [Lanctot et al., 2009, sections 3–4](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf). Our implementation keeps own-prefix correction separate from conditional continuation values. The record is deliberately distinct from `CandidateTargets` and cannot enter the current replay pipeline: sampled visitation changes the regression objective. Integration needs an explicit loss/visitation measure, iteration normalization, checkpoint schema and recovery tests. No clipping, self-normalization, or learned variance baseline is introduced here.

## Frozen local protocol

Committed before measurements. No fitting, rentals, reserved evaluation seeds, or model promotion.

- Exact check: exhaustively enumerate sampling outcomes on a fixed-deal heads-up river tree with stacks `(3, 3)`, deal seed 7. Check every updating-player decision, including absent decisions and zero-own-policy-reach branches, against independent full-tree counterfactual updates. Test uniform and nonuniform policies, exploration 0.5 and 1, and both roles. Compare exact second moments as well as means on the uniform fixture.
- Saved roots: `results/holdem-baseline-v1`, completed iteration 1, sample 0, scenarios 0/1/2 seed 101 traverser 0, and scenario 3 seed 103 traverser 4. Reconstruct the next iteration's original deal, button and frozen profile. Preserve checkpoint/profile/source hashes.
- Arms: existing external sampling (32 independent traversals), outcome sampling with epsilon 0.5 and 1 (256 independent paths each). Seeds are the first eight SHA-256 bytes of `holdem-sampling-comparison-v1/{job}/{arm}/{replicate}`. Streams are reproducible but not paired paths across algorithms.
- Each arm/root cell: at most 50,000 visited nodes and 30 seconds of collection, shared across all requested replicates. A failure invalidates the cell's estimates; retain attempted/completed counts and error. No averages from surviving prefixes. Record collection time separately from setup/reporting and fresh-process peak RSS (includes setup).
- Summaries: complete-cell root-value mean, sample variance and standard error; nodes, terminal paths, decision records; own-prefix inverse weights and largest absolute regret updates. For a root that itself belongs to the traverser, retain action-wise means/variances and sampled action counts. Retain per-replicate summaries, not just aggregates. Outcome samples can have heavy tails; a small observed standard error is not a guarantee of accuracy.
- Run one cell per fresh process, sequentially. Maximum collection allowance across 12 cells: six minutes; no cell extensions after seeing results. The earlier failed large-root profile remains evidence, not an estimate of its exact value.

## Decision rule

Exact expectation checks must pass before the saved-root comparison. Report resource completion and variance together; a fast path is not equivalent to a full external traversal. Do not rank playing strength. If outcome sampling is computationally practical but has large correction weights/variance, recommend a variance-reduction comparison or a carefully weighted integration pilot before substantial training. If it fails correctness or resource limits, preserve the failure and keep the existing sampler.
