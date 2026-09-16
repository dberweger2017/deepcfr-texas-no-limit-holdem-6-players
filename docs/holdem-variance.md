# Sampling variance and the regression objective

## Scope

The [first comparison](reports/holdem-sampling-comparison.md) found extreme late-decision weights despite cheap completed paths. This task tests a frozen value baseline, verifies the regression gradient, and compares a bounded first-decision expansion. It does not change the training default or authorize a strength claim.

## Baseline and branching

For an updating-player action with inclusion probability `q(a)`, use `Q(a) = b(a) + indicator(a sampled) * (V(a) - b(a)) / q(a)`. The baseline is fixed before collection and sees only that player's public candidate input. The saved model's value head supplies `b` in BB; a uniform bootstrap slot supplies zero. A poor baseline can increase variance without biasing the expectation. Opponent nodes still pass the sampled continuation unchanged: another player's value head is not a predictor of the traverser's payoff in multiplayer poker.

This is the control-variate construction from [Schmid et al., VR-MCCFR](https://arxiv.org/abs/1809.03057). We apply it only at traverser decisions; we do not claim the paper's zero-variance result for a baseline missing opponent/chance corrections.

The alternative expands every action at the first traverser decision, then samples one path in each child. The expanded actions have inclusion probability one, so their returned estimates need no action correction and later own sampling reach starts at one in each child. There are at most `A` terminal paths, where `A` is the first decision's menu size; no deeper expansion occurs. Opponents before that decision are sampled once, so the resulting paths form one correlated traversal, not `A` independent replicates. Limits still invalidate the whole traversal.

## Regression measure

For a fixed policy and deal distribution, the intended per-iteration objective sums action-wise squared value and regret errors over all traverser histories, weighted by opponent/chance reach. It excludes the traverser's own policy reach. All action dimensions are summed, as in the existing loss.

At a visited decision let `s` be own sampling reach and `r = Q - sum(sigma * Q)` the **conditional** regret estimate. Use `(sum((predicted_regret - r)^2) + sum((predicted_value - Q)^2)) / s`. Do not use `r/s` as the regression label: that would also distort the prediction term in the gradient. Average by the number of scheduled roots, including roots with no traverser decision, not the number of records or the sum of observed weights. A fixed positive iteration multiplier may be applied outside this loss.

Visitation probability is opponent reach times `s`; multiplying the conditional loss gradient by `1/s` cancels the changed visitation measure. Since the gradient of squared error is linear in the target and `Q` is unbiased, its expected gradient equals the full-tree gradient. Expected loss values need not agree: sampling variance adds a prediction-independent term. This is a gradient identity before optimizer transforms, not a claim about Adam or clipped gradients.

If `m` records are sampled uniformly with replacement from a stream of `N` records (or a uniform reservoir), use `N/m` times their weighted sum, divided by the fixed root count. Reservoir expectation is over both admission and minibatch sampling. Keep the stream count, root count and per-iteration weights in future replay/checkpoints. Mixed-iteration integration must preserve the declared per-iteration root normalization; this PR's loss primitive operates on one collection phase. No self-normalized weights, clipping, priorities, or omission of empty roots.

## Frozen local comparison

Commit this protocol before measurements. No rental or optimizer training; all inputs remain the saved one-iteration width-16 profiles.

- First pass exact checks: full enumeration of the existing two-player river fixture, both roles, uniform/nonuniform and zero-own-reach policies. Verify expected values/regrets and autograd gradients for zero, arbitrary inaccurate, and exact conditional baselines; check both single-path and first-decision expansion. Separately enumerate uniform reservoir admission/minibatches against the full weighted gradient. Check perfect-baseline variance on a tractable fixture without claiming all visitation variance disappears.
- Reuse all four saved roots from the previous comparison: scenarios 0/1/2 seed 101, traverser 0; scenario 3 seed 103, traverser 4. Completed checkpoint iteration 1, next iteration 2, sample 0. Preserve source, checkpoint, profile and rules hashes.
- Three arms: `single-zero`, `single-frozen` (256 replicates each), `first-frozen` (64 replicates). Exploration is fixed at 0.5. Single arms share the previous `outcome-half` action seeds and must execute identical paths. Expanded traversals use SHA-256 first-eight-byte seeds from `holdem-variance-v1/{job}/{replicate}`. No seed search or extensions after looking at outcomes.
- Each cell runs in a fresh process, sequentially, with a shared 50,000-node and 60-second collection-plus-gradient limit. Maximum allowance is twelve minutes across twelve cells. The gradient is evaluated at the same saved role model, without an optimizer step. Record collection and gradient time separately; setup and artifact serialization are outside that timing. A failed cell has no aggregate estimates; retain all completed rows and the error.
- Measure root-value variance, the trace of the full network-gradient sample covariance (sum of parameter variances), gradient norms, conditional target magnitudes, largest regret-update mass, inverse own-prefix reach, nodes/terminals, RSS and baseline magnitudes. Include zero gradients from empty roots. Save per-replicate summaries and the mean-gradient vector; treat each expanded traversal as one replicate. Record path hashes to verify the single-arm pairing.
- Compare gradient variance times mean collection-plus-gradient seconds per replicate. This is an empirical cost/variance diagnostic, not playing strength or a confidence guarantee for heavy tails. The raw control must reproduce the previous per-path values and actions for the declared seeds.

## Decision rule

Any expectation/gradient failure blocks integration. A candidate is eligible for a separate opt-in training/recovery pilot only if every cell completes, its cost-adjusted gradient variance is at least 20% lower on the difficult root and no more than 25% worse on any fixed-stack root. Also report the absolute tail magnitudes and sensitivity to the largest gradient sample; a fragile result is insufficient for a broad training campaign. These point-estimate limits are a diagnostic screen, not a promotion test. If neither qualifies, preserve that outcome and identify the dominant remaining source before fitting a model. Do not silently lower the limits or choose a different baseline after seeing results.
