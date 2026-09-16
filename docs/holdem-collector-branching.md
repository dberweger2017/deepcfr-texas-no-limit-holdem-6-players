# Additional branching and coverage: local protocol

## Decision

Test whether expanding one additional own decision reduces estimation error at equal computational cost. Separately audit whether the frozen policies generate useful postflop decisions across positions and training history. No fitting, production-default change or rental is part of this study.

The [plan](../configs/holdem/collector-branching.json) is committed before measurements. Run one CPU thread, one local job, at most 900 seconds including input loading and reporting. Keep failures and partial output; no outcome-driven retries, extra roots or seeds. Small implementation tests use separate seeds. Abort on invalid actions, non-finite estimates, changed frozen inputs, accounting failure or a resource limit.

## Collector change

Control expands the first traverser decision on each path; candidate expands the first two. Later traverser decisions retain the original exploration-0.5 mixture and baseline residual correction. Opponents follow the same fixed policy. An expanded edge has inclusion probability one; only sampled own edges multiply the own sampling reach. Child values still back up under the playing policy, not the exploration distribution. Every stored regret update retains the inverse own-prefix correction.

Exhaustively enumerate small-tree action sampling to check expected root values and **every** counterfactual update, including later decisions, zero policy reach and nonzero baselines. Check the unchanged default against pre-change numerical/path fingerprints. This is bounded additional branching, not a change to the regret target or a subset-action estimator.

## Matched probes

- Retain the previous 24 finite river contexts and increasing-action profile. Integrate the same declared two-world prior; enumerate exact targets. Compare zero, accounting, historical, learned and oracle baselines at both branching depths, with 16 replicates for each sampling seed 601/607/613.
- Retain the previous four forced call/check roots on each of flop, turn and river at 100 BB, using their original deal-generation plan. Continue under original seed-307 iteration-256 current policies. Compare zero, accounting, historical and learned baselines at both depths with eight replicates per sampling seed. Full-root results remain conditional on each fixed hidden deal, not estimates of hidden-deal uncertainty or full-game bias.
- The learned control is the already failed seed-503 critic for each suite, selected as the first declared critic seed, hash-pinned and never refitted. Sampling seeds do not represent independent trained poker models. The historical head remains the playing profile's Q predictor.
- Use the same action seed across depths, but do not claim identical or strongly coupled trajectories: branching consumes additional draws. At each fixed depth require identical paths and node counts across baseline arms. Rotate arm order by replicate to reduce timing-order effects. The fully expanded final-decision river control must remain unchanged across depths and baselines.

Report each root/seed/baseline/depth: centered root-regret trace variance, exact-reference error where available, nodes, wall time, variance × nodes and variance × seconds. These products estimate the variance of an average at a common asymptotic compute budget; they are **not** an actual equal-time campaign or a confidence interval. Eight full-root replicates make this a mechanism screen, not a precise performance estimate. Retain all cells and regressions.

Primary comparison: depth two versus one with the historical baseline on the frozen full-stack probes. For each sampling seed require both mean variance × nodes and mean variance × seconds ≤0.75 of control, with no street above 1.25 on either metric. Equally weight the four roots per street and then the three streets. Zero/zero is neutral; a completely zero-variance control supplies no evidence of improvement. Other baselines and river results are diagnostic, not alternative ways to pass. A pass supports a fresh bounded online comparison only if the coverage audit shows that the added branching is exercised in natural collection; it does not promote a model.

## Coverage audit

Use the same hash-pinned training checkpoint once. Its archive entry `k` is the profile **after k completed fits**, collected at iteration `k+1`; entry zero is uniform bootstrap. Audit entries 0, 32 and 128, plus the explicit current profile after fit 256. Check their fingerprints against the adjacent training reports to avoid an iteration-label error.

For each profile use two independent deals for each of six buttons, and traverse/play each deal from all six hero seats: 72 hero-hand cases per profile, 288 total. Deals are shared across hero choices and profiles; these are correlated cases, not 288 independent games. Report every physical seat and relative position (button, SB, BB, UTG, HJ, CO).

For each case retain four views:

1. Original first-decision collector with the historical baseline.
2. Additional-branching collector with the same baseline and seed.
3. Natural on-policy self-play: all seats use that fixed profile.
4. Natural hero play against the five fixed evaluation styles, assigned clockwise from hero.

Count own decision records by street, unique own observations per traversal, executions that put the acting player all-in, whether the hero has any postflop decision, and natural-hand terminal outcomes. Keep public execution records with own-decision provenance sufficient to audit the accounting; never feed simulator-only data to the policy. More branch records are not automatically broader coverage. Baseline changes cannot change frozen-policy visitation; additional branching cannot create a decision after the traverser is all-in.

If the primary screen fails, retain the result and do not start a bigger branching sweep. If coverage is the limiting issue, the next collector proposal must explicitly define its target policy distribution and any importance corrections before online training. No silent opponent-policy smoothing or replay reweighting is authorized by this experiment.

## Implementation and verification

`collect_outcome(..., branch_first=True, branch_second=True)` enables the diagnostic. Its per-path expansion counter decreases only at a traverser decision. Returning to another branch restores that branch's counter and own sampling prefix. The default remains `branch_second=False`, and the production sampled-replay converter rejects experimental second-branch traversals.

At an expanded decision, each candidate's estimate is its child's estimate with inclusion probability one. Their policy-weighted mean backs up to the parent. At later sampled decisions the existing fixed-baseline estimator and inverse own-prefix correction apply. Expanding a decision therefore removes that local action-selection randomness without changing the expected conditional values or counterfactual updates. Exact enumeration checks all updates, including those after a third and fourth own decision; pre-change fingerprints check both original default modes.

```bash
python -m scripts.check_collector_branching --out results/collector-branching
python -m scripts.check_collector_branching --verify results/collector-branching
```

Use a new output directory. `samples.jsonl` retains every estimate, cost and paired execution hash. `coverage.jsonl` retains the balanced schedule, collector decision/execution records and complete natural public hand histories. `coverage-progress.json` retains completed-profile summaries if a later stage fails. The verifier checks artifact hashes, sample moments, schedule completeness and the decision screen, then replays natural actions through the engine to verify payments, all-ins, public histories and chip settlement. Input policy and critic artifacts remain separately hash-pinned; no fitting or new checkpoint is produced.
