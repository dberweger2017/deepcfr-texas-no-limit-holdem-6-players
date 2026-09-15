# Strategy fitting: next experiment

**Decision:** diagnose fitting on the twelve saved Leduc replays before collecting
more training data. Spend at most **$2 of the remaining CPU budget**, and keep the
fresh confirmation seeds unused during this experiment. Milestone 3 stays open.

This is the design for the next run. The [machine-readable plan](../configs/solver/strategy-fitting-v1.json)
pins its inputs and settings; its runner still needs to be implemented and
validated. It is not a configuration for the existing study CLI. No rental or
new model fitting was started to prepare this design.

## What we need to learn

The [previous study](reports/strategy-capacity.md) ran twelve seeds for 480
iterations. Every final exact played average and replay average was below the
0.15 exploitability limit. Each larger-network recipe passed 11/12 seeds; its
remaining failure was seed 211. The closest result was 0.151373.

The next experiment asks two narrower questions:

1. Does noisy minibatch optimization account for the remaining fitting gap?
2. Does reducing the learning rate near the end improve the final policy reliably?

Network width, targets, replay, and collection stay fixed so these questions do
not become another capacity or training-duration sweep. A negative result is
useful: it directs the next investigation toward representation, coverage, or a
different fitting method instead of another unchanged long run.

A [read-only input audit](reports/strategy-fitting-inputs.json) confirms that each
replay retains 200,000 samples, but only 283–287 of 288 decision situations are
represented. In seed 211, ten situations have only 1–9 samples each and together
carry **0.00374%** of the weighted loss; two situations are absent. This makes
per-situation diagnosis worthwhile. It does not establish that these situations
cause the exploitability failure. The audit inspected the hash-verified tensors
without fitting models or changing any replay.

## Inputs and reproducibility

Use only the twelve final, iteration-480 snapshots from PR #50. The configuration
pins the archive, source report, training revision, solver fingerprint, individual
checkpoint hashes, and the original width-128/48k policy hashes. Do not substitute
earlier checkpoints, just seed 211, or the old three-seed pilot.

The current training loader intentionally requires the original runtime for
recovery. These snapshots originated on Linux, while local inspection runs on
macOS. Implement a **separate read-only diagnostic import**, with these checks:

- Verify the externally pinned archive/checkpoint hashes before loading. Use
  `torch.load(..., map_location="cpu", weights_only=True)`.
- Verify the recorded training manifest, game, iteration, seed, configuration,
  public information catalog, tensor shapes/dtypes, legal targets, and counts.
  Reuse the existing state validation; do not accept an arbitrary checkpoint's
  self-reported hash or trust a manifest without matching the pinned source.
- Export only the diagnostic data needed: replay arrays and their hashes,
  baseline weights/diagnostic averages, catalog, and source lineage. Preserve
  original provenance and record the importer source and current runtime too.
- Require the original interpretation of features, masks, replay weights, and
  games. Pin the new diagnostic implementation separately. Do not weaken the
  production resume loader or claim this import resumes training.

Use the recorded Python 3.11 / Torch 2.5.1+cpu / NumPy 1.26.4 / SciPy 1.17.1
Linux runtime for the scored experiment. First reproduce the original
width-128/48k fit for seed 211, replicate 0. Its policy hash must match the pinned
result. Reuse this completed fit in the matrix. Every other replicate-0 control
must also reproduce its original hash. A mismatch stops the experiment for
investigation; a platform difference is not permission to silently revise the
control. Local smoke runs are implementation checks, not scored replacements.

## Fixed experiment matrix

All fits start from a fresh **width-128, two-hidden-layer network**, use **48,000
optimizer updates**, and preserve the existing public features, legal-action
softmax, and weighted squared-error target. Keep Adam's beta values (0.9, 0.999),
epsilon 1e-8, zero weight decay, float32 training, gradient norm clipping at 1,
and one Torch/BLAS thread per worker. Minibatch size stays 256.

| Recipe | Gradient calculation | Learning rate | Role |
| --- | --- | --- | --- |
| `minibatch-fixed` | Original uniform replay minibatches | Constant 0.001 | Reproduction control |
| `minibatch-decay` | Same minibatch stream as its paired control | Cosine 0.001 → 0.00001 | Candidate fitting change |
| `exact-fixed` | Full replay objective, grouped by decision situation | Constant 0.001 | Optimization diagnostic |
| `exact-decay` | Same full replay objective | Cosine 0.001 → 0.00001 | Optimization diagnostic |

For zero-based update `k` and `S = 48000`, set the decayed rate **before** that
update to:

```text
lr(k) = 0.00001 + 0.5 × (0.001 - 0.00001) × (1 + cos(pi × k / (S - 1)))
```

Run fitting replicates **0, 1, 2** for every recipe and collection seed:
**12 replays × 4 recipes × 3 replicates = 144 complete fits**. The configuration
defines their random streams. Replicate 0 uses the original strategy-fit seed;
replicates 1 and 2 use a separate named stream. Within each replicate, all four
recipes receive identical initial weights, and the two minibatch recipes receive
identical minibatch indices. Fitting replicates vary initialization and minibatch
randomness together; they do not isolate those two sources of variation.

Record diagnostics after updates **12k, 24k, and 48k**, but screen candidates only
at **48k**. Evaluations must not consume training random streams. Never select the
best initialization or intermediate checkpoint. The twelve collection seeds
remain the independent training units; 36 fits of one recipe are not 36
independent collection trajectories.

### Exact objective, without changing what the model learns

For replay row `s`, let `t_s` be its original iteration, `y_s` its strategy target,
and `i(s)` its decision situation. Let `N` be replay size, `T = 480`, and `p_i` the
network's legal-action probability vector. The existing expected minibatch loss
is:

```text
L = 2 / (N × T) × sum_s t_s × ||p_i(s) - y_s||²
```

Group rows by decision situation, retaining `W_i = sum t_s` and the weighted mean
`m_i = sum(t_s × y_s) / W_i`. The same loss is:

```text
L = 2 / (N × T) × sum_i W_i × ||p_i - m_i||² + C
C = 2 / (N × T) × sum_s t_s × ||y_s - m_i(s)||²
```

`C` does not depend on network weights. The grouped term therefore has the same
unclipped gradient as evaluating all replay rows. Preserve the factor
`2 / (N × T)` exactly: renormalizing by `sum W_i` would change gradient scale and
interact with clipping and Adam. Legal-action masks still apply. Missing
situations have zero objective weight and an unknown target, not a zero-vector
training target. Accumulate grouped statistics in float64, then use the declared
float32 training representation.

This calculation is cheap in Leduc because there are only 288 situations. It
removes minibatch noise, not neural approximation error, and convergence of a
nonconvex optimizer is not guaranteed. A full-objective win does **not** establish
that this implementation is practical in full Hold'em. The exact recipes are
diagnostic controls, not automatically eligible production recipes.

## What to record

For every fit/checkpoint, retain exact exploitability, best-response values,
game value and error against the independent equilibrium reference, weighted
loss/excess MSE, elapsed/CPU time, learning rate, gradient norms/clipping counts,
policy hash, and fitting random-stream identifiers. Save final network weights
outside Git for follow-up analysis; preserve replay hashes before and after.

For **every** decision situation, retain the public descriptor, sample count,
iteration-weight sum, target mean (or null if absent), prediction, legal-action
errors, and contribution to weighted loss. Report these by fixed count bands:
**0, 1–9, 10–99, and 100+**. Include both players and both streets; do not publish
only a ranked list of the worst-looking decisions.

At each checkpoint also evaluate diagnostic policies that replace one observed
count band at a time with its replay means, leaving every other neural decision
unchanged. Do not replace absent targets, train on these hybrids, or promote
them. Recompute a best response for each complete hybrid policy. The change in
exploitability measures that joint intervention; band changes interact and must
not be presented as additive causal contributions.

Report all seeds and fitting replicates, paired recipe differences, and their
observed spread. Avoid confidence intervals that treat reused replay fits as
independent training runs. Hold'em BB/100 and professional-strength claims are
outside this experiment.

## Predeclared decision rule

Require all 144 fits and all scheduled diagnostics. Any invalid state,
non-finite metric, missing fit, changed input, timeout, or reproduction mismatch
makes the screen **inconclusive**; retain the outputs rather than dropping jobs.

Only the two **minibatch** recipes can enter the confirmation shortlist. A recipe
must meet exploitability **≤0.15** and value error **≤0.10** on **every one of its
36 final fits**, and no paired fit may worsen exploitability by more than **0.005**
against `minibatch-fixed`. Rank eligible recipes by worst exploitability, then
mean exploitability, then recipe name. Keep every replicate; do not choose an
initialization as part of the training recipe. The original limits are unchanged;
the regression bound is an additional exploration screen, not a new release gate.

| Outcome | Next action |
| --- | --- |
| A minibatch recipe qualifies reliably | Integrate that fitting schedule with recovery/RNG tests, freeze the production recipe, then declare fresh end-to-end confirmation. |
| Only exact-objective fitting succeeds | Investigate a scalable way to reduce optimization noise while preserving the objective. Do not close milestone 3 using the diagnostic policy. |
| Loss improves but exploitability still fails | Use the per-situation and hybrid-policy evidence to choose the next representation/coverage investigation. |
| Nothing improves, or the experiment is incomplete | Preserve the negative/inconclusive result and revise the hypothesis before spending more. |

Reserved confirmation seeds **401, 409, 419, 421, 431, 433, 439, 443** are not used
here in either game. A later confirmation must rerun the frozen recipe from
scratch with the original Leduc and Kuhn gates and its own committed protocol.
Passing this replay screen alone cannot close milestone 3 or start milestone 4.

## Budget, timing, and execution order

The remaining CPU authorization is conservatively about **$8.78**. This design
caps new spending at **$2**, leaving about **$6.78** if that ceiling is fully used.
Those are spending limits, not targets. There is no GPU rental in this plan.

1. Implement the isolated diagnostic runner and tests locally. Use a single
   bounded smoke job: all four recipes for 1,000 updates on one replay, with a
   combined 120-second deadline. Its shorter cosine schedule is for smoke
   validation only. Local execution stays serial and below the existing
   840-second ceiling. Do not select settings from smoke outcomes.
2. Verify the exact-loss/gradient identity, stream pairing, data validation,
   immutable inputs, report completeness, and failure cleanup before rental.
   Commit and push the runner before running the scored experiment.
3. Obtain a fresh CPU quote, including temporary storage and transfer charges.
   Consider up to eight workers, 16 GB RAM, and 20 GB temporary disk. The previous
   rental's hourly price is historical evidence, not a current offer. Choose
   based on the complete fit workload and cost, not advertised clock speed.
4. Run the full seed-211 reproduction control first and estimate remaining work
   from its measured time. Reuse its result. During the first concurrent batch,
   revise the throughput estimate for contention. Start or continue only when
   the complete remaining matrix plus retrieval fits the available time/spend.
   If it cannot fit, retain an inconclusive attempt; do not drop seeds, shorten
   scored fits, or silently extend the budget.
5. Each fit has a 180-second watchdog. The entire rental has a **60-minute maximum**
   and must stop earlier if necessary to remain below $2. With quoted all-in
   hourly rate `R`, remaining diagnostic allowance `B`, and a $0.25 allowance for
   retrieval/shutdown, use a maximum remaining rental time of
   `min(remaining 60-minute allowance, 3600 × (B - 0.25) / R)`.
   Account separately for any known fixed or variable charges; do not launch if
   the allowance is insufficient. Application process limits do not stop billing.
6. Retrieve and hash-verify results, then stop and terminate the pod and verify no
   billable storage remains. Keep raw artifacts outside Git and commit the
   compact report, all failed fits, source/input hashes, and actual cost record.
   Use a scheduled follow-up near the measured ETA if the assistant session
   pauses; avoid continuous polling. Never leave an idle rental after completion.

The earlier width-128 fits suggest this should be a tens-of-minutes fitting
study, but the new exact-gradient implementation has not been timed. There is
no promised ETA or cost saving until the smoke and rental control are measured.

## Implementation acceptance checks

Before the scored run, the implementation PR must demonstrate:

- Grouped loss plus the constant and parameter gradients match the ungrouped
  full-replay calculation on tiny nonuniform-weight, variable-mask examples.
  Use float64 reference checks and explicit float32 tolerances.
- The fixed minibatch control preserves the existing fitting algorithm and update
  order. Any optional reporting hooks must leave its final policy and random
  streams identical to the uninstrumented implementation.
- All recipes share initial weights within a replicate; paired minibatch streams
  and cosine endpoints are correct. Replay, collection state, and advantage
  weights remain unchanged.
- Altered hashes, provenance, catalog, illegal targets, duplicate/missing jobs,
  and non-finite outputs are rejected. Failures remain in reports.
- Confirmation seeds and exact-objective recipes cannot enter model promotion;
  only complete final-step results can enter the declared screen.
- Worker failure/deadline handling kills and reaps subprocesses. Source,
  configurations, dependencies, and input/output hashes are recorded separately
  from historical training provenance.

After these checks pass, execute this frozen design rather than extending the
previous capacity sweep. If implementation reveals a material design error,
revise and recommit the protocol **before** looking at scored outcomes.
