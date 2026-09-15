# From fitting diagnosis to the Hold'em learner

## Decision

Integrate **cosine learning-rate decay for strategy fitting** as an explicit,
opt-in solver setting. Preserve constant-rate behavior and its regression tests.
Then freeze one end-to-end small-game confirmation campaign before using any
fresh seeds. There is no further automatic width or learning-rate sweep.

Independent Hold'em engineering may proceed while confirmation is pending:
public-history encoding, variable-seat representation, legal bet candidates,
trainer interfaces, and recovery contracts. Substantial Hold'em self-play and
architecture campaigns remain gated on small-game confirmation and the new
Hold'em pipeline's correctness checks.

This changes the sequencing of the roadmap. It does not declare milestone 3
complete, promote a model, authorize another rental, or lower the 1.0 standard.

## Evidence behind the choice

[The completed fitting study](../reports/strategy-fitting.md) used twelve saved
Leduc replays, three paired fitting replicates, and four recipes. Fixed-rate
minibatches passed the absolute limits in 31/36 fits. Decayed minibatches passed
36/36, improved 31 paired outcomes, and reduced mean replay fitting error by
about 90%. Both exact-gradient controls also passed all absolute checks and fit
the observed replay more closely. This makes optimization a better immediate
lead than another increase in width.

The decayed recipe still failed the study's additional paired-regression limit:
seed 239, replicate 0 moved from 0.087157 to 0.102948 exploitability, a worsening
of 0.015791 against the permitted 0.005. All four recipes passed the absolute
limit on that case, including the two exact-gradient controls, which also
regressed against that particular baseline fit. A closer fit to the average
strategy need not improve every approximate policy.

**The historical screen remains `no_candidate`.** The implementation choice is
a new engineering decision based on the complete exploration, including that
failure. It is not an automatic selection or a claim of uniform improvement.

## Alternatives

| Option | Evidence and cost | Decision |
| --- | --- | --- |
| Cosine strategy fitting | Tested directly; preserves network, loss, replay, and inference format. Requires schedule configuration, recovery checks, and fresh end-to-end validation. | Implement next. |
| More width, steps, or learning-rate values | Earlier size increases helped, but the current width already fits the replay closely with better optimization. More experiments would not settle the failed promotion rule. | Defer until a specific new failure justifies them. |
| Exact full-replay gradients | Useful diagnostic in Leduc's 288 information sets. Enumerating the full Hold'em decision space is not practical; a scalable implementation would need a separate design. | Keep as diagnostic evidence. |
| Single Deep CFR | Avoids a separately fitted average-strategy network, but changes policy extraction and the handling of historical advantage networks. Requires its own implementation, storage/inference measurements, and correctness checks. | Retain as a fallback if integrated fitting fails confirmation, not a simultaneous rewrite. |

The [Single Deep CFR paper](https://arxiv.org/abs/1901.07621) supplies the rationale
for the last alternative. Its two-player results do not establish six-player
strength. Our estimate of implementation effort is a judgment about this
repository, not a measured performance comparison.

## Next implementation PR

- Separate the strategy fitting schedule from advantage fitting. Constant rate
  stays the default; enabling cosine decay must be explicit in configuration.
- Match the diagnostic schedule: set the rate before each update, from 0.001 to
  0.00001 over the declared fit budget. Do not change the objective or sampling.
- Include the schedule in configuration validation, provenance, and recovery.
  Fit-local schedules must reset when a fresh fit begins. Preserve the current
  supported recovery boundary between completed iterations.
- Test constant-mode equivalence, cosine endpoints, deterministic paired refits,
  and resumed-versus-uninterrupted execution in a new process. Keep average
  fitting separate from traversal policy construction.
- Declare the next campaign in executable configuration with all seeds, fitting
  settings, collection budgets, evaluation times, and resource/shutdown limits.
  Do not train fresh confirmation seeds merely to choose those settings.

The starting Leduc recipe is the tested width-128 / 48k strategy fit with cosine
decay and unchanged advantage training/collection. Retain the established Kuhn
recipe as a regression check. The implementation PR must specify and freeze both
complete configurations; this decision document alone is not an executable run
protocol.

## Prospective confirmation: what changes and what stays fixed

The next campaign answers **whether a frozen end-to-end learning recipe meets
our existing small-game readiness standard on fresh independent training seeds**.
It does not test whether every new fit improves every older approximation.

Use the eight reserved seeds, 401, 409, 419, 421, 431, 433, 439, and 443, once the
implementation and complete protocol are frozen. Run from scratch rather than
refitting the old replays. Preserve the original final-only absolute criteria:

| Game | Maximum exploitability | Maximum absolute value error |
| --- | ---: | ---: |
| Kuhn | 0.03 | 0.03 |
| Leduc | 0.15 | 0.10 |

Every declared seed must pass. Invalid outputs, missing runs, and exceeded
budgets invalidate or make the campaign inconclusive. Do not select checkpoints,
drop seeds, change settings mid-campaign, or extend a run because it looks close.

**Explicit prospective change:** the fitting exploration's per-fit 0.005
regression veto is not a condition for this readiness campaign. That veto tested
a stronger claim than meeting an absolute convergence tolerance. Retain paired
fixed-rate comparisons and report every regression, using identical collection
and initialization where the implementation establishes they can be shared
without changing the candidate's end-to-end run. Freeze this comparison procedure
and its compute cost in the protocol as well. A readiness pass must not be
reported as a pass of the old screen or proof that decay is uniformly stronger.

This decision is made before the reserved data are used. It does not revise
previous reports, the original absolute limits, the arena's promotion rules, or
the professional-level release requirements. Historical exploration influenced
the recipe choice and remains visible in the final report.

Run **one** such frozen confirmation campaign under a separately declared cap.
If it passes, close the small-game readiness milestone and proceed with the
Hold'em learning experiments. If it fails, preserve the failure and pause scaled
training. Make a bounded architectural decision, including Single Deep CFR,
rather than silently retrying new seeds or searching more widths. Independent
Hold'em engineering can continue; a failed gate is not permission for a large
training campaign.

## First Hold'em experiments

The legacy trainer is not the target of the architecture sweep. First implement
complete public decisions, useful bet candidates with independent regret targets,
consistent self-play for all roles, and full recovery on the corrected engine.
Retain the small-game checks as regression tests for shared learning components.

After profiling, declare a small architecture comparison on the actual game.
A reasonable starting design is three capacities and at least two independent
training seeds per capacity; exact widths, depth, and encoding depend on the
implemented model and measured throughput. The widths discussed with the owner
are hypotheses, not requirements. Compare parameter counts, equal training work,
and equal compute cost. Use common evaluation deals and the same action set.

A short GPU pilot first measures completed collection/fitting cycles, memory,
latency, numerical behavior, and restart. It may not yield enough learning or
evaluation hands to rank playing strength. Benchmark one versus multiple jobs
per GPU before choosing concurrency; choose hardware from measured bottlenecks.
Only extend to a larger comparison with a declared budget and useful expected
evidence. A half-hour run cannot prove that a model size will never converge.

No new CPU rental, GPU budget, or unattended work is created by this document.
The existing conservative CPU balance is $8.57 of the $10 authorization; future
GPU funding remains a separate owner decision after a current cost comparison.
