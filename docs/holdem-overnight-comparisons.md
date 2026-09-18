# Two overnight comparisons against the continuous baseline

## Decision and status

**Launch authorized by the owner.** The owner subsequently requested both
comparison runs alongside the control and explicitly removed RAM caps.
Memory remains measured; disk and phase protections remain enforced. The running [continuous baseline](holdem-continuous-m4.md)
remains unchanged. This document defines two poker training agents, not two
coding agents. The next implementation must satisfy the admission checks below
before starting them.

Use the baseline as the control and change one training setting in each new
arm. Do not combine the changes yet:

| Arm | Sampler | Replay records per role | Question |
| --- | --- | ---: | --- |
| Existing baseline | `first-decision` | 4,096 | Does the current recipe improve with longer training? |
| A: additional branching | `second-decision` | 4,096 | Does collecting less noisy later-decision information improve learning? |
| B: larger replay | `first-decision` | 16,384 | Does retaining more distinct historical decisions improve learning? |

Both candidates start fresh from uniform bootstrap with seed **2026091901**,
matching the current baseline. They do not resume its learned state, copy its
replay, or start with previously selected models. Shared seed namespaces make
the comparison more controlled, but policies and sampled trajectories diverge
as soon as training differs. These are three arms of one seed, not three
independent training replications.

## Why these two changes

The [completed M4 campaign](reports/holdem-local-fullgame.md) still lost heavily
against scripted opponents. A saved-action audit at iteration 256 finds:

- First-action preflop all-in raises in 2,288/6,144 and 2,152/6,144 hands:
  **37.24% and 35.03%**, versus 11.07% for the uniform control.
- Hands containing a preflop all-in account for −733.37 and −693.86 BB/100 of
  the total −1,056.11 and −869.63 BB/100 returns. Those are **69.44% and 79.79%**
  of recorded net losses. The groups are selected by the policy's actions;
  these figures do not measure how much could be recovered by banning shoves.
- In the saved shown-card subset, examples include raising to 100 BB with
  6♦5♣ into a 1.5 BB pot, and with 4♦3♥ facing a raise. The subset is selected
  by showdown visibility, so it cannot estimate the overall holding distribution.

Most collected records already concern preflop. The unresolved issue is whether
the learner receives and retains useful distinctions between those decisions.
More iterations alone may help; the new baseline measures that possibility.
The two additional arms test signal collection and retention separately.

The [multistreet study](reports/holdem-multistreet-campaign.md) also found that a
learned card branch was seed-sensitive and explicit visible features failed
validation. Its restricted 2 BB supervised results do not justify transferring
one of those architectures directly into a 100 BB overnight self-play run.
The earlier clipping/fitting comparison did not establish a consistent playing
benefit either. For this batch, retain the existing architecture and optimizer.

## Arm A: expand one more own decision

Change only:

```json
{"training": {"sampler": "second-decision"}}
```

This is a conceptual override, not a complete runnable configuration. The
existing implementation expands the traverser's first two own decisions and
uses the established corrected estimator afterward. Keep all existing legal
bet sizes, payoff conventions and information boundaries.

The [branching probe](reports/holdem-collector-branching.md) found substantial
cost-adjusted variance reduction. The subsequent
[online comparison](reports/holdem-branching-online.md) improved scripted-pool
results on average but failed the across-seed consistency gate. This is a
longer-horizon retest under the current higher-work recipe, not an untested
mechanism or a previously proven success.

**Expected mechanism:** better action comparisons at later own decisions can
also improve the continuation estimates returned to earlier decisions. It does
not directly guarantee fewer first-action shoves or force postflop visitation.
Measure both rather than assuming either follows.

**Tradeoff:** additional branches increase work per root and can generate more
records. At 128 roots per role, an iteration is no longer equal compute to the
control. Report total nodes and elapsed time alongside iteration count. This
arm tests the implemented collection recipe as a whole; it cannot separate
reduced target noise from its additional record coverage.

A useful outcome would combine lower decision costs on frozen probes, less
costly play, and an acceptable gain per unit of computation. A lower maximum
sampled target alone is insufficient: rare-event maxima are unstable and do
not measure estimator variance.

## Arm B: retain four times more replay

Change only:

```json
{"training": {"capacity": 16384}}
```

Keep the uniform reservoir algorithm, iteration weighting and target semantics.
Do not stratify by street, discard extreme targets, oversample winning hands,
clip regret targets, or add synthetic demonstrations.

**Expected mechanism:** after saturation, a larger reservoir retains more
historical examples and reduces the chance of losing an infrequent but useful
situation from memory. If retained postflop proportions stayed around 4–6%,
4,096 records would contain roughly 164–246 postflop records per role and
16,384 would contain roughly 655–983. These are arithmetic illustrations, not
measured reservoir counts: root visitation and replay-record proportions are
different and must be reported separately.

**What it cannot fix:** it does not change expected street proportions, create
missing situations, make sampled targets less noisy, or guarantee that the
network can generalize across them. A uniform reservoir still retains the
full training history rather than specifically tracking the latest policies.

Keep fitting at 256 steps with batch 32. That is 8,192 sampled record
presentations per role per fit: approximately two per stored record at capacity
4,096 and one half at 16,384. Thus the arm tests whether broader retained data
helps under the existing fitting budget. If it fails while fitting error rises,
that does not disprove the value of larger replay with additional fitting.
Do not silently increase fitting steps during this run.

**Tradeoff:** replay, checkpoint size and serialization costs increase. Total
memory does not necessarily quadruple, because models and archives are separate
allocations; admission requires measurement. Earlier errors may also persist
longer in a larger historical sample. This is a coverage experiment, not a
claim that increasing a buffer fixes unstable targets.

## Settings held fixed

For both new arms, copy the baseline recipe and change only the setting above:

- Six players, 100 BB initial stacks, no rake; same engine and action menu.
- Width 32; Adam learning rate 0.001; norm-1 gradient clipping.
- 256 fresh fitting steps per role, batch 32, 128 traversal roots per role.
- Exploration 0.5; unchanged baseline head, payoff perspective and averaging.
- Save and evaluate every 64 iterations; 1,024 deal blocks with seat rotation.
- Existing scripted pool and random-opponent benchmark; evaluation seed
  2026091810, identical to the baseline.
- Match the control's effective numerical-library thread settings. Verify and
  record them during admission; avoid CPU oversubscription from three workers.

Use separate output directories with arm names and the seed in their names.
Freeze and record the source revision, resolved configuration and hashes for
all arms. Any source change affecting scientific behavior requires a new
comparison; do not silently update the running control.

## What to measure overnight

Use the same four main TensorBoard charts, with three clearly named series.
Preserve the underlying reports; the dashboard is a view, not the evidence.
Add an offline behavior summary from the losslessly retained outcomes.

| Measure | Purpose |
| --- | --- |
| Scripted-pool BB/100, paired by deal block | Primary practical monitoring comparison |
| Random-opponent BB/100 | Secondary basic competence check |
| First-action shove rate and any preflop-all-in rate | Track the observed commitment behavior |
| Returns by preflop/later/never-all-in groups | Descriptive loss accounting, not causal action values |
| Collection roots reaching each street | Measure visited decisions, not dealt showdown boards |
| Actual retained records by street and role | Distinguish replay coverage from root visitation |
| Fitting metrics, target tails, all-negative regret fallback | Explain learning changes without equating MSE with strength |
| Nodes, train/evaluation time, peak RSS, archive size | Compare cost and identify resource limits |

The current runner does not necessarily save every diagnostic in this table.
The implementation must distinguish existing telemetry from newly required
extraction. Do not label a metric as available until its source is verified.

For deeper behavior checks, use the planned first-to-act 100 BB action-value
probe: compare normal raises and shoves for declared weak, medium and strong
holdings under fixed continuation policies. Reference uncertainty must be
reported. This probe is a separate bounded diagnostic, not extra self-play or
a reason to alter the overnight recipe after observing its curve. Merely
reducing shove frequency is not a success criterion; strong poker can require
all-ins in the appropriate situations.

## Comparison boundaries and interpretation

Predeclare **iteration 1,024 as the main research comparison**, using each
arm's saved snapshot-average policy at that boundary. Report 256 and 512 as
intermediate milestones and 2,048 if all arms eventually reach it. Continuing
beyond 1,024 does not move the main comparison to a more favorable checkpoint.
If an arm has not reached 1,024 by morning, mark the main comparison incomplete.
There is no claim that all three will reach it overnight.

Also compare throughput and the last completed scheduled evaluations within
fixed **2-hour and 4-hour elapsed budgets measured from each worker's own
start**. Include evaluation/checkpoint time. If no evaluation exists within a
budget, mark it unavailable. Iteration comparisons measure learning per
iteration; elapsed-budget comparisons measure practical usefulness. Shared-host
contention and the baseline's earlier head start prevent treating these as
clean hardware benchmarks. Do not force or interrupt an evaluation just to hit
a budget boundary.

Pair candidate-minus-control returns at the deal-block level, keeping seat
rotations together. Report both candidate differences, raw win rates and all
intermediate outcomes. With only one training seed and repeatedly reused
validation deals, even a positive interval is exploratory. Selecting a better
arm requires later independent seeds and fresh scripted-pool confirmation;
this overnight run cannot qualify v0.5 or change production defaults.

Possible outcomes:

- A improves reliably within this seed and earns its extra compute: prioritize
  an independent-seed branching confirmation.
- B improves at the same fitting budget: prioritize replay-capacity confirmation.
- Both improve: confirm them separately before testing the combined recipe;
  improvements need not add together.
- Neither improves: retain the null result and prioritize the clean-signal
  preflop probe. Do not extend repeatedly until a favorable evaluation appears.

## Launch admission, retention and stopping

The owner has now authorized implementation and launch. The following checks
apply, with the earlier RAM admission limits superseded by that decision:

1. Generate complete validated configs, verify one-setting diffs against the
   running control, and use a separate pilot seed for bounded resource checks.
   Confirm recovery for both proposed configurations before the long run.
2. Measure all three workers together. Record aggregate RSS without a RAM cap; retain phase deadlines and free-disk protections.
   Do not assume three simultaneous workers fit because two previous workers
   did. If disk admission fails, queue the larger-replay arm. A launch coordinator must account for all workers before
   admitting additional ones.
3. Reserve disk for the combined expected outputs, temporary checkpoint writes
   and compression. The existing 20 GiB free-at-launch check per job is not a
   reservation for three jobs. Account explicitly for larger replay and growing
   policy archives; stop admitting work before crossing the 12 GiB free floor.
4. Preserve reports and compressed raw outcomes for every evaluation. Pin the
   1,024 checkpoint and export for all three arms before rolling retention can
   delete them, in addition to the latest two recovery boundaries. Include
   those pinned files in output limits. This requires retention support or a
   verified archival mechanism; it is not provided by the present two-file
   rolling policy. If the control boundary is already retired, do not claim
   checkpoint recovery or a new policy probe for that boundary is possible.
5. Give both new arms the baseline's owner-stopped continuous lifecycle, with
   no performance-triggered restart or automatic promotion. Existing resource
   and phase limits still apply. Morning review can request a checkpointed
   stop; no autonomous extension, new seeds or paid fallback is implied.

The source and telemetry should make a stop or incomplete result explicit.
A useful overnight experiment can fail its learning hypothesis; losing its
artifacts or confusing a partial run with a completed comparison is avoidable.
