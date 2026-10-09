# HU200 20M–100M paired comparison

Prospective protocol, October 9, 2026. Base: current main `4c4e04c` (merged #216).
This task evaluates only #216's audited opponent-sampled averages at actual nodes
20,002,716 and 100,000,034, seed 2026100905. No new training, recipe change,
live service match or release. Use existing immutable files directly; verify
bytes, SHA256, checkpoint provenance, iteration, seed, game/schema/menu identity,
average rule and uniform fallback against its model index before inference.
Retain #216's HU200 200BB/50–100 blinds/no ante/no rake game, v1 cards, uncapped
min/pot/conditional-jam menu, uniform missing/zero fallback and translation off.
Prior accepted training audits are evidence, not repeated work.

## Frozen inference and sample admission

Five primary contrasts: **100M minus 20M** against each of `random`, `check_call`,
`tight_aggressive`, `loose_aggressive`, `pot_pressure`. For each opponent, a block
contains an identical shuffled deal with the candidate in each seat, for both
checkpoints (four hands). Button alternates across blocks. Deal, candidate-action
and rival-action streams are SHA256-derived from distinct labels, opponent,
block and rotation. Both checkpoints share the same private streams; policies
see only legitimate immutable observations. Tables reset between hands.
Final root **2026100906**; timing-only root **2026100907** is excluded from inference.
Neither reuses #216's smoke or any prior campaign's hands.

Target **2,048 paired blocks/opponent =40,960 hands**. Declare five simultaneous
95% Bonferroni intervals: paired block differences, Student t critical value
`ppf(1-.05/(2*5), n-1)`. Absolute rates receive descriptive unadjusted 95% t
intervals. Block, not hand/rotation, is the independent sampling unit. No
aggregate opponent score or seed-generalized claim. Diagnostics and examples
are exploratory and receive no significance claims. A constant series has a
zero-width arithmetic interval and is flagged as no observed variation.

Timing-only calibration: 32 blocks/opponent/checkpoint, complete replay and
policy reproduction. No calibration payoff summary is produced or inspected.
Measure each model's fixed hash/header/load costs, fixed source snapshot cost,
and per-opponent play, trace, event replay, policy reproduction and native parity
costs separately. Do not multiply loading costs by hand count. After calibration
freeze the largest common sample among **2048, 1024, 512, 256, 128, 64, 32** that
fits `2*(fixed reloads + scaled per-opponent play/replay + native parity)
+600s closeout` in the remaining single 3,600s computation clock. Disk admission
reserves twice projected raw evidence plus 1GiB above the 15.5GiB floor.
No smaller sample is selected after winnings are inspected. If none fits, stop
with timing evidence and no performance claim. After freeze, no optional stopping,
checkpoint selection, restart or budget extension. Failures and valid partials
remain; incomplete comparisons cannot claim the frozen full sample.

One M1 worker and exclusive `/tmp/deepcfr-m1-research.lock`. Record current
threads/process ownership and current parent PR status. AC, normal pressure,
>=15% free memory; initial >=4GiB psutil available. Family soft/hard 3/4GiB,
fixed-baseline swap growth <=512MiB and total <=3,000,000,000 bytes. Sample every
~0.5s, retain actual gaps and kernel command peaks. Soft/hard/time breaches latch
science off and preserve evidence. Evaluation/replay/verification/source snapshot,
local ZIP/readback and closeout share the single clock, beginning at admission.
Source development and small regression checks precede admission. Report/PR/review
and network upload are administrative after local science closeout.

## Diagnostic definitions and verification

For every decision retain street/position, legal bounds/menu/key, lookup and
visits, probabilities/action, actual raise-to and increment/(pot+call), public
pot and own committed chips. Positive mass and visits are separate. Visit bands:
missing, 0, 1, 2–9, 10–99, 100+. Report all wins/ties/losses, not losses alone.
Reuse #201's legal token-tree support witness. A present stored key is a training
witness. A missing key on an observed all-menu path is supported but absent.
Otherwise exhaust the compatible HU200 token/menu tree; only exhaustion proves
unsupported abstract history/menu. Bound each search at 5,000 states and all
search time at 120s/model; exhausted budgets remain unresolved. Off-menu sizing
alone never proves unsupported. Store alternate witnesses and replay their exact
HU200 key. Supported absent keys, stored zero visits, zero mass and covered
positive mass remain distinct, including supported-but-unvisited diagnostics.

Every hand: validate observations/actions, conserve 40,000 chips, reproduce all
candidate and rival actions from recorded traces/private seeds, compare complete
events and settlements. Export every hand's native parity fixture and run #216's
hash-pinned runtime independently. No hidden cards/deck/seeds reach a policy;
those exist only in offline evaluator evidence. Hash raw records; independent
readback recounts coverage, actions, support/visit bands, disjoint hand categories,
paired schedules and payoff arithmetic before inference.

Hand categories: no candidate decision; ever unsupported; unresolved without
unsupported; supported missing without either; zero mass without missing; all
positive. Categories are disjoint hand associations, not causal contributions.
Large pot: maximum pre-settlement pot >=64BB. Stack-off: candidate voluntarily
reaches zero remaining stack; separately record jam/call, street, and final
zero-stack outcome/refunds. Report win/loss counts and payoff contributions over
all hands. Conditional final payoffs are neither action values nor causal effects.

Representative rule, frozen before results: lowest SHA256 of `(opponent,block,
seat)` within each checkpoint/opponent/hand-category/payoff-sign (strict win,
strict loss), plus lowest hash among large-pot losses and stack-off losses.
Deduplicate exact coordinates. Retain full actions, legal observations, seeds,
cards and events for deterministic replay, including winning examples. These
are illustrations, not a selected inferential sample. Form concrete hypotheses
for later controlled confirmation; choose among more training, action-support
work, targeted investigation or Slumbot pilot readiness on the full evidence.

## Storage and handoff

Check this PR's current status before archiving; the owner explicitly authorizes
archival of this task's own open evidence. Create and locally verify one evidence
ZIP (all raw traces, timings/resources, failures/partials, source, plan and input
provenance), place it in `~/Local/Research-Cloud/PR-<number>-HU200-playing-diagnosis/`,
and index archive/member hashes and exact restoration. Reuse #216's model and
runtime archive references; do not duplicate them in this archive. Record actual
Drive folder identity if available and an **upload-pending handoff**. The owner's
updated instruction overrides upload-wait rules: no waiting, cloud acceptance
claim, original deletion or forced offloading. Leave a reviewed, checked PR
unmerged and ready for owner review.
