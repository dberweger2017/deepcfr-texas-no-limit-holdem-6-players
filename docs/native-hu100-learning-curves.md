# HU100 checkpoint learning curves: frozen protocol

This owner-requested PR measures all five #197 verified HU100 averages. It makes
no training, model, checkpoint-selection or release decision. The PR is handed
back after independent review and green CI, without merging.

## Inputs and fixed play

[Configuration](../configs/arena/hu100-learning-curves-v1.json) binds the indexed
average hashes, sizes, checkpoint hashes, iterations and actual completed nodes:
**100,691; 1,001,382; 5,001,210; 10,001,922; 11,042,440**. All belong to seed
2026100601's unchanged linear-CFR/opponent-sampled lineage. The capacity-stop
checkpoint's filename requests 1B; its verified header contains only 11,042,440
completed nodes. No new extraction occurs. The [original model index](reports/native-recovery-hu100-artifacts/followup-model-index.json)
and [RESULTS_INDEX](../RESULTS_INDEX.md) supply archive/member restoration.

One free M4 worker retrieves those five averages from the accepted synced #197
archive into a fresh ignored `results/inputs/`, verifies whole archive and selected
member SHA256s, and checks each header's completed nodes and checkpoint identity.
Historical originals stay untouched; #197's live merged status is checked before
retrieval. Each checkpoint is loaded sequentially in a separate process.

Use unchanged `random`, `check_call`, `tight_aggressive`, `loose_aggressive` and
`pot_pressure`, preserving random's min-raise/all-in behavior. Fixed heads-up
10,000-chip stacks reset each hand; blinds 50/100, no rake. Uniform uses exactly
the same native HU100 menu and original weighted sampler as #197. The existing
arena runner, evaluator, telemetry and all-action/settlement auditor are reused.

## Pairing and sample freeze

Pilot root **2026100820311**, final root **2026100820312**. Both are fresh and
disjoint from #197's pilot/final roots and each other. Each opponent has one
duplicate-deal schedule with two swapped target seats per block. All checkpoint
policies and uniform share its physical deals, button and rotations. Final count
and complete evaluator-private deal/action coordinates are hashed before final
play in `frozen-final.json` and `frozen-schedule.json`.

Private action seeds reuse #197's hash function: root/test/action/namespace,
opponent/block/rotation/arm/logical-player. **Candidate streams are identical
across checkpoints**, separately for target and opponent. Each hand resets its
two private streams; differing trajectories can consume different numbers of
draws, so this pairs stream prefixes rather than semantic decisions. The baseline
arm has distinct streams; neither seeds nor the schedule enter observations.
Telemetry observes after the sampler and consumes no RNG.

Uniform is evaluated once per opponent per stage, in the first checkpoint run.
Later runs play only their candidate hands and reuse those exact baseline action,
decision and settlement rows. The auditor verifies exact reuse. During full
deterministic reproduction the first uniform run is reproduced once; later
reproductions reuse those reproduced rows. Cached uniform lookup telemetry refers
only to the 100,691-node model and is excluded from cross-checkpoint coverage.
All five candidate policies retain their own opponent/street decision telemetry.

Propose **2,048 blocks/opponent/policy =122,880 distinct final hands** for five
checkpoints plus uniform. The separate **16-block cost-only pilot =960 distinct
hands** includes every checkpoint, all replay and full reproduction. The count
selector reads only cost/completeness receipts, never pilot poker results. It
reserves 240 seconds for stage overhead/cross-checkpoint audit/closeout and uses
2× measured fixed loading/snapshot costs plus 2× measured per-block play, replay
and reproduction costs. Uniformly reduce counts in multiples of 32 if needed;
below 32, report no final budget. Freeze once before any final hand; no outcome
extension, checkpoint choice or automatic retry.

## Resource and correctness limits

One fresh **1,800-second cap** starts before retrieval/loading and covers pilot,
all final play, all audits and deterministic reproduction. No previous deadline
is reused or extended. The #197 external supervisor measures the whole owned
family against **10 GiB**, normal system pressure (unknown/warning/critical or
free percentage <15 stops), original **448.81-MiB swap baseline /0.5-GiB maximum
growth**, **15.5-GiB disk floor**, and AC. Each heavy subprocess gets fresh
headroom/deadline admission; the M4 chip and committed configuration/source are
required. Exclusive campaign and worker claims prohibit duplicate launches,
including alternate output roots. Never stop another worker. Correctness or
resource failure stops the campaign, retains partials and receives an honest
incomplete report. No persistent scheduler is created.

## Statistics and evidence

Descriptive per-opponent curves plot **actual completed training nodes** versus
BB/100, with block-based two-sided Student-t 95% intervals and the single reused
uniform reference. The independent sampling unit is the duplicate block, averaging
its two seats. Report every checkpoint and opponent, including losses.

Predeclare one formal family: **20 final-minus-earlier comparisons** (four earlier
checkpoints ×five opponents). Use paired block differences and **Bonferroni FWER
0.05**, two-sided per-comparison alpha 0.0025 /99.75% intervals. Label improvement
only when the adjusted interval is above zero, decline when below, otherwise
inconclusive. Also retain ordinary paired 95% intervals as descriptive quantities.
Uniform differences are descriptive and do not belong to the formal family.
No omnibus, selected-checkpoint, monotonicity or release test is inferred.

Report positive-mass, zero-mass and missing-key counts/rates by checkpoint,
opponent and street. Discuss whether observed gains accompany improved coverage
and shrinking aggressive-opponent losses. Decision-weighted coverage depends on
the policy's trajectories; correlation does not establish causation.

Independently replay every final action/settlement, validate legal observations,
keys/menus/private coordinates, recount telemetry, and recompute block statistics.
Full deterministic reproduction must match all hands and decision traces including
probabilities/classifications, excluding measured latency only. The cross-checkpoint
auditor independently verifies physical schedule, exact uniform reuse and paired
statistics. Preserve source/environment, all input/snapshot hashes, pilot, frozen
counts, traces, resource samples, audits, reproduction and failures in the new PR's
Research-Cloud archive with member manifests and retrieval provenance. Git holds
the report, compact receipts/CSV summaries and exportable plots. One training seed
and one evaluation root make this a limited descriptive study of scripted play.
