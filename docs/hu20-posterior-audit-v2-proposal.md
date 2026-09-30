# Proposed next scientific experiment: paired posterior decision audit

**For owner approval; not executed or authorized by PR #126.** This is a new
attempt after #119's retained two-hour feasibility stop. Its original files,
clock and conclusion remain historical. The ranker experiment supplies
engineering evidence, not a training-mechanism result.

## Inputs, counts and question

Retain #119's exact 24 outcome-blind trained B100M coordinates, all three
saved models/checkpoints and the original selection digest
`578b8b67f21a15d2961e88cf498aeeb64831c9827e578e5ce1db62f85d43e322`.
Use the hash-verified #117 curve archive. Its aggregate returns were previously
opened; selection and this proposed stability subset use no terminal payoff,
hidden opponent cards, future deck or conditional values.

Estimate whether saved-policy gaps at these decisions persist or change when
uniform compatible opponent holdings are replaced by likelihood-conditioned
holdings under the declared original-cap2 LBR. This is a stratified diagnostic
sample, not a representative decomposition of the −73.14-BB/100 loss and not
exact exploitability. Persistent gaps do not identify card, history or menu
abstraction causally by themselves.

For every compatible holding and visible prior LBR action, run **four**
independent LBR simulations with `LBRConfig(4, 5)`. Preserve the attacker
action contract, observed exact raise-to equality, range updates, chance
draws, reduction order and first-index tie. Record the observed-action match
integer and denominator by holding/action, including holdings with zero
preceding estimated posterior mass. This full count is the conservative
58,047-call one-sample inventory; do not save compute by silently omitting
previously zero-weight holdings from the diagnostic records.

Evaluate **96 worlds for uniform and 96 for posterior per decision**:
24 × 2 × 96 = **4,608 primary worlds**. In each range, worlds 0–47 select an
alternative action; worlds 48–95 estimate that already-frozen action's gain
over the saved policy mixture. Report the held-out paired/world interval
from those 48 only. All-96 action means remain descriptive. Pair uniform and
posterior worlds with common inverse-CDF holding uniforms and corresponding
runout/policy/LBR random streams where possible; candidate target actions
within one world share that world's cards and continuation streams. Preserve
failed worlds and never reduce sample counts after looking at values.

RNG roots are declared in the [proposed selection record](reports/hu20-posterior-audit-v2-proposed-selection.json):
main likelihood `202610050126`, independent stability roots
`202610050127/128/129`, higher-count root `202610050130`, and conditional
world root `202610050131`. Derive each substream from selected rank, public
event index, hypothetical holding and replicate/sample index; world streams
also contain range/control and world index. They never use played hidden
cards, original simulator seed or recorded future deck.

## Posterior stability before primary values

The committed outcome-blind subset contains **five** decisions: all four
streets, three seeds and both positions. Among coordinates having a prior LBR
action in the frozen #121 corpus, choose the lowest SHA-256 of
`posterior-stability-v2|<original selected rank>` per street, then add the
lowest-ranked decision for each missing seed and each missing position.
Coordinates are listed explicitly in the proposal JSON before any posterior
or value is generated. This does not change the primary 24 decisions.

For each of these five, estimate three independent **four-sample** likelihood
sets and one independent **16-sample** set over every compatible holding and
public LBR action. This is **28 additional likelihood samples** for that
subset, separately costed from the four primary samples. Do not recycle the
stability estimates into the primary posterior. The 16-sample set is a
comparison with a larger budget, not an exact posterior or accuracy certificate.

Keep every raw match count/denominator, normalized posterior, positive support,
ESS, entropy, largest weight, zero-evidence event and soft-limited batch.
Report all pairwise total-variation distances among four-sample repetitions,
each four-versus-16 distance, ESS ratios, and the 16-sample posterior mass on
holdings given zero mass by each four-sample estimate. A zero empirical match
is **finite-sample zero**, unless a separate exact legal/menu argument proves
the observed action model-impossible. A legal action with probability 0.25
has zero matches in four independent trials with probability 31.6%; at 0.10
it is 65.6%. These are examples, not measurements of these policies.

Proposed conservative stability gate, frozen for review before execution:
no zero-evidence event or incompatible timing result; every four-versus-16 TV
≤0.15; all four-sample pairwise TV ≤0.20; ESS ratios in [0.5, 2]; and ≤0.10
16-sample posterior mass on each four-sample zero-support set. **If any case
fails, stop before primary conditional values**, retain all estimates and
report that four samples do not support a stable posterior on this check.
Do not smooth, increase counts, change the subset or launch training to obtain
a usable answer. The next missing measurement would be a prospectively
specified larger likelihood assessment, requiring a new approval.

Passing this subset does not certify the remaining 19 posteriors. Main-case
zero evidence or unstable/degenerate support remains explicit and blocks
strong mechanism claims for that case. Held-out value intervals condition on
one estimated posterior and cover rollout uncertainty; they do **not** cover
all likelihood/posterior-estimation uncertainty. Report that limitation beside
the gaps. No finite-sample stability result licenses a broad abstraction verdict.

## Executor, timer controls and independent checks

Use the optional ranked shared-cache executor only if PR #126's exact-rank,
336-case, fresh-process, coupled-suit and real-clock checks pass and workload
benefit is measured. Otherwise use the validated #121 original-ranker cache.
Neither choice changes the configured five-second soft rule.

Finite-corpus fixed-work equivalence is not unconditional real-clock
equivalence. Before stability/main posteriors, prospectively compare original
and candidate on every prior public LBR prefix, three hash-ranked compatible
holdings (lowest/median/highest) and two independent seeds, using the same
real-clock work. Retain any completion/action difference. Budget this check
explicitly at **240 seconds before headroom** until measured. All original
recorded B100M prefix batches must also be verified for complete/limited work.
If any timer mismatch or recorded limited prefix appears, stop before values
and return a revised **original-ranker** likelihood plan for approval; do not
silently reinterpret the observed attacker. Monitor main simulated limited
batches too. A later limitation also stops inference. For untested hypothetical
states, faithfulness remains an empirical finite-validation claim; do not
describe host-dependent elapsed-time behavior as a mathematical identity.

The four first-per-street stability coordinates are also coupled global-suit
controls. Recompute their four-sample likelihoods under the same suit/deck
permutation and coupled streams; add their likelihood cost explicitly. Run
96 worlds **per range** per suit control, **768 control worlds** total, and
require matching keys, menus, saved probabilities, coupled posterior weights,
selected actions and action returns within the declared 1e-10-chip numeric
tolerance. The three primary preflop decisions without prior LBR actions are
additional uniform/posterior identity controls; do not omit them as uninformative.

The proposal JSON fixes three river decisions by the lowest SHA-256 of
`posterior-river-reference-v2|<original selected rank>` among primary river
coordinates; their full coordinates and 100,000-node ceiling are recorded.
Use an independent native-settlement enumeration of compatible holdings for
terminal fold/call and matched checkdown action-value subproblems. Freeze
that boundary and a 100,000-node-per-reference ceiling before running it.
It is an independent check of rank/chip/value plumbing, **not an exact full
continuation reference** where future strategic actions remain. Retain a
reference that hits its node ceiling as incomplete. No material unexplained
control/reference discrepancy can support a training recommendation.

## Host, safety deadline and recovery proposal

The completed PR #126 report supplies measured workload projections,
including main likelihood, the 28-sample stability work, separate coupled-suit
likelihood work, the 240-second timer cross-check, historical 1,728-second
primary-value proxy, the 1,200-second control/report reserve and 1.25×
headroom. The value/reference allowances and full cache growth are not fresh
measurements. A run longer than ten hours is acceptable if that is the measured
need; duration alone is not a feasibility veto.

Use **one M4 worker by default**: no hourly rental, measured native build and
source/model parity, existing durable archives and clear recovery. For approval,
set the new absolute deadline to the first heavy scientific start plus the
report's rounded-up safety window; record its UTC/Madrid timestamp before
launch. Never reset it. Stop at least 30 minutes before that timestamp for
reporting/seal. Keep AC/caffeinate, aggregate RSS ≤10.5 GiB, swap growth
≤0.5 GiB and free disk ≥8 GiB. Full-workload cache growth remains guarded.
Coordinate every phase through `/tmp/DR_RESEARCH_M4_COORDINATION.txt`.

Atomically commit per-coordinate/action/holding/sample likelihood records
with code/model/input hashes. Checkpoint every completed coordinate and at
least every 15 minutes. On an interruption, validate the last committed shard
and continue only missing deterministic substream IDs under the same deadline;
never rerun completed simulations or select an earlier/favorable checkpoint.
Save all 96 raw world rows/range with explicit failure rows and atomic completion
indexes. M4 supervises, reports and seals independently of an SSH/M1 connection.
Keep raw archives on M4 and retrieve compact manifests/reports only to M1.

A paid CPU pilot would require separate authorization, a **specific live
CPU/RAM quote**, one-worker parity/timing before measured safe parallelism,
aggregate model/cache memory, persistence, transfer and automatic shutdown.
No price, account balance, paid worker or multicore scaling was verified in
this task; the balance is unknown. Do not assume eight workers, GPU benefit,
Claude's old balance or his $3–8 estimate. Unless M4 availability becomes a
material constraint, the measured single-process plan is the preferred executor.

**One next scientific experiment:** this stability-gated paired posterior
comparison. Return its evidence before recommending or starting a training
intervention. An inconclusive result does not authorize same-recipe continuation,
an abstraction change, richer sizing, search, a paid run or promotion.
