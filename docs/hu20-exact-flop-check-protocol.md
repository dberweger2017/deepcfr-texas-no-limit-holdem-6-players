# HU20 exact flop-subgame diagnostic protocol

Design and selection frozen on **2026-10-02**, before any main-run value was
computed. **Execution is not admitted:** every real preflight spot exceeds the
measured M4 budget, including the cap-three fallback. This protocol authorizes
no paid host. Changing host, memory ceiling, tree, ranges or selection requires
a recorded amendment before main results are inspected.

## Population and frozen inputs

Use the six exports in
[`configs/diagnostics/hu20-exact-flop-check-inputs.json`](../configs/diagnostics/hu20-exact-flop-check-inputs.json):
B500M current and the retained #141 stored-average extraction for seeds
2026093001, 2026093002 and 2026093003. Verify every SHA-256 before use. Report
current and stored-average results separately; current is primary. B100M is
omitted because the cost preflight fails. No policy is trained or promoted.

Set A is the **214 unique roots** behind the pinned **273 v1 LBR-facing flop
decisions** in #143. Its retained manifest also contains source-file hashes,
source-line/action references and empirical opponent holding counts grouped
by public preflop line and opponent position. Condition those counts on each
root board, report compatible sample counts and do not smooth a missing range
into an apparently observed empirical range. This secondary range is sparse
and comes from the B100M evaluation, not a B500M LBR evaluation.

Set B is **150 unique roots** sampled from 10,000 fresh preflop self-play deals
using the first current B500M export, with both players using that same policy.
Deal/action/selection seeds are **202610020001 / 202610020002 / 202610020003**.
Stop each deal at its live flop root; inspect no postflop action or profit.
There are 5,665 root occurrences and 5,512 unique roots in the source corpus.
Sample uniformly within pot type and physical button rotation; weight each
selected root by its corpus multiplicity divided by inclusion probability.
Evaluate **both target positions** at every root. Rotation is a reproducibility
stratum, not an additional strategically distinct poker position.

| Pot type | Button 0 | Button 1 |
| --- | ---: | ---: |
| Limped | 19 | 19 |
| Minimum single raise | 20 | 20 |
| Pot single raise | 18 | 18 |
| Three-bet or deeper | 18 | 18 |

Set A and B have no shared selected root. The planned corpus is 364 roots ×
six policies = 2,184 conditional subgames, with two target seats each. Preserve
the common corpus across policies and report incomplete coverage. Never treat
lineages, seats, branches or runouts of one root as independent spot samples.
The manifests are [`set-a.json`](reports/hu20-exact-flop-check-artifacts/set-a.json)
and [`set-b.json`](reports/hu20-exact-flop-check-artifacts/set-b.json).

## Values and projections

Root ranges are `public_ranges` from the evaluated policy's preflop likelihoods.
Primary values use those ranges for both players. Secondary values replace the
exploiter root factor with the compatible empirical LBR factor while retaining
the target's blueprint factor; recompute the equilibrium baseline under that
same joint range law. Missing empirical support is reported as missing.

Use native uncapped `choices()` lines and exact integer street raise-to amounts,
100 chips/BB, no rake, solver seat 0 OOP and seat 1 IP. Freeze the upstream solver
at `9d1509fe5077d019825f833eed04b16d342dfda1`. Center solver and native utilities
by subtracting half the root pot. The centering cancels in exploitability gains.

Solve to ≤0.2% root-pot equilibrium exploitability. Record achieved residuals
for every spot. A spot above 0.5% is retained and excluded from the decision
rule; label spots between 0.2% and 0.5% as target misses. No convergence-based
checkpoint selection across policies. Set a declared iteration/wall ceiling in
the execution amendment once real admitted solve timings exist.

For each target, lock only its nodes and leave the responder unlocked when
calling `compute_mes_ev`. `e_bp` is responder BR minus responder equilibrium
value, in BB and % pot. With both players locked, use `compute_current_ev` for
the end-to-end native Monte Carlo comparison. MES respects the responder's own
locks, so locking both players would not measure the requested BR.

The actual v1 key also aliases **different betting lines**, for example a 100-
and 200-chip river bet in the retained
[`alias fixture`](reports/hu20-exact-flop-check-artifacts/public-line-alias.json).
The requested per-line projection is therefore a relaxation. Retain both:

- `e_v1proj`: pool equilibrium action distributions by the **full actual v1
  key**, over aliased lines and every turn/river runout within this flop. This is
  the feasible v1 projection used by the corrected primary ratio.
- `e_v1proj_line`: the literal requested projection within each exact line,
  pooled over runouts. Report its separate ratio as a sensitivity analysis;
  do not describe it as a feasible policy for the full v1 key.

Weight groups by root holding mass × equilibrium own-action reach. Within a
street, constant chance factors cancel. Zero-reach groups use uniform action
probabilities. Blocked holdings have zero weight and no card/bucket key. Preserve
action-name correspondence when solver action order differs from native order.
This projection still relaxes the real key's pooling across different flops.

Preview equity projections use K = **50 and 200** per street for this flop.
Build **20 equal-width equity histogram bins**, exhaust all legal final boards,
and calculate river equity against uniform compatible opponent holdings. Flop
histograms average over final boards; turn histograms average over legal rivers.
Cluster flop contexts and all turn contexts separately with mean centroids and
L1 distance between cumulative histograms, seed **202610020004 + K** (turn +1),
up to 100 iterations. This specifies an EMD-assignment k-means convention,
not an assertion that arithmetic means minimize the EMD objective. River
quantiles pool all final-board/holding contexts for this flop. Keep ties together;
report occupied buckets when fewer than K are occupied. Preserve the same
public-history aliases as v1 so the comparison changes card information.

At every target node facing a flop bet, compare blueprint and equilibrium fold
probabilities on the **same equilibrium-conditioned compatible joint range**.
Retain per-node frequencies and excess blueprint-fold probability grouped by
uniform-opponent flop equity decile and v1 descriptor. Pool the Set A fold gap
using root corpus multiplicity and decision reach; show the pinned selected
nodes separately. This is not an MDF test. Report zero-reach nodes as unavailable.

## Gates and M4 execution admission

K: ≥100,000 production/factored-key samples, zero mismatches. V1: all declared
solver lines, terminal flags and action/chip amounts match native replay, zero
mismatches. V2: integer terminal settlements match native fixtures; retain f32
rounding error. Early all-in terminals average undealt cards, so a single fixed
runout is not their expected payoff. V3: unlocked responder MES agrees with
native river `profile_quality`. V4: both-locked EV is inside the 95% interval of
≥20,000 independent native deals from the same root/ranges. V5: retain every
equilibrium residual and apply the exclusions above. Missing/failed gates stop
admission; preserve each attempt and fix/revalidate before continuing.

Before work retain other processes, RSS, memory pressure and swap. The M4 budget
is **min(6 GiB, floor(80% measured reclaimable GiB))**, where reclaimable is free
+ inactive + speculative pages, with an absolute ceiling of 10 GiB. Measured
admissions were 6 GiB initially and 5 GiB on the instrumented repeat. Use one
nice-10 solver and `RAYON_NUM_THREADS=2`, admitted from the initial idle-core
snapshot. Recheck idle cores if other work starts. Stop owned work on aggregate
RSS above budget, swap growth >1 GiB from the recorded run baseline, non-finite
values or gate failure. Write results atomically; verify retained request hashes
before reuse and preserve a new attempt when tool or budget changes.

Call `memory_usage()` before strategy allocation. The pinned constructor
allocates its node arena **before this API can be called**. If a provable metadata
lower bound already exceeds budget, reject construction and explicitly report
lower bounds rather than fabricate API estimates. The preflight's rainbow flop
has no suit isomorphisms under the pinned source conditions.

Fallback ladder: 16-bit compression; remove only zero-weight holdings; declare
per-street cap three with legal jams retained; otherwise oversize. A capped spot
also needs blueprint and equilibrium removed-target-reach audits, with >2%
excluded. The preflight capped spots remain oversize before strategy allocation,
so no such audit is claimed. **No paid compute without a separate approved
quote.** Shrinking Set B cannot solve per-spot oversize. Projected solve time is
unknown, not zero; main run cannot start under the 24-M4-hour condition.

An admitted execution amendment must provide solve ceilings and projected total
wall time. If it exceeds approximately 24 M4 hours, present a smaller Set B for
owner decision before starting. Interleave complete cycles over set, pot stratum,
lineage and extraction with ordering seed **202610020006**, and both target seats
within each subgame. Keep partial counts balanced and monitoring provisional.

## Inference and monitoring

Set B current B500M pooled is primary. Report per-lineage and pooled weighted
means; bootstrap independent roots with **2,000** resamples, seed **202610010905**,
resampling within sampling strata and reusing each sampled root's
lineages/targets together. A singleton stratum has no reported bootstrap interval.
Report current and
stored-average separately. Keep exclusion counts, residuals and partial coverage
beside every summary. Bootstrap intervals quantify root variation, not solver
rounding or systematic range error.

Freeze the requested thresholds exactly:

- R = mean `e_v1proj` / mean `e_bp`.
- R ≥ **0.7** with material `e_bp` (descriptive flop share ≥**25%**) → H1,
  abstraction-limited; recommend fixed-K equity buckets plus exact-card flop
  search rather than longer v1 training.
- R ≤ **0.3** → H2, training-limited; recommend a native fast trainer first.
- Otherwise → mixed.
- Flop share <**10%** → H3; recommend a preflop range audit.
- Set A equilibrium fold frequency within **3 percentage points** of blueprint
  → H0 for the selected overfold question.
- Report mean `e_eq200` / mean `e_v1proj` as bucket headroom.

Report these as heuristic H1/H2-consistent classifications: a projection is a
feasible strategy whose loss **upper-bounds** the minimum achievable abstract
loss. High R cannot prove v1 incapable of better play; low R provides a
constructive feasible witness. State the same limitation for the per-line
relaxation. Give H3 priority when flop share is below 10%; do not divide by a
zero/nonpositive mean `e_bp` or classify an unrun/incomplete experiment.

For descriptive accounting use live flop-root reach **385/768 = 50.13%**;
also show all dealt-flop reach **405/768 = 52.73%**, including 20 preflop all-in
showdowns outside this subgame population. Multiply live-root reach by mean
`e_bp`, compare with the supplied approximate **0.65 BB/hand** LBR loss and label
it descriptive. Different policies, opponent ranges and root occupancy prevent
causal street attribution. Ranges are taken as given, excluding preflop errors;
projections are not abstraction equilibria; per-flop buckets favor this preview
over global blueprint buckets.

JSONL progress and `status.md` identify the stage, resources, residuals, counters
and failures. TensorBoard means/intervals are labelled **monitoring only**, split
by set and extraction. The existing M4 server already listens on
127.0.0.1:6006; append this job's separate run without restarting it. From the M1:

```sh
ssh -N -L 6006:127.0.0.1:6006 m4
```

Open <http://localhost:6006>, run `hu20-exact-flop-check-20261001`. Main progress
comments at 25/50/100% are emitted only if an admitted main run reaches them.
