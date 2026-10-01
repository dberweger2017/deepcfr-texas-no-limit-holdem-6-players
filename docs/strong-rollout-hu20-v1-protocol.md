# Range-aware HU20 heuristic and decision diagnostics

I build `strong-rollout-hu20-v1` as a transparent bounded heuristic. “Strong” is
a goal to check, not a capability claim. One M1 worker, no training/paid compute,
M4 compute or changes to the active #136 campaign. Doctor Research authorizes
only minimal read-only transfers; verify original compressed bytes locally.

## Declared v1 behavior

The [configuration](../configs/diagnostics/strong-rollout-hu20-v1.json) freezes
position-aware SB opens, BB calls, value/blocker re-raises, calls versus re-raises
and stackoff categories. Opening frequency .90; value re-raise .85; blocker
re-raise .25; stackoff .90; allowed range calls .95, otherwise check/fold.
Restricted mode uses the existing uncapped min/pot/conditional-jam menu. Native
mode uses 2.5BB opens, 3× re-raises and postflop min/one-third/two-thirds/conditional
jam. Exact native bounds and engine validation remain authoritative.

A hand-local, 128-particle uniform compatible prior excludes only the observer's
own cards. Public cards remove holdings. Observed rival actions update weights
using a declared heuristic likelihood: preflop range/mixing probabilities;
postflop made-hand/draw tier, pot odds and wager size. This is an **assumed** range,
not the target policy's posterior. A .02 likelihood floor retains alternatives.
Any depleted particle reset is explicit and replays public likelihood updates.
There is no target-policy dependency or cross-hand learning.

Postflop 64 common sampled pair/runout worlds estimate checkdown equity. Raises
are scored with estimated fold mass and equity weighted by estimated continuation
probability, using the exact HU matched-contribution ledger. Log range effective
sample size, equity, pot odds, effective stacks/SPR, board texture/draw/blocker
features, continuation/fold estimates, chip scores and mixing probabilities.
Value/draw/blocker gates and near-best seeded mixing are heuristics, not a solve.

## Calibration criteria, before results

Use both positions on each fixed deal and report modes separately. Development:
16 paired blocks per each of 12 controls; confirmation: 64 fresh blocks each.
Controls are uniform restricted random, six existing styles (project their
preferred legal action to the restricted menu only in restricted mode), frozen
selective-stackoff, min-raise, check/call, largest-raise and unconditional native
jam (restricted jam control takes largest available restricted raise).
All control projection/sizing is recorded; never pool modes.

Engineering acceptance: all actions legal, zero-sum native settlement, independent
replay, deterministic seeds and information-isolation tests. Confirmation quality
goal: positive profit against random/call/minraise in both modes with Bonferroni
family-95% paired intervals for those six checks, and nonnegative point profit
against at least six of the nine other controls in each mode. All per-control
95% intervals, positions, denominators and adverse tails remain visible. Failure
means competence is unvalidated; it does not justify tuning until a target loses.

At most one declared development revision before confirmation, only if necessary;
retain original evidence and describe the reason. No tuning on confirmation or
B100M/B500M. Freeze source/config hashes and calibration outcome before opening
any model comparison. Correctness fixes after freezing require a new freeze and
rerun affected evidence; strategy improvements belong to v2.

## Outcome-blind decision scoring

Sample target decisions by a deterministic coordinate hash, stratified by street
and position, before inspecting their payoff. Each decision's own observation
supplies its assumed compatible range; shared independently sampled worlds and
reset continuation streams compare every candidate and the saved-policy mixture.
Keep native alternatives and restricted alternatives separate. Actual engine
wagers are unchanged; no action translator is added.

The target continues via its saved policy using only its observation; its rival
uses frozen v1 from its own observation. First-half worlds select an alternative;
second-half worlds estimate its paired advantage over the root policy mixture,
with a conditional sampling interval. Never score the maximum on its selection
sample as an unbiased gain. Preserve all world/alternative returns, range and
continuation assumptions, fallback exposure, negative/inconclusive gaps and
representative traces selected without using losses. One realized hidden world
is not expected value, and the assumed-range result is not a proven policy error.

## Runtime and comparison freeze

First run an outcome-free generated-policy timing pilot. After calibration/freeze,
measure a bounded real-policy pilot; keep pilot deals separate from the final
fresh comparison. Freeze practical hand/world/decision counts from runtime before
opening final outcomes. Compare B100M/B500M for each original lineage on identical
fresh position-balanced deals; no seed or checkpoint selection. Restricted and
native rival-sizing modes are separate panels. Report paired block uncertainty,
positions, lineage differences, large-pot/full-stack tails, fallback, conditional
decision-gap estimates and exact traces. No model promotion or strength claims.

Publish one draft PR, all configurations/hashes, generated-fixture checks,
calibration failures/successes, replayable simulator records, resources and
reproduction commands. Preserve raw artifacts and report incomplete tasks honestly.

### Exact postflop coefficients

The assumed continuation base by own made tier is strong .98, top-pair .85,
lower-pair .55, board-pair .30, board-only .18, weak .12. Draws raise it to at
least .62 at price <.32, otherwise .35. Subtract 1.1×max(0, price−.20), bound
.03–.99. Preflop continuations are .96 inside the declared price-dependent range,
.06 otherwise. Public postflop raise likelihood is .55 for strong hands, .22 for
top-pair/draws, .06 otherwise; wagers above one pot multiply it by .75 for strong
hands or .25 otherwise. Check likelihood is 1−aggression; call is continuation
×(1−aggression); fold is 1−continuation, all floored at .02.

Postflop raises qualify at continuing equity ≥.58, a draw with whole-range equity
≥.40, a nut-suit blocker with estimated folds ≥.35, or estimated folds ≥.55.
Select among eligible actions within 50 chips of the best checkdown score using
exp((score−best)/35). Fold scores −own current contribution. Call/check scores
(2×equity−1)×matched contribution. A raise scores fold_probability×rival current
contribution + continue_probability×(2×continuing_equity−1)×matched contribution.
These are explicitly approximate one-step/checkdown utilities, not full strategic
rollouts or a calibrated model of a particular saved blueprint.
