# Stored CFR-average extraction experiment

I compare the three original B500M current exports with a diagnostic extraction
of the retained trainer average, on M1. No training, promotion, paid compute,
M4 computation or change to Doctor Research's #136 campaign. Closed checkpoint
transfers require coordination and local byte/hash verification.

## Extraction and verification, before evaluation

`solver._collect_root` accumulates `iteration * own_reach * policy[action]` at
traverser nodes. Own reach multiplies only that player's prior action
probabilities; opponents are externally sampled. Complete iteration deltas are
added to `Node.average`. Extraction normalizes that stored nonnegative vector.
This is the trainer's lifetime reach/iteration-weighted average, not a windowed
policy, mean of checkpoint strategies or proof of equilibrium convergence.
Sampling frequency and the existing abstraction remain part of the algorithm.

Keep the production short-stack current-only export guard **unchanged**. Add a
separate versioned, diagnostic-only HU20 average artifact and adapter. Production
`FrozenBlueprint` must reject it. No current model replacement or UI integration.

Before any comparison, verify all checkpoint hashes, game/schema/table/seed/
iteration identities, unique keys and action labels; require finite nonnegative
accumulators and nonnegative integer visits. Per node, total average mass must
not exceed `last_iteration * visits` within a stated floating tolerance, since
each visit adds `iteration * own_reach <= last_iteration`. This bound cannot
reconstruct every historical increment; report that limitation. Zero mass has
undefined normalized average and gets explicit uniform fallback within that
known menu; distinguish zero mass from missing-key fallback. Verify normalized
probabilities directly against **every** retained accumulator. Independently
match current regret-matched probabilities and coverage against the same retained
checkpoint. Use exact synthetic traversal fixtures to test own-reach propagation,
iteration weights, accumulation and normalization without new campaign training.

## Prospective measurement

First use a small, separate timing pilot; do not use pilot returns to choose the
final budget. Then commit a fresh paired deal root and fixed block counts before
final evaluation. Compare all three current/average pairs on identical deals,
two seat rotations per block, separate deal/target/opponent streams. Reset 20BB
per hand, no rake/ante. Keep panels separate: uncapped uniform random, existing
passive/minraise/restricted-pressure controls, six existing styles, exact native
pressure and frozen selective-stackoff. Clearly identify the original-cap2
reactive menu controls and native style sizing. No action translation.

Include the optional exact-ranker/cached bounded LBR only if a measured pilot
projects the fixed panel within a 20-minute final-work allowance. Use four chance
samples/holding and the existing five-second soft batch budget; retain timing
limits and incomplete batches. A limited response is not an exact best response.
One policy loaded at a time on M1, process peak guard 6GiB, absolute final worker
limit 30 minutes. No outcome-dependent extension, seed selection or tuning.

Report paired average-minus-current overall/position BB/100 and intervals by
lineage; aggregate the three fixed lineages within shared deal blocks, never pool
different panels. Exploratory unadjusted Student-t 95% intervals, no strength
claim. Preserve all negative/inconclusive results, hand records/native replays,
large-action opportunity denominators and responses, full-stack wins **and**
losses, street/mass/fallback coverage, distinct reached keys, model/checkpoint
hashes, environment and resources. Whole-hand subgroup returns are not bet EV.
One focused draft PR, all evidence/reproduction instructions, no automatic merge.

## Final budget, frozen from the timing pilot

The separate first-lineage timing pilot completed 208 hands across current/average
and all 13 panels in 32.08 seconds including both loads. Eight LBR hands took
1.992 seconds current and 3.615 seconds average. Using the slower measured LBR
rate, the maximum cheap-panel rate, six loads and 1.25× headroom projects about
18.3 minutes. Pilot payoffs are excluded from final inference.

The [final plan](../configs/diagnostics/b500-cfr-average-comparison.json) fixes
256 paired blocks/policy for each of the 12 cheap/style/native-pressure/stackoff
panels and 128 for LBR: **38,400 hands**, all three lineages, root `202610050201`.
Both positions are played in each block. LBR stays a separate bounded diagnostic;
its sampled range/checkdown assumptions and any limited batches are reported.
No outcome-dependent budget changes. Absolute final worker guard remains 30 minutes.
