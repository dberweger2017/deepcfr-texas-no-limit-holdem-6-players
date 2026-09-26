# Frozen conditional river comparison for PR #108

The [development curves](reports/river-development-m4.md), [nonuniform
amendment](reports/river-range-amendment-m4.md), and [high-world rollout
calibration](reports/river-rollout-highworld-calibration-m4.md) select this
comparison before any confirmation returns are inspected. The solver, river
action menu, 12M blueprint, and active-range approximation are fixed. No
blueprint training, paid host, model promotion, or turn/multiway feature is
part of this run.

## Candidate and controls

- **Candidate:** solve the complete declared public range at the river root,
  using the own-reach-weighted average profile after a maximum of 30 seconds
  including range and tree construction. Require at least 32 complete sweeps;
  cap at 100,000. Retain that one profile through all on-tree actions. An
  off-tree action delegates to same-law corrected rollout with the normal
  budget and is counted.
- **Normal control:** existing corrected rollout mechanism with eight worlds
  and 0.5 seconds per decision, using the same declared compatible-card law.
- **Compute control:** the same corrected rollout mechanism with 8,192 worlds
  and 30 seconds per decision. The development calibration measured
  22.5–25.7 seconds for 8,192 worlds on three roots. Actual time, completed
  worlds and fallback are reported on confirmation. The 8,192-world allowance
  is local to this evaluation adapter; the ordinary player cap is unchanged.

The CFR budget is **once per public root** and the rollout budgets are **per
hero decision**. The comparison records this cost distinction; it does not
claim equal total experiment CPU or identical live-use latency on hands with
multiple hero decisions. Each player sees only its observation. The evaluator
uses hidden deals to run the native game, never as policy inputs.

## Freeze and analysis

The [confirmation cases](../configs/blueprint/river-confirmation-cases.json)
contain 32 held-out boards: eight each from dry, paired, four-flush, and
connected families. They vary root pots (2, 4, 6, 12 BB), hero position,
stacks, and six scripted opponent styles. None of their complete boards is in
the earlier reference or development sets. The
[confirmation plan](../configs/blueprint/river-conditional-confirmation-m4.json)
predeclares eight independent deal repetitions per root (256 paired deals),
three arms, independent deterministic seeds for deal, opponent, and each
player, a five-hour overall wall guard, 10.5-GiB peak process RSS guard, and
30-GiB free-disk guard. Sample each complete deal from the normalized
product-compatible public joint distribution `Q`; use that same deal and
opponent seed in all three arms. Persist the sampled deal and every attempt.
No confirmation result may be used to change the cases, seeds or budgets.

The primary estimate is the equal-root-weighted mean of the eight paired
candidate-minus-control incremental river returns in BB, reported separately
against each control. Resample the 32 **roots**, keeping all eight deals
within each root, for 20,000 fixed-seed percentile bootstrap replicates.
Use 97.5% two-sided intervals for each of the two comparisons (Bonferroni
family coverage at least 95%). Report pot-normalized differences, individual
arm returns, root variation, position/style breakdowns, legal actions,
solver and rollout fallback/completion, wall time, RSS, swap, and all failures.
An incomplete run is an incomplete comparison, never silently reduced to
surviving hands. A positive lower interval bound against both controls, with
complete valid hands and resource compliance, would justify considering a
later integration decision. It does not by itself qualify six-player
whole-hand strength or v0.5. A nonpositive bound or adverse effect is retained
as an inconclusive or negative result.

The [preflight plan](../configs/blueprint/river-conditional-preflight-m4.json)
uses two previously examined roots and one deal each to exercise all three
arms before the long confirmation. Its returns are resource and legality
checks only; they are excluded from the confirmation estimate. The long M4
budget requires the owner's decision under ROADMAP.md's **How we work**.
