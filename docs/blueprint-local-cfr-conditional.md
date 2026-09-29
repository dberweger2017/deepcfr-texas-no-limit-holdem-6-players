# Conditional local CFR diagnostic

PR #106 delivered an opt-in first-decision flop solver. Its revised source
completed only three eligible solves in the short M4 smoke. This diagnostic
selects eligible observations directly, so it can measure the solver without
spending most of the budget on full hands where the pilot never runs. It does
not estimate BB/100, compare poker opponents, or promote a player.

## Frozen design

The [M4 plan](../configs/blueprint/local-cfr-conditional-m4.json) fixes the
12M blueprint hash, 12 deal seeds, four public flop-action patterns per deal,
and one search seed for each of the 48 situations. The [case file](../configs/blueprint/local-cfr-conditional-cases.json)
was generated and committed before loading the checkpoint or solving. It
contains each deal seed, button, public prefix, target position, hero seat,
search seed and observation hash. The deal seed lets the evaluator reproduce
hidden cards, but only the resulting `Observation` is passed to the solver.
The generator uses fixed card-independent preflop actions to leave three
players at the flop root. It covers the hero's first action, one preceding
check, one preceding minimum raise, and two preceding checks, with 12 cases
each. Every case is checked for pilot eligibility and exact observation
reproduction before either solver condition runs.

For each observation, run `targeted_traversal=true` and `false` at the same
five-second limit, 96 sampled holdings per nonhero seat, 32 minimum and 128
maximum full per-player cycles. Both conditions use the same starting search
seed; their later random streams can diverge because the targeted pass
consumes extra draws. Alternate condition order across cases. The overall
wall limit is 15 minutes, process RSS limit 10.5 GiB, free-disk floor 30 GiB,
and no paid host. Keep every completed row and explicit stop status. TensorBoard
flushes condition summaries after each eight case pairs.

## Measures and interpretation

Record completion, timeout, target-information-set absence, cycles, latency,
peak RSS, sampled nodes, leaf choices, number of hero action information sets,
visits to the actual decision, and distinct hero holdings visited at that
public decision. Sum the public *root-range prior mass* of those visited
holdings. This is a useful-work measure, **not** a collision-conditioned or
opponent-reach-weighted probability of covering the true decision range.
Record the actual decision policy and L1 movement from cycle 64 to 128,
plus trained/untrained/off-tree continuation lookup calls. Lookup fractions
count repeated calls, not distinct leaves. Report how often both targeting
conditions produce a playable target policy and their policy L1 difference
where both do.

The conditional sample describes these 48 scripted public situations. It
cannot establish a whole-game completion rate or playing strength; 128 cycles
and policy stability are diagnostics rather than convergence certificates.
Targeting off may fail to visit the actual hero information set. That is an
expected, informative result, not a reason to discard the row or extend the
run. A later continuation-quality experiment should change the continuation
model as its own controlled factor rather than infer causality from lookup
coverage alone.

```sh
python -m scripts.diagnose_local_cfr \
  --plan configs/blueprint/local-cfr-conditional-m4.json \
  --cases configs/blueprint/local-cfr-conditional-cases.json \
  --checkpoint /Users/dberweger/Local/blueprint-04-backups/checkpoints/c6649a3e8ab061c70c5f6192d91f1f4090f15d68f4821cf3b50f2746f816c845.json.gz \
  --out results/local-cfr-conditional-m4
```
