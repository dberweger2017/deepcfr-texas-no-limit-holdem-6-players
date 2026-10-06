# HU20 trainer bench protocol

#149 found that board-blind pooling is not what keeps the HU20 blueprint weak on its limped, checked-through turn line (D=0.19). A v1 strategy fitted on half the boards loses 0.66 BB on the other half, while the trained blueprint loses 1.42 BB. Better v1 strategies exist; the trainer does not find them. This bench asks why, without a new full-game training run.

## Question

When the production CFR update rules train densely on these exact turn spots, with ranges held fixed, do they reach the quality of #149's held-out witness? Or do they land near the blueprint even here?

- **If they reach the witness,** the rules are sound, and the full-game training conditions are the bottleneck: too few visits per key, budget, ranges that keep moving.
- **If they land near the blueprint,** the rules themselves converge to a poor v1 strategy. Then the averaging variant separates an averaging defect from a deeper problem with CFR in this abstraction.

## Design

- **Spots and inputs:** the frozen #149 corpus of 40 limped, checked-through turn roots and its two frozen halves. One lineage (B500M seed 2026093001) supplies the ranges, taken unchanged from #149's prepared requests: the ranges its exact equilibria used.
- **Trainer:** `src/diagnostics/subgame_bench.py` applies the production rules from `src/blueprint/solver.py`. That means external sampling with the traverser branching on every action, opponent and chance sampled, regret and average weights linear in the iteration, regret matching, and the production v1 information keys and action menus. Each iteration deals one root from the training half (uniform; all corpus weights are equal), samples both holdings jointly from the frozen ranges with card removal, deals a fresh river, and traverses once per seat. Updates are applied after both traversals, as in production.
- **Two averages from one run:** regrets do not depend on how the average is kept, so one run records both.
  - **Traverser-reach:** the production rule. The average is accumulated at the traverser's nodes with weight t times own reach. Because those nodes are reached through sampled opponent and chance actions, the average is also weighted by that sampled reach.
  - **Opponent-sampled:** the standard external-sampling average, with t times the current policy added at each sampled opponent node.

  Exports cover both averages and the current (regret-matched) policy.
- **Budget:** one training run per held-out half, each on the other 20 boards, with 3,000,000 iterations and checkpoints at 100k, 300k, 1M and 3M. On the M4 that is about 200 iterations per second per process, so about 4 hours with both halves in parallel.
- **Scoring:** the #149 lock-only evaluator, with the native binary, prepared compact trees and main-06 collect equilibria as references. One native pass per held-out root scores all 12 exported policies with uniform fallback on absent keys, exactly like #149's held-out witness. That is 40 roots at about 2–3 minutes each.

## Decision rule (fixed before outcomes)

On the 40 held-out boards, paired with the same boards' #149 values for this lineage (B = blueprint, L = per-root witness, P = held-out witness), compute each policy's loss E and its placement **Q = (E − P) / (B − P)**, with a 2,000-draw paired board bootstrap.

The primary estimate is the **3M traverser-reach average**, the production rule:

| Outcome | Reading | Next step |
|---|---|---|
| Primary Q ≤ 0.3 | Production rules reach witness quality when trained densely on fixed ranges | Full-game conditions are the bottleneck: native trainer throughput, visit allocation to turn/river, pruning |
| Primary Q ≥ 0.7, opponent-sampled Q ≤ 0.3 | Averaging defect | Fix the average and retrain |
| Both ≥ 0.7 | CFR's v1 fixed point is poor even with dense training | Abstraction-aware training (e.g. CFR-BR, asymmetric abstraction) or a better abstraction |
| Otherwise | Mixed | Read the learning curve: still falling at 3M means budget-limited |

The 100k/300k/1M checkpoints show whether the result is still moving. The current policy is reported, not used to classify.

## Limits

- This is one public line, one lineage and fixed ranges. It isolates turn/river learning from range co-evolution and earlier streets, which is the point, but it does not reproduce full-game training.
- The witnesses are feasible constructions, not optimal v1 strategies.
- The bench trains on 20 boards; the full game pools many more boards per key.

## Resources and outputs

The run uses the M4 only: two single-thread trainers (about 150 MB each), then one native evaluation at a time (about 4.5–6 GiB, 7 GiB ceiling, at least 20 GiB free disk). No paid compute. `scripts/run_subgame_bench_m4.sh` runs everything and resumes after an interruption. Results go to `runs/` (exported policies, state, status) and `eval/` (per-root native results and `summary.json`) in the dated work folder.
