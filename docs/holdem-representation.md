# Reliable-target card representation comparison

## Question

Can a wider, deeper or card-separated network transfer reliable six-player river
regret targets better than the current width-32 model? This is a finite regression
diagnostic, not full-game self-play, exploitability or a model promotion.

The [plan](../configs/holdem/representation.json) is frozen before campaign fits.
Use the real engine, observation/candidate schema and utility convention. Reuse
the legal shallow river prefix: six players start with 2 BB, call/check to the
river with 1 BB remaining. Test unopened and facing-all-in roots. Full 100 BB
betting, draws on earlier streets and varied strategic histories are outside this
benchmark. In particular, this cannot choose a history encoder for long games.

## A coherent conditional range

Two declared ten-card templates assign holdings to the five nonhero seats in
seat order. Each template and all 24 global suit permutations have equal prior
mass. Condition this same 48-component joint prior on compatibility with the
hero holding and board, retaining multiplicities. Do not select fresh arbitrary
opponent assignments for each board. This preserves suit symmetry and defines
one predictable target function of visible cards under known fixed assumptions.
It is a small correlated joint range, not realistic independent opponent ranges
or a posterior inferred from real opponents' betting. The forced prefix defines
the diagnostic starting game; it is not evidence about these opponents' actions.

All compatible assignments are enumerated exactly, along with all candidate
continuations under uniform and increasing-index public policies. Opponent reach
weights are included in own-information values. Combine the two profile targets
with weights 1 and 2, as in the previous reference. Regress accumulated normalized
regrets and Q values without recentering or sample-dependent target scaling.
Averaging these frozen targets does not reproduce an online CFR training history.

24 declared boards span pairs, two pairs, trips, connected cards and flush
textures. Deterministic holding construction includes Ac Kd when compatible,
pocket pairs matching board ranks, then suited/random holdings; it uses no target
values. Six holdings per board × two situations gives 288 contexts: 192 training
(16 boards), 48 validation (4 boards), 48 test (4 boards). All worlds of a board
stay in one split; complete boards equivalent under suit renaming or reveal-order
permutation are rejected. Related rank/texture examples across splits are
intentional transfer tests, not independent poker samples.

The earlier Qs Js 8d 5c 2h / Ac Kd case is included in the test set under the new
shared range. Its old 1 BB fold advantage belonged to a different range and must
not be assumed here. The old regression test/results remain intact.

## Five models

| Variant | Parameters | Change |
| --- | ---: | --- |
| original | 25,602 | Current width-32 topology and inputs |
| scaled | 25,602 | Same topology; fixed numerical transformation |
| wide | 69,634 | Scaled inputs, width 64 |
| deep | 68,738 | Scaled inputs, width 48; two residual context blocks and one residual action block |
| cards | 73,698 | Scaled inputs, width-64 history; separate two-layer card branch and numerical branch |

Scaling applies asinh to BB-valued inputs (including event/action amounts), and
divides table-size counts by six. Ratios and categorical/card indicators remain
unchanged. No normalization is fitted to validation or test data, and targets
remain in BB. The card branch consumes the existing four canonical card bags;
it does not receive hand strength labels, ranges or hidden holdings. All variants
retain legal action-conditioned heads and the existing regret-matching rule.
The larger models are within 6% of the wider model's parameter count. Their
runtime is measured separately; this is not an equal-compute comparison or a
pure causal isolation of depth from bottleneck width.

## Fitting and selection

Three initialization seeds: 811, 821, 823. For every variant, use 1,024 Adam steps,
LR 0.001, clipping norm 1, batch size 32 and the existing joint regret/value loss.
Within each seed, all models see the same independently seeded minibatch-index
sequence. Original/scaled share initial parameters; different shapes cannot share
identical initial functions. Fresh fits use only training contexts. Record train
and validation metrics at 0, 128, 512 and 1,024 steps. No early stopping, fit
extensions or learning-rate tuning based on validation results.

Report relative regret RMSE, regret-matching policy TV, fixed-continuation
decision cost in BB, per-board/per-holding predictions, parameters, clipping and
wall time. Decision cost is max Q minus the policy-weighted Q under the declared
weighted continuations; it is not exploitability or CFR's training objective.
Also show the exact regret-policy cost so imperfect agreement with a greedy Q
policy is not automatically called a regression failure.

A candidate qualifies on validation only if, for each of the three seeds, its
final mean decision cost improves over original by at least 0.02 BB and at least
10%, its relative regret RMSE is no more than 0.02 above original, and its training
relative RMSE is at most 0.20. These are finite diagnostic thresholds, not
statistical population confidence or a professional standard. Average contexts
equally (each board contributes the same number). Pick the qualifying candidate
with lowest mean validation decision cost, then parameter count, then name.
Record the selection before opening test predictions. Only that candidate and
original are evaluated on test, once at the final checkpoint. Require the same
cost/error criteria there; no replacement candidate after a test failure. If none
qualifies, evaluate original alone to retain a baseline and report no selection.

Test references are computed and stored during reference generation, but never
used in fitting, selection, calibration or architecture tuning. Do not inspect
their metrics before selection. Retain every fitted model and validation result,
including failures. A test pass licenses discussion of an integration/online
comparison, not automatic production promotion. A failure ends this bounded
comparison without a broader architecture sweep.

## Cost, validation and recovery

CPU only, one thread, local. A separate calibration uses two training contexts,
seed 809 and eight fitting steps per model; only timings and parameter counts
inform admission. Require projected reference work below 1,800 seconds and
projected fits below 1,800 seconds, each with 50% allowance. Main stages have
separate hard deadlines of 1,800 seconds each; preserve partial results and do not
extend them after outcomes. No rental, GPU usage or original campaign seeds in
the calibration. Save source/config/environment fingerprints, context/range
hashes, exact targets, learning curves, all 15 weights and a selection record.

Test hidden-world observation equality, full range suit invariance, compatible
card removal, split separation, exact weighted values, unchanged baseline model
behavior and deterministic fitting. Reload all model weights in a fresh process
and reproduce stored metrics. Verify artifact hashes and recompute selection
before committing the report. The refactor of the existing river builder must
preserve its old contexts and reference tests. Review and merge one results PR
after exact-head CI passes; update ROADMAP.md with the actual outcome.

```bash
python -m scripts.check_representation --plan configs/holdem/representation.json \
  --out results/representation
```
