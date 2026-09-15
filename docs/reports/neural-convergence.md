# Neural convergence and resume validation

## Declared protocol

Commit this protocol and both `neural-*-convergence-v1.json` configurations before running their training seeds. These checks complete roadmap milestone 3 only if every declared seed passes. The prior single-seed pilots and frozen-replay diagnosis remain separate results.

### Acceptance runs

| Game | Seeds | Iterations | Traversals per player/update | Advantage steps per update | Strategy steps per evaluation | Reservoir capacity | Final exploitability limit | Final value-error limit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kuhn | 11, 29, 47 | 100 | 1,024 | 1,000 | 6,000 | 100,000 | 0.03 | 0.03 |
| Leduc | 11, 29, 47 | 120 | 1,024 | 4,000 | 6,000 | 200,000 | 0.15 | 0.10 |

All runs use the existing two-hidden-layer network, hidden size 64, batch size 256, Adam 0.001, alternating updates, uniform reservoirs, and iteration-weighted losses. Evaluate every 20 iterations and at the declared final iteration. Keep the initial policy convention and sampling method unchanged. The greater advantage fitting budget follows the previous frozen-replay diagnosis; it was not chosen from these new seeds.

Exploitability is half NashConv, in antes per hand, evaluated using exact information-set best responses to the **actual neural average policy**. Value error is the absolute difference between its first-player value and the independently solved equilibrium value. The limits match the earlier sampled tabular reference's acceptance limits. They establish a small-game convergence gate, not optimal play or a six-player strength guarantee.

Every seed must meet both limits at the final iteration. Do not select intermediate checkpoints, omit failed seeds, pool results to rescue a failed seed, extend budgets, or relax thresholds after viewing results. Report each seed, the mean, sample standard deviation, and range. These are three training seeds, not a precise estimate of all possible training outcomes.

### Reference and diagnostics

Pin `docs/reports/tabular-validation.json` by SHA-256 in both campaign configurations. Check that its underlying solver/game/evaluation source files still match the retained reference. Recompute the sequence-form equilibrium for each game and require primal/dual agreement, constraint residuals, exact exploitability, and agreement with the stored equilibrium value within 1e-8.

Compare neural results with the full-tree and external-sampling tabular results already retained in that report. This is a correctness/convergence comparison, not a paired-seed or equal-compute race. The tabular solvers use simultaneous updates, uniform iteration averaging, exact strategy tables, and different traversal/iteration budgets; the neural implementation uses alternating updates and linear weighting. Do not infer relative large-game efficiency from toy-game timings.

Retain all scheduled neural, empirical-memory, and exact-played-average evaluations, every advantage fit, final strategy fitting error, sample-noise MSE, replay coverage/counts, artifact hashes, and runtime/source/configuration fingerprints. Explain failures using those diagnostics without replacing the measured playing policy with a diagnostic table.

### Resume and resource limits

First require tests of complete replay/RNG restoration on Kuhn and Leduc, actual reservoir replacement, fresh-process CLI resume, byte-identical final inference files, and equality of every deterministic report field. Simulate an interrupted update and verify recovery uses the last completed snapshot. Check hash/contract mismatch rejection, preserved elapsed budget, and atomic publication without overwriting old snapshots.

During the Leduc seed-11 campaign, stop at iteration 60 and continue in a new process from its saved state. Retain the paused bundle and resumed bundle. The saved budget carries forward; this split does not permit additional iterations, optimizer steps, or a new seed. Exact uninterrupted/resumed equivalence is checked by the short regression scenarios, rather than claiming that this one split alone proves it at campaign scale.

Run each seed as a separate sequential local CPU job, with one Torch thread and an 800-second training budget (40 seconds of margin under the 840-second command target). Maximum declared training allocation is 4,800 seconds across six seeds; this is six bounded jobs, never concurrent. The iteration-60 split shares seed 11's original budget. Stop on invalid states, non-finite values, contract changes, or exhaustion. Retain a failure/timeout and continue the other predeclared seeds; no paid compute or GPU rental.

Snapshots are written at scheduled evaluation boundaries and explicit stops. Keep immutable files and a pointer to the last fully published snapshot. Interrupted work after that snapshot must be repeated. Recorded elapsed compute through the snapshot carries forward; work lost to an abrupt process kill cannot be recovered from that file. Keep the original failed attempt for accounting. This implementation resumes between completed iterations, not halfway through an optimizer fit.

## Results

Results will be appended after the declared checks and runs.

## Follow-up declared after Leduc seed 11

Seed 11 completed 120 iterations with neural exploitability **0.169868**, above the 0.15 limit. The empirical strategy-memory average was **0.145379** and the exact played average **0.137689**. Final strategy fitting excess MSE was **0.004197**, with 287 of 288 information sets represented in its retained strategy replay. Keep this campaign result as failed.

Before running any refits, declare a separate diagnostic on that seed's final **frozen strategy memory**. Use `neural-strategy-refit-v1.json`: fit hidden sizes 64 and 128 for 24,000 steps each, preserving the original learning rate, batch size, iteration weighting, initialization seed, and minibatch stream. The 64-wide fit isolates additional optimizer steps; the 128-wide fit also changes capacity. Check the loaded baseline against the retained evaluation and hash replay before/after. Record both excess MSE and exact exploitability, including whether each reaches the original seed's 0.15 limit.

This is a diagnosis on the first declared Leduc seed, not a new convergence campaign or a model promotion. There is no best-checkpoint selection, additional self-play, policy export, or change to the original seed's assessment. A result below the limit would motivate a separately declared multi-seed confirmation, not retrospectively pass this campaign. Retain both refits whether they help or hurt. Run the diagnostic once, sequentially after the acceptance seeds, with a 180-second CPU limit and no paid compute.
