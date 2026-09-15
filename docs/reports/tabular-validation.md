# Tabular CFR validation

## Declared protocol

Commit this protocol and `configs/solver/reference-v1.json` before running convergence experiments. Thresholds are in ante units per hand. Exploitability is half NashConv: the mean gain available to the two players by switching individually to an exact best response.

- Check the game trees, legal betting, card removal, net utilities, information-set boundaries, and perfect recall.
- Require the Kuhn sequence-form equilibrium to have first-player value -1/18. For both games, independently solved primal/dual values, constraint residuals, and best-response evaluation must agree within 1e-8.
- Check exact best response against exhaustive pure-strategy enumeration in Kuhn. Enumerate every external-sampling outcome for a frozen Kuhn profile and compare its expected regret update with full CFR, including zero-probability actions.
- Run simultaneous vanilla CFR with exact own-reach-weighted averaging for 10,000 Kuhn iterations and 5,000 Leduc iterations. Require final exploitability/value error at most 0.01/0.01 and 0.05/0.05 respectively.
- Run external-sampling regrets with the same exact averaging for seeds 7, 19, and 43: 20,000 Kuhn iterations (limits 0.03/0.03), 50,000 Leduc iterations (limits 0.15/0.10). Every seed must pass; no seed selection or pooled rescue.
- Evaluate at iteration zero, every 1,000 iterations, and the final iteration. Acceptance uses the declared final iteration, never the best intermediate result. Retain failures and timeouts without extending budgets.
- Cap each run and the complete sequential campaign at 840 seconds, leaving margin within the agreed 15-minute local job limit. Execute runs sequentially with no GPU or paid compute. Record source/configuration/strategy hashes, runtime versions, seeds, wall time, iterations, and all evaluations.

The sampled solver deliberately retains a full-tree averaging pass. This is a correctness reference, not a throughput benchmark or a claim of fully sampled scaling. These are two-player, fixed-limit toy games; their equilibrium checks do not certify six-player no-limit strength. The neural baseline and multi-seed neural comparisons remain separate roadmap tasks.

## Results

The campaign ran on September 15, 2026 at clean revision `20c90feaada5fe7aa1e331616e4230bd6119adae`. All eight declared runs completed and passed without a budget extension or threshold change. Total runner wall time was **93.89 seconds** on the local CPU. Full-tree runs are deterministic; seed zero identifies them, not an independent source of training variation.

The [compact report](tabular-validation.json) retains resolved inputs, source/environment fingerprints, independent oracle results, final evaluations, strategy hashes, and timings. [Every convergence measurement](tabular-curves.jsonl) is retained, including intermediate regressions. Raw strategy bundles are under local ignored `results/tabular-reference-v1`; they are not included in a fresh checkout.

| Game | Solver | Seed | Iterations | Final exploitability | Limit | Value error | Limit | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |

| Kuhn | full | 0 | 10,000 | 0.002318 | 0.01 | 0.000009 | 0.01 | passed |
| Leduc | full | 0 | 5,000 | 0.014454 | 0.05 | 0.002342 | 0.05 | passed |
| Kuhn | external | 7 | 20,000 | 0.003586 | 0.03 | 0.000098 | 0.03 | passed |
| Kuhn | external | 19 | 20,000 | 0.008222 | 0.03 | 0.000368 | 0.03 | passed |
| Kuhn | external | 43 | 20,000 | 0.007199 | 0.03 | 0.000327 | 0.03 | passed |
| Leduc | external | 7 | 50,000 | 0.064428 | 0.15 | 0.001581 | 0.1 | passed |
| Leduc | external | 19 | 50,000 | 0.064993 | 0.15 | 0.001990 | 0.1 | passed |
| Leduc | external | 43 | 50,000 | 0.077651 | 0.15 | 0.005673 | 0.1 | passed |

The complete eight-run campaign was reproduced in `results/tabular-reference-v1-replay`: all strategy files matched byte for byte, and all deterministic report fields (including every scheduled evaluation and oracle result) matched exactly. Only timings differ. The replay is the same experiment, not additional independent training evidence.

### Independent checks

Kuhn's maximizing/minimizing linear programs give −0.055555555556 (−1/18). Leduc's give −0.085606424078 for the specified two-bet-cap game. Both programs' constraint residuals were zero at reported float64 precision; exact best responses to the equilibrium profiles give exploitability below 1e-14. This agreement checks the independent solution/evaluation paths, while the separate rule tests check their shared tree.

The local regression suite passes **230 tests**, including 25 new game, solver, and experiment tests. They cover exact chip payoffs, card removal, public information and history, perfect recall, hand-calculated regrets, own-reach averaging, all pure Kuhn best responses, exhaustive expectations of sampled updates, independent equilibrium solutions, repeatable random streams, output preservation, reproduction mismatch rejection, and failed/time-limited runs. CI also executes and reproduces the short solver smoke.

### Interpretation

Full-tree CFR and all three sampled seeds meet these declared reference tolerances. That supports using the implementation to check the upcoming neural solver. It does not establish the neural learner's correctness, six-player no-limit strength, or a multiplayer equilibrium guarantee. External sampling still uses exact full-tree averaging; its runtime is not a benchmark for large-game collection. No neural training, GPU rental, or paid compute was used.
