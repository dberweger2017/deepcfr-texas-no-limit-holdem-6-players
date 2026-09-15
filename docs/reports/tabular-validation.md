# Tabular CFR validation

## Declared protocol

Commit this protocol and `configs/solver/reference-v1.json` before running convergence experiments. Thresholds are in ante units per hand. Exploitability is half NashConv: the mean gain available to the two players by switching individually to an exact best response.

- Check the game trees, legal betting, card removal, net utilities, information-set boundaries, and perfect recall.
- Require the Kuhn sequence-form equilibrium to have first-player value -1/18. For both games, independently solved primal/dual values, constraint residuals, and best-response evaluation must agree within 1e-8.
- Check exact best response against exhaustive pure-strategy enumeration in Kuhn. Enumerate every external-sampling outcome for a frozen Kuhn profile and compare its expected regret update with full CFR, including zero-probability actions.
- Run simultaneous vanilla CFR with exact own-reach-weighted averaging for 10,000 Kuhn iterations and 5,000 Leduc iterations. Require final exploitability/value error at most 0.01/0.01 and 0.05/0.05 respectively.
- Run external-sampling regrets with the same exact averaging for seeds 7, 19, and 43: 20,000 Kuhn iterations (limits 0.03/0.03), 50,000 Leduc iterations (limits 0.15/0.10). Every seed must pass; no seed selection or pooled rescue.
- Evaluate at iteration zero, every 1,000 iterations, and the final iteration. Acceptance uses the declared final iteration, never the best intermediate result. Retain failures and timeouts without extending budgets.
- Cap each run at 840 seconds, leaving margin within the agreed 15-minute local job limit. Execute runs sequentially with no GPU or paid compute. Record source/configuration/strategy hashes, runtime versions, seeds, wall time, iterations, and all evaluations.

The sampled solver deliberately retains a full-tree averaging pass. This is a correctness reference, not a throughput benchmark or a claim of fully sampled scaling. These are two-player, fixed-limit toy games; their equilibrium checks do not certify six-player no-limit strength. The neural baseline and multi-seed neural comparisons remain separate roadmap tasks.

Results will be appended after the declared checks.
