# Small-game Deep CFR implementation checks

## Declared protocol

This PR implements the small-game neural baseline. The multi-seed convergence and saved-resume gate is the next roadmap task. Commit the configurations and this protocol before running the diagnostic pilots; do not interpret a completed pilot as proof of strong poker play.

1. Require exhaustive Kuhn traversal expectations to agree with full CFR regrets. Verify that strategy-memory visits are weighted by chance and the sampled player's own reach. Test uniform reservoir admission by enumerating every replacement outcome on a small stream.
2. Verify public-only, injective Kuhn/Leduc features; stable legal output slots; paper-style iteration-weighted losses; alternating player updates; fresh network fitting; highest-legal-advantage fallback; separate random streams; and frozen observation-only inference.
3. Before the self-play pilots, fit controlled advantage and strategy labels over every information set in each game. Use the exact sequence-form equilibrium as the strategy label and an analytically evaluated frozen uniform profile's conditional advantages as the value label. This is a supervised capacity/optimizer check only: these labels must never enter self-play memories. With hidden size 64, seed 101, Adam 0.001, batch size 256, and 3,000 steps per fit, require excess MSE at most 0.02 for advantages and 0.01 for strategy probabilities (sum over legal actions, averaged over information sets).
4. Run the fixed pilots in `configs/solver/neural-kuhn-v1.json` and `neural-leduc-v1.json`, one at a time. Retain each scheduled evaluation of the actual policy network, the empirical strategy-memory mean, and the exact played-strategy average. Record sample noise separately from fitting excess MSE, memory coverage/replacement counts, losses, hashes, seeds, and environment. These are single-seed diagnostics, with no model promotion or convergence pass/fail threshold.
5. Fit strategy networks for evaluations using isolated fit seeds so evaluating more often cannot change subsequent collection or advantage fitting. Verify deterministic short-run reproduction and inference export/reload. Strategy files contain inference weights only; they are not resumable training checkpoints.
6. Use deterministic one-thread CPU execution. Cap each command at 840 seconds; keep raw artifacts in ignored `results/`. Retain failures/timeouts without extending a run or selecting the best intermediate iteration. No GPU rental or paid compute.

The implementation follows Algorithms 1–2 and the linear-weighted losses of [Brown et al., Deep CFR](https://arxiv.org/pdf/1811.00164). The small network and public feature encoding are explicit changes for these toy games. Exact tree enumeration is used for evaluation, caching public network predictions between fits, and diagnostic averages; it does not supply regret or strategy training targets during self-play.

## Follow-up declared after the first pilots

The original Leduc pilot completed at source revision `66dfb6b`, but its final advantage excess MSE was 3.566871 for player 0 and 3.237039 for player 1, despite small controlled-fitting errors. The next diagnostic tests optimizer budget on the **same frozen replay**, without changing self-play results or extending their declared iterations.

Reconstruct `neural-leduc-v1.json` with the same seed, traversals, 30 iterations, and 500-step advantage fits. Strategy evaluation fits may be omitted during reconstruction because their generators are isolated; verify the final baseline fitting records exactly match the retained pilot. Refit each final advantage memory from the same initialization and minibatch stream for **4,000 steps**, changing no other fitting parameter. Hash replay before/after to verify unchanged data. Require each refit's excess MSE to fall to at most **25%** of its 500-step value. Retain the result whether it passes or fails. The command remains capped at 840 seconds and exports no revised playing policy.

This diagnostic was declared before its refits. It does not retrospectively change the first pilot's budget, data, results, or interpretation.

## Results — 15 September 2026

The [machine-readable results](neural-validation.json) retain all four manifests, scheduled evaluations, fitting checks, and refit comparisons. [Every self-play advantage fit](neural-advantage-fits.jsonl) is retained separately, with its run name. Raw bundles and inference weights are in ignored `results/`; these committed reports include the artifact hashes, not the binaries.

The original fitting check and both pilots ran from clean revision `66dfb6b3aed4cdc43f7e73a3ccd96c90bbba3455`. The refit diagnostic ran from `0bea5140038518ea54e1bedd3c91af27534b1a35`, with uncommitted documentation changes recorded by its dirty flag. Its source fingerprints include the committed refit helper. All runs used Python 3.11.15, NumPy 1.26.4, SciPy 1.17.1, and Torch 2.5.1 on one deterministic CPU thread. No paid compute was used.

### Controlled fitting: passed

| Game | Advantage excess MSE (limit 0.02) | Strategy excess MSE (limit 0.01) |
| --- | ---: | ---: |
| Kuhn | 0.000000 | 0.000014 |
| Leduc | 0.005662 | 0.001372 |

All four fits passed after their declared 3,000 optimizer steps. The check covered all 12 Kuhn and 288 Leduc information sets and took 6.80 seconds. This establishes that the small network and optimizer can fit these controlled labels; it does not establish self-play convergence.

### Self-play pilots: completed, with substantial Leduc error remaining

Exploitability below is the mean of the two exact information-set best-response gains, in antes per hand; lower is better. Every scheduled checkpoint is shown, including the temporary increase in Kuhn's neural error.

| Game | Iteration | Neural average | Empirical memory average | Exact played average |
| --- | ---: | ---: | ---: | ---: |
| Kuhn | 10 | 0.065933 | 0.065657 | 0.066514 |
| Kuhn | 20 | 0.052672 | 0.039135 | 0.037053 |
| Kuhn | 30 | 0.053105 | 0.029374 | 0.028015 |
| Kuhn | 40 | 0.031164 | 0.024610 | 0.026701 |
| Leduc | 10 | 0.673265 | 0.670783 | 0.668166 |
| Leduc | 20 | 0.569226 | 0.515960 | 0.520685 |
| Leduc | 30 | 0.368155 | 0.362302 | 0.360558 |

Kuhn completed in 17.83 seconds and Leduc in 26.68 seconds. Both used seed 7. Final policy-network excess MSE was 0.002642 for Kuhn and 0.005953 for Leduc. Leduc's strategy reservoir retained 200,000 of 405,050 seen samples, exercising replacement in an actual pilot. Its final advantage memories covered 144 and 141 information sets for players 0 and 1 respectively, out of 144 each.

The Leduc error also appears in the exact played average, so fitting a better final strategy network alone would not resolve it. The final advantage fits had excess MSE of 3.566871 and 3.237039, above their controlled-fitting errors. Their sample-noise MSE was 22.029641 and 24.073470; treating that entire noisy loss as network fitting error would obscure the diagnosis.

### Frozen-replay refit: passed

The reconstructed pilot's final baseline fitting records match the original Leduc pilot **exactly**. Both replay hashes stayed unchanged during refitting, and the sample-noise values were identical before and after.

| Player | Excess MSE after 500 steps | Excess MSE after 4,000 steps | Reduction | Required reduction |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3.566871 | 0.406146 | 88.61% | at least 75% |
| 1 | 3.237039 | 0.250214 | 92.27% | at least 75% |

The diagnostic completed in 24.61 seconds. Increasing optimizer steps substantially reduced error on the same retained samples, so the next campaign should allow more advantage fitting and measure its effect on subsequent self-play. This does not prove that a 4,000-step training run will converge or play better: no revised self-play policy was produced by this diagnostic.

### Implementation validation and next gate

The regression suite passes 249 tests, including exhaustive traversal expectations, uniform reservoir admission, public-feature injectivity, loss gradients, alternating updates, inference leakage checks, random-stream isolation, exact short-run reproduction, and frozen-replay reconstruction. CI also runs the neural CLI smoke plan and exact replay alongside the existing engine, session, arena, and tabular checks.

This completes the small-game neural implementation task. It does not complete milestone 3. The next PR must add resumable training state and compare neural/tabular results across predeclared seeds and exploitability tolerances. Preserve this pilot as the original fixed-budget result; declare the stronger fitting budget before collecting that campaign's data.
