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

Results will be appended after these checks.
