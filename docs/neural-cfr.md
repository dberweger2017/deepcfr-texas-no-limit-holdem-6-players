# Small-game neural Deep CFR

This implements the second task in roadmap milestone 3: an auditable neural baseline on the [Kuhn/Leduc reference](solver-reference.md). Its collection and fitting loop follows [Deep CFR, Algorithms 1–2](https://arxiv.org/pdf/1811.00164). The current Hold'em agents and trainer are unchanged. This implementation is not yet the variable-player no-limit agent.

## Algorithm decisions

| Concern | Implementation |
| --- | --- |
| Player updates | Alternate player 0 then player 1. One reported iteration includes both updates. Player 1 collects against the newly fitted player 0 network. |
| Initial advantages | Both networks return exactly zero. |
| Traversal | Sample chance and opponent actions; enumerate every traverser action. Freeze the profile throughout that player's collection phase. |
| Advantage samples | Store the traverser's public information key, iteration, and sampled action-minus-policy continuation values in net ante units. No extra own/chance/opponent reach multiplication. |
| Strategy samples | Store the acting opponent's probability vector at each opponent node, before sampling its action. No strategy sample is inserted at a traverser node. |
| Replay | One uniform reservoir per player's advantages and one shared strategy reservoir. Entries survive across iterations unless Algorithm R replaces them. |
| Iteration weights | Uniform minibatch draws with replacement; multiply each sample's loss by `2 * sample_iteration / current_iteration`. Sum errors over legal actions, then average over the minibatch. |
| Advantage fitting | Initialize a fresh network and Adam optimizer for every player update. Fit raw sampled targets, with gradient norm clipped to 1. |
| Regret matching | Normalize positive legal predictions. If none is positive, select the highest legal prediction; ties use the first output slot. |
| Average policy | Fit a separate network to stored strategy vectors using weighted squared error on masked-softmax probabilities. Its parameters do not influence traversal. |
| Evaluation | Evaluate the average policy with exact information-set best responses, in ante units per hand. No model promotion occurs automatically. |

This differs from the tabular reference's simultaneous updates, uniform iteration weights, and uniform fallback when no regret is positive. These conventions are deliberate; neural trajectories should not be expected to reproduce the earlier tabular trajectories. The learning loop and loss scaling follow the paper, while the network and features below are changes for small games. The original paper's large-game hyperparameters and published performance are not reproduced by these local pilots.

### Why sampling weights matter

External sampling already includes chance/opponent reach in the frequency of collected advantage samples. Weighting those samples by those reaches again would bias the targets. During an opponent's traversal, a player's strategy-memory visit frequency instead contains that player's own reach. With a uniform reservoir, the conditional iteration-weighted mean estimates the required average strategy. Exhaustive Kuhn tests check the regret expectation and strategy-visit probabilities against the full tree, including off-policy traverser branches.

Sample insertion and minibatch sampling have separate generators. Reservoir capacity does not change the traversal random stream; different capacities can still change later learned policies because they retain different training data. Iteration weights enter the loss once, rather than also biasing minibatch selection. [OpenSpiel's public implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/pytorch/deep_cfr.py) is a useful independent reference, not a dependency of this code.

## Features and network

The encoder accepts an immutable `InformationSet`, never a privileged `State`. Its 48 float32 features represent:

- Game and acting player.
- Own card rank and the public board rank, with an explicit absent-board value.
- Current round.
- Complete public action history in four slots per round, retaining check/call distinctions and round boundaries.
- The legal fold, check/call, and raise mask.

The encoding is injective across the 300 Kuhn/Leduc information sets. It contains no opponent card, undealt board, deck order, random seed, simulator node number, or replay index. Suits are strategically irrelevant in Leduc and remain omitted under the reference game's symmetry convention.

The network has two ReLU hidden layers and three output slots: fold, check/call, raise. The default hidden size is 64. A value network's outputs are advantages; a policy network's outputs are logits. The legal mask excludes unavailable actions from regret matching, softmax, and fitting loss. Final inference probabilities are normalized in float64 for the exact evaluator.

The small tree lets us cache predictions for every public information set between fits. Replay stores an index into that public feature table, but the index is never a neural input. Collection still samples trajectories and learns only from their samples. This cache and full-tree diagnostics are conveniences for small-game validation; they do not provide a large-game traversal architecture.

## Fitting diagnostics

A noisy replay sample need not equal the best prediction for its information set. The runner groups stored samples by information set using their iteration weights and decomposes empirical squared error into:

1. **Sample noise:** weighted variation of targets around the memory's conditional mean.
2. **Fitting excess MSE:** weighted squared distance between network predictions and that mean.

Their sum is the empirical MSE, with errors summed over legal actions. For advantages, MSE is in squared ante units; strategy MSE is on probabilities. These are diagnostics on retained training data, not held-out generalization scores. Coverage and stored/seen sample counts make missing information sets and reservoir replacement visible.

Each evaluation reports three separate strategies:

- **Neural average:** the actual fitted policy network, which is the proposed playable artifact.
- **Empirical memory average:** the conditional mean in the strategy reservoir, using a uniform legal fallback where no sample exists. This exposes replay/sampling error separately from network fitting error.
- **Exact played average:** an iteration- and own-reach-weighted table of the policies that generated strategy samples, aligned with alternating updates. It is a diagnostic only and is never read by the learner or exported as the neural policy.

These gaps are informative but not additive exploitability components: different errors can sometimes offset each other. An empirical average can outperform the exact played average in a finite sample. Keep the actual neural result even when another diagnostic looks better.

A separate controlled fitting check uses fixed equilibrium policy labels and analytically computed conditional advantages on every information set. It checks model capacity and optimization before self-play. Those labels are created in separate memories and never enter a self-play run.

## Commands and artifacts

```bash
python -m scripts.check_deep_cfr --fitting-check --out results/neural-fitting
python -m scripts.check_deep_cfr --plan configs/solver/neural-smoke.json --out results/neural-smoke
python -m scripts.check_deep_cfr --reproduce results/neural-smoke --out results/neural-replay
python -m scripts.check_deep_cfr --plan configs/solver/neural-kuhn-v1.json --out results/neural-kuhn
python -m scripts.check_deep_cfr --plan configs/solver/neural-leduc-v1.json --out results/neural-leduc
python -m scripts.check_deep_cfr --refit-plan configs/solver/neural-leduc-v1.json --out results/neural-leduc-refit
```

Run jobs sequentially. Each command is bounded by at most 840 seconds, checked between traversals and optimizer steps. Small setup, evaluation, and final writes may finish after the last deadline check; this is not an operating-system kill timer. A partial failed iteration cannot be continued. Timeouts/errors remain explicit and do not export a partial policy.

The [declared validation protocol and results](reports/neural-validation.md) distinguish controlled fitting acceptance from single-seed self-play diagnostics. A pilot's `completed` status means its declared work finished, not that it passed a convergence or strength threshold. Multi-seed convergence thresholds remain a separate gate.

New output directories contain `manifest.json`, `report.json`, and a final `policy.pt` after successful completion. The manifest pins resolved configuration, source files, Python/platform/NumPy/SciPy/Torch versions, and algorithm conventions. Reports include every scheduled evaluation, every advantage fit, memory counts, hashes, and wall time. Fitting-check bundles have the same manifest/report structure but no playable policy.

The separate `--refit-plan` diagnostic reconstructs the plan's advantage replay, omitting independent strategy fits. It refits each final advantage memory for 4,000 steps from the original initialization/minibatch stream, verifies unchanged replay hashes, and requires at least a 75% reduction in excess MSE. Compare its baseline fitting records with the retained original pilot before interpreting the result. This command exports no revised policy and does not change the original self-play result.

Inference exports include the format/feature version, game, completed iteration, configuration/seed, and policy weights. `load_policy(path, digest, player=..., seed=...)` verifies the entire file hash before weights-only CPU decoding and rejects unsupported metadata or non-finite/mismatched weights. Policy instances share frozen weights and own separate sampling streams; they accept only their owner's information sets. These exports support Kuhn/Leduc only, not legacy Hold'em checkpoints or the Hold'em arena adapter.

Exact pilot reproduction checks source/configuration/environment/protocol fingerprints, verifies the original policy file hash, and compares every deterministic report field, including serialized policy identity. Timings are excluded. The controlled fitting command is rerunnable but does not use the pilot `--reproduce` path. Raw bundles and binary weights stay in ignored `results/`; compact reports belong in git.

Execution uses one CPU thread with deterministic Torch algorithms and restores previous process settings even after errors. Network initialization preserves the CPU random generator; collection, reservoir admission, fit initialization, and minibatches have separate streams. Run the CLI in its own process: Torch runtime settings are process-wide. GPU execution, concurrent threaded training in that process, and cross-environment bitwise reproduction are outside this implementation's contract.

## Remaining gate

The baseline has inference exports, not resumable training checkpoints. It must next pass multiple predeclared neural seeds, reservoir/replay persistence and interrupted-resume checks, and neural-versus-tabular comparisons with enough fitting and sampling diagnostics to explain failures. Architecture changes or Single Deep CFR should be evaluated separately. Strong play in six-player no-limit requires the later representation, sizing, self-play, and search work in the roadmap; these small-game checks do not establish that strength.
