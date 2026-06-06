# DeepCFR Poker AI

Deep CFR for 6-player no-limit Texas Hold'em, built on top of the `pokers` environment and focused on practical training workflows from source.

![Poker AI](https://raw.githubusercontent.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/refs/heads/main/images/testing_different_iteration_values/Screenshot%202025-03-04%20at%2014.39.24.png)

## Project Update (March 2026)

This repo has moved past the March 2025 state described in the original article and early README.

Current status:

- The project is source-first and ready to train from the repository.
- The all-in edge cases in the underlying poker engine are handled by pinning to the patched `pokers` fork in [requirements.txt](./requirements.txt).
- Six-player all-in side-pot check cycles are normalized in the shared action wrapper, so tournaments and evaluation no longer hang when all remaining active players are all-in.
- The Phase 2 self-play and Phase 3 mixed-training regressions from issue `#22` have been fixed on `main`.
- Regression coverage now exists for both poker-engine integration and training-path smoke tests.

What this means in practice:

- Standard Deep CFR now has a clean 3-stage progression: random, self-play, mixed checkpoints.
- Opponent-modeling training now exposes the same 3-stage CLI progression.
- `--checkpoint` now consistently means "continue from this checkpoint" across both public training entrypoints.
- Mixed checkpoint discovery now works recursively, so stage-based subdirectories are supported directly.
- Opponent modeling is still more experimental in learning quality than the standard track, but the workflow is no longer a separate one-off path.

The Medium article is still useful for background, but the code has evolved. Prefer this README and the current scripts over the article when they differ.

## Installation

Recommended workflow: run directly from source.

```bash
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git
cd deepcfr-texas-no-limit-holdem-6-players

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- The GUI requires `PyQt5`, which is already included in [requirements.txt](./requirements.txt).
- The documented commands below use `python -m ...` or `python scripts/...` from the repo root. That is the maintained workflow.

## What Works Today

- Deep CFR training against random opponents
- Checkpoint continuation
- Self-play continuation against a fixed checkpoint snapshot
- Mixed training against a rotating checkpoint pool
- Opponent-modeling training with the same 3-stage progression
- Checkpoint evaluation via CLI
- CLI play against saved checkpoints or random agents
- PyQt GUI play
- Tournament visualization across checkpoints
- Regression tests for known `pokers` and training-path failures

## Current Verification Goal

The main goal of the current training and evaluation examples is to demonstrate that the code works end to end:

1. training fills replay memory and writes checkpoints
2. checkpoints reload correctly
3. TensorBoard logs are written
4. fixed-seed evaluation runs without invalid game states
5. saved checkpoints can play tournaments against each other
6. result files and plots are generated without editing scripts by hand

This is a working-code verification milestone, not a claim that the current models are already strong poker agents. Poker quality still needs longer training, variance-aware evaluation, and checkpoint-vs-checkpoint comparisons.

## Architecture

The current implementation uses:

- A 6-player no-limit Texas Hold'em environment from [`pokers`](https://github.com/Reinforcement-Poker/pokers)
- A fixed-length state encoding including hole cards, board cards, stage, pot, positions, player states, min bet, legal actions, and previous action
- A shared feed-forward network body with two heads:
  - action head: `Fold`, `Check/Call`, `Raise`
  - sizing head: continuous raise sizing in roughly `0.1x` to `3.0x` pot
- Prioritized replay for advantage training
- Separate strategy-memory storage for policy updates
- Optional opponent-modeling variants built around GRU-based action-history encoding

This repo no longer uses the older 4-action "half-pot / pot raise" architecture described in earlier versions of the README. The current action model is 3 action types plus continuous raise sizing.

## Training

All commands below are run from the repository root.

There are now two clean training tracks:

- standard Deep CFR: [train.py](./src/training/train.py)
- opponent-modeling Deep CFR: [train_opponent_modeling.py](./src/training/train_opponent_modeling.py)

The older files:

- [train_with_opponent_modeling.py](./src/training/train_with_opponent_modeling.py)
- [train_mixed_with_opponent_modeling.py](./src/training/train_mixed_with_opponent_modeling.py)

are now internal implementation modules. Use the two entrypoints above when training from the command line.

Both follow the same 3-stage progression:

1. random opponents
2. self-play against a fixed checkpoint
3. mixed checkpoint training

Shared core flags:

- `--iterations`
- `--traversals`
- `--save-dir`
- `--log-dir`
- `--checkpoint`
- `--self-play`
- `--mixed`
- `--checkpoint-dir`
- `--model-prefix`
- `--refresh-interval`
- `--num-opponents`
- `--strict`
- `--progress-interval`

Flag meaning:

- `--checkpoint` means "continue from this checkpoint"
- `--checkpoint --self-play` means "continue from this checkpoint and use that same checkpoint as the fixed opponent snapshot"
- `--checkpoint --mixed` means "continue from this checkpoint while sampling opponents from `--checkpoint-dir`"
- `--progress-interval` controls compact terminal summaries during Phase 1 training. The default is `100`; use `0` to keep only the progress bar and milestone messages.

Recommended directory layout:

```text
models/
  standard/
    phase1/
    selfplay/
    mixed/
  opponent_modeling/
    phase1/
    selfplay/
    mixed/
```

### Phase 1: Train Against Random Opponents

A 1000-iteration run is useful as a smoke test, but it is not a serious poker-training target. Use it only to verify that the environment, checkpointing, TensorBoard logging, and evaluation scripts work:

```bash
python -m src.training.train \
  --iterations 1000 \
  --traversals 200 \
  --log-dir logs/standard/phase1 \
  --save-dir models/standard/phase1
```

A more useful first baseline is a longer Phase 1 run:

```bash
python -m src.training.train \
  --iterations 20000 \
  --traversals 200 \
  --save-dir models/standard/phase1_20k \
  --log-dir logs/standard/phase1_20k
```

For long runs, the trainer uses a `tqdm` progress bar in an interactive terminal and prints compact summaries every `--progress-interval` iterations. To print fewer summaries:

```bash
python -m src.training.train \
  --iterations 20000 \
  --traversals 200 \
  --save-dir models/standard/phase1_20k \
  --log-dir logs/standard/phase1_20k \
  --progress-interval 500
```

Notes:

- Checkpoints are saved every 100 iterations by default.
- Replay memory is not saved in checkpoints. Continuing from a checkpoint continues model weights, but starts with fresh replay memory. For a true uninterrupted Phase 1 baseline, run the full target iteration count in one process.

### Continue Training From a Checkpoint

```bash
python -m src.training.train \
  --checkpoint models/standard/phase1/checkpoint_iter_1000.pt \
  --iterations 1000 \
  --traversals 200 \
  --log-dir logs/standard/continued \
  --save-dir models/standard/continued
```

### Phase 2: Self-Play Against a Fixed Checkpoint

```bash
python -m src.training.train \
  --checkpoint models/standard/phase1/checkpoint_iter_1000.pt \
  --self-play \
  --iterations 2000 \
  --traversals 400 \
  --log-dir logs/standard/selfplay \
  --save-dir models/standard/selfplay
```

Example continuing from a stronger Phase 1 candidate:

```bash
python -m src.training.train \
  --checkpoint models/standard/phase1_20k/checkpoint_iter_3000.pt \
  --self-play \
  --iterations 10000 \
  --traversals 400 \
  --save-dir models/standard/selfplay_from_3000 \
  --log-dir logs/standard/selfplay_from_3000
```

### Phase 3: Mixed Training Against a Checkpoint Pool

```bash
python -m src.training.train \
  --checkpoint models/standard/selfplay/selfplay_checkpoint_iter_3000.pt \
  --mixed \
  --checkpoint-dir models/standard \
  --model-prefix "*checkpoint_iter_" \
  --refresh-interval 1000 \
  --num-opponents 5 \
  --iterations 10000 \
  --traversals 400 \
  --log-dir logs/standard/mixed \
  --save-dir models/standard/mixed
```

### Opponent-Modeling Training

Stage 1: random opponents

```bash
python -m src.training.train_opponent_modeling \
  --iterations 1000 \
  --traversals 200 \
  --save-dir models/opponent_modeling/phase1 \
  --log-dir logs/opponent_modeling/phase1
```

Stage 2: self-play against a fixed checkpoint

```bash
python -m src.training.train_opponent_modeling \
  --checkpoint models/opponent_modeling/phase1/checkpoint_iter_1000.pt \
  --self-play \
  --iterations 2000 \
  --traversals 400 \
  --save-dir models/opponent_modeling/selfplay \
  --log-dir logs/opponent_modeling/selfplay
```

Stage 3: mixed checkpoint training

```bash
python -m src.training.train_opponent_modeling \
  --mixed \
  --checkpoint models/opponent_modeling/selfplay/selfplay_checkpoint_iter_3000.pt \
  --checkpoint-dir models/opponent_modeling \
  --model-prefix "*checkpoint_iter_" \
  --iterations 10000 \
  --traversals 200 \
  --refresh-interval 1000 \
  --num-opponents 5 \
  --save-dir models/opponent_modeling/mixed \
  --log-dir logs/opponent_modeling/mixed
```

Notes:

- Standard mixed training should usually point at `models/standard` so it only samples standard checkpoints.
- Opponent-model self-play requires an opponent-model checkpoint created by `src.training.train_opponent_modeling`.
- Opponent-model mixed training can use:
  - `--checkpoint-dir models/opponent_modeling` for OM-only pools
  - `--checkpoint-dir models` for a mixed pool containing both standard and opponent-model checkpoints

### Monitor Training

```bash
tensorboard --logdir=logs
```

Then open `http://localhost:6006`.

For one run only:

```bash
tensorboard --logdir=logs/standard/phase1_20k
```

## Evaluating Checkpoints

Use the evaluation CLI to compare checkpoints with fixed seeds instead of editing training scripts by hand.

Example:

```bash
python scripts/evaluate_models.py \
  --checkpoint-dir models/standard \
  --pattern "*checkpoint_iter_" \
  --games-random 100 \
  --games-pool 100 \
  --json-out reports/evaluation.json \
  --csv-out reports/evaluation.csv
```

For a Phase 1 run, evaluate stable checkpoint slices explicitly while training continues. This avoids accidentally loading a checkpoint file while the training process is writing it:

```bash
python scripts/evaluate_models.py \
  --checkpoints \
    models/standard/phase1_20k/checkpoint_iter_500.pt \
    models/standard/phase1_20k/checkpoint_iter_1000.pt \
    models/standard/phase1_20k/checkpoint_iter_1500.pt \
    models/standard/phase1_20k/checkpoint_iter_2000.pt \
    models/standard/phase1_20k/checkpoint_iter_2500.pt \
    models/standard/phase1_20k/checkpoint_iter_3000.pt \
  --games-random 5000 \
  --games-pool 1000 \
  --json-out results/standard_phase1_20k_500_3000.json \
  --csv-out results/standard_phase1_20k_500_3000.csv
```

What it reports:

- average profit vs random opponents
- average profit vs the checkpoint pool
- completed hands
- invalid-state counts
- optional machine-readable JSON / CSV summaries

## Playing Against the Models

### CLI

```bash
python scripts/play.py --models-dir models/standard/selfplay
```

Useful options:

- `--model-pattern "*.pt"` to filter checkpoint files
- `--num-models 5` to control how many checkpoint opponents are sampled
- `--position 0` to choose your seat
- `--no-shuffle` to keep the same sampled models across games
- `--strict` to raise on invalid game states instead of logging and continuing

### GUI

```bash
python scripts/poker_gui.py --models_folder models/standard/selfplay
```

### Tournament Visualization

```bash
python scripts/visualize_tournament.py \
  --checkpoints models/standard/phase1/checkpoint_iter_1000.pt models/standard/selfplay/selfplay_checkpoint_iter_3000.pt \
  --num-games 100
```

For checkpoint-vs-checkpoint comparisons during Phase 1:

```bash
python -m scripts.visualize_tournament \
  --checkpoints \
    models/standard/phase1_20k/checkpoint_iter_500.pt \
    models/standard/phase1_20k/checkpoint_iter_1000.pt \
    models/standard/phase1_20k/checkpoint_iter_1500.pt \
    models/standard/phase1_20k/checkpoint_iter_2000.pt \
    models/standard/phase1_20k/checkpoint_iter_2500.pt \
    models/standard/phase1_20k/checkpoint_iter_3000.pt \
  --num-games 1000 \
  --output-dir results/tournament_phase1_20k_500_to_3000_step500
```

For a larger Phase 1 versus self-play comparison:

```bash
python -m scripts.visualize_tournament \
  --checkpoints \
    models/standard/phase1_20k/checkpoint_iter_3000.pt \
    models/standard/selfplay_from_3000/selfplay_checkpoint_iter_3500.pt \
    models/standard/selfplay_from_3000/selfplay_checkpoint_iter_4000.pt \
    models/standard/selfplay_from_3000/selfplay_checkpoint_iter_4500.pt \
    models/standard/selfplay_from_3000/selfplay_checkpoint_iter_5000.pt \
    models/standard/selfplay_from_3000/selfplay_checkpoint_iter_6000.pt \
  --num-games 10000 \
  --output-dir results/tournament_selfplay_from_3000_3500_to_6000_10k
```

Tournament output includes raw data plus plots:

- `tournament_data.csv`
- `cumulative_profit.png`
- `final_performance.png`
- `segment_heatmap.png`
- `stack_sizes_over_time.png`
- `zero_sum_validation.png`

Use tournament results as a robustness signal, not as the only selection metric. A single six-player table can be noisy; compare it with fixed-seed evaluation versus random opponents and checkpoint pools.

The tournament script shows a `tqdm` progress bar with the current leader and average actions per hand. It also has `--max-actions-per-hand` to fail clearly if a hand does not terminate.

## Testing and Regression Coverage

The repo now includes targeted regression tests for the issues that have caused the most damage recently.

Run them with:

```bash
python3 scripts/run_regression_suite.py
```

What these cover:

- `tests/test_evaluation_cli.py`
  - checkpoint evaluation CLI behavior
- `tests/test_training_opponent_modeling_regressions.py`
  - opponent-model self-play smoke test
  - opponent-model self-play rejects standard checkpoints
  - unified OM training CLI dispatch
- `tests/test_pokers_regressions.py`
  - all-in and legal-action regressions inherited from the `pokers` library
- `tests/test_training_regressions.py`
  - self-play and mixed-training smoke tests
  - mixed-training continuation from checkpoint
  - replay-memory shape consistency
  - explicit `.pt` save-path handling
- `tests/test_logging_regressions.py`
  - UTF-8 log writing
  - tournament invalid-state logging
  - six-player all-in side-pot check-cycle resolution
- `tests/test_state_scenarios.py`
  - deterministic edge-case hand scenarios

## Notes on Results

Some older README claims and article screenshots implied a more stable training outcome than the current repo can honestly guarantee.

What is safe to say today:

- the main training paths run
- the known training-path bugs from issue `#22` are fixed
- the all-in legal-action bugs that were breaking games are fixed in the pinned `pokers` fork
- six-player all-in side-pot check cycles are resolved before they can hang tournament or evaluation loops

Recent local benchmark data from a standard Phase 1 run and a self-play continuation from `checkpoint_iter_3000.pt`:

```text
10k tournament, fixed six-seat lineup:
phase1 checkpoint_iter_3000.pt:            +642629.55
selfplay selfplay_checkpoint_iter_3500.pt:  +27121.85
selfplay selfplay_checkpoint_iter_4000.pt: -126573.81
selfplay selfplay_checkpoint_iter_4500.pt: -278549.67
selfplay selfplay_checkpoint_iter_5000.pt: -183321.37
selfplay selfplay_checkpoint_iter_6000.pt:  -81306.54

Fixed-seed evaluator:
checkpoint                         random EV    checkpoint-pool EV
checkpoint_iter_3000.pt            +32.35       +57.70
selfplay_checkpoint_iter_3500.pt    +6.51        -7.32
selfplay_checkpoint_iter_4000.pt    -1.21       -20.03
selfplay_checkpoint_iter_4500.pt    -9.94       -16.57
selfplay_checkpoint_iter_5000.pt    -8.28        -9.29
selfplay_checkpoint_iter_6000.pt    -3.46        -2.07
```

This specific self-play run did not improve over the Phase 1 `3000` checkpoint by iteration `6000`. The later self-play checkpoints partially recovered versus the worst middle checkpoints, but the Phase 1 anchor still dominated both the tournament and fixed-seed evaluator.

What is still an open research / tuning question:

- exact profitability numbers versus the article
- how robust the learned strategy is across seeds and training schedules
- whether opponent-modeling variants outperform the simpler baseline consistently

If you care about reproducibility, run multiple seeds and compare checkpoints rather than relying on a single training curve.

## Future Work

The forward-looking backlog lives in [FUTURE_IMPROVEMENTS.md](./FUTURE_IMPROVEMENTS.md). It has been trimmed to items that still make sense after the recent architecture and training fixes.

## References

1. Brown, N., and Sandholm, T. (2019). [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164).
2. Zinkevich, M., Johanson, M., Bowling, M., and Piccione, C. (2008). [Regret Minimization in Games with Incomplete Information](https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information.pdf).
3. Heinrich, J., and Silver, D. (2016). [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121).

## License

This project is licensed under the MIT License. See [LICENSE.txt](./LICENSE.txt).

## Acknowledgments

- The maintainers of [`pokers`](https://github.com/Reinforcement-Poker/pokers)
- The community members who reported and reproduced training and game-state bugs
- The PyTorch ecosystem for making iteration on this kind of project practical
