# DeepCFR Poker AI

Deep CFR for 6-player no-limit Texas Hold'em, built on top of the `pokers` environment and focused on practical training workflows from source.

![Poker AI](https://raw.githubusercontent.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/refs/heads/main/images/testing_different_iteration_values/Screenshot%202025-03-04%20at%2014.39.24.png)

## Project Update (March 2026)

This repo has moved past the March 2025 state described in the original article and early README.

Current status:

- The project is source-first and ready to train from the repository.
- The all-in edge cases in the underlying poker engine are handled by pinning to the patched `pokers` fork in [requirements.txt](./requirements.txt).
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

Flag meaning:

- `--checkpoint` means "continue from this checkpoint"
- `--checkpoint --self-play` means "continue from this checkpoint and use that same checkpoint as the fixed opponent snapshot"
- `--checkpoint --mixed` means "continue from this checkpoint while sampling opponents from `--checkpoint-dir`"

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

```bash
python -m src.training.train \
  --iterations 1000 \
  --traversals 200 \
  --log-dir logs/standard/phase1 \
  --save-dir models/standard/phase1
```

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
- `tests/test_state_scenarios.py`
  - deterministic edge-case hand scenarios

## Notes on Results

Some older README claims and article screenshots implied a more stable training outcome than the current repo can honestly guarantee.

What is safe to say today:

- the main training paths run
- the known training-path bugs from issue `#22` are fixed
- the all-in legal-action bugs that were breaking games are fixed in the pinned `pokers` fork

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
