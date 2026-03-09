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

- Basic random-opponent training works.
- Continuing training from a checkpoint works.
- Self-play against a checkpoint works.
- Mixed checkpoint training works again.
- Opponent-modeling training scripts are still available, but should be treated as experimental compared with the main training path.

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
- Self-play training against a fixed checkpoint
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

### Phase 1: Train Against Random Opponents

```bash
python -m src.training.train --iterations 1000 --traversals 200 --log-dir logs/phase1 --save-dir models/phase1
```

### Continue Training From a Checkpoint

```bash
python -m src.training.train \
  --checkpoint models/phase1/checkpoint_iter_1000.pt \
  --iterations 1000 \
  --traversals 200 \
  --log-dir logs/continued \
  --save-dir models/continued
```

### Phase 2: Self-Play Against a Fixed Checkpoint

```bash
python -m src.training.train \
  --checkpoint models/phase1/checkpoint_iter_1000.pt \
  --self-play \
  --iterations 2000 \
  --traversals 400 \
  --log-dir logs/selfplay \
  --save-dir models/selfplay
```

### Phase 3: Mixed Training Against a Checkpoint Pool

```bash
python -m src.training.train \
  --mixed \
  --checkpoint-dir models \
  --model-prefix t_ \
  --refresh-interval 1000 \
  --num-opponents 5 \
  --iterations 10000 \
  --traversals 400 \
  --log-dir logs/mixed \
  --save-dir models/mixed
```

### Opponent-Modeling Training

Stage 1: random opponents

```bash
python -m src.training.train_opponent_modeling \
  --iterations 1000 \
  --traversals 200 \
  --save-dir models_om \
  --log-dir logs/deepcfr_om
```

Stage 2: self-play against a fixed checkpoint

```bash
python -m src.training.train_opponent_modeling \
  --checkpoint models_om/om_checkpoint_iter_1000.pt \
  --self-play \
  --iterations 2000 \
  --traversals 400 \
  --save-dir models_om_selfplay \
  --log-dir logs/deepcfr_om_selfplay
```

Stage 3: mixed checkpoint training

```bash
python -m src.training.train_opponent_modeling \
  --mixed \
  --checkpoint-dir models_om \
  --model-prefix om_checkpoint_iter_ \
  --iterations 10000 \
  --traversals 200 \
  --refresh-interval 1000 \
  --num-opponents 5 \
  --save-dir models_mixed_om \
  --log-dir logs/deepcfr_mixed_om
```

Notes:

- Opponent-model mixed training can evaluate and train against opponent-model checkpoints and standard checkpoints.
- Standard mixed training should still use standard checkpoints.

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
  --checkpoint-dir models \
  --pattern "*.pt" \
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
python scripts/play.py --models-dir models/phase1
```

Useful options:

- `--model-pattern "*.pt"` to filter checkpoint files
- `--num-models 5` to control how many checkpoint opponents are sampled
- `--position 0` to choose your seat
- `--no-shuffle` to keep the same sampled models across games
- `--strict` to raise on invalid game states instead of logging and continuing

### GUI

```bash
python scripts/poker_gui.py --models_folder models/phase1
```

### Tournament Visualization

```bash
python scripts/visualize_tournament.py \
  --checkpoints models/phase1/checkpoint_iter_1000.pt models/selfplay/checkpoint_iter_2000.pt \
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
