# DeepCFR Poker AI

Deep CFR for 6-player no-limit Texas Hold'em, built on top of the [`pokers`](https://github.com/Reinforcement-Poker/pokers) environment. The focus here is a training workflow you can actually run from source, not a polished black box.

![Poker AI](https://raw.githubusercontent.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/refs/heads/main/images/testing_different_iteration_values/Screenshot%202025-03-04%20at%2014.39.24.png)

## Where things stand (March 2026)

This repo has come a long way since the March 2025 version described in the original Medium article. If the article and this README ever disagree, trust the README and the current scripts.

The short version: you can clone this, install it, and start training from the repo without fighting the tooling first. The nasty all-in edge cases in the poker engine are handled by pinning a patched `pokers` fork in [requirements.txt](./requirements.txt), and the six-player all-in side-pot check cycles that used to hang tournaments and evaluation are now normalized in the shared action wrapper. The Phase 2 self-play and Phase 3 mixed-training regressions from issue `#22` are fixed on `main`, and there's regression coverage for both the poker-engine integration and the training paths.

In practice that means standard Deep CFR has a clean three-stage flow now — random, self-play, mixed — and the opponent-modeling track exposes the same three stages instead of being a separate one-off. `--checkpoint` means the same thing everywhere ("continue from this checkpoint"), mixed checkpoint discovery walks subdirectories recursively, and opponent modeling, while still more experimental on learning quality, at least follows the same workflow as everything else.

The Medium article is still good background reading, but the code has moved on.

## Installation

Run it from source. That's the path I actually maintain.

```bash
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git
cd deepcfr-texas-no-limit-holdem-6-players

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

FastAPI and uvicorn (for the web play client) are in `requirements.txt`. All the commands below assume you're at the repo root and use `python -m ...` or `python scripts/...`.

## What works today

Most of the moving pieces are in place: Deep CFR training against random opponents, checkpoint continuation, self-play against a fixed checkpoint snapshot, and mixed training against a rotating checkpoint pool. The opponent-modeling track runs through the same three stages. On top of training you get checkpoint evaluation from the CLI, terminal and web play against saved checkpoints or random agents, tournament visualization across checkpoints, and regression tests for the `pokers` and training-path failures that have bitten this project before.

## What the current examples are meant to prove

The training and evaluation examples in this README are here to show the code works end to end, not to claim the models are already strong. Concretely, they demonstrate that training fills replay memory and writes checkpoints, those checkpoints reload correctly, TensorBoard logs get written, fixed-seed evaluation runs without hitting invalid game states, saved checkpoints can play tournaments against each other, and result files and plots get generated without anyone hand-editing scripts.

That's a working-code milestone. Real poker strength still needs longer training, variance-aware evaluation, and careful checkpoint-vs-checkpoint comparison.

## Architecture

The setup is a 6-player no-limit Texas Hold'em environment from [`pokers`](https://github.com/Reinforcement-Poker/pokers), fed a fixed-length state encoding that covers hole cards, board cards, stage, pot, positions, player states, min bet, legal actions, and the previous action.

The network is a shared feed-forward body with two heads: an action head (`Fold`, `Check/Call`, `Raise`) and a sizing head for continuous raise sizing in roughly the `0.1x` to `3.0x` pot range. Advantage training uses prioritized replay, policy updates draw from a separate strategy memory, and the opponent-modeling variants add a GRU-based action-history encoder on top.

This is no longer the old 4-action "half-pot / pot raise" model from earlier versions. It's three action types plus continuous sizing.

## Training

Everything below runs from the repo root. There are two clean training tracks: standard Deep CFR in [train.py](./src/training/train.py), and opponent-modeling Deep CFR in [train_opponent_modeling.py](./src/training/train_opponent_modeling.py). The older [train_with_opponent_modeling.py](./src/training/train_with_opponent_modeling.py) and [train_mixed_with_opponent_modeling.py](./src/training/train_mixed_with_opponent_modeling.py) are now internal modules — use the two entrypoints above from the command line.

Both tracks follow the same three stages: random opponents, then self-play against a fixed checkpoint, then mixed checkpoint training.

The shared core flags are `--iterations`, `--traversals`, `--save-dir`, `--log-dir`, `--checkpoint`, `--self-play`, `--mixed`, `--checkpoint-dir`, `--model-prefix`, `--refresh-interval`, `--num-opponents`, `--strict`, and `--progress-interval`. A few of them carry specific meaning worth spelling out:

- `--checkpoint` means "continue from this checkpoint"
- `--checkpoint --self-play` means "continue from this checkpoint and use that same checkpoint as the fixed opponent snapshot"
- `--checkpoint --mixed` means "continue from this checkpoint while sampling opponents from `--checkpoint-dir`"
- `--progress-interval` controls the compact terminal summaries during Phase 1. Default is `100`; use `0` to keep only the progress bar and milestone messages.

A directory layout that keeps the stages tidy:

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

### Phase 1: train against random opponents

A 1000-iteration run is fine as a smoke test, but don't mistake it for real training — use it to confirm the environment, checkpointing, TensorBoard logging, and evaluation scripts all work:

```bash
python -m src.training.train \
  --iterations 1000 \
  --traversals 200 \
  --log-dir logs/standard/phase1 \
  --save-dir models/standard/phase1
```

A longer Phase 1 run makes a more useful first baseline:

```bash
python -m src.training.train \
  --iterations 20000 \
  --traversals 200 \
  --save-dir models/standard/phase1_20k \
  --log-dir logs/standard/phase1_20k
```

For long runs the trainer shows a `tqdm` progress bar in an interactive terminal and prints compact summaries every `--progress-interval` iterations. To print fewer:

```bash
python -m src.training.train \
  --iterations 20000 \
  --traversals 200 \
  --save-dir models/standard/phase1_20k \
  --log-dir logs/standard/phase1_20k \
  --progress-interval 500
```

Two things to keep in mind. Checkpoints save every 100 iterations by default. And replay memory isn't stored in checkpoints — continuing from one resumes the model weights but starts with fresh replay memory, so for a true uninterrupted Phase 1 baseline, run the full iteration count in a single process.

### Continue training from a checkpoint

```bash
python -m src.training.train \
  --checkpoint models/standard/phase1/checkpoint_iter_1000.pt \
  --iterations 1000 \
  --traversals 200 \
  --log-dir logs/standard/continued \
  --save-dir models/standard/continued
```

### Phase 2: self-play against a fixed checkpoint

```bash
python -m src.training.train \
  --checkpoint models/standard/phase1/checkpoint_iter_1000.pt \
  --self-play \
  --iterations 2000 \
  --traversals 400 \
  --log-dir logs/standard/selfplay \
  --save-dir models/standard/selfplay
```

Continuing from a stronger Phase 1 candidate instead:

```bash
python -m src.training.train \
  --checkpoint models/standard/phase1_20k/checkpoint_iter_3000.pt \
  --self-play \
  --iterations 10000 \
  --traversals 400 \
  --save-dir models/standard/selfplay_from_3000 \
  --log-dir logs/standard/selfplay_from_3000
```

### Phase 3: mixed training against a checkpoint pool

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

### Opponent-modeling training

Same three stages, different entrypoint.

Stage 1, random opponents:

```bash
python -m src.training.train_opponent_modeling \
  --iterations 1000 \
  --traversals 200 \
  --save-dir models/opponent_modeling/phase1 \
  --log-dir logs/opponent_modeling/phase1
```

Stage 2, self-play against a fixed checkpoint:

```bash
python -m src.training.train_opponent_modeling \
  --checkpoint models/opponent_modeling/phase1/checkpoint_iter_1000.pt \
  --self-play \
  --iterations 2000 \
  --traversals 400 \
  --save-dir models/opponent_modeling/selfplay \
  --log-dir logs/opponent_modeling/selfplay
```

Stage 3, mixed checkpoint training:

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

A couple of notes on the pools. Standard mixed training should usually point at `models/standard` so it only samples standard checkpoints. Opponent-model self-play needs an opponent-model checkpoint created by `src.training.train_opponent_modeling`. And opponent-model mixed training can either stay OM-only with `--checkpoint-dir models/opponent_modeling`, or draw from a mixed pool of both standard and OM checkpoints with `--checkpoint-dir models`.

### Monitoring

```bash
tensorboard --logdir=logs
```

Then open `http://localhost:6006`. For a single run, point it at that run's directory:

```bash
tensorboard --logdir=logs/standard/phase1_20k
```

## Evaluating checkpoints

The evaluation CLI compares checkpoints with fixed seeds, so you don't have to hand-edit training scripts to measure progress.

```bash
python scripts/evaluate_models.py \
  --checkpoint-dir models/standard \
  --pattern "*checkpoint_iter_" \
  --games-random 100 \
  --games-pool 100 \
  --json-out reports/evaluation.json \
  --csv-out reports/evaluation.csv
```

While a Phase 1 run is still training, evaluate stable checkpoint slices explicitly. Naming the files keeps the evaluator from loading a checkpoint mid-write:

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

It reports average profit versus random opponents, average profit versus the checkpoint pool, completed hands, invalid-state counts, and optional JSON/CSV summaries.

## Playing against the models

CLI:

```bash
python scripts/play.py --models-dir models/standard/selfplay
```

Handy options: `--model-pattern "*.pt"` to filter checkpoint files, `--num-models 5` to control how many checkpoint opponents get sampled, `--position 0` to pick your seat, `--no-shuffle` to keep the same sampled models across games, and `--strict` to raise on invalid game states instead of logging and continuing.

Web client (replaces the old PyQt GUI):

```bash
python scripts/play_web.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765), pick a checkpoint folder under `models/`, choose your seat, and play. The server loads up to five random checkpoints from that folder as opponents and surfaces load failures instead of silently falling back.

Handy options: `--host` and `--port` to bind elsewhere, `--reload` for development.

### Tournament visualization

```bash
python scripts/visualize_tournament.py \
  --checkpoints models/standard/phase1/checkpoint_iter_1000.pt models/standard/selfplay/selfplay_checkpoint_iter_3000.pt \
  --num-games 100
```

Comparing checkpoint against checkpoint during Phase 1:

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

A larger Phase 1 versus self-play comparison:

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

Each tournament run drops raw data (`tournament_data.csv`) alongside plots: `cumulative_profit.png`, `final_performance.png`, `segment_heatmap.png`, `stack_sizes_over_time.png`, and `zero_sum_validation.png`.

Treat tournament results as a robustness signal, not the only metric. A single six-player table is noisy, so cross-check it against fixed-seed evaluation versus random opponents and the checkpoint pool. The script shows a `tqdm` bar with the current leader and average actions per hand, and `--max-actions-per-hand` makes a non-terminating hand fail loudly instead of hanging.

## Testing and regression coverage

The repo carries targeted regression tests for the failures that have done the most damage. Run them all with:

```bash
python3 scripts/run_regression_suite.py
```

What they cover:

- `tests/test_evaluation_cli.py` — the checkpoint evaluation CLI
- `tests/test_training_opponent_modeling_regressions.py` — OM self-play smoke test, OM self-play rejecting standard checkpoints, and the unified OM training CLI dispatch
- `tests/test_pokers_regressions.py` — all-in and legal-action regressions inherited from `pokers`
- `tests/test_training_regressions.py` — self-play and mixed-training smoke tests, mixed-training continuation from checkpoint, replay-memory shape consistency, explicit `.pt` save-path handling
- `tests/test_logging_regressions.py` — UTF-8 log writing, tournament invalid-state logging, six-player all-in side-pot check-cycle resolution
- `tests/test_state_scenarios.py` — deterministic edge-case hand scenarios

## A note on results

Some of the older README claims and article screenshots implied a steadier training outcome than this repo can honestly promise. What's safe to say today is that the main training paths run, the issue `#22` training-path bugs are fixed, the all-in legal-action bugs that were breaking games are fixed in the pinned `pokers` fork, and the six-player all-in side-pot check cycles get resolved before they can hang a tournament or evaluation loop.

Some recent local benchmark numbers, from a standard Phase 1 run and a self-play continuation off `checkpoint_iter_3000.pt`:

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

Read honestly, this self-play run didn't beat the Phase 1 `3000` checkpoint by iteration `6000`. The later self-play checkpoints clawed back some ground against the worst middle ones, but the Phase 1 anchor still won both the tournament and the fixed-seed evaluator.

What's still open: the exact profitability numbers versus the article, how robust the learned strategy is across seeds and schedules, and whether the opponent-modeling variants consistently beat the simpler baseline. If reproducibility matters to you, run multiple seeds and compare checkpoints rather than trusting a single training curve.

## Future work

The forward-looking backlog lives in [FUTURE_IMPROVEMENTS.md](./FUTURE_IMPROVEMENTS.md), trimmed down to the items that still make sense after the recent architecture and training fixes.

## References

1. Brown, N., and Sandholm, T. (2019). [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164).
2. Zinkevich, M., Johanson, M., Bowling, M., and Piccione, C. (2008). [Regret Minimization in Games with Incomplete Information](https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information.pdf).
3. Heinrich, J., and Silver, D. (2016). [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121).

## License

MIT. See [LICENSE.txt](./LICENSE.txt).

## Acknowledgments

Thanks to the maintainers of [`pokers`](https://github.com/Reinforcement-Poker/pokers), the people who reported and reproduced the training and game-state bugs, and the PyTorch ecosystem for making iteration on something like this practical.
