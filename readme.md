# DeepCFR Poker AI

Deep CFR for 6-player no-limit Texas Hold'em, built on top of the [`pokers`](https://github.com/dberweger2017/pokers) environment. The focus here is a training workflow you can actually run from source, not a polished black box.

## Development direction

We are rebuilding toward a strong no-limit Hold'em agent that plays by a documented ruleset and sees exactly the game information available to a human in its seat. Six-handed play is the main target, with four- and five-handed tables, changing lineups, and unequal stacks included in the design.

The [roadmap](./ROADMAP.md) sets the implementation order, PR acceptance checks, training milestones, and model promotion criteria. The first priorities are correct poker rules, a tested boundary between public observations and hidden simulator state, reproducible evaluation, and a validated CFR baseline. Range-aware search and opponent adaptation follow once that foundation works. Backwards compatibility is not required.

The workflows below describe the existing implementation. Passing its tests does not establish complete rules compliance or strong play; the roadmap records the gaps the rewrite must close.

## Where things stand (September 2026)

This repo has come a long way since the March 2025 version described in the original Medium article. If the article and this README ever disagree, trust the README and the current scripts.

The first rules-engine rewrite is in place. The pinned Rust fork now handles integer chips, minimum raises and reopening, short calls, side pots, heads-up order, and automatic all-in runouts. Logging no longer repairs game state. See the [rules profile](docs/rules.md) and [engine audit](docs/engine-audit.md) for evidence and limits. The [observation interface](docs/observations.md) now keeps engine state out of policy calls and records complete public history. The [session manager](docs/sessions.md) now carries bankrolls and identity-owned histories through changing lineups. The [evaluation arena](docs/evaluation.md) now provides reproducible schedules and paired reports. The [benchmark suites](docs/benchmarks.md) add varied style opponents and hash-pinned historical checkpoints. A validated learning rewrite and competitive trained opponents remain on the roadmap.

In practice that means standard Deep CFR has a clean three-stage flow now — random, self-play, mixed — and the opponent-modeling track exposes the same three stages instead of being a separate one-off. `--checkpoint` means the same thing everywhere ("continue from this checkpoint"), mixed checkpoint discovery walks subdirectories recursively, and opponent modeling, while still more experimental on learning quality, at least follows the same workflow as everything else.

The Medium article is still good background reading, but the code has moved on.

## Installation

Run from source using Python 3.10 or 3.11 and a Rust toolchain. Python 3.11 is the development default; newer Python versions need a separate PyO3 upgrade.

```bash
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git
cd deepcfr-texas-no-limit-holdem-6-players

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyQt5 (for the GUI) is already in `requirements.txt`, so there's nothing extra to install. All the commands below assume you're at the repo root and use `python -m ...` or `python scripts/...`.

## Reproducible evaluation

The [evaluation arena](docs/evaluation.md) saves declared schedules, hand-level results, provenance, and paired confidence intervals. It supports independent fixed-stack hands and persistent bankroll sessions, with separate random streams and seat rotations. Versioned four-/five-/six-player suites cover stack depths and sessions, with separate training/evaluation pools and an observation-only adapter for frozen standard checkpoints. The style opponents are exploitable controls; archived models retain their documented limitations.

```bash
python -m scripts.run_arena --plan configs/arena/smoke.json --out results/arena-smoke
python -m scripts.run_arena --reproduce results/arena-smoke --out results/arena-replay
```

See the [benchmark guide](docs/benchmarks.md), [opponent/checkpoint validation](docs/reports/benchmark-validation.md), and [declared sensitivity check](docs/reports/arena-validation.md) for usage and evidence. The existing `scripts.evaluate_models` command remains a legacy evaluator.

## Small-game solver reference

The [Kuhn/Leduc reference](docs/solver-reference.md) validates tabular CFR and external-sampling regret updates against exact best responses and an independent equilibrium solver. It reports exploitability in ante units and retains declared budgets, seeds, average strategies, and reproducible results. These toy-game checks are the first part of the learning rewrite; they do not establish six-player playing strength.

```bash
python -m scripts.check_solver --plan configs/solver/smoke.json --out results/solver-smoke
python -m scripts.check_solver --reproduce results/solver-smoke --out results/solver-replay
```

## Public observation interface

Policies now receive immutable player views rather than the Rust simulator. The new API uses integer chips and explicit raise-to actions, while a public-only adapter supports the current model feature layout. Folded and mucked cards stay private, including in CLI and GUI results. [Interface documentation](docs/observations.md) covers events, replay, history ownership, showdown choices, and remaining limits.

Run a short headless check with `python -m scripts.check_game --players 6 --hands 20`. For bankrolls, departures, arrivals, and sit-outs, run `python -m scripts.check_session --hands 30`. This exercises the interface with random policies; it does not measure playing strength.

## What works today

Most of the moving pieces are in place: Deep CFR training against random opponents, checkpoint continuation, self-play against a fixed checkpoint snapshot, and mixed training against a rotating checkpoint pool. The opponent-modeling track runs through the same three stages. On top of training you get checkpoint evaluation from the CLI, CLI and PyQt GUI play against saved checkpoints or random agents, tournament visualization across checkpoints, and regression tests for the `pokers` and training-path failures that have bitten this project before.

## What the current examples are meant to prove

The training and evaluation examples in this README are here to show the code works end to end, not to claim the models are already strong. Concretely, they demonstrate that training fills replay memory and writes checkpoints, those checkpoints reload correctly, TensorBoard logs get written, fixed-seed evaluation runs without hitting invalid game states, saved checkpoints can play tournaments against each other, and result files and plots get generated without anyone hand-editing scripts.

That's a working-code milestone. Real poker strength still needs longer training, variance-aware evaluation, and careful checkpoint-vs-checkpoint comparison.

## Architecture

The setup is a 6-player no-limit Texas Hold'em environment from [`pokers`](https://github.com/dberweger2017/pokers), fed a fixed-length state encoding that covers hole cards, board cards, stage, pot, positions, player states, min bet, legal actions, and the previous action.

The network is a shared feed-forward body with two heads: an action head (`Fold`, `Check/Call`, `Raise`) and a sizing head for continuous raise sizing in roughly the `0.1x` to `3.0x` pot range. Advantage training uses prioritized replay, policy updates draw from a separate strategy memory, and the opponent-modeling variants add a GRU-based action-history encoder on top.

This is no longer the old 4-action "half-pot / pot raise" model from earlier versions. It's three action types plus continuous sizing.

## Training

Everything below runs from the repo root. There are two clean training tracks: standard Deep CFR in [train.py](./src/training/train.py), and opponent-modeling Deep CFR in [train_opponent_modeling.py](./src/training/train_opponent_modeling.py). The older [train_with_opponent_modeling.py](./src/training/train_with_opponent_modeling.py) and [train_mixed_with_opponent_modeling.py](./src/training/train_mixed_with_opponent_modeling.py) are now internal modules — use the two entrypoints above from the command line.

Both tracks follow the same three stages: random opponents, then self-play against a fixed checkpoint, then mixed checkpoint training.

The shared core flags are `--iterations`, `--traversals`, `--save-dir`, `--log-dir`, `--checkpoint`, `--self-play`, `--mixed`, `--checkpoint-dir`, `--model-prefix`, `--refresh-interval`, `--num-opponents`, `--strict`, `--progress-interval`, and `--checkpoint-interval`. A few of them carry specific meaning worth spelling out:

- `--checkpoint` means "continue from this checkpoint"
- `--checkpoint --self-play` means "continue from this checkpoint and use that same checkpoint as the fixed opponent snapshot"
- `--checkpoint --mixed` means "continue from this checkpoint while sampling opponents from `--checkpoint-dir`"
- `--progress-interval` controls the compact terminal summaries during Phase 1. Default is `100`; use `0` to keep only the progress bar and milestone messages.
- `--checkpoint-interval` controls checkpoint save cadence. Default is `1000`; use `0` to save only the final checkpoint.

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
  --log-dir logs/standard/phase1_20k \
  --checkpoint-interval 1000
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

Two things to keep in mind. Checkpoints save every 1000 iterations by default. And replay memory isn't stored in checkpoints — continuing from one resumes the model weights but starts with fresh replay memory, so for a true uninterrupted Phase 1 baseline, run the full iteration count in a single process.

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

GUI:

```bash
python scripts/poker_gui.py --models_folder models/standard/selfplay
```

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
- `tests/test_engine_integration.py` — strict sizing and logging adapters over 300 unequal-stack hands with four to six players
- `tests/test_pokers_regressions.py` — all-in and legal-action regressions inherited from `pokers`
- `tests/test_training_regressions.py` — self-play and mixed-training smoke tests, mixed-training continuation from checkpoint, replay-memory shape consistency, explicit `.pt` save-path handling
- `tests/test_logging_regressions.py` — UTF-8 log writing, tournament invalid-state logging, automatic all-in settlement without logging repair
- `tests/test_state_scenarios.py` — deterministic edge-case hand scenarios

## A note on results

The main training paths run, but the learning algorithm and evaluation still need the work described in the roadmap. The engine now plays a corrected game, so results and checkpoints from the old rules are historical records, not strength benchmarks for this version.

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

The development plan lives in [ROADMAP.md](./ROADMAP.md). It replaces the earlier feature backlog with staged PRs, correctness checks, and measured training campaigns.

## References

1. Brown, N., and Sandholm, T. (2019). [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164).
2. Zinkevich, M., Johanson, M., Bowling, M., and Piccione, C. (2008). [Regret Minimization in Games with Incomplete Information](https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information.pdf).
3. Heinrich, J., and Silver, D. (2016). [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121).

## License

MIT. See [LICENSE.txt](./LICENSE.txt).

## Acknowledgments

Thanks to the maintainers of [`pokers`](https://github.com/Reinforcement-Poker/pokers), the people who reported and reproduced the training and game-state bugs, and the PyTorch ecosystem for making iteration on something like this practical.
