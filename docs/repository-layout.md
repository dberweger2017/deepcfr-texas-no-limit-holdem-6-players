# Repository layout and supported workflows

The repository is a research source checkout. Run the commands below from its
root after installing `requirements-dev.txt` as described in the README.

| Code | Responsibility | Main command |
| --- | --- | --- |
| `src/game` | Rules boundary, observations, hand/session lifecycle | `python -m scripts.check_game` / `scripts.check_session` |
| `src/arena` | Schedules, fixed opponents, reports and reproducibility | `python -m scripts.run_arena` |
| `src/solver` | Exact small games and tabular reference | `python -m scripts.check_solver` |
| `src/solver/neural` | Small-game learning, snapshot averaging and recovery | `python -m scripts.check_deep_cfr` |
| `src/holdem` | Current no-limit encoding, actions, collection and training | `python -m scripts.train_holdem` |
| `configs` | Versioned experiments and resource limits | Pass the relevant plan to its runner |
| `docs/reports` | Compact experimental evidence, including failed results | Read together with the frozen protocol |

The CPU pilot, strategy studies and readiness runners remain available for
reproduction. They are not authorizations to repeat paid campaigns. Raw models,
training checkpoints and retrieved rental archives remain outside Git under
`results/` or their recorded artifact locations; cleanup does not delete them.

## Retired code

The pre-rewrite trainers, prioritized replay path, opponent-modeling experiment,
desktop GUI, terminal play loop, tournament visualizer, old evaluator, Telegram
notifier and their utilities have been removed. The stale improvement lists and
legacy workflow guide have also been removed; the roadmap is the development
plan. Tests specific to those removed implementations are retired with them.
Useful game-state and observation regressions now exercise the current hand and
policy interfaces instead.

The old `setup.py`, source manifest and automatic PyPI release workflow advertised
removed entrypoints and the earlier package layout. They have been retired.
Source-checkout installation remains supported; a packaged headless release and
new play/inspection interface belong to the release milestone. This cleanup
neither publishes a package nor renames the planned 1.0 release.

The removed implementation is recoverable in
[Git history before cleanup](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/tree/642aeab163aba990619f21c8775b3a40b030fe6a).
Historical experiment reports retain their original results and source hashes.
Reproducing those reports requires checking out their recorded revision.

## Historical opponents

`src/arena/historical.py` retains only the standard network architecture and its
public-observation encoding. `FrozenNetwork` verifies a checkpoint's bytes and
metadata, then runs that architecture read-only. No old trainer, simulator view
or action-sanitization path is needed. The current Hold'em learner uses its own
complete-history representation and separate action candidates.

A retained fixture records feature hashes for 52 decisions across four-, five-
and six-player unequal-stack hands, captured from the old implementation before
removal. Tests require exact float32 feature agreement, legal sizing (including
half-chip rounding), hidden-world invariance and deterministic arena replay.
Historical opponents remain useful controls; unknown training seeds and weak
old results do not become release evidence.

## Cleanup validation

All **453 remaining tests pass** in a fresh Python 3.11 environment installed
from the reduced requirements with Torch 2.5.1. The suite no longer requires Qt,
TensorBoard, plotting/dataframe packages or notifier dependencies. The lower
count reflects tests retired with removed workflows; current solver, arena,
engine, observation and session coverage remains. A 20-hand six-player check and
a 30-hand changing-lineup session also complete with chip accounting intact.
All local Markdown documentation links resolve.
