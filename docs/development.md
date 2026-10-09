# Repository map and supported workflows

Start with [the roadmap](../ROADMAP.md) for current decisions, [rules](rules.md)
and [observations](observations.md) for game contracts, and
[artifact storage](artifact-storage.md) before adding or moving research files.
The [research history](research-history.md) indexes completed experiments.

## Current runtime

- `src/game/`: engine-owned hands, legal observations, public replay and sessions.
- `src/blueprint/`: information keys, Python reference trainer, current/average
  policy readers, compact storage and turn/river search. `average.py` owns the
  stored-average reader and its validation; `compact_policy.py` owns storage.
- `src/policies/`: released v0.4.0/v0.4.1/v0.4.2 artifact pins, verification and loading.
  Adding a released model here or to the web catalog still requires release authorization.
  `hu100_research.py` is an explicit research-only pin; [local HU100 play](hu100-local-play.md)
  records its retrieval gate, fixed table and off-by-default translation option.
- `src/play_api/` and `apps/poker-web/`: durable local sessions, HTTP and the table.
  `versions.py` has the explicit release catalog. Defaults never silently fall
  back when an artifact is missing or corrupt; sessions keep their model.
- `native/hu20-trainer/`: native training, game/key parity, exports and bench.
  Equal-stack HU20/HU100 preparation and future operator commands are in
  [the native HU100 protocol](native-hu100-preparation.md); it authorizes no runs.
  `native/hu20-buckets/`: bucket construction and the native card evaluator.
- `src/arena/`: paired evaluation, scripted opponents, reports and checkpoint
  adapters. Historical network/snapshot adapters still have consumers.

The average reader is shared by play and research. Its existing diagnostic-v1
format IDs are part of released files and stay unchanged. Extraction and
accumulator audits remain in `src/diagnostics/cfr_average.py`; inference does not
import that package. CLI verifiers delegate to `src/policies`, not the reverse.

## Install and check

Use Python 3.11. For released tabular play, install `requirements-play.txt` and
follow [the readme](../readme.md#install-and-play) for pinned model downloads.
This environment needs NumPy and the pinned engine, not PyTorch or SciPy.
`--o-candidate` remains the existing single-model v0.4.1 CLI option; its historical
name does not authorize a new candidate or change the default.

For research and tests, install `requirements-dev.txt` (full requirements plus
pytest and monitoring). The full `requirements.txt` remains unchanged because
historical opponent freezes pin its bytes; keep the common engine/NumPy pins in
`requirements-play.txt` synchronized when intentionally updating dependencies.
From the repository root:

```sh
python -m pytest -q
python -m scripts.check_game --players 6 --hands 20
python -m scripts.check_session --hands 30
python -m scripts.check_repository_artifacts
```

The last command checks **staged Git objects**, so run it after `git add`.
Use `--revision HEAD` for committed content or `--inventory` for a read-only
machine-readable inventory. CI runs storage checks in its scope job on every PR.
[CI](../.github/workflows/tests.yml) also runs paired arena reproduction and
small-game solver/resume checks. Local browser-driven checks have additional
requirements documented in [the play guide](play-web.md).

## Research versus reusable code

`src/solver/` retains small-game CFR/reference and neural solvers. `src/holdem/`
retains earlier neural Hold'em work; `src/arena/snapshots.py` still imports it.
These are research/reference components, not the released tabular training recipe.
Do not delete them based solely on their age or a failed strength experiment.

`scripts/` mixes maintained entry points and frozen campaign launchers. Before
reusing a campaign, read its protocol and report: many embed exact roots, source
hashes, budgets, machine requirements and stopping conditions. A past permission
or resource quote does not authorize a new run. In particular, do not launch an
old controller simply because its script is still present.

For native training start with the [native CLI](../native/hu20-trainer/src/main.rs)
and the command/source pins in the current campaign protocol; for inference extraction see
`scripts/extract_hu20_cfr_average.py`; for paired average-policy evaluation see
`scripts/evaluate_hu20_v041_arena.py` and its experiment protocol. A new campaign
still predeclares inputs, roots, comparison, stopping rule and cost.

Shared behavior belongs in a library, with scripts as entry points. Prefer one
coherent refactor with regression coverage over renaming all historical modules.
New imports should use the current reader/release modules; to reproduce an old
campaign, use its exact recorded source revision. Recovery source checks include
the moved average reader, compact storage and checksum helper.

The [maintenance audit](reports/necessary-cleaning.md) records the remaining
retirement candidates and why evidence/worktree removal is not automatic.
