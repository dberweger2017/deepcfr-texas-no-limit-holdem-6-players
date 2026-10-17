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
- `src/policies/`: released HU20/HU100 artifact pins and HU200 bundle verification/loading.
  Adding a released model here or to the web catalog still requires release authorization.
  `hu100_release.py` loads a verified HU100 release bundle (v0.5.0–v0.5.3; v0.5.3 is unpublished owner review);
  `hu100_bundle.py` pins each release's model and provenance and is the bundles'
  standalone verifier. `--hu100-release` runs a bundle as its own table, outside
  the HU20 catalog. [v0.5.2 install](releases/v0.5.2/INSTALL.md), [v0.5.3 split model/K50 install](releases/v0.5.3/INSTALL.md).
  `hu100_research.py` is an explicit research-only pin; [local HU100 play](hu100-local-play.md)
  records its retrieval gate, fixed table and off-by-default translation option.
- `src/play_api/` and `apps/poker-web/`: durable local sessions, HTTP and the table.
  `versions.py` has the explicit release catalog. Defaults never silently fall
  back when an artifact is missing or corrupt; sessions keep their model.
- `native/hu20-trainer/`: native training, game/key parity, exports and bench.
  `src/nplayer.rs` is an isolated 2–6-player rules core; default training uses
  the existing HU path. Its [Phase 1 parity gate](nplayer-phase1-parity.md) runs with
  `cargo build --release --locked --manifest-path native/hu20-trainer/Cargo.toml --example nplayer_parity`
  followed by `python -m scripts.check_nplayer_parity --out results/nplayer-parity`.
  The separate [N-player traversal gate](nplayer-traversal-protocol.md) uses
  `multi_cfr.rs` for both Kuhn/Leduc and the cash adapter. Build experimental
  commands with `cargo build --release --locked --manifest-path native/hu20-trainer/Cargo.toml --example multi_known --example nplayer_train`.
  `multi_known kuhn 3 2000000 2026101501 NEW_OUTPUT_DIRECTORY` runs the declared
  small-game experiment. Leduc failed its first fixed endpoint; the separately
  owner-approved three-seed confirmation passes at 2M iterations with the same
  limit. The owner cleared the independent traversal review at PR289 head
  `5df55c52`; the [size/smoke report](reports/nplayer-size-smoke.md) follows.
  The cash command requires
  explicit `--players`, `--nodes`, `--max-entries`, `--max-seconds`, `--card-buckets`
  and `--out`; optional `--stack-bb`, `--stacks`, blinds, `--last-average-street`
  (0=preflop through 3=river), `--freeze-at-cap`, `--telemetry` and `--stop-file`
  control a bounded run. Resume requires `--resume` plus `--resume-sha256` and
  identical game/seed/averaging/cap settings. This version supports only the exact
  HU v1 sizing convention, including its conditional jam and duplicate removal.
  N-player checkpoints use a distinct N-tagged format; legacy HU commands and
  checkpoints remain supported through their unchanged entry point.
  Build `--example nplayer_size --example hu_size_reference` for the
  [fixed size study](nplayer-size-smoke-protocol.md); run its guarded pilot/main
  phases with `python -m scripts.run_nplayer_size_study --phase pilots|main --out NEW_DIRECTORY`.
  `python -m scripts.analyze_nplayer_sizes RESULTS_JSON --out PROJECTIONS_JSON` emits growth scenarios.
  Build `--example nplayer_policy` for minimal N-player research inference:
  `nplayer_policy export CHECKPOINT SHA256 current|average|scoped-average NEW_OUTPUT`.
  Full `average` requires all streets; `scoped-average` explicitly uses current
  regret-matching probabilities after the stored averaging scope. Missing keys
  and zero-mass rows use uniform probabilities over the exact legal menu.
  `nplayer_policy play K50 HANDS SEED NEW_TRACE POLICY SHA256 [POLICY SHA256 ...]`
  takes one independently seeded instance per seat, with matching game identities.
  Only acting-seat observations enter policy methods; the replay harness owns
  the privileged trace. Verify it with `python -m scripts.check_nplayer_policy_replay TRACE`
  and every export row with `python -m scripts.audit_nplayer_policy CHECKPOINT POLICY`.
  The fixed smoke runner requires its completed size/key/test admission receipts
  and the external on-host watchdog; it is not a general campaign launcher.
  Equal-stack HU20/HU100/HU200 identities are supported. HU100 operator commands
  are in [the native HU100 protocol](native-hu100-preparation.md); it authorizes no runs.
  The [HU200 M1 feasibility report](reports/hu200-feasibility.md),
  [frozen protocol](hu200-feasibility.md) and [Slumbot readiness note](hu200-slumbot-readiness.md)
  pin its distinct 200BB research lineage and remaining external work.
  `scripts/play_slumbot.py` plays bounded live hands against Slumbot with verified
  settlements; see the [Slumbot client report](reports/slumbot-client.md).
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

Build the native trainer with `python -m scripts.build_native_trainer`. A plain `cargo build`
embeds the checkout's absolute paths, so the same source gives different bytes in each
worktree. This build gives the same bytes from any checkout, given the same toolchain, linker
and Cargo home, and prints a receipt naming them. Compare binaries by their receipts: a
different toolchain or linker explains a different hash.

For native training start with the [native CLI](../native/hu20-trainer/src/main.rs)
and the command/source pins in the current campaign protocol; for inference extraction see
`scripts/extract_hu20_cfr_average.py`; for paired average-policy evaluation see
`scripts/evaluate_hu20_v041_arena.py` and its experiment protocol. A new campaign
still predeclares inputs, roots, comparison, stopping rule and cost.

HU100's [independent stages protocol](hu100-independent-stages.md) records
`scripts/run_hu100_independent_stages.py`: two-size cost-only calibration,
separate training/evaluation admission and a single fixed guarded budget.
Its one-use launcher is campaign evidence, not authorization for another run.
The [historical #211 report](reports/hu100-seed-qualification.md) retains that
attempt's older combined quote and 1M partial; it contains no new 1B results.
For archived assets, `scripts/restore_hu100_staged_archive.py` verifies the
whole ZIP, manifest and selected members, reconstructs pinned hardlink aliases
into a fresh destination, and never launches science. See RESULTS_INDEX for
the owning archive IDs and required hashes.

Shared behavior belongs in a library, with scripts as entry points. Prefer one
coherent refactor with regression coverage over renaming all historical modules.
New imports should use the current reader/release modules; to reproduce an old
campaign, use its exact recorded source revision. Recovery source checks include
the moved average reader, compact storage and checksum helper.

The [maintenance audit](reports/necessary-cleaning.md) records the remaining
retirement candidates and why evidence/worktree removal is not automatic.

## Three-player Phase 2 research

The [Phase 2 protocol](nplayer-blueprint-phase2-protocol.md) fixes the sequential
seeds, all-street averages and duplicate evaluation. Build `--example nplayer_arena`.
Its CLI is `nplayer_arena K50 BLOCKS ROOT NEW_TRACE SINGLETON PAIR`, where a player
specification is `policy=PATH@SHA256` or `bot=random|check-call|tight-aggressive|pot-pressure`.
Each block plays all six seat permutations. `scripts.check_nplayer_policy_replay`
checks every transition and the duplicate schedule; `scripts.analyze_nplayer_arena`
requires that receipt and a complete identical reproduction before calculating
block-cluster intervals. `scripts.machine_qualification.vrm_watchdog` is the
exclusive four-CPU on-host guard while cygnus-x1's VRM heatsink is absent; it logs
Thermistor 15 and fan1 and supplies `NPLAYER_STOP_FILE` to its child. Campaign
entry points consume the fixed protocol and do not authorize another run.
