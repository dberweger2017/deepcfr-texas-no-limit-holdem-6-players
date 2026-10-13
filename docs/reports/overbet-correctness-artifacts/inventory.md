# Menu consumer and native launcher inventory

Baseline main `9fc168414f5205f7483d0c6d474919afd7f9b777`. Python AST inspection finds **79 abstraction `choices(...)` calls in 56 files** under `src/` and `scripts/`. RNG `.choices(...)` calls are samplers, not menu constructors, and are excluded. The locations below name baseline lines; changed lines can move. Fixed opponent/reference trees have no model-schema argument and keep their declared v1 contract; they never label those menus as v2 policy keys.

| Call site on main | Handling in this PR |
|---|---|
| `scripts/audit_hu20_cards_v2.py:53` | Fixed HU20 card descriptor v1/v2 comparison, both use menu-v1 (card-v2 is not menu-v2). |
| `scripts/audit_hu20_cfr_plus_arena.py:106` | Fixed HU20_UNCAPPED_SCHEMA key; frozen v1 campaign model loader. |
| `scripts/audit_hu20_k50_first_pass_direct.py:56` | Each replay actor uses its own schema; all equity schemas, including K50/menu-v2, discover and receive explicit pinned tables. |
| `scripts/audit_hu20_turn_search.py:63` | Fixed v1 search protocol; source/search admission rejects v2. |
| `scripts/audit_native_hu100_baseline.py:106` | Candidate schema/tables; v1 baseline keeps historical candidate-key coverage context, while v2 uses the uniform reference's own fixed v1 menu/keys. |
| `scripts/check_exact_lbr_ranker.py:45` | Independent v1 attacker-menu diagnostic; no schema argument. |
| `scripts/continue_hu20_history.py:87` | Explicit v2 rejection; comparison corpus requires a shared v1 menu. |
| `scripts/diagnose_native_hu100.py:132,228` | Explicit native card-v1 admission in replay; support-witness history/fixtures are v1. |
| `scripts/evaluate_blueprint_symmetry.py:81` | Schema sizes in probe; historical v1 unchanged. Canonical mode still requires verified legacy schema. |
| `scripts/evaluate_hu100_direct.py:165` | Each replay actor uses its own policy schema and tables. |
| `scripts/evaluate_hu20.py:51` | HU20_SCHEMA-only manifest admission; Uniform is an explicit v1 reference. |
| `scripts/evaluate_hu200_diagnosis.py:108,132,138` | Explicit HU200 card-v1 admission (schema-less test fixtures retain their v1 default); support-witness signatures/native fixtures are v1. |
| `scripts/evaluate_hu200_translation.py:45,45,63,88,114` | AveragePolicy(expected_schema=HU200_SCHEMA); translation remains v1-only. |
| `scripts/evaluate_hu20_cfr_average.py:33,62` | Uniform/scripted opponent snapshots use an explicit v1 reference menu; target uses source.distribution. |
| `scripts/evaluate_hu20_river.py:105` | Independent rival menu; target owns its menu and river-search admission rejects v2. |
| `scripts/evaluate_hu20_turn_search.py:73` | Explicit HU20_UNCAPPED_SCHEMA source admission; rival reference menu v1. |
| `scripts/evaluate_robustness.py:28,68,69` | Target schema sizes; original cap2/attacker menu deliberately v1. Schema-less fixtures use the declared/default v1 schema. |
| `scripts/evaluate_tp20.py:24` | TP20_SCHEMA-only source admission; independent uniform reference v1. |
| `scripts/hu20_reopening_common.py:30` | Model-independent cap2/native v1 fixture trajectories; no schema argument. |
| `scripts/native_parity_fixtures.py:43` | Explicit --menu determines both schema and menu sizes. |
| `scripts/play_hu20.py:81` | HU20_SCHEMA-only manifest admission. |
| `scripts/play_hu20_native.py:38` | Human menu derives from base policy schema; bot already owns its distribution. |
| `scripts/play_tp20.py:75` | TP20_SCHEMA-only source admission. |
| `scripts/pot_pressure_diagnostic.py:161` | Candidate schema and tables; scripted rival owns its separate action contract. |
| `scripts/preflight_hu20_history.py:68` | Explicit v2 rejection; comparison corpus requires a shared v1 menu. |
| `scripts/probe_robustness_rivers.py:26` | Independent v1 attacker probe; target distribution owns its schema. |
| `scripts/replay_native_hu100_examples.py:20,78,81,87` | Explicit native card-v1 admission for archived examples/witnesses. |
| `scripts/report_audit_hu20_zero_mass_pressure.py:63` | Explicit HU20 card-v1 admission; historical four-arm protocol. |
| `scripts/report_hu20_reopening.py:68,68` | Pinned target-spec schema sizes; original cap2 menu deliberately v1. |
| `scripts/report_luna_browser.py:100` | Fixed v1 browser release menu transcript; play service rejects v2. |
| `scripts/resolve_hu20_search_stackoff.py:365` | Fixed HU20 v1 search root witness; search admission rejects v2. |
| `scripts/validate_flop_check.py:37` | Fixed HU20 v1 local tree fixture; blueprint_support requires HU20_UNCAPPED_SCHEMA. |
| `scripts/validate_reverse_lbr_acceleration.py:138` | Independent v1 attacker-menu diagnostic; target distribution owns its schema. |
| `src/arena/endgame_quality.py:105` | Independent tiny RiverGame/reference action tree; no policy schema parameter. |
| `src/arena/policies.py:55` | NativeHU100Uniform is an explicitly versioned v1 reference, not a model-schema consumer. |
| `src/blueprint/abstraction.py:148` | schema_choices derives menu_sizes(schema). |
| `src/blueprint/action_translation.py:91` | Existing version_for_schema permits only HU100/HU200 card-v1. |
| `src/blueprint/artifact.py:393` | Already uses menu_sizes(self.abstraction). |
| `src/blueprint/average.py:150` | Already uses menu_sizes(self.abstraction). |
| `src/blueprint/evaluation.py:23` | Schema sizes; native no-free-fold for v2, historical v1 unchanged. |
| `src/blueprint/hu20_turn_tree.py:84` | Turn-search wrapper explicitly rejects v2; compile_tree has no policy schema parameter. |
| `src/blueprint/local_cfr.py:308` | Local solver/player explicitly reject v2; local search tree is v1. |
| `src/blueprint/lookup.py:99` | Schema sizes, trainer cap and no-free-fold for native schemas. |
| `src/blueprint/river_game.py:217` | River search/conditional adapters explicitly reject v2; game constructor has no policy schema parameter. |
| `src/blueprint/solver.py:382` | Schema sizes, existing cap/free-fold convention; K50 step still refused. |
| `src/blueprint/windowed.py:78,246` | Explicit v2 rejection in collector/reader; extraction identity is v1-only. |
| `src/diagnostics/action_translation.py:27` | Nearest-raise wrapper explicitly rejects v2; legacy history labels. |
| `src/diagnostics/flop_check.py:53,56,114,161,200` | Fixed HU20_UNCAPPED_SCHEMA keys; blueprint_support rejects other schemas. |
| `src/diagnostics/history_alias.py:87,167` | Fixed HU20_UNCAPPED_SCHEMA keys and corpus; no model-schema argument. |
| `src/diagnostics/robustness.py:24,152` | Independent v1 ReactiveAttack/LBR attacker contract; target uses its own distribution. |
| `src/diagnostics/selective_stackoff.py:56` | Independent scripted opponent contract; no policy schema argument. |
| `src/diagnostics/stackoff_tails.py:98` | RecordingOpponent snapshots independent scripted v1 opponent; RecordingTarget uses source.distribution. |
| `src/diagnostics/strong_rollout.py:32,34` | Independent restricted/native attacker menu; target uses its own distribution. |
| `src/diagnostics/subgame_bench.py:166` | Accepts only HU20 native card-v1/K50-v1; export identity is v1-only. |
| `src/play_api/service.py:276,435` | validate_policy admits only HU20/HU100 card-v1; human restricted menu is v1. |
| `src/play_api/uniform_random.py:45` | Versioned independent uniform v1 runtime reference; no policy schema argument. |

The LiveBlueprint wrapper rejects v2 at its boundary; the additional search entry points (`search.SearchPlayer`, `HU20TurnSearchPolicy`, `HU20RiverPlayer`, `RiverProfileCache`, `public_ranges`, `RiverCFRPlayer` and `ConditionalRiverBestResponse`) now refuse menu-v2 before building v1 trees. K50 reference training remains explicitly refused by `step()`.

## Native command admission

| Command | Positionals | Value flags | Boolean flags | Menu |
|---|---|---|---|---|
| train | none | nodes, iterations, seed, roots-per-seat, out, milestones, stack-bb, max-entries, max-seconds, stop-file, telemetry, resume, resume-sha256, completed-nodes, average-rule, regret-floor, card-buckets, menu | recovery, card-tables-unpinned | v1/v2 supported |
| parity | fixture path before flags | card-buckets, menu | card-tables-unpinned | v1/v2 supported |
| bench-train | none | roots, seed, iterations, lineage, variant, out, checkpoints, regret-floor, dcfr, card-buckets | card-tables-unpinned | fixed v1; --menu rejected |
| traversal-parity / run-parity | one fixture path | none | none | v1 fixture format; --menu rejected |
| export | checkpoint path before flags | current, average, zero-mass | none | uses checkpoint identity; --menu rejected |
| engine-deal | one or more integer seeds | none | none | no tree; --menu rejected |

Unknown/wrong-command flags, duplicates, missing values, misplaced positionals and unparseable values fail with exit 2 and a message on stderr before outputs. Numeric/enum/list arguments are validated even when they would not affect an iteration-limited/resumed run. Resume flags must appear together, `completed-nodes` requires resume, and unpinned tables require a table path. Valid execution prints exactly one resolved configuration JSON line on stderr (resume seed/roots/averaging inherited from the checkpoint). Raw lineage/save-threshold requests are separate from effective options; lineage and schedules consumed by the runner are the receipt values, with completed train milestones removed on resume. Cheap DCFR domains are shared with numerical validation and checked before inputs. Existing stdout and serialization code are unchanged; runtime failures report their cause without panic backtraces. Resource-stop exit 3 is unchanged.

## Binary invocations on main

AST lists, appended options, wrapper builders, shell files and CI were inspected. These are the files that invoke or construct commands for the trainer; archive-only binary references and unrelated Python `train` subcommands are excluded. No historical native launcher with a silently ignored flag was found. `subgame_bench --option` forwards optional bench flags (current recipes use regret-floor/DCFR); unsupported future options now fail. The apparent `--control` launchers in `dr2x2_pod.py` and `hu20_500m_pod.py` run Python worker subcommands, not this binary.

| Launcher file | Commands; accepted flags |
|---|---|
| `scripts/benchmark_hu_export_audit.py` | export; --average, --current |
| `scripts/prepare_native_hu_campaign.py` | export, train; --average, --average-rule, --current, --max-entries, --max-seconds, --milestones, --nodes, --out, --roots-per-seat, --seed, --stack-bb, --zero-mass |
| `scripts/prepare_native_hu_execution.py` | export, train; --average, --average-rule, --completed-nodes, --current, --max-entries, --max-seconds, --milestones, --nodes, --out, --resume, --resume-sha256, --stack-bb, --stop-file, --telemetry, --zero-mass |
| `scripts/run_hu100_3b_campaign.py` | export, train; --average, --average-rule, --current, --max-entries, --nodes, --out, --roots-per-seat, --seed, --stack-bb, --stop-file, --telemetry, --zero-mass |
| `scripts/run_hu100_4b_seed_ladder.py` | export, train; --average, --average-rule, --current, --max-entries, --nodes, --out, --roots-per-seat, --seed, --stack-bb, --stop-file, --telemetry, --zero-mass |
| `scripts/run_hu100_4b_seed_replication.py` | export, train; --average, --average-rule, --current, --max-entries, --nodes, --out, --roots-per-seat, --seed, --stack-bb, --stop-file, --telemetry, --zero-mass |
| `scripts/run_hu100_independent_stages.py` | export; --average, --current, --zero-mass |
| `scripts/run_hu100_k50_2b.py` | export, train; --average, --average-rule, --card-buckets, --current, --max-entries, --nodes, --out, --roots-per-seat, --seed, --stack-bb, --telemetry, --zero-mass |
| `scripts/run_hu100_seed_qualification.py` | export, train; --average, --average-rule, --current, --max-entries, --nodes, --out, --roots-per-seat, --seed, --stack-bb, --stop-file, --telemetry, --zero-mass |
| `scripts/run_hu200_diagnosis.py` | parity; positionals only |
| `scripts/run_hu200_feasibility.py` | export, train; --average, --average-rule, --current, --max-entries, --max-seconds, --nodes, --out, --seed, --stack-bb, --telemetry, --zero-mass |
| `scripts/run_hu200_k50_first.py` | export, train; --average, --average-rule, --card-buckets, --current, --max-entries, --nodes, --out, --roots-per-seat, --seed, --stack-bb, --telemetry, --zero-mass |
| `scripts/run_hu200_translation.py` | parity; positionals only |
| `scripts/run_hu20_equity_bench.py` | bench-train; --checkpoints, --iterations, --lineage, --out, --roots, --seed, --variant |
| `scripts/run_hu20_k50_first_pass_gates.py` | train; --average-rule, --card-buckets, --completed-nodes, --max-entries, --milestones, --nodes, --out, --resume, --resume-sha256, --seed, --telemetry |
| `scripts/run_hu20_k50_first_pass_training.py` | export, train; --average, --average-rule, --card-buckets, --current, --max-entries, --nodes, --out, --seed, --telemetry, --zero-mass |
| `scripts/run_hu20_zero_mass_fallback.py` | export; --average, --zero-mass |
| `scripts/run_native_hu100_diagnosis.py` | parity; positionals only |
| `scripts/run_native_hu100_growth.py` | export, train; --average, --average-rule, --current, --max-entries, --max-seconds, --nodes, --out, --resume, --resume-sha256, --roots-per-seat, --seed, --stack-bb, --stop-file, --telemetry, --zero-mass |
| `scripts/run_native_hu100_growth_1b.py` | export, train; --average, --average-rule, --current, --max-entries, --milestones, --nodes, --out, --roots-per-seat, --seed, --stack-bb, --stop-file, --telemetry, --zero-mass |
| `scripts/run_roots_per_seat.py` | export, train; --average, --average-rule, --current, --max-entries, --nodes, --out, --roots-per-seat, --seed, --stack-bb, --stop-file, --telemetry, --zero-mass |
| `scripts/subgame_bench.py` | bench-train; --checkpoints, --iterations, --lineage, --out, --roots, --seed, --variant |
| `scripts/train_hu20_cfr_plus_lineages.py` | export, train; --average, --average-rule, --current, --milestones, --nodes, --out, --regret-floor, --roots-per-seat, --seed |
| `tests/diagnostics/test_compact_policy.py` | export, train; --average, --nodes, --out, --seed |
| `tests/diagnostics/test_streaming_cfr_audit.py` | export; --average |
| `tests/test_diagnose_hu20_checkpoints.py` | train; --nodes, --out, --seed |
| `tests/test_k50_evaluation.py` | export, train; --average, --card-buckets, --card-tables-unpinned, --current, --iterations, --nodes, --out, --seed, --stack-bb |
| `tests/test_native_bench_parity.py` | bench-train, train; --card-buckets, --card-tables-unpinned, --checkpoints, --dcfr, --iterations, --lineage, --nodes, --out, --roots, --seed |
| `tests/test_native_hu100_preparation.py` | export, parity; --average, --current |
| `tests/test_native_hu200.py` | train; --average-rule, --iterations, --nodes, --out, --resume, --resume-sha256, --seed, --stack-bb |
| `tests/test_native_hu_save_stop.py` | train; --average-rule, --max-seconds, --milestones, --nodes, --out, --seed, --stop-file, --telemetry |
| `tests/test_native_trainer_parity.py` | export, parity, train, traversal-parity; --average, --average-rule, --card-buckets, --card-tables-unpinned, --current, --iterations, --menu, --nodes, --out, --regret-floor, --roots-per-seat, --seed, --stack-bb, --zero-mass |

Appended train flags also include `--resume`, `--resume-sha256`, `--completed-nodes`, `--milestones`, `--recovery`, `--max-seconds`, `--stop-file`, `--regret-floor`, `--average-rule`, `--card-buckets`; bench appends `--card-buckets` and forwards the declared `--option` words. All are covered by the command table. `run_hu100_independent_stages.py` delegates train command construction to `run_hu100_seed_qualification.fixture_command`; growth-50M/other campaign wrappers delegate to `prepare_native_hu_execution`/`run_native_hu100_growth`. `run_hu100_3b_ladder.py` constructs model specs, not trainer argv. There are no direct native trainer invocations in `src/` or maintained shell files. CI builds via `python -m scripts.build_native_trainer`, runs `cargo test --lib`, then executes the Python native suites; the new subprocess CLI tests cover argument admission in that same suite. No open-PR-only launcher was modified.
