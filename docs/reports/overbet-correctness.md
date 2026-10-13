# Overbet-menu consumer and CLI correctness

[PR #258](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/258) closes three gaps in merged #256: Python reference traversal chose v1 actions under v2 keys; independent evaluation replay reconstructed v1 menus; native commands silently ignored unsupported/malformed flags.

Reference training and lookup now derive sizes from the configured schema. Replay uses the schema and card tables owned by each actor's policy, so v1, v2 and both mixed-policy directions replay and reproduce exactly. Fixed scripted/attacker references retain their declared v1 menus. Unfinished search, translation, windowed extraction, bench exports and historical support-witness formats explicitly refuse v2. Card-descriptor-v2 remains distinct from menu-v2. [Every call site and native launcher](overbet-correctness-artifacts/inventory.md) is inventoried against main `9fc16841`: 79 menu constructors in 56 files.

Native argument admission is command-specific. Unknown/wrong-command flags, duplicates, absent values, malformed integers/floats/lists/enums and misplaced positionals fail clearly on stderr with exit 2, without panic backtraces. `train` and hand `parity` support `--menu v1|v2`; `bench-train`, traversal/run parity, export and engine-deal reject that option. Bench/traversal/run fixtures remain v1-only. Each valid command emits one resolved JSON configuration line to stderr; resumed seed, roots and averaging reflect the checkpoint. Serialization, stdout formats and resource-stop exit 3 are unchanged. Timing fields and telemetry's executed-binary hash naturally remain run-specific; no telemetry fields were changed.

## Exact v1 harness comparison

Main and branch each ran 8 duplicate blocks (16 hands) through the direct harness at root `2026100922201`, rung `fixture`, with the existing one-node synthetic average from `tests.test_native_hu100_baseline.model_fixture`, under names `candidate-model`/`reference-model`. The pot-pressure smoke used that candidate, arm `v1` (translation off), 8 blocks and root `20261013100101`. Both independently replayed. Hashes cover **every decompressed JSONL byte**, including hand outcomes, menus, keys, action streams and traces. Gzip timestamps/headers are excluded because these harnesses historically write timestamped gzip; compressed-container identity is not claimed. Exact reconstruction recipes are in the archive.

| Stream | Main SHA256 | Branch SHA256 |
|---|---|---|
| direct/decisions.jsonl.gz (JSONL bytes) | `9e990d4ef5ec5ed5d9b061a747c75e6d75ff0fa9ef40498b6c6fd4f318352e90` | `9e990d4ef5ec5ed5d9b061a747c75e6d75ff0fa9ef40498b6c6fd4f318352e90` |
| direct/hands.jsonl.gz (JSONL bytes) | `88ae31167b1e31713685bdabb7c3c3f68b2258174847548e033c647f05fdc43e` | `88ae31167b1e31713685bdabb7c3c3f68b2258174847548e033c647f05fdc43e` |
| pressure/decisions.jsonl.gz (JSONL bytes) | `c891d2c2548e32f59d701cf95e7c11c19ae43cf94422f2e034ba9029f0b0a491` | `c891d2c2548e32f59d701cf95e7c11c19ae43cf94422f2e034ba9029f0b0a491` |
| pressure/hands.jsonl.gz (JSONL bytes) | `9b1e0c25b665f9a4ec87786c9da6fcfa07d34867c8044beabf721e915cf9f48d` | `9b1e0c25b665f9a4ec87786c9da6fcfa07d34867c8044beabf721e915cf9f48d` |

The same small matches were also repeated on pristine main and the branch with the gzip clock fixed to `1234567890`, using the same output filenames. **All complete compressed file bytes match**, without normalization. The gzip clock is fixed to make headers reproducible; game/deal/action seeds and harness code are unchanged, and the JSONL bytes also match the original normal-clock runs. Independent replay passes on both. [Supplemental archive receipt](overbet-correctness-artifacts/fixed-gzip-archive-receipt.json).

| Complete gzip stream | Main SHA256 | Branch SHA256 |
|---|---|---|
| direct/decisions.jsonl.gz | `0e1c798415ebdd49d9d59c44e56c97641575fe76f3a1e352fc4dc33d466241a2` | `0e1c798415ebdd49d9d59c44e56c97641575fe76f3a1e352fc4dc33d466241a2` |
| direct/hands.jsonl.gz | `539505c6f03e5f5b74fec8534ca414ef0b81b668b729b67aef47c319136259b1` | `539505c6f03e5f5b74fec8534ca414ef0b81b668b729b67aef47c319136259b1` |
| pressure/decisions.jsonl.gz | `63d646209a35dc21a22d916a3f87b29cfaf401d72d5c99add14bc6f9b71b6068` | `63d646209a35dc21a22d916a3f87b29cfaf401d72d5c99add14bc6f9b71b6068` |
| pressure/hands.jsonl.gz | `64ec5e8f10195138d2a90d802e7a96a1cbf723d4a4036a0e09221f24fdef0370` | `64ec5e8f10195138d2a90d802e7a96a1cbf723d4a4036a0e09221f24fdef0370` |

## Exact native output comparison

Both binaries were built with `python -m scripts.build_native_trainer`, on the M1, Rust 1.96.0, using the same lockfile/linker. No M4 or long benchmark was used. Main binary SHA256 is `37625009eadae1a6633346a828ecad299de5b0b13be0654b2e72c92cb66d6e46`; branch is `0e50a7d681a8491763f668efef8d751581901b9e4efd8f5fd853e767dc5ceb7b`. [Complete build and identity receipts](overbet-correctness-artifacts/identity.json) retain toolchain details. The branch was built before commit; its source is bound to committed native tree `a56ce73696e81a89e4a0fd2c66b8626f7ff91f35` at code revision `070207f377901814535ca04f10c39a793667c3cf` in the archive receipt.

For each HU20/HU100 × v1/v2 cell: `train --stack-bb BB --menu MENU --nodes 10000 --seed 11 --average-rule opponent-sampled --recovery --out CHECKPOINT`, then `export CHECKPOINT --current CURRENT --average AVERAGE`. The HU100/v2 checkpoint was additionally resumed to 20,000 nodes with its SHA256 and exported again. All 15 **complete compressed files** are byte-identical to main. Each hash below is the main hash **and** the branch hash; the JSON receipt records both separately.

| File | SHA256 on both binaries |
|---|---|
| hu100-v1-average.gz | `ad02de0a4a4ff9ed4cb232efc2d550ed57442fd4c4fc454514d9b20fe808a0ac` |
| hu100-v1-current.gz | `4ece8fa5eef3c2bb095236b8d424cf1324be907666869e4818e0debf6e9d53d8` |
| hu100-v1.json.gz | `54a9fe16608262f4608481ab67d355840f265522dcdfa44e918a86e936a633f4` |
| hu100-v2-average.gz | `d7aa4056dde8f36f953c9496f7d5180a3654b630b19f944d162095abd4eef86b` |
| hu100-v2-current.gz | `0d402202065209fde95b67f390596f2e503e9ad354d5aed0b6d577a0c037b0cd` |
| hu100-v2-resumed-average.gz | `5410b9a0094ad5138f31c65e489b878b6c29fb0db32aac900685fb9ab7167fb1` |
| hu100-v2-resumed-current.gz | `6663021c86ca434d448cecae8a949aecb940972a4844b40a530ee7852e35128d` |
| hu100-v2-resumed.gz | `ac1ceddecd7f14473fdfa3c90b82c34fe798957118ef3eec09240a979528d718` |
| hu100-v2.json.gz | `2313e77ebcafe90fada873e03b247e5387e7d6de0737d5eb61588d1b116d6d03` |
| hu20-v1-average.gz | `42276fde895eefd8f400bba439fc3d680b35dc612b1489d40ebebdb47ec257ec` |
| hu20-v1-current.gz | `5fe0226329d6f09a4e0a8ddf26146ebdebfba91c6ad0c0318b89e77c74e87913` |
| hu20-v1.json.gz | `fcb97e5349be39b36286427c03efa9ba1841a1b08586065bd784fdfc236bf446` |
| hu20-v2-average.gz | `dbd65dc9d07538f3af70d2627e2342c91f1edcc92e328ccf9e7e2d429594cfd3` |
| hu20-v2-current.gz | `1abc5c365327a986acfaf7b4745b65a7da857437728cdce358ed8ac43fb7c716` |
| hu20-v2.json.gz | `4f6a5fcec6cefced568a887eef8d69645a2b5663186c326ae1a1cdbd413b9f72` |

## Regression evidence

- Synthetic direct play, tracing, independent replay and reproduction: v1/v1, v2/v2 and both mixed directions. Pot-pressure v1/v2 replay. Tiny Python v2 step stores half, pot-1.5 and pot-2 names; every stored key has a witnessed schema-derived menu. Unsupported consumers fail before work: **9 tests pass**.
- CLI subprocess argument admission/resolved resume configuration: **26 pass**; Rust CLI unit tests: **2 pass**.
- `cargo test --locked --manifest-path native/hu20-trainer/Cargo.toml`: **14 library + 2 CLI tests pass**.
- Final `tests/test_native_trainer_parity.py`: **19 pass**, including engine off-menu/history keys and v2 plain-card/K50 parity, v1 run/output parity and v2 resume.
- CLI, native bench, v2 consumers and saved-policy diagnosis combined: **62 pass**; clean native baseline/search/robustness/conditional-river adapters: **113 pass**; consumer/lookup/windowed/K50 replay/zero-mass audits: **24 pass**. Native HU100/HU200/save-stop tests also passed in the initial 87-test set.

The initial CI run also caught historical Target/Uniform wrappers that do not expose `abstraction`; pinned target-spec schemas and fixed v1 fixture defaults preserve their interfaces. All **33 affected local regressions pass** after that correction. [CI failure logs, corrected source patch and local rerun evidence](overbet-correctness-artifacts/ci-archive-receipt.json) are preserved.

Failures are preserved in the archive: an initial harness recipe omitted PYTHONPATH; a synthetic v2 accumulator exceeded its test visit bound; two schema-less robustness fixtures needed their existing schema fallback; a DCFR diagnostic wording mismatch was corrected; dirty-source qualification correctly refused work and passed on the clean revision; one test selection named absent files. These were small engineering checks, not campaign hands or scientific retries. No training campaign, external-service calls, policy adoption, release, tag or merge is included. Final GitHub CI status is shown on the PR.

[Archive receipt](overbet-correctness-artifacts/archive-receipt.json) and [retrieval index](../../RESULTS_INDEX.md) retain the raw comparisons, binaries, test logs (including failures), source/provenance and exact reproduction recipes. Research payloads stay out of Git.

Archive upload is pending in native Drive (no conflicts); independent cloud discovery has no item IDs yet. Parent [Research-Cloud](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s) is confirmed. No cloud upload acceptance or remote-byte verification is claimed; all originals remain.

## Owner review corrections

The four remaining owner-review findings are fixed at `e59bdf4202371b754ade2e6e56f3e1569362bbde`:

| Finding | Result and regression |
|---|---|
| LiveBlueprint hid v2 from search guards | The actual live wrapper rejects menu-v2 before it can reach SearchPlayer, LocalCFRPlayer, HU20RiverPlayer or RiverCFRPlayer. Tests construct real HU20/HU100/HU200 v2 trainers; v1 wrapped distribution is unchanged. |
| K50/v2 replay missed its tables | Discovery and key construction recognize equity schemas and explicitly pass the pinned tables. Passive postflop v2/card-v1 matches in both directions and K50/v2-vs-K50/v1 replay to river with global registration cleared. |
| Resolved receipts reflected requested values | `requested` preserves raw lineage and thresholds; `options` holds the parsed lineage and sorted, deduplicated, eligible schedule. Runner and receipt use the same resolved values. Resume removes completed milestones and records initial nodes/iteration. Early resource/iteration stops can truncate the eligible schedule. Tests compare receipt lineage/iterations against actual exports and train/resume stdout. |
| DCFR domains were checked after input loading | Admission calls the same cheap `Options.check_domain` used by numerical `Options.check`. Alpha, gamma and iteration restrictions reject before opening nonexistent roots or emitting configuration. Full discount-prefix validation remains authoritative and separate from the cheap domain checks. |

Post-review verification: **32 Python consumer/replay/search tests**, **71 native CLI/bench/parity tests**, and **14 Rust library + 4 CLI tests** pass. The new binary was built with the path-independent builder: SHA256 `1d907872b96346ba42aff696908099a659df7ac614559f4cdbacb1e1ae4e6416`, committed native tree `0aeb9974f32a99b141e9eabe3e641e819320d242`. All 15 complete native checkpoint/export/resume files and all four fixed-clock compressed harness streams retain the main hashes above. Six tiny bench exports (numeric lineage 3, requested checkpoints 2,2,0,99 → effective 2,3) also match main's binary byte-for-byte. [Review build, output hashes and model-member receipts](overbet-correctness-artifacts/review-identity.json), [archive receipt](overbet-correctness-artifacts/review-archive-receipt.json). Upload acceptance remains pending; originals are retained. This verification changes no training recipe, policy or release.
