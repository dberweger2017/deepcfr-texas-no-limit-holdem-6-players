# Frozen native training tables

[PR262](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/262), October 13, 2026. Engineering support for the owner's HU200 K50 continuation; no campaign, match, release or strength claim. The owner decides the merge.

## Semantics

`train --freeze-at-cap` is a boolean accepted only by `train`, including its `--resume` form. Without it, entry stopping is unchanged. With it, a fresh run completes the first iteration reaching or overshooting `--max-entries`, applies **all** that iteration's updates, records the boundary, and continues. Starting with the next iteration, each missing key's merged delta is discarded. Every existing key receives exactly its usual regret, visit and average updates in the original task/addition order. Membership alone decides retention; zero traverser visits never suppress opponent-sampled average updates.

An unfrozen checkpoint already at or above the caller's cap freezes before the first resumed step, at its saved iteration. A frozen checkpoint stays frozen even if a later caller supplies a larger cap; `--freeze-at-cap` is mandatory when resuming it. Node/time/stop-file limits retain their existing meaning and atomic complete-iteration saves. This mode changes no game rules, observations, key/menu schemas, RNG streams or regret formula.

This is CFR on a restricted game: keys outside the retained table play uniformly for **both** players forever. Counterfactual traversal still explores their continuations, and stored ancestors learn against that fixed continuation. Keys that become important later can never be added. More visits to retained keys are not proof of stronger poker, convergence in the original game, or solved exploitability. Missing-key shares are diagnostics, not a strength gate.

## Recovery and receipts

Only an actually frozen checkpoint adds this header field; non-frozen headers and all row formats stay unchanged:

```json
{"frozen_table":{"version":1,"since_iteration":8728,"entries":7883678,"discarded_traverser_visits":23763,"discarded_updates":63052}}
```

`since_iteration` identifies the last iteration allowed to insert keys. `entries` fixes the cardinality and is checked against loaded rows. `discarded_traverser_visits` is cumulative and repairs recovery accounting. `discarded_updates` counts one missing key's merged delta per complete iteration, including opponent-only deltas; it does not count scalar additions or sampled decisions. Version/field/type checks reject malformed extensions. Frozen HU20 saves always carry `native_state`, including when `--recovery` was omitted.

Recovery verifies, with overflow checks:

```text
coverage_start[2] + sum(traverser_visits_by_street)
  == sum(stored node visits) + discarded_traverser_visits
```

A changed discarded-visit count is rejected. Existing checkpoint admission still compares rows against `min(caller max_entries, saved config.max_entries)`; freezing never relaxes it. The existing save raises the saved admission cap to the actual row count after an iteration's overshoot.

Each `--telemetry` save adds a `frozen_table` receipt with cumulative state, interval start iteration, discard counts **since the preceding save**, and separate traverser/opponent decisions, table misses and missing-decision shares by preflop/flop/turn/river. Opponent-only stored keys are hits even when they have no allocated regrets. Zero-decision shares are null. After recovery the first interval begins at process admission; checkpoint discard totals persist, while receipt-only miss counters restart. A first interval may straddle the freeze boundary; later intervals permit the diagnostic comparison of whether missing-key shares rise. These counters never enter non-frozen checkpoint bytes.

The existing `process_rss_bytes_after_save` remains in every receipt. Fixed entries do **not** imply fixed RAM: an existing opponent-only key can allocate regret storage when it first receives nonzero traverser regrets. At #250's 44.29% zero-visit density, several tens of millions of short-menu regret vectors could consume roughly another GB, plus allocator/chunk overhead. This is a planning estimate, not a measured bound; keep the campaign's RSS/pressure/swap/disk supervision.

Python's checkpoint reader loads frozen tables for evaluation and refuses `step()`, including traverser-reach checkpoints. Native/Python current/average export, `AveragePolicy`, and every-row accumulator audits accept the extra field. K50 fixture tests substitute pins only within tests; production pin admission remains strict.

## Resume #250 tonight

#250's nominal cap was **85,373,862**, but its complete-iteration terminal contains **85,373,907** entries. Use the **actual** count so loading succeeds and freezing happens before the first step. Its source filename is target-named `1000000000.gz`, containing **752,593,175 actual nodes**, not a reached 1B checkpoint. Its pinned recovery SHA256 is `4b25e8029ed0ebfbeeb42ac02dee3179303e7f2300c23c21d32d27382f390da4`.

From the checkout containing this change, after building with `python -m scripts.build_native_trainer`, the command is below. Paths name #250's indexed M4 originals; if restoration is needed, use its [model/retrieval index](hu200-k50-first-artifacts/model-index.json) and verify those hashes. This command was **not executed**. It assumes the owner's existing campaign resource supervision and a fresh output directory; a fresh stop-file request saves at a complete iteration. No M4 paths or processes were accessed for these gates.

```sh
mkdir results/hu200-k50-frozen-20b
native/hu20-trainer/target/release/hu20-trainer train \
  --stack-bb 200 --menu v1 --seed 2026101302 --roots-per-seat 1 \
  --average-rule opponent-sampled \
  --card-buckets /Users/dberweger/Local/hu200-k50-first-20261013/results/hu200-k50-first/tables \
  --resume /Users/dberweger/Local/hu200-k50-first-20261013/results/hu200-k50-first/training/1000000000.gz \
  --resume-sha256 4b25e8029ed0ebfbeeb42ac02dee3179303e7f2300c23c21d32d27382f390da4 \
  --max-entries 85373907 --freeze-at-cap --nodes 20000000000 \
  --out 'results/hu200-k50-frozen-20b/{nodes}.gz' \
  --telemetry results/hu200-k50-frozen-20b/telemetry.jsonl \
  --stop-file results/hu200-k50-frozen-20b/stop.request
```

The node target is total lineage nodes, not another 20B. The command schedules only the final save; the owner's wrapper may supply budgeted intermediate `--milestones` and a duration cap. A forced kill cannot recover unsaved iterations.

## Verification and M1 measurements

Full `cargo test --locked --manifest-path native/hu20-trainer/Cargo.toml`: **16 library and 5 CLI tests pass**; the two strengthened frozen unit tests pass again. Native CLI, frozen-table and existing `tests/test_native_trainer_parity.py`: **63 passed**. The final frozen-only suite, including an added traverser-reach Python read-only regression: **12 passed**. Existing native save-stop, recovery-equivalence and HU200 suites add **21 passed**. The CLI, frozen-table and parity runs together cover 64 distinct passing tests; the 21 recovery tests are additional. Rust checks serial and parallel shard application against an unfrozen reference supplied with exactly the same starting key set, covering opponent-only updates and first regret allocation.

Cap-not-reached, fresh freeze versus cap-stop/resume, and interrupted frozen continuation are byte-identical. Overshoot admission, frozen-without-flag rejection, changed discard counts, key lists, live regrets/averages, receipt intervals and both players' street misses are checked. HU20 and HU200 each pass native/Python export equality, compact `AveragePolicy` load, read-only checkpoint loading and full accumulator/current-regret audit with v1 cards and synthetic K50 tables through `--card-tables-unpinned`.

Main (`ac19e29a`) and candidate were both built with `python -m scripts.build_native_trainer` on the same M1/toolchain. Fixed 1M-node v1 HU20/HU100/HU200 checkpoints match main byte-for-byte, with and without the flag when no freeze occurs. [Compact validation](frozen-table-artifacts/validation.json) names exact hashes, saved states, operations and table inputs. Build receipts, full logs, telemetry and sample histories are retained in the indexed archive.

Small fixed-table runs: seed 5, opponent-sampled average, cap 20,000, saves near 500k and 1M nodes, two traversal workers. Throughput is native elapsed **including writes**; family peak is sampled every 200 ms including the lightweight Python gate supervisor. These brief runs are instrumentation checks, not throughput forecasts for the 85M-entry table.

| Run | Actual terminal nodes | Fixed entries | Nodes/s incl. writes | Process RSS after save | Sampled family peak | Last-interval missing decisions |
|---|---:|---:|---:|---:|---:|---:|
| HU20 v1 | 1,000,575 | 20,025 | 1,027,653 | 8.6 MiB | 36.9 MiB | 80.25% |
| HU100 v1 | 1,001,295 | 20,561 | 1,419,643 | 11.1 MiB | 40.5 MiB | 97.01% |
| HU200 v1 | 1,000,140 | 21,237 | 1,442,815 | 11.4 MiB | 40.9 MiB | 98.57% |

The real-table gate reads only `~/Local/slumbot-k50-20261013/tables/`. All three full SHA256s match #163: flop `084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111`, turn `c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f`, river `70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629`.

Fresh HU200 K50 seed **2026101302**, one root/seat, linear iteration weighting, opponent-sampled average and v1 menu reproduce #250's **20,002,695-node /8,728-iteration** 20M gate exactly: **270,143,830 bytes**, SHA256 `4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`. It never freezes. Native training before serialization took **14.88 s**, write **42.44 s**, throughput including write **348,986 nodes/s**. Entire guarded operation including table load, hashing and telemetry took **65.00 s**, sampled family peak **1.919 GiB**.

A resume frozen at those **7,883,678 entries** reaches **20,200,258 nodes /8,811 iterations**. Both sorted key digests are `e2a21320e123229a52d0941d200c07fe885dafa104fe9cdc3087269c2c9dbafb` (SHA256 of the concatenated 32-character serialized keys). Stored traverser visits rise **3,892,221 → 3,907,064**; **23,763** discarded visits and **63,052** merged updates are recorded, and recovery accounting reconciles exactly. The frozen checkpoint is **270,343,154 bytes**, SHA256 `6835968c723cfb5247a8f2b6402493ccb73ee996300ef2b39efd74a80e63f1a2`. Native segment **0.158 s /1.248M nodes/s before writing**, write **42.99 s**; whole load/train/save/receipt operation **63.85 s**, sampled family peak **1.833 GiB**. The short segment's save dominates its 4,578 nodes/s inclusive rate. macOS page residency varies: after-save RSS alone is not a peak or a fixed-table memory bound.

Each trainer operation was bounded to 240 seconds, 5 GiB sampled family RSS and 32 GiB free disk. No guard, hash, key-set or scientific gate failed. No scientific retry occurred; complete logs remain in the archive. No Slumbot/external-service calls, large-model continuation, M4 access or other worktree changes occurred.

## Evidence

All generated checkpoints, both executables, build receipts, test logs, gate commands, telemetry, source snapshot and resource samples are archived with member manifests in `~/Local/Research-Cloud/PR-262-frozen-table/`. Tables are excluded and refer to #163's accepted archive and exact hashes. [RESULTS_INDEX](../../RESULTS_INDEX.md#pr262-frozen-native-training-table--october-13) records whole/member SHA256s, restoration commands and native/independent Drive acceptance. Originals remain in the isolated PR root; no cleanup or synced-folder eviction is authorized by this task.
