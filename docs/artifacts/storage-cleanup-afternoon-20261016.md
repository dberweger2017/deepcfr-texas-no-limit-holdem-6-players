# M1-prioritized storage cleanup — October 16 afternoon

Owner-requested cleanup after checking recent PRs: #312 (cache policy) and #313 (contributor guide) are merged. New uploads go only through bb128; Mac Drive stays read-only. The deployed cache tool is still the original SHA256 `d03af1d5b0d3161723d44cbadf57cee62f5df60baaf3bd7b1fc32e340d49dcce`; this task does not deploy fixes or change cron. Open #301, #306 and #314 and their inputs remain protected.

| Host | Removed paths / unique inodes | Allocated bytes retired (upper bound) | Free at pass verification |
|---|---:|---:|---:|
| M1 | 40 / 16 | 14,747,889,664 | 16,854,740,992 (16.855 GB) |
| M4 | 4 / 4 | 4,693,065,728 | 57,726,742,528 (57.727 GB) |

GB is decimal. Allocation counts each inode once, including complete hardlink groups; APFS sharing can reduce physical recovery. Other activity changed free space during the task: the M1 passes observed +12,889,051,136 and +3,453,304,832 bytes, exceeding selected allocations in the first pass. Those observations are not an attribution of unrelated space changes. M4 observed +4,696,653,824 bytes. Subsequent free space may differ.

## Acceptance, unchanged files and dependencies

Current read-only `rclone lsjson --hash --files-only` on bb128 confirmed the indexed IDs, names, sizes and MD5s for PR298 evidence, PR272 model archives, PR163 tables, PR250 averages and PR211 evidence. Existing accepted upload/member proofs were reused; no research payload was downloaded, hydrated or rehashed for cleanup. Published v0.5.3 assets also match GitHub’s current IDs/sizes/SHA256 digests and the existing successful public-download verification. Owning PRs #211/#250/#283/#298/#309 were checked merged before removal.

Selection checked source paths, sizes, modification/change history, inodes/link counts and existing manifest/retrieval provenance. `lsof +D` returned no matches and no warnings for selected roots, repeated immediately before removal. Open PR301/306 planning had no references to selected copies. The old Slumbot model, tables and journals stay available for PR314 analysis. M4 PR306’s running evaluation, its source/environment/protected inputs and policy cache were untouched.

M1 package/publication retained-file checks passed for 625 files; seven canonical files passed the additional copy-cleanup check. M4 checks passed for 333 retained files, including the canonical terminal input. Entire source checkouts, shared Git, failures/probe records and the existing archive backup remain. No whole worktree was removed.

## Exact removed paths

All prefixes below are absolute nonsynced local paths. The linked full receipt records each size, full SHA256, mtime/ctime/inode/link group, provenance, restoration and per-path fsynced removal events.

### M1 package and publication copies

Prefix: `/Users/dberweger/Local/`.

| Relative removed path | Bytes | Restore rule |
|---|---:|---|
| `hu100-v053-release-package/results/v053-release-package/assets/river-k50.bin` | 1,231,562,564 | P |
| `hu100-v053-release-package/results/v053-release-package/assets/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-002` | 1,073,741,824 | P |
| `hu100-v053-release-package/results/v053-release-package/assets/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-003` | 1,048,807,903 | P |
| `hu100-v053-release-package/results/v053-release-package/assets/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-001` | 1,073,741,824 | P |
| `hu100-v053-release-package/results/v053-release-package/assets/turn-k50.bin` | 139,600,524 | P |
| `hu100-v053-release-package/results/v053-release-package/assets/flop-k50.bin` | 12,867,944 | P |
| `hu100-v053-release-package/results/v053-release-package/assets-attempt-000/river-k50.bin` | 1,231,562,564 | P |
| `hu100-v053-release-package/results/v053-release-package/assets-attempt-000/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-002` | 1,073,741,824 | P |
| `hu100-v053-release-package/results/v053-release-package/assets-attempt-000/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-003` | 1,048,807,903 | P |
| `hu100-v053-release-package/results/v053-release-package/assets-attempt-000/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-001` | 1,073,741,824 | P |
| `hu100-v053-release-package/results/v053-release-package/assets-attempt-000/turn-k50.bin` | 139,600,524 | P |
| `hu100-v053-release-package/results/v053-release-package/assets-attempt-000/flop-k50.bin` | 12,867,944 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-identity/river-k50.bin` | 1,231,562,564 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-identity/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-002` | 1,073,741,824 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-identity/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-003` | 1,048,807,903 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-identity/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-001` | 1,073,741,824 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-identity/turn-k50.bin` | 139,600,524 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-identity/flop-k50.bin` | 12,867,944 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-model/river-k50.bin` | 1,231,562,564 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-model/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-002` | 1,073,741,824 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-model/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-003` | 1,048,807,903 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-model/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-001` | 1,073,741,824 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-model/turn-k50.bin` | 139,600,524 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-model/flop-k50.bin` | 12,867,944 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-table/river-k50.bin` | 1,231,562,564 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-table/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-002` | 1,073,741,824 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-table/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-003` | 1,048,807,903 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-table/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-001` | 1,073,741,824 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-table/turn-k50.bin` | 139,600,524 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-table/flop-k50.bin` | 12,867,944 | P |
| `hu100-v053-release-package/results/v053-release-package/models/v052/O4B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz` | 2,121,415,396 | P |
| `hu100-v053-release-package/results/v053-release-package/runtime-refusals/wrong-model-model/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz` | 2,121,415,396 | P |
| `v053-publication/assets/river-k50.bin` | 1,231,562,564 | R |
| `v053-publication/assets/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-002` | 1,073,741,824 | R |
| `v053-publication/assets/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-003` | 1,048,807,903 | R |
| `v053-publication/assets/O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-001` | 1,073,741,824 | R |
| `v053-publication/assets/turn-k50.bin` | 139,600,524 | R |
| `v053-publication/assets/flop-k50.bin` | 12,867,944 | R |

### M1 duplicate retrieval copies

Prefix: `/Users/dberweger/Local/`.

| Relative removed path | Bytes | Restore rule |
|---|---:|---|
| `slumbot-k50-20261013/archive/hu200-k50-first-averages-M4-20261013.zip` | 2,221,371,477 | A |
| `cygnus-x1-qualification/results/cygnus-qualification/table-inputs/river-k50.bin` | 1,231,562,564 | T |

### M4 duplicate pilot inputs

Prefix: `/Users/dberweger/Local/`.

| Relative removed path | Bytes | Restore rule |
|---|---:|---|
| `hu100-seed-qualification-20261009/results/hu100-seed-qualification/pilot/old-terminal/models/47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9.json.gz` | 1,173,264,021 | Q |
| `hu100-seed-qualification-20261009/results/hu100-seed-qualification/pilot/old-terminal-on/models/47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9.json.gz` | 1,173,264,021 | Q |
| `hu100-seed-qualification-20261009/results/hu100-seed-qualification/pilot-reproduction/old-terminal/models/47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9.json.gz` | 1,173,264,021 | Q |
| `hu100-seed-qualification-20261009/results/hu100-seed-qualification/pilot-reproduction/old-terminal-on/models/47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9.json.gz` | 1,173,264,021 | Q |

## Restoration

- **P — PR298 package/probe copies:** retain canonical `hu100-v053-release-package/results/v053-release-package/models/v053/` and `models/tables/`. At pinned source `ffa9ff7d9b68cd96c9cec218088f5a62fc6dcca2`, use `python3.11 -m scripts.build_hu100_bundle --release v0.5.3 --source MODEL --card-tables TABLES --out FRESH_DIRECTORY --source-sha ffa9ff7d9b68cd96c9cec218088f5a62fc6dcca2`. The retained `real-refusals.py` defines the wrong-table byte flip and hardlinked dependencies. The v0.5.2 assembled wrong-model fixture restores from its published model, SHA256 `0ed7f9512996bb3e9535a7258a5d0e4bc8c00f22734d511c642c0e5b95cbd3c0`. Small original manifests/checksums/probe outcomes remain; they are not regenerated or removed.
- **R — publication copies:** `gh release download v0.5.3 --repo dberweger2017/CFR-NLHE-6max --dir FRESH_DIRECTORY`; verify the retained release checksums/verifier and [published asset hashes](../releases/v0.5.3/PUBLICATION.md). The same model/tables also remain in the package’s canonical directories.
- **A — PR250 duplicate ZIP:** [Drive archive](https://drive.google.com/file/d/1DMz8kmY4DIQfO-NSIdEiOu9k-tl1hzrB/view), 2,221,371,477 bytes, SHA256 `4e6fc9f34b93f8a4e08628e31ba72ebe0b13f08605c02adc62d6354127c33165`; manifest SHA256 `78b893820f2fcfb18af69e3da99ee68798d373ac90a543dc8c9205edb27e10db`. `rclone copy gdrive:PR-250-hu200-k50-first/hu200-k50-first-averages-M4-20261013.zip FRESH_DIRECTORY` on bb128, then verify the indexed hashes. The extracted model remains on M1.
- **T — PR283 duplicate river table:** copy the retained canonical package or Slumbot `river-k50.bin`; require SHA256 `70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629`. Drive fallback: [PR163 archive](https://drive.google.com/file/d/1Gl1JYWN0F7KFJbwoKtcA0zB_rUsWWZ6T/view), `rclone copy gdrive:PR-163-equity-buckets/ FRESH_DIRECTORY`; require archive SHA256 `03d4828f030082f213c8ce82e29fa0fa8fd97288bd8035ad0056de85bc0816ca`, then extract `hu20-equity-buckets-20261005/river-k50.bin`.
- **Q — PR211 pilot copies:** copy retained `hu100-seed-qualification-20261009/results/hu100-seed-qualification/inputs/terminal/average.gz` to each recorded model path; require SHA256 `47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9`. Drive fallback: [PR211 evidence](https://drive.google.com/file/d/1MkVUkiRMZE_7UK790J7Pu6ObfPw8gMtr/view), `rclone copy gdrive:PR-211-hu100-seed-qualification/ FRESH_DIRECTORY`; archive SHA256 `2357e9eb3e74df35ee9f4c3421812ef77bd90e0e59e7de0a2e6c087928835e5e`, manifest SHA256 `707e95387d56c1134b9f243ae1bbc10c2820965f0e15ffd746258537f2ba9783`, member `research/inputs/terminal/average.gz`.

## Receipt and administrative failures

[Full receipt ZIP on Drive](https://drive.google.com/file/d/1vEmYHa3c63-YWoWFCuuE81JlmWEsUJSm/view),
`storage-cleanup-afternoon-20261016-receipt.zip`, **84,338 bytes**.
SHA256 `c369b620fafaf3c7f9ef6d1f7fa8c19d83f4cbdd13041c6c3db53e4c6193e0cb`;
Drive/local MD5 `d6cf7038f54b2a640458037c06916171`;
embedded `ARCHIVE-MANIFEST.json` SHA256
`acd9f514549dac787c84fb9dcb2e1ce710222454062c8c957f8c13cb1397ed07`.
All 18 ZIP members passed local readback. The uncompressed evidence and ZIP
remain cached at `/research/PR-315-storage-cleanup-afternoon-20261016/` on bb128.
The live tool's immutable upload compared Drive MD5/size and wrote
`.drive-receipt.json`, covering **all 19 local files** (18 members plus ZIP),
with no uncovered files. Its returned Drive ID is the link above.

Always-available restoration on bb128:

```sh
rclone copy gdrive:PR-315-storage-cleanup-afternoon-20261016/ FRESH_DIRECTORY
```

Verify the ZIP/manifest hashes, then extract `evidence/complete-receipt.json`
for all metadata pins, cloud confirmations and protected snapshots, and the
per-host `evidence/*-removal.jsonl` journals. The original local receipt also
remains in this cleanup worktree's ignored `planning/cleanup-afternoon/`.

The M1 helper’s final receipt event used a duplicate `at` keyword after all 38 removals and retained-file checks completed. Its fsynced journal proved every removal; absence and retained metadata were checked again and the receipt recovered, without repeating deletion. The helper was corrected before M4 removal. A later extra-pass invocation ran before its asynchronous plan existed and failed in preflight before mutation; it was rerun only after the plan completed. Both administrative failures are retained in the full receipt.

Validation: exact-path metadata/provenance and complete hardlink groups, fresh merged-owner checks, current cloud/release metadata, no-handle/dependency review, per-path journals, retained-file snapshots, post-removal absence/free-space checks and receipt upload acceptance. Staged artifact and whitespace checks passed. This documentation-only PR changes no game or storage-tool code.
