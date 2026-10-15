# Completed-work storage cleanup — October 15, 2026

Owner-requested cleanup reclaims **19.342 GB** across both Macs.
Current owning-PR status, upload acceptance, dependencies, open handles and exact
file identity/modification history were checked before removal. No research jobs
were signaled. PR278 was still open for upload/administrative closeout; its
work roots, scientific originals, models, inputs and separate server caches remain.
The owner's October14 standing permission applies to inactive rebuildable caches.

| Machine | Native uploaded cache released | Unsynced duplicate/build/policy cache removed | Total reclaimed | Verified free |
|---|---:|---:|---:|---:|
| M1 | 6.745 GB | 3.988 GB | 10.733 GB | 37.846 GB at 2026-10-15T06:07:23.285104+00:00 |
| M4 | 3.162 GB | 5.447 GB | 8.609 GB | 47.149 GB at 2026-10-15T06:07:53.037572+00:00 |

Allocation counts use actual filesystem blocks and count internal hardlinks once.
Native dataless placeholders can retain bookkeeping blocks, which are excluded
from reclaimed bytes. Free space is a snapshot and can change with other writes.

## Native cache paths and restoration

Only normal per-file `FileManager.evictUbiquitousItem` was used. Every same-size
placeholder and independently confirmed Drive file survived. There was no raw
synced deletion, force eviction, unpinning or folder-wide action. Exact native
paths are in the accepted receipt; the corresponding Research-Cloud alias is
`~/Local/Research-Cloud/` plus each relative path below. Download normally from
that alias or its unchanged Drive link when needed.

| Machine | Research-Cloud relative path | Confirmed logical bytes | Drive file |
|---|---|---:|---|
| M1 | `PR-283-cygnus-x1-qualification/cygnus-main-20261015.zip` | 3,553,618,907 | [11bu2M3AY66AscCgF5onO4tHampdxErMf](https://drive.google.com/file/d/11bu2M3AY66AscCgF5onO4tHampdxErMf/view?usp=drivesdk) |
| M1 | `PR-276-277-slumbot-frozen-20b-comparison/slumbot-frozen-20b-comparison-20261014.zip` | 26,922,026 | [1fBk0ozoB99NpQVNDOCtqOyOWmBPlk6wq](https://drive.google.com/file/d/1fBk0ozoB99NpQVNDOCtqOyOWmBPlk6wq/view?usp=drivesdk) |
| M1 | `PR-283-cygnus-x1-qualification/cygnus-closeout-20261015.zip` | 1,973,287 | [1e7WOsx1JNQfSway6AuAcB9BUpTay3bK4](https://drive.google.com/file/d/1e7WOsx1JNQfSway6AuAcB9BUpTay3bK4/view?usp=drivesdk) |
| M1 | `PR-278-hu200-v055-confirmation/hu200-v055-seed-2026101421-final-average-20261014.zip` | 3,162,672,661 | [1OAKKAxH91_flAKK9eWa4peCy_jUImhDl](https://drive.google.com/file/d/1OAKKAxH91_flAKK9eWa4peCy_jUImhDl/view?usp=drivesdk) |
| M4 | `PR-278-hu200-v055-confirmation/hu200-v055-seed-2026101422-final-average-20261014.zip` | 3,162,066,289 | [1nOhwFtDzJakpkB7MnHdk5JgY75CRnQjT](https://drive.google.com/file/d/1nOhwFtDzJakpkB7MnHdk5JgY75CRnQjT/view?usp=drivesdk) |

Current native uploaded/no-pending/no-conflicts/unpinned/no-download/no-pause/no-
trash/no-exclusion and exact size were required. Cloud ID/name/size/direct parent
matched before and after. These uploaded ZIP caches are unused output copies;
no open PR input or research root was removed. Cloud archive payloads were not
downloaded or rehashed for cleanup.

## Unsynced removals and rebuilds

All paths below begin with `~/Local/`. All owning PRs for these removals are
currently merged. Exact recursive path/size/time/inode/device inventories,
internal hardlink accounting and open-handle checks are in the private receipt.

| Machine | Removed path under Local | Owner | Regeneration or restoration |
|---|---|---|---|
| M1 | `deepcfr-texas-no-limit-holdem-6-players/native/hu20-trainer/target/debug` | #204 | Source `7e51349ecb886a69b2c4efafdafad0998884a6d1`; `cargo build --locked --manifest-path /Users/dberweger/Local/deepcfr-texas-no-limit-holdem-6-players/native/hu20-trainer/Cargo.toml` (then `cargo test --locked` for test output). Tracked Cargo.toml/Cargo.lock unchanged and retained. |
| M1 | `deepcfr-texas-no-limit-holdem-6-players/.claude/worktrees/deepcfr-repo-access-830ea6/native/hu20-trainer/target/debug` | #281 | Source `ca1b8b66814d5e08fc9b087bb303bbf5d1df9ee8`; `cargo build --locked --manifest-path /Users/dberweger/Local/deepcfr-texas-no-limit-holdem-6-players/.claude/worktrees/deepcfr-repo-access-830ea6/native/hu20-trainer/Cargo.toml` (then `cargo test --locked` for test output). Tracked Cargo.toml/Cargo.lock unchanged and retained. |
| M1 | `cygnus-x1-qualification/results/cygnus-qualification/main-archive/cygnus-main-20261015.zip` | #283 | Download [same accepted ZIP](https://drive.google.com/file/d/11bu2M3AY66AscCgF5onO4tHampdxErMf/view?usp=drivesdk) into a fresh unsynced path; 3,553,618,907 bytes, SHA256 `35c53b9460d0ec1c5492b8e824c53e51c1bcd98757fd4b61ea29e78087f3925c`. Coordinator/cloud-staging verification and unchanged pre-verification size/mtime/ctime establish this duplicate; no repeat payload hash. |
| M4 | `hu100-growth-50m-20261008/native/hu20-trainer/target/debug` | #204 | Source `d3033822d865e077093809e82633a1ad7fad19ba`; `cargo build --locked --manifest-path /Users/dberweger/Local/hu100-growth-50m-20261008/native/hu20-trainer/Cargo.toml` (then `cargo test --locked` for test output). Tracked Cargo.toml/Cargo.lock unchanged and retained. |
| M4 | `hu100-growth-20261008/native/hu20-trainer/target/debug` | #203 | Source `ae8d9b83638cdfe96df0849f5ed6bfe6874d53eb`; `cargo build --locked --manifest-path /Users/dberweger/Local/hu100-growth-20261008/native/hu20-trainer/Cargo.toml` (then `cargo test --locked` for test output). Tracked Cargo.toml/Cargo.lock unchanged and retained. |
| M4 | `hu200-frozen-evaluation-20261014/results/hu200-frozen-evaluation/policy-cache/9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6-hu200-native-reopening-ordered-history-equity-k50-v1-compact-average-cache-v1` | #268 | Rebuild from retained PR265 final average and retained K50 tables as below. This local cache is separate from the protected server cache. |

The removed PR268 cache's source is `cc98b63858ed508ccff3491fc86c9e5f8bca87f6`.
Its manifest pins model SHA256 `9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6`,
3,164,563,975 bytes, schema `hu200-native-reopening-ordered-history-equity-k50-v1`.
The source average remains at `/Users/dberweger/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/frozen/20000000000.average.gz` with matching size/recorded mtime;
all tables in `/Users/dberweger/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/tables` remain unchanged.
Model restoration is Drive archive [`1iVND4K_trvJ8S6gnDnyDc4tj5fNR2mDh`](https://drive.google.com/file/d/1iVND4K_trvJ8S6gnDnyDc4tj5fNR2mDh/view),
whole SHA256 `fe3465bea7591e82a762943e197d88d89fa36c8002b4143e6ebf70a887b2d943`,
manifest SHA256 `d349217a651c7d28ac536124be40609778c8b55827d4b000e6840e49544eddc5`,
member `averages/20000000000.average.gz` /the model size/hash above.
Table archive/member hashes and retrieval commands remain in the
[PR163/PR265 index](../../RESULTS_INDEX.md#pr265--exact-hu200-k50-frozen-continuation-october-1314).
After restoring and verifying inputs if necessary, rebuild in a fresh unsynced
cache directory using the pinned source's existing API (`FRESH_NONSYNCED_CACHE` is a new empty
path chosen for the rebuild):

```python
from pathlib import Path; from src.blueprint.average import AveragePolicy; AveragePolicy('/Users/dberweger/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/frozen/20000000000.average.gz', '9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6', expected_schema='hu200-native-reopening-ordered-history-equity-k50-v1', card_buckets=Path('/Users/dberweger/Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/tables'), sorted_rows=True, cache=Path(FRESH_NONSYNCED_CACHE))
```

The build-cache source commits remain in Git. Restore a source into a fresh
checkout before rebuilding; never overwrite another working checkout.

## Preservation and accepted receipt

The failed seed1 canonical full ZIP, its still-uploading
`hu200-v055-seed-2026101421-full-retry-20261015.zip` and Lost and Found recovery
copy are retained. At the final check the retry is uploaded0/uploading1, so it
is not accepted or offloaded. The original canonical ZIP retains its initial
size/mtime/inode and resident allocation. Unknown historical caches, unique
scientific evidence, all worktrees/shared Git, models/tables and personal files
remain. No upload retry/repair or monitoring schedule was created by this pass.

[Accepted private per-path receipt](https://drive.google.com/file/d/14AXtp0Bl3JbPNCHuDvdU-7SwT1ch5nP9/view?usp=drivesdk), ID `14AXtp0Bl3JbPNCHuDvdU-7SwT1ch5nP9`,
**3,182,322 bytes**, SHA256 `cf174b03d1780d48c546042dbbc958f7707dfe7f35ff6a305cfae4142bf2fec8`,
root folder `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`.
Independent Drive metadata readback confirmed its exact name/size/direct parent.
It records each deletion/release, source/input pins, native/cloud acceptance,
modification-history checks, retained dependencies and full rebuild inventories.
Local receipts are under `~/Local/storage-vacuum-20261015-completed/` on both Macs.
