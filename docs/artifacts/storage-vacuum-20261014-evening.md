# M4 storage for v0.5.5 admission — October 14 evening

The owner requested another storage pass and a comment on PR278 once enough
M4 disk was available. PR278's 45 GiB admission was short by about 9.5 GiB.
This pass reclaimed **15.132 GB**: 14.737 GB
of uploaded ZIP cache and 0.395 GB of inactive debug build
output. At **2026-10-14T15:44:29Z**, M4 reported **53,216,952,320 free bytes
(49.562 GiB /53.217 GB)**, above the 45 GiB
threshold by **4.562 GiB**. This is a dated filesystem
reading; per-path accounting excludes other work and residual provider blocks.

## Explicit scope and checks

The full and average PR272 ZIPs were already uploaded and unused, but PR272
remained open. The owner explicitly approved: **“Offload those two uploaded ZIPs
(Recommended)”**, in response to the exact two-ZIP question promising to preserve
its worktree/models/tables/Drive files. That authorization applies only to these
two cache paths. The cleaner retained all other open-PR roots/dependencies and
did not remove any research original.

Fresh native acceptance requires uploaded1/uploading0/conflicts0, unpinned,
not downloading/paused/excluded/trashed and exact document size. Independent
Drive metadata confirms both IDs/names/sizes/direct parents. Exact local
size/inode/device/times and fresh lsof checks precede ordinary per-file
`FileManager.evictUbiquitousItem`. The helper permits the two specifically
authorized open-PR paths while retaining all other original protections.
Fresh cloud readback and same-ID/same-size dataless local placeholders prove
both files remain. Counted cache deltas exclude 30,236,672 residual provider
bytes; these were not forced away.

All **69 checked research/model/table gzip/bin files** under the PR272 run and
PR265 frozen model root match before/after path, byte count, nanosecond mtime
and inode. PR272's roots, originals, models/tables, Slumbot dependencies and
shared Git are untouched. PR277/278 roots and other inputs were protected.
No research jobs were signaled and no training/server preparation was launched
by the cleaner. No synced deletion, forced offload, unpinning, payload download,
repeated whole/member hash audit, personal cleanup or scheduling.

## Same-file restoration

Open/download each linked unchanged Drive archive to restore its native cache.
For research use, follow the owning index's existing whole/member hash checks
into fresh ignored nonsynced storage before extracting or computing.

| Native cache path | Unchanged Drive file | Logical bytes | Allocated bytes released |
| --- | --- | ---: | ---: |
| `~/Local/Research-Cloud/PR-272-hu100-v2-frozen-5b/hu100-v2-frozen-5b-average-M4-20261014.zip` | [hu100-v2-frozen-5b-average-M4-20261014.zip](https://drive.google.com/file/d/1p7LmF9xRGXroTLDqNiY4eRwPuMC2IRiX/view?usp=drivesdk) | 3,196,298,276 | 3,196,301,312 |
| `~/Local/Research-Cloud/PR-272-hu100-v2-frozen-5b/hu100-v2-frozen-5b-M4-20261014.zip` | [hu100-v2-frozen-5b-M4-20261014.zip](https://drive.google.com/file/d/1SKEvKCzuoWeMvBSYY8_iafuKQcb6pmT7/view?usp=drivesdk) | 11,540,258,992 | 11,540,262,912 |

## Build-cache restoration

- `~/Local/hu100-1b-growth-20261008/native/hu20-trainer/target/debug`: 231,215,104 allocated file bytes, merged PR207. Retained source `~/Local/hu100-1b-growth-20261008`, HEAD `fbe19c17f0fbd474e5ea26393cc16c9ddeafaf27`. Run `cargo build --manifest-path native/hu20-trainer/Cargo.toml` there with the retained Rust toolchain to regenerate debug output.
- `~/Local/native-recovery-hu100-20261008/native/hu20-trainer/target/debug`: 163,876,864 allocated file bytes, merged PR197. Retained source `~/Local/native-recovery-hu100-20261008`, HEAD `fa84b84c6a32caaf557cd3823b885f015bb7d524`. Run `cargo build --manifest-path native/hu20-trainer/Cargo.toml` there with the retained Rust toolchain to regenerate debug output.

Fresh cargo/rustc/open-file admission and an unchanged full file/inode/time
inventory preceded build removal. No symlink or external hardlink was selected;
internal hardlinks are counted once per unique file inode. Source/checkouts and
release executables remain; only the two exact `target/debug` paths are removed.

## Delivery and receipt

The owner authorized a comment on PR278 once its disk requirement was met.
[The posted, independently read-back Vacuum Cleaner comment](https://github.com/dberweger2017/CFR-NLHE-6max/pull/278#issuecomment-6172261185)
records the exact timestamp/free-space reading, admission margin and cleanup
scope. It points to the on-M4 per-file receipts under
`~/Local/storage-vacuum-20261014-evening/`.

[Full private receipt](https://drive.google.com/file/d/16ZKIrqlQIwtp9vOjcb5QlJKYFJsJ0enx/view?usp=drivesdk), ID `16ZKIrqlQIwtp9vOjcb5QlJKYFJsJ0enx`, **79,609 bytes**,
SHA256 `98d0c8010a3539b78eb07672e453c1012adae52b7ff35f21cff59e66e9d7ee1d`. Independent metadata readback confirms the exact
receipt name/size and Research-Cloud parent `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`.
It preserves owner authorization, before/after acceptance and file stats,
69 protected-file pins, rebuild/source provenance and comment delivery.
M1 was inspected read-only; no M1 cache or research path was removed in this pass.
