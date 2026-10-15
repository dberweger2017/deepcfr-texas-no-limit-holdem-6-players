# M4 storage for HU100 seed qualification — October 15, 2026

Owner-requested cleanup reclaims **17.826 GB** from five
unchanged, archived PR278 seed-2 HU200 files outside sync. M4 reports
**64,976,728,064 bytes free (64.977 GB /60.514 GiB)**
at 2026-10-15T06:33:23.596817+00:00. Immediate observed free-space increase is
17,826,959,360 bytes; later research writes can change the snapshot.

PR278 is currently **merged** at 2026-10-15T06:27:46Z,
head `98cf3ace6faca9eeb6368f23986ce914409a4395`. Its existing full seed-2 archive has
complete local member readback and current upload acceptance. Exact source
paths/sizes/mtime match both the published model index and pre-archive source
selection; ctime/mtime precede that selection snapshot. Fresh open-handle and
size/time/inode/device checks pass before each exact-path unlink.

## Removed originals and restoration

Local prefix: `~/Local/hu200-v055-confirmation-20261014/results/hu200-v055-confirmation/`.
Each row names the original path relative to that prefix and its archived member.
This receipt supersedes earlier local-retention statements only for these paths.

| Removed relative path | Archive member | Logical bytes | Member SHA256 |
|---|---|---:|---|
| `seed-2026101422/frozen/20000000000.current.gz` | `research/seed-2026101422/frozen/20000000000.current.gz` | 2,301,944,453 | `1bfd938c992591a79cec66ad93065996856fbc456096127fe893b235131d91d5` |
| `seed-2026101422/frozen/20000000000.gz` | `research/seed-2026101422/frozen/20000000000.gz` | 4,957,316,628 | `e265c384686f003521e49158c5162e57175d3b53e405140cb494d30a8ec9660c` |
| `seed-2026101422/frozen/5000000000.gz` | `research/seed-2026101422/frozen/5000000000.gz` | 4,354,826,440 | `67a819934d85bf45fdcc308cb2bd6f4dcc6ce9cf4d571857125714262094dc3a` |
| `seed-2026101422/stage1/1000000000.gz` | `research/seed-2026101422/stage1/1000000000.gz` | 3,445,287,989 | `94422a4a48305468dabfebeb36440282a5dd954376ff5cdb627b1773ff706317` |
| `seed-2026101422/stage1/500000000.gz` | `research/seed-2026101422/stage1/500000000.gz` | 2,726,310,329 | `45144a01a799b81d185c20e15772fe70793518a3d35b03fa84eb6db0bfa46966` |

All five restore from [the accepted seed-2 full archive](https://drive.google.com/file/d/1xAg4LpkRM_XLre2TsQcDkn0voN48dAwO/view?usp=drivesdk),
Drive ID `1xAg4LpkRM_XLre2TsQcDkn0voN48dAwO`, name `hu200-v055-seed-2026101422-full-20261014.zip`,
**20,953,054,295 bytes**, whole SHA256 `73e76b94333ae9236ab4b280a04b33d5c619abfa333cda1b7486160949347fbb`;
embedded `ARCHIVE-MANIFEST.json` SHA256 `6c8cafeca8e855e7d8251bf0110f09ff1b3cc3e0e722254479c06a9a3ac553b1`.
Folder [`1-0y8ETC4xqzfGzSMOnt9BWMJAUkqpNiP`](https://drive.google.com/drive/folders/1-0y8ETC4xqzfGzSMOnt9BWMJAUkqpNiP).
Current native uploaded1/uploading0/no-conflicts/no-download/no-pause/no-trash/
no-exclusion and exact size were checked on M1, which sealed/uploaded the archive.
Independent cloud ID/name/exact-size/direct-parent metadata match before and after
retirement. The cloud file is unchanged; no archive payload was downloaded or
rehashed for cleanup. Existing complete local readback/manifests are trusted.

Download the pinned Drive archive into a fresh ignored nonsynced directory and
verify its whole SHA256 before extracting. For any exact member above:

```sh
shasum -a 256 FULL_ARCHIVE.zip
python3 -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extract(sys.argv[2],sys.argv[3])' FULL_ARCHIVE.zip EXACT_MEMBER FRESH_RESTORE_ROOT
shasum -a 256 FRESH_RESTORE_ROOT/EXACT_MEMBER
```

Compare the restored member's size/SHA256 with its table pin and verify the
embedded manifest against its listed SHA256. The
[full PR278 model/retrieval index](../reports/hu200-v055-confirmation-artifacts/seed-2026101422-model-index.json)
contains all original lineage/archive pins and a complete checked retrieval command.
Never overwrite active inputs or another working tree.

## Preserved dependencies

The new M4 task uses HU100 PR272/PR260, v0.5.2 and shared PR163 K50 tables.
It does not consume these HU200 PR278 seed-2 checkpoints. Its new root
`~/Local/hu100-v053-three-seeds-20261015/`, all source/worktrees/shared Git,
released inputs and shared tables are untouched. No processes were signaled.

Both PR278 final averages remain, including seed2
`seed-2026101422/frozen/20000000000.average.gz`, 3,162,064,602 bytes,
SHA256 `b191c5ed8b943d138e6076fb4a1b0117c98964d92dd2d6253e157f696f691f6f`.
All seed1 originals, the failed canonical full ZIP, retry ZIP and Lost and Found
copy remain. Seed1 full retry still lacks native upload acceptance; its full
checkpoints/current export cannot be retired on that pending cloud copy.
All seven retained PR278 model files match prior size/mtime/inode after cleanup.
Separate prepared server models/caches/launchers are untouched. No raw synced
deletion, force offload, upload repair/retry or unattended monitor was performed.
Optional extra-cache discovery found no further eligible compact policy cache.

## Accepted receipt

[Private per-path receipt](https://drive.google.com/file/d/1OSxDRcAd6hfmwf5cX9lJzOvycoqWZskG/view?usp=drivesdk), ID `1OSxDRcAd6hfmwf5cX9lJzOvycoqWZskG`,
**39,825 bytes**, SHA256 `847837f8987c157acc4eef7ffb5866e39dfa27fd1a70ddb93da904a95befa11e`,
parent `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. Independent metadata readback confirms
its exact name/size/direct parent. It retains source pins/selection history,
native/cloud acceptance, current merged status, exact removal journal, failed
preflight helper details and protected-model verification. Local records are at
`~/Local/storage-vacuum-20261015-m4-admission/` on both Macs.
