# Merged PR286 cleanup on M4 — October 15, 2026

Owner-requested cleanup after PR286 merged reclaims **50.124 GB**:
eight archived model originals **28.261 GB** and two uploaded
model ZIP caches **21.863 GB**. Verified M4 free space is
**78,787,035,136 bytes (78.787 GB)** at 2026-10-15T13:45:10.055519+00:00.
PR286 current GitHub state is merged at 2026-10-15T13:38:21Z, head
`d328f75ab38192e3288f4d5780b43ed1338670ff`. No process is signaled or restarted.

## Retired originals and exact restoration

All eight paths below are relative to
`~/Local/hu100-v053-three-seeds-20261015/results/hu100-v053-seeds/`.
These exact paths supersede earlier originals-retained snapshots. Both final
inference averages and all stage1/final checkpoints/current exports remain
recoverable from the pinned model/average archives. Source and audit lineage
are retained in the [published model index](../reports/hu100-v053-three-seeds-artifacts/model-index.json)
and [archive/member catalog](../reports/hu100-v053-three-seeds-artifacts/archive-catalog.json).

| Removed relative path | Archive ID | Exact member | Logical bytes /SHA256 |
|---|---|---|---|
| `2026101501/5000000000.average.gz` | [`1pSFspz05rmjRU57FUEfOxR6-ov8pZxtT`](https://drive.google.com/file/d/1pSFspz05rmjRU57FUEfOxR6-ov8pZxtT/view?usp=drivesdk) | `averages/2026101501/5000000000.average.gz` | 3,190,242,513 /`83771d644b423dc3e32fd313b0bb127b18920b6eb58d7ec6b6f90a7036087000` |
| `2026101501/5000000000.current.gz` | [`1DyUOyR9PGV7z5GjUO1H9qeUUsjACog4C`](https://drive.google.com/file/d/1DyUOyR9PGV7z5GjUO1H9qeUUsjACog4C/view?usp=drivesdk) | `research/2026101501/5000000000.current.gz` | 2,587,233,270 /`d547a13568a887f0bba235e66935c1846f6f2a725ddd79c69cb112c58463fbea` |
| `2026101501/stage1-requested-500000000.checkpoint.gz` | [`1DyUOyR9PGV7z5GjUO1H9qeUUsjACog4C`](https://drive.google.com/file/d/1DyUOyR9PGV7z5GjUO1H9qeUUsjACog4C/view?usp=drivesdk) | `research/2026101501/stage1-requested-500000000.checkpoint.gz` | 3,484,422,859 /`14cc0c322ed7a5d4b22000d3a3fecafb2f7d1fac8872d3c2b3c5714b2e9a88fe` |
| `2026101501/stage2-requested-5000000000.checkpoint.gz` | [`1DyUOyR9PGV7z5GjUO1H9qeUUsjACog4C`](https://drive.google.com/file/d/1DyUOyR9PGV7z5GjUO1H9qeUUsjACog4C/view?usp=drivesdk) | `research/2026101501/stage2-requested-5000000000.checkpoint.gz` | 4,859,625,626 /`ea7107e6c1722e1cb85192e70fc8b10797021dc7838d58ffdeefe886a660086b` |
| `2026101502/5000000000.average.gz` | [`1thfF0zPFYQT5QxAa0pav_RWZpYE66p0u`](https://drive.google.com/file/d/1thfF0zPFYQT5QxAa0pav_RWZpYE66p0u/view?usp=drivesdk) | `averages/2026101502/5000000000.average.gz` | 3,187,447,872 /`a476afa7334968087e5dfdb268d3897f9c2afba13905694c054bca2db4a59c19` |
| `2026101502/5000000000.current.gz` | [`1p5qbkxd2dbh_YyArMhM_JfYkMM92uJi2`](https://drive.google.com/file/d/1p5qbkxd2dbh_YyArMhM_JfYkMM92uJi2/view?usp=drivesdk) | `research/2026101502/5000000000.current.gz` | 2,587,283,689 /`00b4bbf8c364cfb8ed39787ca55e668f07ee0cbe7c61f453e06eeb9b0fa9baac` |
| `2026101502/stage1-requested-500000000.checkpoint.gz` | [`1p5qbkxd2dbh_YyArMhM_JfYkMM92uJi2`](https://drive.google.com/file/d/1p5qbkxd2dbh_YyArMhM_JfYkMM92uJi2/view?usp=drivesdk) | `research/2026101502/stage1-requested-500000000.checkpoint.gz` | 3,485,365,726 /`86407477b095d5098355555be7504deaffcce621343ae04b746b2c2bebacb6ef` |
| `2026101502/stage2-requested-5000000000.checkpoint.gz` | [`1p5qbkxd2dbh_YyArMhM_JfYkMM92uJi2`](https://drive.google.com/file/d/1p5qbkxd2dbh_YyArMhM_JfYkMM92uJi2/view?usp=drivesdk) | `research/2026101502/stage2-requested-5000000000.checkpoint.gz` | 4,858,779,254 /`a2f403d966ef4f2bfc394fbf1d13bf2c7dafd886483acc858aeaeb4b0cd9fd96` |

Current native uploaded/no-pending/no-conflicts/no-download/no-pause/no-trash/
no-exclusion/exact-size status and independent cloud ID/name/size/direct-parent
checks pass for all four restoration archives. Independent cloud metadata matches
after retirement too. Existing complete local member readback explicitly matches
audit pins; no remote archive bytes or model payloads are downloaded/rehashed.

Each original's exact published path/size/hash agrees across the model index and
archive catalog. Its current mtime/ctime precede a conservative sealing-start
bound: ZIP content mtime minus the full recorded source-hash/pack/readback/whole-
hash duration. Fresh no-open-handle, single-link and exact path/size/mtime/ctime/
inode/device checks pass before journaling and unlinking. Uncertain files stay.

| Archive | Bytes | Whole SHA256 | ARCHIVE-MANIFEST.json SHA256 |
|---|---:|---|---|
| [hu100-v053-seed-2026101501-average.zip](https://drive.google.com/file/d/1pSFspz05rmjRU57FUEfOxR6-ov8pZxtT/view?usp=drivesdk) | 3,190,246,314 | `19b1510446d5f5b76603e91d01dab347aa92d2e21115ebe339ab2eb827ff31ed` | `e0e79eeda29f02f8e2bb3fa858e0edc7c2c36f680b0088b370d2910b64009b0e` |
| [hu100-v053-seed-2026101501-models.zip](https://drive.google.com/file/d/1DyUOyR9PGV7z5GjUO1H9qeUUsjACog4C/view?usp=drivesdk) | 10,931,284,102 | `d992cabf4ab34253a7304f1c3e69abb02006471c9d221f27913b6de15e4093ce` | `82e3d69d536a657e3e602f4379bef157a00abe6996613c22c26d21fd32bca67a` |
| [hu100-v053-seed-2026101502-average.zip](https://drive.google.com/file/d/1thfF0zPFYQT5QxAa0pav_RWZpYE66p0u/view?usp=drivesdk) | 3,187,451,673 | `275f7ed7ca143f05ca64a27d93163a08e829c2e7ca2c9d71b1a2a9791fc6f0df` | `e3f56f9b972009dd96c83e5f4d66842514c2bbd7bd421e5b611c080baffd027b` |
| [hu100-v053-seed-2026101502-models.zip](https://drive.google.com/file/d/1p5qbkxd2dbh_YyArMhM_JfYkMM92uJi2/view?usp=drivesdk) | 10,931,431,016 | `4bf18e7d43fe85767acab9ad12876431a77ea6a47df4cf2c0d06cb0724182f29` | `2601251e90e707c5bf91f1d6f4e546d49779006a286378dd0d38e6697dea3f4c` |

Folder: [`1xZlQ09yZnQ9VekGtRlBE5sd1U1pCqVQW`](https://drive.google.com/drive/folders/1xZlQ09yZnQ9VekGtRlBE5sd1U1pCqVQW).
Download the selected archive to a fresh ignored unsynced directory; require its
whole SHA256 above before extracting. Verify the embedded manifest and each
restored member's size/SHA256 against the pins above:

```sh
shasum -a 256 ARCHIVE.zip
python3 -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extract(sys.argv[2],sys.argv[3])' ARCHIVE.zip EXACT_MEMBER FRESH_RESTORE_ROOT
shasum -a 256 FRESH_RESTORE_ROOT/EXACT_MEMBER
```

Restoration does not authorize resuming the terminal campaign, rerunning failed
hands or overwriting active inputs. Its incomplete global readout barrier,
partial status and owner merge/release decisions remain unchanged.

## Uploaded native cache release

Only the two `hu100-v053-seed-202610150{1,2}-models.zip` cache files in the PR286
Research-Cloud folder are normally evicted with `FileManager.evictUbiquitousItem`.
Both had current uploaded/no-pending/no-conflicts/unpinned/inactive status and
independent exact cloud acceptance. Same-size placeholders and Drive IDs remain;
post-checks confirm dataless flags. Remaining bookkeeping blocks are excluded
from reclaimed bytes. Download normally by the unchanged Drive links above to
restore residency. No raw synced deletion, force eviction, unpinning or folder
operation is used. No new archive is created and no payload is rehashed.

## Preserved work and accepted receipt

Open PR292's bb128/M1 dependency chain is untouched. Source/worktrees/shared Git,
native binaries, original-seed `inputs/2026100601.average.gz`, the shared v0.5.2
release model and canonical K50 tables stay. All pilot/final raw gzip traces,
partials, failures, resource logs and metadata remain. All **109** selected
protected files match size/mtime/inode after cleanup. PR286's owning agent had
already removed its policy caches; this pass deletes none. All M1/bb128 working
data remains. No cleanup or monitoring schedule is created.

[Accepted private exact-path receipt](https://drive.google.com/file/d/1HKyQUkGffO1_UH-PzllfayS0eM7_N7AL/view?usp=drivesdk),
ID `1HKyQUkGffO1_UH-PzllfayS0eM7_N7AL`, **84,810 bytes**, SHA256
`ea0a16e468b58b0a907e2eb7385eb80ac5ccf02d83ad6cb82dd652c3c779a06f`, parent `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`.
Independent Drive metadata confirms its exact name/size/direct parent. It contains
current merged state, original/source-history pins, native/cloud acceptance,
per-path durable journals, actual block accounting and protected-file checks.
Local records: `~/Local/storage-vacuum-20261015-pr286/` on both Macs.
