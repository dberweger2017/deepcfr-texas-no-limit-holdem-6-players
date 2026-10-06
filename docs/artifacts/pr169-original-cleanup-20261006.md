# PR169 M4 originals removed — October 6, 2026

PR169 was confirmed **merged** at `2026-10-06T06:23:05Z`, merge commit `eb3067a4937dd601569d78cdfd6eb141324e4f99`. Under the owner's existing authorization to remove research originals after confirmed upload, its completed M4 scoring evidence was cleaned. Open #166 and #171 remained protected; M1 originals and shared upstream inputs were unchanged.

## Archive and checks

The canonical [scoring ZIP](https://drive.google.com/file/d/17X9Fpi1gNL0BVU24iN6xGBuoIMG8_Jpa/view) remains in [PR-169 / scoring-6fd63e0](https://drive.google.com/drive/folders/1j9i0_HYXfiDjJYmwqvIxdkicPowWcYhT), available on both Macs through Research-Cloud. It preserves the frozen declaration, all 54 exports, 40 held-out roots, raw solver output, references, copied inputs, failures, runtimes, summaries, plots and exact source `6fd63e0f3576d2139afcace7f5528cd676cba8bb`. Both options help on the declared bench; neither closes the gap. Results and scientific limits remain in [PR169's final scoring comment](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/169#issuecomment-6005846837).

- ZIP: **1,869,797,825 bytes**, SHA256 `198301d2c14a8abf35a1567c241e96254294c10941ebca7efc96ae92c15170c7`.
- Embedded/adjacent manifest SHA256: `b791f688d16fe1761fc55ebfe21e67a8b03cd59bbf82b158053aa4f110cef57e`.
- **676 source members / 6,133,755,821 bytes** independently streamed from the ZIP and checked against their manifest sizes/SHA256s; fresh source hashes and stable identities matched every member.
- Current native Drive `isUploaded=1`, `isUploading=0`, no unresolved conflict, exact size and item ID matched current cloud name/size/parent readback. The locally available synced ZIP and separate original ZIP were freshly rehashed. This uses the owner's requested desktop upload criterion; no full cloud-byte re-download is claimed.

No original had an open file handle. The active M4 #166 river/arena directories contained no references or symlinks to this scoring root, and no M4 #171 run directory was present. #171's declared evidence is a separate full-game implementation/test task. The reviewed deletion set contained only these exact manifested scoring paths and the separate original ZIP. PR169 merge status, native upload acceptance, open handles and identities were rechecked immediately before unlinking. The verifier first expected a flat ZIP layout; it was corrected to use the actual `pr169-scoring-6fd63e0/` prefix before any removal, without changing the archive.

## Removal and retrieval

Removed **677 paths / 8,003,553,646 logical bytes** from M4 `~/Local/pr169-scoring-6fd63e0/` and `~/Local/pr169-scoring-6fd63e0.zip`. All removed paths were confirmed absent, with zero changed/missing/open-file exceptions. Free space rose from **46,176,845,824 to 51,018,608,640 bytes**: **4.84 GB reclaimed / 51.0 GB free**. APFS clones and shared copied inputs make reclaimed bytes smaller than logical removed bytes.

The synced ZIP, its manifest, restore README, presentation files, prior review archives and other Drive records remain intact. The excluded Git/source checkout, build caches, plotting environment, unmanifested final receipts and directory shells remain. No process was stopped and no synced file was deleted or evicted. Historical archive records saying originals were retained are superseded only for the exact paths in this cleanup receipt.

To restore, download the canonical ZIP, verify its SHA256 and unzip into a fresh directory with adequate space. Members are under `pr169-scoring-6fd63e0/`. Verify all payloads against `ARCHIVE-MANIFEST.json`, then follow `RESTORE-README.txt`: the pinned source is in `source-6fd63e0.tar.gz`; original absolute request/run paths must be reconstructed or rebound in a separate restored copy with recorded provenance.

[Cleanup receipt](https://drive.google.com/file/d/1S5wkafKgADZvhsR3cOBDjYGy9njIhfvG/view), SHA256 `70bf5d21e038fc5cd86ada9fe2f7b5de7a41a99bd43c7137951cb24f63f01d32`, records the merged status, archive/native proofs, per-file identities/hashes, deletion set and before/after free bytes. The [receipt folder](https://drive.google.com/drive/folders/1xeOUW53rNxh2_op2-CYjUEwSi6989cJZ) preserves the unlink journal, one-time helper, merge-status readback and prior shared index. Receipt sizes/parents were checked through cloud metadata readback. This does not schedule unattended cleanup or authorize changes to open PR dependencies.
