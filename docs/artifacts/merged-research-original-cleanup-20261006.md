# Verified merged research copy cleanup — October 6, 2026

The owner authorized removing local copies for completed PRs. Fresh GitHub checks found #165, #171/#173, #175, #176 and #179 merged; #166 and #182 remain open. Only regular files whose stable identity, size and SHA256 matched accepted Drive archives were removed.

| Mac | Files removed | Removed logical bytes | Measured free-space increase during removal | Free at final check |
|---|---:|---:|---:|---:|
| M1 | 2,134 | 28.84 GB | 27.88 GB | 57.23 GB |
| M4 | 1,572 | 19.04 GB | 19.04 GB | 38.64 GB |

Free-space deltas were measured around each removal batch. Archive verification fetched streamed Drive files into the native cache, and #182 training continued during the audit; these figures do not imply that final free space increased by the same amount since the start of the session. APFS sharing and concurrent writes can also affect allocation. GB uses decimal bytes.

The [full cleanup receipt and audit bundle](https://drive.google.com/file/d/1P_KUWSE-gM2SwMCCvkn6TFONBslc6qz7/view?usp=drivesdk) records every removed path, original member SHA256, archive ID/URL/member, file identity, removal time, executable audit scripts, per-batch plans, PR status, current Drive metadata, native upload acceptance and dependency review. Receipt bundle SHA256: `f85893c228207f431733f05165897c05476f3909ea1d7cd833e4d3375d7b68d2`. Durable nonsynced copies remain on each host under `~/Local/storage-cleanup-receipts/merged-pr-copies-20261006/`. No unattended cleanup is configured.

Both active work roots remain intact: `~/Local/hu20-turn-search-arena-20261005/` (#166) and `~/Local/hu20-learning-curve-20261006/` (#182). The complete #165 original root remains for current #182 checkpoint provenance and #171 review symlink targets. #175 historical policies remain for #182 preparation. Its three O checkpoints and historical R1 were rehashed, and the M4 trainer remained running. Shared Git, source checkouts, executable orchestration, retained manifests/receipts, changed or unarchived files, and all synced files remain. No cache eviction was requested.

| PR | Verified canonical archive | Archive SHA256 |
|---|---|---|
| #165 | [hu20-native-average-complete-20261005.tar.gz](https://drive.google.com/file/d/1xenvcfa5QfhGutNe5xNtDhFLYRqKcyxI/view) | `b5b43d83cdfb9e9b25679eef94b9350c3ef4a76f8c4dad4d076090bb2c6aa898` |
| #165 | [hu20-v041-audit-closeout-20261005.tar.gz](https://drive.google.com/file/d/1olH4RanIwqGhmPae7Pw8GtvIwRr0e14n/view) | `dcfd4fbc1fa360dd765f6db0481353a64646d1893497d4d71b2060b4f89083c5` |
| #171 | [hu20-cfr-plus-complete-20261006.zip](https://drive.google.com/file/d/1kn-i1ZhOA1sftQPC0nYek1MI1XuJBOGS/view) | `8ec9dbee5fd392b058d8a5c33e00a81a414f95189562d8dd0f9e9c66348b982c` |
| #175 | [hu20-floor-control-complete-20261006.zip](https://drive.google.com/file/d/1Dd-jOoJ9ZcZvYCB0yQ_qbkH98wtqImZW/view) | `f350afc81cb8090d24ca125413df8d58259e1d1787912193d13de291904a6962` |
| #176 | [hu20-v041-O-confirmation-complete-20261006.zip](https://drive.google.com/file/d/1VS62w5iLeYlgif_mnNxCgWEUSxm1qa3H/view) | `9f473aa22b16037549716e8fd938b4b47670ca89e29b398bf241b1448fe597a3` |
| #179 | [hu20-zero-mass-fallback-complete-20261006.zip](https://drive.google.com/file/d/1P7oGDwFToL9z2-ctEXKKOdTDm_xoEVc4/view) | `2c18809a712434865c12ae77dd4ca281f10422f77014d3534063b4b5d50ef9b1` |

Every archived research member was freshly read and hashed: #171 1,065, #175 147, #176 792, #179 272, #165 campaign 123 and closeout 35. ZIP/tar archive hashes match their pinned receipts. #165 verification used its separate local duplicate archives; other verification read the native Drive archive. Current Drive metadata matches exact archive name, size and parent; FileProvider reports uploaded=1, uploading=0 and no unresolved conflict. Remote cloud byte re-download is not claimed. For #179, the pinned manifest hash refers to the decompressed `ARCHIVE-MANIFEST.json.gz` content; the source member hashes still refer to exact stored gzip bytes.

To retrieve evidence or a model, look up its original path in the receipt, check the owning PR’s current status, then download/read its canonical Drive archive into a fresh ignored `models/`, `results/` or `planning/` destination. Verify archive SHA256 before extraction and the exact member SHA256 afterwards. ZIP members retain their recorded bytes; do not additionally decompress a checkpoint or policy gzip when checking its stored hash. For a ZIP member, a retrieval command is:

```sh
mkdir -p models/retrieved-pr171
archive_path="$HOME/Local/Research-Cloud/PR-171-HU20-cfr-plus/hu20-cfr-plus-complete-20261006.zip"
gh pr view 171 --json state,mergedAt
shasum -a 256 "$archive_path"
unzip -p "$archive_path" "<member from receipt>" > models/retrieved-pr171/checkpoint.json.gz
shasum -a 256 models/retrieved-pr171/checkpoint.json.gz
```

Compare both printed hashes with the receipt before using the file. For #165 tar members use `tar -xOf "$archive_path" "<member from receipt>" > <ignored destination>` with the same hash checks. Its removed nonsynced archive copies can be restored by copying the matching canonical Drive archive and verifying the whole archive hash. Historical “originals retained” statements in earlier index/report entries describe their original closeout; this receipt is the authoritative later removal record.
