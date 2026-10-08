# v0.4.2 publication verification

**Published October 8, 2026 at 09:19:50 Madrid (07:19:50 UTC), stable Latest.** [Release](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/v0.4.2), [reviewed release PR #198](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/198).

Tag `v0.4.2`, the GitHub release target and publication manifest all identify **`a53f167fa736a481339034d1a7235202d599418c`**, the actual reviewed green-check merge. The merged tree exactly equals approved PR head `b6ae359f216ec7e1bba24718ef708729ca0e82a0`. All five checks passed; independent review has no unresolved findings. Release ID **406532927**; Latest API returns that ID/tag with draft=false and prerelease=false.

The canonical #188 model remains unchanged: fixed seed **2026100601**, **249,237,403 bytes**, SHA256 **`15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae`**. No retraining, re-extraction, arena or model selection. Preparation source `1d80bd02c2e8acf2dc99b75f6e0349bb55a941d8`, its original unpublished manifest/checksums and archive/member provenance remain preserved.

## Download and source verification

A clean M4 detached checkout at the merge built a fresh explicitly approved publication package. Tagging followed exact-source verification. All seven assets were attached to a draft, downloaded into a fresh ignored directory through authenticated GitHub downloads, and independently checked on M4 before publication. After stable Latest publication, all seven assets were downloaded again into a separate fresh ignored M4 directory from public browser URLs, without credentials. Both stages passed complete byte counts/SHA256 against the approved package and GitHub asset digests. Model card, notes, verifier and catalog bytes also equal their exact tagged source files.

The downloaded standalone verifier passes with `--expect-source a53f167fa736a481339034d1a7235202d599418c --require-publication`. The publication manifest has explicit approval true, `package_source_commit` and `approved_release_source_commit` equal the tag, and `release_tag` v0.4.2. The fixed runtime catalog manifest hash matches both the publication manifest and checked-in runtime pin.

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| `catalog-manifest.json` | 1,126 | `b53205ed70fdbf3172f7331c4a64f3122bf0f7e812b58ecc2722a2c82c97cc5c` |
| `MODEL_CARD.md` | 3,790 | `639f78654ab0efa46173ff210589c84cc7c42f36c764a3320ba1eeffcca6105d` |
| `O10B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz` | 249,237,403 | `15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae` |
| `release-manifest.json` | 2,150 | `0257d1c3d724a0b3a81747a4868c1f236858b4fcc2c4d234797988daa618d82c` |
| `RELEASE_NOTES.md` | 2,421 | `3d7e919e6919882dbcaedd648088e06ae30b315e80abb6089ccb14cce92ccda7` |
| `SHA256SUMS` | 553 | `c1731cc43035101b4f90802431237d04b2d4a661b36a2b41dd9830b0e83078ae` |
| `verify_v042_bundle.py` | 8,146 | `adf0e2893acaa5ff4c4f857183be09cf0a1d7f0527ea80f7527b36ebaec1286a` |

## Published-model runtime check

The supported versioned runtime was loaded from the exact tagged source on M4 using the freshly public-downloaded model. Its default is v0.4.2, with the exact intended model SHA256; v0.4.0/v0.4.1 remain in the catalog and human/spectator routes. Four human hands /37 actions /16 policy positions cover both seats, restricted/free sizing and an exact 201-chip raise. Four spectator hands /15 decisions cover v0.4.2 against each older release. Every action, distribution and settlement independently audits and replays. This deterministic smoke is an integration check, not a new arena or strength estimate.

Elapsed **81.47 seconds**, peak RSS **2,977,005,568 bytes /2.77 GiB** below the 3-GiB guard; minimum free disk **98,666,082,304 bytes** above 15 GiB. The detached M4 process exited successfully. No heavy M4 release process remains running.

## Older releases and evidence

v0.4.0 (release ID 399925883, four assets) and v0.4.1 (404820543, five assets) remain public. Their release IDs, publication dates, asset IDs/names/sizes/SHA256 and download URLs match before/after publication. Every old asset URL returns HTTP 200 with its exact declared byte length. Older model bytes were publicly downloaded and hash-verified for the M4 smoke; no older asset or tag was replaced.

The [research decision table](../../reports/hu20-v042-lbr-confirmation.md) is unchanged: direct **+3.50 [+1.63, +5.37] BB/100**, all four declared gates pass, fresh aggregate LBR **−1.092 [−4.965, +2.782]**, passing narrowly. No LBR improvement, each-seed non-regression, HU100 or professional-strength claim follows.

## Retained operational proof

All release retrieval/publication/download/runtime receipts, original preparation and private smoke journals remain in own ignored nonsynced M4 `~/Local/v042-release-20261008/`; the exact tagged checkout is `merged-source/`. Draft download is `merged-source/results/v042-release/draft-download-01/`; fresh public download is `merged-source/results/v042-release/public-download-01/`; runtime journals are under `merged-source/results/v042-release/public-runtime-smoke-01/`. Lightweight M1 copies remain in this release worktree’s ignored `planning/v042-release/` and `results/v042-release/`. Terminal receipts are posted on #198. No model binary is in Git; no research originals, synced archive or other PR work root was cleaned, removed or evicted.

The first attempt to read draft metadata via the tag API returned HTTP 404; authenticated lookup by release ID and fresh authenticated asset download succeeded. This was a metadata-access attempt, not a failed package or scientific check. The separate original #188 failed package attempt remains excluded. No extra research or cleanup follows publication verification.
