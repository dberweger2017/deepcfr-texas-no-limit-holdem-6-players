# M4 research artifact organization — October 5, 2026

The large recent M4 folders are traced to PRs #149, #162, #163, #165 and #166. Both Macs already use `~/Local/Research-Cloud`, a symlink to `~/Library/CloudStorage/GoogleDrive-dberweger2017@gmail.com/My Drive/deepcfr-research-results`. The [project Drive folder](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s) and its existing `RESULTS_INDEX.md` are the shared retrieval entry points. Existing folder names, IDs and permissions are preserved.

## Existing archives verified

PR histories and current main's results index identify the raw-run roots, exact frozen source commits and their whole archives. Current native FileProvider acceptance (`isUploaded=1`, `isUploading=0`, no unresolved conflict, exact document size), current cloud ID/name/size/parent and freshly recomputed staged SHA256 all agree for these three archives. This verifies the established desktop upload acceptance criterion; no cloud-byte re-download is claimed.

| PR | Original M4 folder(s), under `~/Local` | Whole archive | Bytes | SHA256 |
| --- | --- | --- | ---: | --- |
| [#149](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/149) | `hu20-board-pooling-20261004`, `hu20-board-pooling-closeout-06-20261005` | [Board-pooling campaign](https://drive.google.com/file/d/1NNSkCO811USN6U9L2p6lBbti1-6Q9r2X/view) | 16,802,192,861 | `20ad0f67df71464e7c06bdb1cf63451c243944c14e732b83650861f8f9bbcaed` |
| [#162](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/162) | `hu20-trainer-bench-20261005`, `hu20-trainer-bench-closeout-20261005` | [Trainer bench](https://drive.google.com/file/d/1_6dRapLReZ9-sAMPaePNX-nehXowGwki/view) | 1,745,796,815 | `67d51196af43202d6e2ebff5222ccfd6a264c1d9d0eb4efdc7ccfa1fa78d3dd3` |
| [#163](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/163) | `hu20-equity-buckets-20261005`, `hu20-equity-buckets-closeout-20261005` | [Equity tables](https://drive.google.com/file/d/1Gl1JYWN0F7KFJbwoKtcA0zB_rUsWWZ6T/view) | 1,991,949,530 | `03d4828f030082f213c8ce82e29fa0fa8fd97288bd8035ad0056de85bc0816ca` |

These archives total 20,539,939,206 compressed bytes. They preserve 5,740 / 2,809 / 29 manifested source members respectively, together with failures, source/input provenance, member hashes and restore instructions. Their manifests were independently verified at the owning-task closeouts; current staging hashes match those immutable receipts. Exact runtime source commits are `e0c91d25` (#149), `74ba3202396128a3823b654d05d0b002b3c4ceed` (#162) and `4ade72cc18e512ed9f11fa6afa83c701a55069c8` (#163).

PR165's existing [campaign](https://drive.google.com/file/d/1xenvcfa5QfhGutNe5xNtDhFLYRqKcyxI/view) and [audit/source snapshot](https://drive.google.com/file/d/1olH4RanIwqGhmPae7Pw8GtvIwRr0e14n/view) already cover the retained M4 `hu20-native-average-20261005` records, transferred into the M1 campaign. Current cloud ID/name/size/parent readbacks match its published acceptance receipt. They are reused rather than uploading another copy; their current staged hashes were not recomputed by this task.

## PR166 M4 river-validation snapshot

The original M4 root `~/Local/hu20-turn-search-river-20261005` contains completed river validation, pilot bundles, idle retiming, copied inputs, admission/launch/budget records and source checkout `befa6ba0df74a2c393c08a11897ade5ff492b561`. No owned process was running in that root at snapshot time. The subsequent production arena is a separate active task; this snapshot does not claim its completion.

A new lossless whole archive contains **3,982 source members / 3,379,584,375 logical bytes**, with seven preserved hard-link references. Compressed bytes: **1,517,967,697**. SHA256: `bb21fa48867ab622e35486ec732ada86a0e1726dcb3dccfd348c5b5392d3821c`; embedded manifest SHA256: `bb1b24dc26be869bc424029304fac0529bc1d5a4cb51329b222f161abfdfb6b8`. Every source and streamed archive member independently passes size/SHA256 verification. Git administration, Python caches/environments and node modules are excluded; working source and research binaries remain. Initial verification could not resolve streamed tar hard links; corrected verification reused the unchanged archive, verified earlier link targets and rehashed every source. The receipt records this packaging incident.

Staged location: `~/Local/Research-Cloud/PR-166-HU20-turn-search-arena/M4-river-validation-20261005/`. The [PR166 Drive folder](https://drive.google.com/drive/folders/1iVCttvjpYo8X4jD9C3tcflLyT_Y9QBqO) contains the archive, full member manifest, archive/staging receipts, restore README and archival/verification helpers. Native staging SHA256 matches. **Confirmed uploaded**: current native acceptance and exact cloud ID/name/size/parent match. [Whole snapshot](https://drive.google.com/file/d/1AcdU_6DI-F4SABL1ell4GP82oa_Lskh6/view), [run folder](https://drive.google.com/drive/folders/1KPsr-sitP-pn-xNdUt-i0WJV0S2-49_e) and [acceptance receipt](https://drive.google.com/file/d/1GZOHo_6Eh_6U_LvEbM4bAsRGMzwq4eTP/view) preserve the readback.

## Retrieval and retention

Download the relevant whole archive, check its archive SHA256, extract into a fresh directory and verify every manifested member's size/SHA256 before analysis. Use the adjacent restore README to map archive prefixes back to original paths. The root index links to PRs, archive folders and exact files, and labels historical pending states separately from current acceptance.

The old 81,115-byte Drive index is preserved as a receipt before updating the same existing index file ID. Its nonblank lines are all retained in current main's index, which adds the intervening closeout records. The Drive copy expands repository-relative Markdown links to exact GitHub links so report links also work outside a checkout.

Original working roots, input dependencies and synced payloads remain. This task does not delete local evidence or force cache eviction, so organizing the archives does not reclaim the originals' disk space. Its new PR166 snapshot consumes about 1.5 GB locally; the staged copy uses APFS cloning. Detailed receipts and the previous root index are retained in [Archive-receipts / M4-organization-20261005](https://drive.google.com/drive/folders/1lknF5OpM4MUCweagu3em2tnk4h4lDvkp).
