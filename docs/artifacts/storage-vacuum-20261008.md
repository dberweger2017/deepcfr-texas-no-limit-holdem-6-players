# Storage cleanup — October 8, 2026

The owner requested another cleanup. The selected nonsynced copies belong to currently merged PRs #197, #200–#205. Canonical Google Drive archives remain present and uploaded; no synced file was deleted or offloaded. Xcode remains installed, and the earlier simulator leftovers are already absent.

- **M1:** 52 files /1.545 GB logical removed; measured batch free-space gain **1.059 GB**, with **41.343 GB free** immediately afterward.
- **M4:** 154 files /23.868 GB logical removed; measured batch free-space gain **23.884 GB**, with **77.664 GB free** immediately afterward.

M4 removal covers eight completed local ZIP copies (including sealed files named `archive.partial.zip`) and large archived PR202/PR205 outputs. PR202 input copies and PR205 retrieved inputs remain. M1 removal covers the PR201 ZIP copy and manifest-mapped large archived diagnosis outputs. Smaller metadata, reports, source checkouts and unknown-provenance files remain. Allocated-file totals and measured free-space gains differ on M1; concurrent restore/build activity and shared filesystem blocks can affect those measurements.

## Acceptance and dependency checks

Current GitHub merged status was checked before planning and immediately before removal. For each of nine canonical archives, current native status was uploaded=1, uploading=0, conflicts=0, trashed=0, with the expected document size; independent Drive metadata matched ID, name, size and parent. Existing accepted manifests and hash receipts supplied provenance. Research payloads were neither downloaded nor re-hashed for this cleanup.

Every selected path retained its recorded byte count and unchanged size/mtime/ctime/inode/device between plan and removal. Archived originals matched their member size; PR201/PR202 ZIP timestamps also matched modification times. PR205's deterministic ZIP timestamps are 1980, so its recorded absolute original paths and source timestamps preceding the accepted seal were used instead. Live open files were excluded. Only regular, singly linked nonsynced files were removed.

M4's running HU100 1B pilot and its own gate checkpoint/current/average copies remain; the original PR203/PR204 terminal checkpoints and averages also remain. M1's running turn-search restore, PR166 inputs, native bucket-key work, PR190/board-pooling inputs, all shared Git and active build roots remain. Their retained paths were checked after removal. Cleanup did not stop, restart or change a research process. No unattended cleanup was scheduled.

## Exact removal receipt and restoration

[Full per-path receipt](https://drive.google.com/file/d/1I-5Z5G7l26h3g37YwUndSgGWalx3vy4_/view) — 242,597 bytes, SHA256 `8a88a0b1c754fc8ed927ed322a75689cfd71048a546a52443aa8534e7741088f`. It records each removed absolute path, original size and timestamps, Drive archive ID/URL/hash, exact archive member and member hash when applicable, plus both machine measurements. The receipt was uploaded to Research-Cloud and its current ID/name/size/parent independently read back.

To restore a duplicate whole ZIP, download its recorded Drive ID into a fresh ignored nonsynced directory. To restore an original, download the same archive and extract the receipt's exact member into a fresh destination. Verify the indexed whole archive and required member SHA256 before research use; never overwrite an active input. The owner's trust in upload acceptance applies to cleanup, while verification on research retrieval remains required.

Restoration archives:

- M1 PR201: [1FBfquL5UGU3E2MGAl7WUD27e75ENfRlr](https://drive.google.com/file/d/1FBfquL5UGU3E2MGAl7WUD27e75ENfRlr/view); 52 removed paths /1,544,534,864 logical bytes.
- M4 PR202: [1-XHLJ-dM_DFMFTtXZYJZW7OmBeW2VLem](https://drive.google.com/file/d/1-XHLJ-dM_DFMFTtXZYJZW7OmBeW2VLem/view); 59 removed paths /8,930,394,776 logical bytes.
- M4 PR205: [1Lkme7gGFIsiCBkquKwt0gRgUIFFbCy7G](https://drive.google.com/file/d/1Lkme7gGFIsiCBkquKwt0gRgUIFFbCy7G/view); 89 removed paths /6,243,823,218 logical bytes.
- M4 PR203: [1hC98K7KSU-pFEdPD7lfFMT17EcxpW_Re](https://drive.google.com/file/d/1hC98K7KSU-pFEdPD7lfFMT17EcxpW_Re/view); 1 removed path /1,143,009,755 logical bytes.
- M4 PR203: [1Qs_8_7YUCMccw6A44jNld_s0ksKfTM-b](https://drive.google.com/file/d/1Qs_8_7YUCMccw6A44jNld_s0ksKfTM-b/view); 1 removed path /2,078,090,291 logical bytes.
- M4 PR204: [12LEcZWKJCwtTCCYcMxW578Gm3rDy9emK](https://drive.google.com/file/d/12LEcZWKJCwtTCCYcMxW578Gm3rDy9emK/view); 1 removed path /1,641,001,686 logical bytes.
- M4 PR204: [1D8BRCrV2BDAgKB8tipRYptABERZv88xt](https://drive.google.com/file/d/1D8BRCrV2BDAgKB8tipRYptABERZv88xt/view); 1 removed path /2,032,348,910 logical bytes.
- M4 PR197: [1QEHJJ_yPJbfHYkX1Taa3wM4RnblVPdK1](https://drive.google.com/file/d/1QEHJJ_yPJbfHYkX1Taa3wM4RnblVPdK1/view); 1 removed path /574,958,858 logical bytes.
- M4 PR200: [1aZ5HtPGEY0-qCtQkoJbbY122S6xxFISg](https://drive.google.com/file/d/1aZ5HtPGEY0-qCtQkoJbbY122S6xxFISg/view); 1 removed path /1,224,575,891 logical bytes.
