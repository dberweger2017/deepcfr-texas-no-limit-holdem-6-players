# M4 archived originals removed — October 5, 2026

The owner instructed: “delete the originals as soon as Google Drive says its uploaded.” This authorizes removal of archived research originals, subject to the earlier instruction to protect open PR work and its dependencies. This cleanup covers M4; M1 originals were unchanged.

Fresh GitHub reads confirmed #149, #162 and #163 merged. #166 and #169 were open. #169's real-input parity helper uses the M1 `hu20-board-pooling-m4-retrieval-04-20261004/prepared-03` copy. That M1 copy was untouched. M4 preparation/input/source/tool folders were retained conservatively, alongside all open PR roots and shared Git.

## Verification and removal

Before removal, every archive passed fresh native Drive acceptance (`isUploaded=1`, `isUploading=0`, no unresolved conflict, exact document size and cloud ID), current cloud name/size/parent readback and SHA256 of the locally available synced archive. Each adjacent manifest matched its immutable SHA256. All selected original files and the three separate local archive copies then passed fresh size/SHA256 and stable identity checks. No selected file had an open handle. The entire deletion set was reviewed, owning PR states were rechecked, and native acceptance/open handles/identities were checked again immediately before unlinking.

| Merged PR | Removed original records | Canonical archive retained in Drive |
| --- | --- | --- |
| #149 | Main-06 results, earlier qualification/main/engineering results, archived logs/receipts and the separate local compressed archive | [Board-pooling campaign](https://drive.google.com/file/d/1NNSkCO811USN6U9L2p6lBbti1-6Q9r2X/view) |
| #162 | Training/evaluation/smoke outputs, archived closeout provenance and the separate local compressed archive | [Trainer bench](https://drive.google.com/file/d/1_6dRapLReZ9-sAMPaePNX-nehXowGwki/view) |
| #163 | Manifested table/build/verification records and the separate local compressed archive | [Equity buckets](https://drive.google.com/file/d/1Gl1JYWN0F7KFJbwoKtcA0zB_rUsWWZ6T/view) |

**3,399 file paths / 70,575,119,313 logical bytes** were removed. There were no changed/missing-file exceptions in the selected set. Free space rose from **19,226,726,400 to 69,267,038,208 bytes**, approximately **50.0 GB reclaimed / 69.3 GB free**. Logical deleted bytes exceed reclaimed bytes because of original hard links and APFS archive clones shared with retained Drive copies. Uploading or removing a clone does not necessarily reclaim that clone's logical size.

All 3,399 removed paths were subsequently confirmed absent. All 5,182 excluded manifest paths were confirmed present. The #166 M4 river and arena roots remained present, and no process was stopped. #149 prepared/input/source/tool folders, #162 input/source folders, Git administration, unarchived files and synced archives remain. No Drive file was deleted or evicted. M1 and #165 originals were outside this removal set. Empty directories may remain.

## Retrieval and audit

Use the canonical Drive archives, adjacent manifests and restore READMEs linked above. Check the published archive SHA256, extract into a fresh directory with adequate free space, then verify member hashes. #149's `campaign/` prefix restores its original relative paths; #162's archive preserves the work root and frozen inputs; #163's archive preserves all six tables and the build/verification evidence. Historical reports describing originals as retained are superseded for the exact paths in this receipt, not rewritten as if cleanup had occurred earlier.

The [cleanup receipt](https://drive.google.com/file/d/1nldLqO9B2Ln4E-jCLfE3UseCqK2G7exX/view) records archive/member hashes, native acceptance, exclusions, original identities, every removed path and before/after free space. Receipt SHA256: `ce8525dc79fdedfd871fb691cd543335e97cda7ec8832bef2a19c5c5cdf501ea`. The [receipt folder](https://drive.google.com/drive/folders/1fI2S4LaPJqMTlxEfSgA5SBX_43G5ZJFF) also preserves the unlink journal, one-time verification/removal helper and previous shared index. Cloud metadata readback confirmed receipt sizes/parents. No full cloud archive re-download was performed; current native upload acceptance plus current cloud metadata and fresh locally available archive hashes implement the owner's requested acceptance criterion.

The owner's authorization is recorded in AGENTS.md for subsequent merged research closeouts. Future deletion still requires current PR/upload status, matching hashes and dependency review. It does not authorize removing active inputs or open PR files and does not install an unattended cleanup service.
