# M4 uploaded-original cleanup — October 6 evening

Owner-authorized cleanup removed **6,025 verified local files /13.779 GB logical bytes** belonging to merged PRs #166 and #181. Measured free space rose from **17.975 GB to 30.173 GB**, a **12.198 GB** gain; a later read showed 30.233 GB free. APFS shares some blocks, so summed file allocation is not the reclaimed space. Decimal GB throughout.

[Full cleanup receipt and restore helper](https://drive.google.com/file/d/1k6fw_iM91tvTfGCgocbADpt-L0Cx8dF3/view) — **2,576,866 ZIP bytes**, SHA256 `125a71d61c6f19324e1b9d1f03bf4b72e974f2f38860256f1d18662621659ed3`. [Compact verification summary](m4-uploaded-original-cleanup-20261006-evening.json). Copies of the receipt remain outside sync on both Macs at `~/Local/storage-cleanup-receipts/m4-uploaded-copies-20261006-evening/` (M1 has the M4 records under `M4/`).

## Eligibility and preservation

Owning PR166 and PR181 were freshly confirmed **MERGED** immediately before deletion. Open PR185 and PR186 work roots and dependencies were read-only throughout this cleanup. Their current configs/helpers, symlinks, processes and open file handles on both Macs show no dependency on the selected paths. References to these folders in copied RESULTS_INDEX snapshots are historical archival descriptions. PR185's M4 root remains **12,680,400 KiB allocated**, unchanged from the initial audit.

The exact cleanup is **5,991 PR166 files /13,489,478,895 bytes**, plus **34 PR181 files /289,566,925 bytes**. This includes archived partials/failures, raw timing/request/result evidence, completed frozen bundle copies and three separate nonsynced archive copies. Canonical synced archives, shared Git, source checkouts/helpers, root inputs/policies/models, board-pooling/trainer/native-average/control dependencies, credentials and all unarchived files remain. No eviction or forced offloading occurred. Seven empty hard-linked logs were retained when removing their first link changed their metadata.

The PR181 preparation archive embeds an explicit `m4-source-manifest.json` and `m4-evidence/<relative-path>` transfer map. Only those exact members were eligible: six protected source/model files remain and 34 verified originals were removed. Closeout/publication archives also verify, but their remaining M4 originals are retained pending exact transfer mapping; basename/hash matching alone was not used.

## Upload and byte checks

Current independent Drive ID/name/size/parent metadata matches every archive. Native FileProvider reports uploaded=1, uploading=0, no unresolved conflicts and exact document size immediately before removal. Full archive SHA256 and every member size/SHA256 were read back: **10,365 PR166 members** from matching staged copies or cached M4 canonical archives; **165 PR181 members** from cached M1 canonical archives. No independent cloud-byte redownload is claimed.

| Canonical archive | Verified members | Archive SHA256 |
| --- | ---: | --- |
| [stage-4-closed-M4-20261005.zip](https://drive.google.com/file/d/16sVfuvK3iglvsbQ_diZYuvlJFm2-a68o/view) | 659 | `d4ff034cd19d6e6fd8db39313ceba16ff76031d86bc5bc8b74dfea87a87defc6` |
| [fixed-work-proposal-M4-20261005.zip](https://drive.google.com/file/d/1EjIofd_xGULxEIYJKHfsUm1PtI9ruA_O/view) | 1,768 | `e4987d936053eb8276120eff235e21c42356cee33ab181590408e7ef2031fc0b` |
| [fixed-work-approved-stock-attempt-M4-20261005.zip](https://drive.google.com/file/d/1rcFlqfsIRmY5yHzH6wl0uPobD9xv0lAW/view) | 2,303 | `af02f9c59ac85b9741cf1fbc69ff98c98c883757e675c0cc36a48082a49f7f32` |
| [fixed-work-equivalent-closeout-M4-20261005.zip](https://drive.google.com/file/d/16OeY_hNxd9SOMOp627D9jGQC-VYw-R_e/view) | 1,653 | `e1243c2cb6a98f6ebf32a79122cc22aa15665ad1b34c85c908d0ba78add51517` |
| [hu20-turn-search-river-m4-snapshot-20261005.tar.gz](https://drive.google.com/file/d/1AcdU_6DI-F4SABL1ell4GP82oa_Lskh6/view) | 3,982 | `bb21fa48867ab622e35486ec732ada86a0e1726dcb3dccfd348c5b5392d3821c` |
| [v0.4.1-release-preparation-20261006.zip](https://drive.google.com/file/d/1YPdy5Fro9B2BAXlyHOCcpavv1u77gsXE/view) | 58 | `2ad0a8a92a9812a13067fce242547494a00246803cafc49ef2b983e7685e3a0e` |
| [v0.4.1-release-closeout-20261006.zip](https://drive.google.com/file/d/1PonQG3Cc-lcFogpc8warbiroHSHO2KKd/view) | 54 | `d4b5d5ded66bdec1ab4c198af6258eda19ecc5f57258532cc211a4282942c23d` |
| [v0.4.1-release-publication-20261006.zip](https://drive.google.com/file/d/1yM_-tdx2kuk8MD4eNgwZFK0cankWNmHR/view) | 53 | `b7cdb80379227357743ebefdaed76595b98736baaa237e7798e93a180f073689` |

## Restoration

The receipt ZIP contains `M4/cleanup-receipt.json` with every removed absolute path, Drive ID, full archive hash, exact member path, member size/hash and original filesystem identity. `M4/removal-journal.jsonl` durably records each intent/removal; `M4/cleanup-plan.json` records selected and retained manifested files. Other unmanifested files were never selected.

Extract the receipt ZIP into an ignored nonsynced directory. Restore only when needed, using the included helper:

```sh
python3 restore.py \
  --original /Users/dberweger/Local/v041-release-pr181-20261006/clean-checkout-download.json \
  --out results/retrieved/pr181/clean-checkout-download.json
```

The helper reads the canonical native Research-Cloud archive, checks its full SHA256, extracts the exact member into a fresh nonsynced file, then checks restored size/SHA256. `--archive LOCAL_DOWNLOADED_ARCHIVE` selects a separately downloaded copy; `--receipt PATH` selects the receipt location. A null member means the removed file was a separate archive copy. The example was restored and its 1,396 bytes/SHA256 verified successfully. Research binaries remain ignored; only explicitly approved release artifacts may enter Git.

Historical originals-retained claims are superseded **only for the exact paths in this receipt**. M1 research originals and all other M4 evidence remain. This record schedules no unattended cleanup.
