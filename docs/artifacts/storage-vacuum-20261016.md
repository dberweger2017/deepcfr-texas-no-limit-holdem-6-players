# Storage cleanup after PR297 and PR298 merged

Owner-requested cleanup on October 16, 2026. Current GitHub status confirmed
PR297 merged at 06:12:01 UTC and PR298 at October 15, 23:09:44 UTC before
selection and again immediately before removal. PR292 archival dependencies
and open PR301 remain protected. PR292 merged at 06:30:22 UTC while this record
was integrated; its incomplete archive still prevents cleanup of its originals. No research, upload worker, schedule or application was stopped.

| Machine | Selected paths /unique inodes | Allocated bytes retired, upper bound | Measured free-space increase | Free after removal |
|---|---:|---:|---:|---:|
| M1 | 44 /43 | 6,568,140,800 | 6,526,529,536 (6.527 GB) | 17,459,183,616 (17.459 GB) |
| M4 | 5 /5 | 19,407,470,592 | 19,408,011,264 (19.408 GB) | 83,393,691,648 (83.394 GB) |

Free-space changes are filesystem observations during ongoing activity; allocated
bytes are counted once per selected inode and can include shared APFS extents.
These numbers exclude all earlier cleanup passes. GB means decimal bytes.

## Exact receipt and acceptance

[Complete private receipt](https://drive.google.com/file/d/1Z_XSRBqoRzIDgxz3NMFr1lOJWH-TfRhs/view?usp=drivesdk),
`storage-vacuum-20261016-receipt.json.gz`, **78,255 bytes**, SHA256
`0c37861cdb05e6fa775eb8f764a0783f8b261e66e24485913a1f812c6db8569c`.
Direct project parent: `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`.
Connector upload success, independent exact ID/name/size/parent readback and
current native uploaded=1/uploading=0/no-conflicts/documentSize=78255 agree.
The receipt also remains in the nonsynced
`~/Local/storage-vacuum-20261016/` directory.

The receipt names every removed absolute path, source size/SHA256, inode/link
group, modification/change times, source-history cutoff, exact restoration
archive/member, current native/cloud acceptance and fsynced removal journal.
It includes the existing PR298 member manifest, PR297 input proofs, selected
plans, retained-file snapshots and cleanup helpers. Research payloads were not
downloaded or rehashed for cleanup. Only existing manifest metadata was read.

## M1: completed package evidence and redundant copies

All 44 paths are within
`~/Local/hu100-v053-release-package/results/v053-release-package/`:

- Forty archived files of at least 1 MB under `science/`, `runtime-smoke/` and
  `runtime-smoke-attempt-000/`: final raw traces/reproductions and runtime
  journals. Their source path, size and exact modification time still match
  the existing manifest; change times predate accepted sealing/upload.
- Two hardlinked assembled-model copies under `assets-model/` and
  `assets-attempt-000-model/`, counted once. Their original exclusion inventory
  pins the exact sizes/times. The unchanged canonical model under `models/v053/`
  remains, SHA256
  `c3a8edc4cc78098f07ac3e4ecbbb3dacae298a266b9fcd5fc788d0e2b6fcfc8e`.
- Two old compatibility downloads under `compatibility-download-cache/`,
  for v0.5.0 and v0.5.1. Their recorded sizes/times match the sealed exclusion
  inventory; published release hashes and retrieval commands are in the receipt.

Archived originals restore from the accepted
[PR298 evidence ZIP](https://drive.google.com/file/d/1V_ma54XVi0PTAFUDLe8nJJt1L5ADfGGd/view):
603,730,754 bytes, whole SHA256
`8704a092b928de163bc8955dbd5fa2d4b2b965f4f0455996a1f857c420418599`,
embedded `ARCHIVE-MANIFEST.json` SHA256
`c11cd598f329e905d4f63c3f7ab3100d01e74d10796645321edfe8060569184b`.
Each receipt row gives its exact `research/...` member and SHA256.

Assembled copies can be copied again from the retained canonical model and
verified against the hash above. Compatibility inputs restore from
[v0.5.0](https://github.com/dberweger2017/CFR-NLHE-6max/releases/tag/v0.5.0)
and [v0.5.1](https://github.com/dberweger2017/CFR-NLHE-6max/releases/tag/v0.5.1)
with `gh release download TAG --repo dberweger2017/CFR-NLHE-6max --dir FRESH_DIRECTORY`,
then published size/SHA256 verification. Their hashes remain in their model
cards and the cleanup receipt.

Canonical models/tables, both existing package asset directories, publication
assets under `~/Local/v053-publication/`, small failure/audit receipts, source,
private access tokens and the original evidence ZIP backup remain.

## M4: five restored blend inputs

All five paths belong to the merged PR297 working checkout
`~/Local/hu200-v055-frozen-blend/results/`. No open campaign uses these M4
copies; PR301 transfers from its separate native Drive archives on M1.

| Relative removed path | Bytes | Canonical restoration |
|---|---:|---|
| `hu200-v055-blend/inputs/2026101302/checkpoint.gz` | 4,960,844,586 | PR265 full ZIP, `research/frozen/20000000000.gz` |
| `hu200-v055-blend/inputs/2026101302/average.gz` | 3,164,563,975 | PR265 full ZIP, `research/frozen/20000000000.average.gz` |
| `hu200-v055-blend/inputs/2026101422/checkpoint.gz` | 4,957,316,628 | PR278 seed1422 full ZIP, `research/seed-2026101422/frozen/20000000000.gz` |
| `hu200-v055-blend/inputs/2026101422/average.gz` | 3,162,064,602 | PR278 seed1422 full ZIP, `research/seed-2026101422/frozen/20000000000.average.gz` |
| `hu200-v055-blend-inputs/seed-2026101421.average.gz` | 3,162,671,080 | PR278 seed1421 final-average ZIP, `averages/seed-2026101421/final.average.gz` |

Accepted restoration archives:

- [PR265 full ZIP](https://drive.google.com/file/d/1dnVW4lVL_FjnlkrhdU5D5K1eqiXAKlsI/view),
  26,698,069,587 bytes, whole SHA256
  `5fe98dddba46eff70b402ded1448c39280902b58cdded109d3cd3dec00aa404f`.
- [PR278 seed1422 full ZIP](https://drive.google.com/file/d/1xAg4LpkRM_XLre2TsQcDkn0voN48dAwO/view),
  20,953,054,295 bytes, whole SHA256
  `73e76b94333ae9236ab4b280a04b33d5c619abfa333cda1b7486160949347fbb`.
- [PR278 seed1421 final-average ZIP](https://drive.google.com/file/d/1OAKKAxH91_flAKK9eWa4peCy_jUImhDl/view),
  3,162,672,661 bytes, whole SHA256
  `27790a65a55f4e95e7818511118ac62b09387c41f432727a102522f53611f85f`.

Fresh native uploaded/no-pending/no-conflicts checks and independent exact
cloud ID/name/size/direct-parent readback match for all three. Existing input
verification and exact-average reproduction proofs establish the source pins;
current sizes and modification/change history remain within those proofs.
The complete receipt supplies each manifest hash and extracted-member hash.

To restore any original, download the named archive to a fresh ignored
directory, verify its whole SHA256, extract the exact member above, then verify
the member size/SHA256 from the receipt before use. Large private archives may
be retrieved through the signed-in native Drive client and transferred over SSH.

## Preservation and administrative failures

All selected paths were regular nonsynced files with no live handle. Every
hardlink group was selected completely and rechecked before each unlink.
**267 retained files** pass size/mtime/inode checks after removal: 41 on M1
and 226 on M4. PR292 archival dependencies, active PR301 roots, archive spools, transfers and callbacks,
shared Git, failed/uncertain originals and all synced paths remain untouched.

PR278 seed1421's original full cloud object is still only 592,464,795 bytes,
versus the 20,952,808,971-byte local archive. It is not accepted for cleanup.
Its raw checkpoints/current export, protected clone and all related local
originals remain. Only its separately accepted final-average copy was removed.

M1 completed all removals and retained-file checks, then its final receipt
serialization failed on a duplicate `at` keyword. The fsynced per-path journal
was used to reverify absence and retained metadata and recover the final receipt;
no removal was repeated. M4's first attempt found no `gh` in its SSH PATH and
removed nothing; fresh coordinator GitHub readback was supplied under a
120-second age check. Earlier setup failure also remains in the private receipt.
The first receipt build ran before the small M4 receipt transfer completed;
no file was uploaded until the completed receipt existed.
