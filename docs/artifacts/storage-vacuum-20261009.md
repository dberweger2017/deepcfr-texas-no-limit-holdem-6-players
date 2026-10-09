# Storage cleanup — October 9, 2026

Owner-requested cleanup removed **392 M4 files /6.014 GB logical** and **41 M1 files /1.098 GB logical**. Measured removal-batch free-space gains were **6.035 GB M4** and **1.086 GB M1**; immediate free space was **31.619 GB M4** and **35.481 GB M1**. Concurrent writes and the temporary documentation checkout can change later free-space readings.

[Full exact-path restoration receipt](https://drive.google.com/file/d/1KJU6cRbeHcuBv6asxGcMxZIQv79znWcx/view?usp=drivesdk) · [compact measurements](storage-vacuum-20261009.json). Full receipt: **539,270 bytes**, SHA256 `3d3803b1d60dcd66ea726b257758b8e2a47517e132995e911f224627729b8dfe`; Drive ID/name/size/parent metadata confirmed after connector upload. Local receipts, journals and helpers remain at `~/Local/storage-vacuum-20261009/` on both hosts.

## Scope and evidence

M4 removals cover sealed inactive pilot checkpoint/export copies, per-panel export copies and large raw final/reproduction hand, decision, seed and timing traces from merged #197/#200/#203/#204. Canonical recovery checkpoints, terminal/current/average inputs and source/build environments remain. M1 removals cover sealed inactive #165 pilot copies and arena traces, plus #208 closeout staging analysis/request/profile copies and its separate completed nonsynced ZIP. Original #208 small derived summaries, solver, source and input provenance remain.

All owning PRs were freshly confirmed MERGED. Each canonical archive passed current native uploaded=1/uploading=0/unresolved-conflicts=0/trashed=0/exact-size status plus independent connected cloud ID/name/size/parent acceptance. Existing manifests and provenance were used without archive downloads or repeated archive/member payload hash audits. Source sizes and pre-seal modification/change history matched; #165 originals also matched their recorded device/inode/mode/mtime. Current open-PR source/script searches and open-handle checks cleared selected paths. Regular singly linked nonsynced file identities were rechecked before unlink, restoration journals fsynced first, and absence confirmed afterward.

Open #210/#211/#212/#213 roots and dependencies remain untouched. The **entire #207 1B HU100 root**, all synced partials and archives, canonical recovery/model inputs, retrieved/prepared inputs, external solver, shared Git and other working trees remain. Open #211 explicitly preserves #207; its recent upload acceptance does not override that dependency. No unrelated app removal, synced deletion/offloading, process interruption or unattended cleanup.

## Restoration archives

| Owning PR | Archive | Drive restoration link |
| --- | --- | --- |
| #165 | `hu20-native-average-complete-20261005.tar.gz` | [Accepted archive](https://drive.google.com/file/d/1xenvcfa5QfhGutNe5xNtDhFLYRqKcyxI/view?usp=drivesdk) |
| #208 | `hu20-search-stackoff-diagnosis-M1-20261008.zip` | [Accepted archive](https://drive.google.com/file/d/1-6l6MbF3pkPSWZe_9V6kJHAYp2H7lHJg/view?usp=drivesdk) |
| #203 | `hu100-growth-stage1-M4-20261008.zip` | [Accepted archive](https://drive.google.com/file/d/1hC98K7KSU-pFEdPD7lfFMT17EcxpW_Re/view?usp=drivesdk) |
| #203 | `hu100-growth-stage2-M4-20261008.zip` | [Accepted archive](https://drive.google.com/file/d/1Qs_8_7YUCMccw6A44jNld_s0ksKfTM-b/view?usp=drivesdk) |
| #204 | `hu100-growth-50m-stage1-M4-20261008.zip` | [Accepted archive](https://drive.google.com/file/d/12LEcZWKJCwtTCCYcMxW578Gm3rDy9emK/view?usp=drivesdk) |
| #204 | `hu100-growth-50m-stage2-M4-20261008.zip` | [Accepted archive](https://drive.google.com/file/d/1D8BRCrV2BDAgKB8tipRYptABERZv88xt/view?usp=drivesdk) |
| #197 | `native-hu100-playing-baseline-20261008.zip` | [Accepted archive](https://drive.google.com/file/d/1QEHJJ_yPJbfHYkX1Taa3wM4RnblVPdK1/view?usp=drivesdk) |
| #200 | `hu100-learning-curves-20261008.zip` | [Accepted archive](https://drive.google.com/file/d/1aZ5HtPGEY0-qCtQkoJbbY122S6xxFISg/view?usp=drivesdk) |

The full receipt maps every removed absolute path to its exact archive ID, member path, original member SHA256 and indexed whole-archive SHA256. Download the pinned archive into a **fresh ignored nonsynced** folder, verify indexed hashes before research use, and extract only the named member. Use `python3 -m zipfile -e <archive.zip> <fresh-root>` for ZIPs, or `tar -xzf <archive.tar.gz> -C <fresh-root>` for #165. Do not overwrite an active input. Whole-archive duplicates restore by downloading their pinned Drive file.

Historical retention claims are superseded only for these receipted removed paths. Upload status is confirmed; remote archive bytes were not redownloaded during this cleanup.
