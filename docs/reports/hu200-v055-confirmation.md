# v0.5.5 M4 confirmation seeds

[Draft PR278](https://github.com/dberweger2017/CFR-NLHE-6max/pull/278) ·
[Owner-approved acceptance](../hu200-v055-acceptance-protocol.md).

**Both seeds reached 20B and passed every-row current/average audits on the 16-GiB M4.**
Seed **2026101422 started October 14 at 20:22:19 UTC /22:22:19 CEST**, after
seed 1’s audits, fresh exclusive nonblocking admission on both research locks,
no campaign workers, **58.662 GiB free** and **7.582 hours remaining**.
It finished science at **October 15 00:38:20 UTC /02:38:20 CEST**, within
the original clock. Both server preparations are complete. The owner-authorized retry of seed 1’s
full ZIP is **visibly uploading**; the owner assigned upload acceptance to
Vacuum Cleaner and instructed this campaign not to wait.

## Fixed recipe, source and clock

Both seeds use current main `9c449929e4b41e8c60383e682bea1e0e1e4c01d5`
on the 16-GiB Apple M4 (Mac16,1), in a separate unsynced worktree,
`~/Local/hu200-v055-confirmation-20261014`, ignored results
`results/hu200-v055-confirmation`. The M1 performs only administration/archive
I/O; it does not train these seeds.

The detached first build begins **October 14 15:57:16 UTC /17:57:16 CEST**.
The fixed 12-hour deadline is **October 15 03:57:16 UTC /05:57:16 CEST**;
training stops by **01:42:16 UTC /03:42:16 CEST**, reserving 15 minutes for a
terminal save and two hours for exports/audits. Each stage receives remaining-clock
`--max-seconds`; a detached wall guard includes checkpoint loading. Seed 2
was admitted above the fixed 5.5-hour minimum. No budget extension or retry.

The path-independent build exactly matches #265: **1,492,512 bytes**, SHA256
`4f6abcbb954c3c744448c2a3542ab3164e0e1a015cb05ecebeda077525991cd4`.
[Build](hu200-v055-confirmation-artifacts/build-receipt.json),
[clock and one swap baseline](hu200-v055-confirmation-artifacts/baseline.json),
[fresh initial admission](hu200-v055-confirmation-artifacts/readmission.json).
All three #163 K50 source/copy SHA256s match; campaign copies are read-only.
[Input provenance and full hashes](hu200-v055-confirmation-artifacts/inputs.json).

Recipe is unchanged apart from the independent seed: HU200, menu v1, K50,
R=1, opponent-sampled lifetime average, linear CFR; growing stage has #250's
3B target and 500M/1B/1.5B/2B/2.5B saves, cap **85,373,862**, without freeze.
The cap terminal is the first complete iteration past it. Frozen continuation
resumes that seed's hash-pinned terminal, fixes its actual entry count, uses
recovery, telemetry and a fresh stop file, and saves at 2B/5B/10B/20B.
Target-named files do not imply target nodes were reached.

## Seed 2026101421: complete science

Growing terminal: **754,080,075 nodes /496,692 iterations /85,373,916 entries**,
54-entry overshoot. Its target-named `1000000000.gz` is this cap terminal;
the growing 1B and 3B targets were not reached. Frozen terminal:
**20,000,000,322 nodes /16,745,107 iterations**, exactly the same retained keys.
Every frozen save passes saved-byte/count/sorted-key verification against stage 1.
Sorted serialized key SHA256:
`85ab441df298c21d2f47c93798f89f69d9690709304a60f4f737eb782c8d55df`.

| Save | Actual nodes | Iterations | Entries | Mean visits/key | Zero visits | After-save RSS GiB | Write s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 500M | 500,000,695 | 320,364 | 69,180,237 | 1.4317 | 46.49% | 4.924 | 236.14 |
| Cap | 754,080,075 | 496,692 | 85,373,916 | 1.7483 | 44.24% | 3.613 | 292.61 |
| 2B frozen | 2,000,001,206 | 1,415,679 | 85,373,916 | 4.1481 | 34.89% | 3.776 | 283.85 |
| 5B frozen | 5,000,000,230 | 3,813,020 | 85,373,916 | 9.7582 | 26.54% | 1.750 | 300.98 |
| 10B frozen | 10,000,000,162 | 8,053,874 | 85,373,916 | 19.0021 | 21.19% | 2.076 | 314.48 |
| 20B frozen | 20,000,000,322 | 16,745,107 | 85,373,916 | 37.3670 | 16.64% | 1.657 | 325.22 |

[Stage-1 verification](hu200-v055-confirmation-artifacts/seed-2026101421-stage1-verified.json),
[all per-save statistics, discarded visits/updates and street/role missing-decision shares](hu200-v055-confirmation-artifacts/seed-2026101421-learning-curve.json),
[full frozen lineage and key checks](hu200-v055-confirmation-artifacts/seed-2026101421-frozen-complete.json).
Only the redundant 2B/10B checkpoints were removed after verified successors,
as #265 permits; cap, 5B and final checkpoints remain resumable.

Final current and average were exported with **uniform zero-mass play**.
`src.diagnostics.cfr_average.audit` verified **all 85,373,916 rows**:
**66,350,746 positive-mass /19,023,170 uniform-fallback rows**.
Audit took **1,653.47 s**; unweighted current/average TV was **0.3305011**.
This checks stored accumulators and current regrets, not historical increments
or playing strength. Science finished **20:05:35 UTC /22:05:35 CEST**, inside
its original clock, with supervisor return code zero and no guard stop.
[Every-row audit](hu200-v055-confirmation-artifacts/seed-2026101421-final.audit.json),
[science completion](hu200-v055-confirmation-artifacts/seed-2026101421-science-complete.json).

| Final file | Bytes | SHA256 |
|---|---:|---|
| Checkpoint | 4,957,752,665 | `c70e8e01b3b27c7fa8b8e0561bdbe5b77b321125de646f04614c82ecacacc676` |
| Current | 2,301,203,484 | `bbac06244ccb0815751417afe3c13c28ef56f7aa3b97621f9cab634e3a9c01df` |
| Average | 3,162,671,080 | `c08d86a9615022df7015d34e5dc3de2705bbb8346ed6976bd72284e71d91f444` |

The closed **2,966 five-second readings** have sampled family peak
**8.777 GiB**, minimum free memory **33%**,
minimum disk **42.263 GiB**, maximum swap growth
**67,108,864 bytes**, pressure levels **[1, 2]**,
and **0 stop readings**. Maximum observed gap is
**5.08 s**. These are sampled values.
[Closed resource summary](hu200-v055-confirmation-artifacts/seed-2026101421-resource-summary.json).
The small completion observer has separate retained readings; it does no training.

## Seed 2026101422: complete science

[Admission and detached launch](hu200-v055-confirmation-artifacts/seed-2026101422-launch.json)
confirm seed 1's successful audit, released locks and worker exit before launch.
The same executable, K50 copies, clock, swap baseline and guard thresholds remain.
Growing terminal: **755,276,076 actual nodes /494,507 iterations /
85,373,928 entries**, 66-entry complete-iteration overshoot.
Stage 2 resumes that terminal SHA256 with `--max-entries 85373928`
and frozen/recovery mode. Latest verified frozen save:
**20,000,000,096 nodes /16,827,728 iterations**, all **85,373,928 keys**
and the stage-1 sorted-key digest unchanged:
`1fc63ca8eb21d6246a87da3745598b68d57cd40c4cf6331dbcf2f333750860b1`.
[Stage-1 bytes/counts/keys](hu200-v055-confirmation-artifacts/seed-2026101422-stage1-verified.json),
[verified frozen terminal and lineage](hu200-v055-confirmation-artifacts/seed-2026101422-frozen-complete.json).

| Save | Actual nodes | Iterations | Entries | Mean visits/key | Zero visits | After-save RSS GiB | Write s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 500M | 500,000,820 | 320,010 | 69,055,433 | 1.4340 | 46.44% | 2.992 | 259.28 |
| Cap | 755,276,076 | 494,507 | 85,373,928 | 1.7507 | 44.22% | 3.967 | 304.41 |
| 2B frozen | 2,000,001,235 | 1,396,928 | 85,373,928 | 4.1401 | 34.82% | 2.213 | 297.39 |
| 5B frozen | 5,000,000,239 | 3,772,546 | 85,373,928 | 9.7493 | 26.43% | 1.806 | 314.63 |
| 10B frozen | 10,000,002,576 | 8,041,286 | 85,373,928 | 18.9896 | 21.11% | 1.378 | 320.92 |
| 20B frozen | 20,000,000,096 | 16,827,728 | 85,373,928 | 37.3934 | 16.63% | 1.648 | 332.36 |

Current and average exports use **uniform zero-mass play**. The final audit
verifies **all 85,373,928 rows**: **66,371,144 positive-mass /19,002,784
uniform-fallback rows**. Audit takes **1,654.83 s**; unweighted current/average
TV is **0.3305658**. It checks stored accumulators/current regrets, with the
same limits as seed 1. Science finished **00:38:20 UTC /02:38:20 CEST**, inside
the original deadline; supervisor exited zero, no reasons.
[Every-row audit](hu200-v055-confirmation-artifacts/seed-2026101422-final.audit.json),
[science completion](hu200-v055-confirmation-artifacts/seed-2026101422-science-complete.json).

| Final file | Bytes | SHA256 |
|---|---:|---|
| Checkpoint | 4,957,316,628 | `e265c384686f003521e49158c5162e57175d3b53e405140cb494d30a8ec9660c` |
| Current | 2,301,944,453 | `1bfd938c992591a79cec66ad93065996856fbc456096127fe893b235131d91d5` |
| Average | 3,162,064,602 | `b191c5ed8b943d138e6076fb4a1b0117c98964d92dd2d6253e157f696f691f6f` |

Closed **3,058 five-second readings**: peak family RSS **8.808 GiB**,
minimum free memory **40%**, minimum disk **30.783 GiB**,
maximum swap growth **340,650,886 bytes**, pressure levels **[1, 2]**,
**zero stop readings**, maximum observed gap **5.05 s**.
At the 02:51 CEST check, the correct second-seed supervisor identity is exited,
both nonblocking research locks are free and no trainer remains.
[Closed resources](hu200-v055-confirmation-artifacts/seed-2026101422-resource-summary.json),
[release check](hu200-v055-confirmation-artifacts/seed-2026101422-release-confirmation.json),
[per-save statistics and missing-decision shares](hu200-v055-confirmation-artifacts/seed-2026101422-learning-curve.json).

## Server preparation and archives

Seed 1's final average has passed remote size/SHA256 verification in
`~/deepcfr-slumbot/inputs/model-seed-2026101421/final.average.gz`.
Its **5,115,570,300-byte** cache in `~/deepcfr-slumbot/cache-seed-2026101421`
passes the verified read-only reload, including every cache payload hash.
Cache preparation takes **945.26 s**, with **190 five-second samples**,
peak worker RSS **4.825 GiB** and minimum host available
memory **2.970 GB**, above the unchanged 2-GB stop threshold.
No guard stop/failure, play, login or credentials. Existing sessions/paths remain untouched.
[Cache receipt](hu200-v055-confirmation-artifacts/server-seed-2026101421/cache-receipt.json),
[server resources summary](hu200-v055-confirmation-artifacts/server-seed-2026101421/supervisor-receipt.json).

**Prepared only:** `~/deepcfr-slumbot/run-v055-seed-2026101421.sh`, shell syntax checked,
not executed. It pins the completed-hands client source and audited model/cache;
10,000 completed hands, translation 512/128, timeout 30 s, abandoned/rejected
limits 200/50, policy seed 20261014055201, account `deepcfr-v055-s21`, and a fresh
output path under `v055-runs-pr278`. The owner logs in and starts it.
[Exact launcher](hu200-v055-confirmation-artifacts/server-seed-2026101421/run-v055-seed-2026101421.sh),
[launcher receipt](hu200-v055-confirmation-artifacts/server-seed-2026101421/launcher-receipt.json).
Seed 2's final average also passes remote bytes/SHA256 verification. Its new
**5,115,678,680-byte** cache passes every payload hash through read-only reload.
The cache guard exits zero after **840.19 s /169 five-second readings**,
peak worker RSS **4.826 GiB**, minimum available memory **2.783 GB**, no stop.
**Prepared only:** `~/deepcfr-slumbot/run-v055-seed-2026101422.sh`, shell syntax
checked, never executed. Same completed-hands source and options as seed 1,
account `deepcfr-v055-s22`, policy seed **20261014055301**, pinned own average
and read-only cache, fresh own output path. Both server preparations are complete;
existing sessions/paths remain untouched.
[Cache receipt](hu200-v055-confirmation-artifacts/server-seed-2026101422/cache-receipt.json),
[resource summary](hu200-v055-confirmation-artifacts/server-seed-2026101422/supervisor-receipt.json),
[exact launcher](hu200-v055-confirmation-artifacts/server-seed-2026101422/run-v055-seed-2026101422.sh),
[launcher receipt](hu200-v055-confirmation-artifacts/server-seed-2026101422/launcher-receipt.json).

[Canonical PR278 Research-Cloud folder](https://drive.google.com/drive/folders/1-0y8ETC4xqzfGzSMOnt9BWMJAUkqpNiP):
`~/Local/Research-Cloud/PR-278-hu200-v055-confirmation/`, confirmed under project
folder `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. Seed 1's full and separate
final-average ZIPs pass member-hashed transfer and complete local readback:

| Seed-1 ZIP | Bytes | Whole SHA256 | Manifest SHA256 | Members |
|---|---:|---|---|---:|
| Full | 20,952,808,971 | `88554b5a0698bfe1eb9b7a596340df71500ba7d1caf6bf1d41114704676b932d` | `6fe3742f89c50288381045d87f5309c04538ea9eb91e8806b5fc974764383b39` | 52 |
| Final average | 3,162,672,661 | `27790a65a55f4e95e7818511118ac62b09387c41f432727a102522f53611f85f` | `678205c4f4894d643f75e4b08d064649e2e983bc0f5e7901505d640997de8f25` | 1 |

[Full readback](hu200-v055-confirmation-artifacts/seed-2026101421-full-archive-receipt.json),
[average readback](hu200-v055-confirmation-artifacts/seed-2026101421-average-archive-receipt.json),
[model/member pins and restoration command](hu200-v055-confirmation-artifacts/seed-2026101421-model-index.json).
The **[final-average ZIP](https://drive.google.com/file/d/1OAKKAxH91_flAKK9eWa4peCy_jUImhDl/view?usp=drivesdk) is upload-accepted**:
native uploaded1/uploading0/no conflicts, plus independent matching ID/name/
3,162,672,661-byte size/direct parent. **Full ZIP upload is blocked**: the
06:21 CEST read-only Drive UI shows “Can’t sync file, moved it to ‘Lost and
Found’ folder.” Native uploaded0/uploading1 remains set, but independent Drive
size is still the earlier 592,464,795-byte partial. Do not restore that version.
The complete verified **20,952,808,971-byte local archive remains at its
canonical path**, modification time unchanged since sealing. The separately
reported Lost and Found copy is **1,481,657,243 bytes**, an earlier partial;
it is retained. The original failure is preserved; the authorized retry below
retains both copies.
[Sync error and exact local paths](hu200-v055-confirmation-artifacts/seed-2026101421-sync-error.json).
[Native observation](hu200-v055-confirmation-artifacts/seed-2026101421-native-upload-observation.json),
[independent metadata](hu200-v055-confirmation-artifacts/seed-2026101421-cloud-upload-observation.json).
No remote archive bytes were downloaded or verified. M4 originals remain.
Seed 2 archives use the same canonical folder, split across the two Macs to fit
available space: **full ZIP sealed on M1**, **separate average ZIP sealed on M4**.
Before launch, M1 free **23,907,180,544 bytes** exceeds full-only admission
**23,100,509,657 bytes**, including 2 GiB headroom. No cleanup is required.
The M4 average ZIP is completely member-readback/whole-hash verified before
an atomic same-filesystem rename into Research-Cloud:

| Seed-2 ZIP | Bytes | Whole SHA256 | Manifest SHA256 | Status |
|---|---:|---|---|---|
| Final average | 3,162,066,289 | `76a82e8c1f757f295a7f2498b429047eff23befb29fe3f7d48580f4cdb84332e` | `14ec89c65d748a7fd2204546793108a152515d24933aded0753404ec29539089` | Local readback and upload accepted |
| Full | 20,953,054,295 | `73e76b94333ae9236ab4b280a04b33d5c619abfa333cda1b7486160949347fbb` | `6c8cafeca8e855e7d8251bf0110f09ff1b3cc3e0e722254479c06a9a3ac553b1` | 48 members; local readback and upload accepted |

[Average readback](hu200-v055-confirmation-artifacts/seed-2026101422-average-archive-receipt.json),
[native upload observation](hu200-v055-confirmation-artifacts/seed-2026101422-average-native-upload-observation.json),
[model pins and average restoration command](hu200-v055-confirmation-artifacts/seed-2026101422-model-index.json).
The **[second average ZIP](https://drive.google.com/file/d/1nOhwFtDzJakpkB7MnHdk5JgY75CRnQjT/view?usp=drivesdk) is upload-accepted**:
native uploaded1/uploading0/no conflicts, independently matching ID/name/
3,162,066,289-byte size/direct parent. No remote archive bytes downloaded.
[Independent acceptance](hu200-v055-confirmation-artifacts/seed-2026101422-average-cloud-upload-acceptance.json).
At **04:21 CEST**, the original M1 archive worker is exited and the complete
**48-member** full ZIP passes all source/member SHA256s, local readback and
whole hashing (**2,926.60 s** sealing/readback). No failure receipt; M1 has
**5.82 GiB free**. Both scientific/full archive lineages are now locally sealed.
The [second full ZIP](https://drive.google.com/file/d/1xAg4LpkRM_XLre2TsQcDkn0voN48dAwO/view?usp=drivesdk)
is **upload-accepted at 05:51 CEST**: native uploaded1/uploading0/no conflicts
and independent exact ID/name/20,953,054,295-byte size/direct-parent match.
Seed 1’s full upload is blocked by the sync error above.
[Full local readback](hu200-v055-confirmation-artifacts/seed-2026101422-full-archive-receipt.json),
[native observation](hu200-v055-confirmation-artifacts/seed-2026101422-full-native-upload-observation.json),
[independent metadata](hu200-v055-confirmation-artifacts/seed-2026101422-full-cloud-upload-observation.json),
[closed worker observation](hu200-v055-confirmation-artifacts/seed-2026101422-full-archive-progress.json).
The original-baseline M4 archive guard exits zero, **4 readings /15.13 s**,
peak family RSS **0.109 GiB**, no breach. Its first helper invocation failed
before opening a ZIP because `PYTHONPATH` was missing; that source/log/failure
is retained, and the corrected environment passes an import preflight. This is
administrative work after completed science, with no repeated training/export.
[Both archive attempts](hu200-v055-confirmation-artifacts/seed-2026101422-average-archive-guard.json).
The owner subsequently authorized the upload retry and directed a handoff as
soon as uploading was verified, without waiting for acceptance. A completed
APFS clone of the intact verified ZIP was staged outside Drive and atomically
published into the same folder as
`hu200-v055-seed-2026101421-full-retry-20261015.zip`.
The **20,952,808,971 bytes**, whole SHA256 and 52-member manifest are identical
to the original verified ZIP; no reseal or repeat hashing occurred. The original
ZIP, old partial cloud version and Lost and Found partial remain untouched.
Drive's sync activity visibly progresses from **0% to 10% uploaded**,
19.51 GiB; the retry's native state is uploaded0/uploading1. **Upload completion
and acceptance are not claimed.**
[Retry receipt](hu200-v055-confirmation-artifacts/seed-2026101421-upload-retry.json),
[native observation](hu200-v055-confirmation-artifacts/seed-2026101421-retry-native-upload-observation.json),
[Vacuum Cleaner handoff](hu200-v055-confirmation-artifacts/upload-handoff.json).

The [closeout supplement](https://drive.google.com/file/d/1Pg4fSvilquKi53C0EIFfjE0SPf6qgLhi/view?usp=drivesdk) is **locally verified and upload-accepted**:
**873,329 bytes**, whole SHA256
`14f92de8e5423bd743af96ea4d14b47d81fa55d2f6bc041fda77585b7152f458`,
manifest SHA256 `b6b1dacebf5b9ddd48ce70658f0ae645dcbdd4614b089bf56994e255ab9c1edb`,
**159 members**. Complete local member readback, native uploaded1/uploading0/no
conflicts and independent exact ID/name/size/direct-parent match all pass.
It retains closed administrative/server/observer sources, failures and receipts,
all **190/169 server**, **2,906 observer**, and **2/4 failed/corrected archive-guard**
readings. Both full scientific archives separately retain all scientific readings.
The private callback is excluded; no remote archive bytes were downloaded.
[Local readback](hu200-v055-confirmation-artifacts/closeout-archive-receipt.json),
[independent acceptance](hu200-v055-confirmation-artifacts/closeout-cloud-upload-acceptance.json).

The owner explicitly assigns the large retry’s completion/acceptance to Vacuum
Cleaner. No monitor is recreated, no cleanup is performed, and the PR remains
draft for the owner’s merge decision. Two retry setup errors occurred before publication: this Python lacks
`os.listxattr`, and `cp` rejects combined `-c -X`; native `xattr` and a clone
followed by sync-identity checks corrected them. Their receipt is retained.
No scientific retry, budget extension, cleanup, synced deletion or eviction.

## Guards, supervision and preserved administrative failures

Both research locks are held during each detached seed. Five-second observations
stop after 60 continuous observed seconds above 13 GiB family RSS, critical
pressure, free memory below 8%, or swap growth above 4 GB. Disk below 16 GiB
requests a clean stop immediately. Gaps cannot establish a sustained breach;
owned process creation times are checked before signaling. No other jobs signaled.

The owner-requested 30-minute checks caught completion and upload acceptance.
At the terminal sync block, **the interval and disabled completion webhook were
deleted**, as the owner required. No follow-on schedule or observer exists.
[Schedule deletion receipts](hu200-v055-confirmation-artifacts/schedule-removal.json).
The earlier webhook's three HTTP errors remain
[preserved](hu200-v055-confirmation-artifacts/completion-callback-failure.json).
Private routing capability data was not read and is excluded from Git/archives.

Initial disk admission at 35.474 GiB failed before the build/clock; owner-cleared
storage passed fresh admission. The unsupported Python 3.9 artifact-check
invocation and missing psutil in two setup environments are preserved; supported
Python 3.11/research environments pass. A quoting error was found in the
administrative server-launch wrapper; its source/failure is retained and a
corrected launch verified the existing transfer and completed the first cache build,
with no repeated training/export. All closed administrative logs/source will enter
the closeout supplement.

Nine focused clock/recipe/key-order/guard tests and staged repository artifact
checks pass. No unresolved source-review finding; only the owner-directed retry upload
acceptance remains with Vacuum Cleaner. No Slumbot play,
credentials, release, tag, default change or merge; the owner decides merge.
