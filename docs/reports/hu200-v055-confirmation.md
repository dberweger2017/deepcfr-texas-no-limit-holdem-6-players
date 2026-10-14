# v0.5.5 M4 confirmation seeds

[Draft PR278](https://github.com/dberweger2017/CFR-NLHE-6max/pull/278) ·
[Owner-approved acceptance](../hu200-v055-acceptance-protocol.md).

**Seed 2026101421 has reached 20B and passed every-row current/average audits.**
Seed **2026101422 started October 14 at 20:22:19 UTC /22:22:19 CEST**, after
those audits, fresh exclusive nonblocking admission on both research locks,
no campaign workers, **58.662 GiB free** and **7.582 hours remaining**.
Its growing-table stage is running. It is not yet a 20B result.

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

## Seed 2026101422: running

[Admission and detached launch](hu200-v055-confirmation-artifacts/seed-2026101422-launch.json)
confirm seed 1's successful audit, released locks and worker exit before launch.
The same executable, K50 copies, clock, swap baseline and guard thresholds remain.
Actual cap/frozen counters, sorted-key digest and final audits will be added when
available; no requested milestone is relabelled complete prematurely.

## Server preparation and archives

Seed 1's audited final average is transferring to the new db-ocean directory
`~/deepcfr-slumbot/inputs/model-seed-2026101421/`. Remote SHA256 verification
and a new guarded `cache-seed-2026101421` build follow. No cache success or
launcher readiness is claimed yet. Seed 2's server work awaits its audits.
Cache preparation uses the existing venv and verified #259 cache path, stops
below 2 GB available memory, and records five-second readings. Launchers will
use the completed-hands client and exact protocol options; the owner logs in
and starts them. Existing sessions/runs/inputs/caches remain untouched.

[Canonical PR278 Research-Cloud folder](https://drive.google.com/drive/folders/1-0y8ETC4xqzfGzSMOnt9BWMJAUkqpNiP):
`~/Local/Research-Cloud/PR-278-hu200-v055-confirmation/`, confirmed under project
folder `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. Closed per-seed full and separate
final-average ZIPs will stream from M4 into this folder on M1, with member hashes
and complete local readback. Archive hashes, restoration commands and separate
native/independent cloud upload acceptance are pending. M4 originals remain.
No synced deletion, eviction or other open-PR cleanup occurs here.

## Guards, supervision and preserved administrative failures

Both research locks are held during each detached seed. Five-second observations
stop after 60 continuous observed seconds above 13 GiB family RSS, critical
pressure, free memory below 8%, or swap growth above 4 GB. Disk below 16 GiB
requests a clean stop immediately. Gaps cannot establish a sustained breach;
owned process creation times are checked before signaling. No other jobs signaled.

Owner-requested **30-minute T3 checks remain enabled** for this campaign only.
The completion webhook returned three HTTP errors after seed 1; it is disabled
and that failure is [preserved](hu200-v055-confirmation-artifacts/completion-callback-failure.json).
The 22:21 CEST periodic check caught the completed audit. No duplicate scientific
job was started. Both schedules will be removed at closeout; no follow-on research.
Private routing capability data is excluded from Git and archives.

Initial disk admission at 35.474 GiB failed before the build/clock; owner-cleared
storage passed fresh admission. The unsupported Python 3.9 artifact-check
invocation and missing psutil in two setup environments are preserved; supported
Python 3.11/research environments pass. A quoting error was found in the
administrative server-launch wrapper; its source/failure is retained and a
corrected launch follows the existing transfer, with no repeated training/export.

Nine focused clock/recipe/key-order/guard tests and staged repository artifact
checks pass. Final checks/review remain for the completed PR. No Slumbot play,
credentials, release, tag, default change or merge; the owner decides merge.
