# v0.5.5 confirmation seeds: training in progress

[Draft PR278](https://github.com/dberweger2017/CFR-NLHE-6max/pull/278).

Seed **2026101421 is training**, after owner-cleared storage passed fresh
admission with **49.235 GiB free**. Seed **2026101422 is unstarted** and will
start only after seed 1's final exports/audits, with at least 5.5 hours left.
The first detached build began **October 14 15:57:16 UTC /17:57:16 CEST**;
the fixed deadline is **October 15 03:57:16 UTC /05:57:16 CEST**.
Training must stop by **01:42:16 UTC /03:42:16 CEST** to reserve the terminal
save and exports/audits. Neither seed is yet labelled 20B or complete.

The build uses current main `9c449929e4b41e8c60383e682bea1e0e1e4c01d5`,
including #275. Its binary exactly matches #265: SHA256
`4f6abcbb954c3c744448c2a3542ab3164e0e1a015cb05ecebeda077525991cd4`,
1,492,512 bytes. All three #163 K50 tables were copied into the ignored M4
results root, SHA256 checked at source and destination, and made read-only.
Both locks remain held during detached training. One campaign swap baseline,
five-second resource readings, sustained-memory guards and immediate disk-stop
requests are retained on M4. The wall guard bounds checkpoint loading as well
as native training; native commands also receive remaining-clock max-seconds.

[Readmission](hu200-v055-confirmation-artifacts/readmission.json),
[clock/baseline](hu200-v055-confirmation-artifacts/baseline.json),
[build](hu200-v055-confirmation-artifacts/build-receipt.json),
[table provenance](hu200-v055-confirmation-artifacts/inputs.json).

## Preserved initial admission block

At **2026-10-14 15:30:37 UTC /17:30:37 CEST**, `ssh m4` identified
`dberwegerch-m4.local`, Mac16,1, **17,179,869,184 bytes /16 GiB RAM**.
The attached workstation is the M1; no training was run there.

- Both `~/Local/.hu20-m4-research.lock` and
  `~/Local/.hu100-m4-research.lock` admitted nonblocking exclusive `flock`
  simultaneously, held only during observation and released on exit.
- Process enumeration found no campaign matches. The exact process command,
  search terms and transport exclusions are in the receipt.
- `shutil.disk_usage(~/Local)` reported **35.474 GiB free**, **9.526 GiB short**
  of the required 45 GiB; a separate `df -k` observation is retained.
- Live GitHub checks found #272 **OPEN** and #274/#275 **MERGED**. Compute
  availability does not authorize changes to #272's research root.

[Machine-readable admission receipt](hu200-v055-confirmation-artifacts/admission.json).
No jobs were signaled and no research files were copied, archived, moved,
deleted or evicted. The first admission observation held no reservation.

## Scope and outstanding work

M4 worktree: `~/Local/hu200-v055-confirmation-20261014`, detached at the
pinned current main; ignored results: `results/hu200-v055-confirmation`.
Controller source and exact native commands are retained. Both seeds follow
#250's fresh 3B target/save schedule/cap, followed by #265's hash-pinned frozen
continuation to 20B on that seed's actual terminal key count. Per-save byte,
counter and sorted serialized key verification precedes final current/average
exports and every-row accumulator/regret audits with uniform zero-mass play.

Only #265-authorized redundant 2B/10B checkpoints may be removed after a
successor passes full byte/count/key checks; cap, 5B and final remain retained.
Disk headroom for the second seed and both archives will be checked before
proceeding. No other PR root or synced files are changed.

Server preparation is **not started** and depends on successful final audits.
The offline preparation controller checks model/table SHA256s, builds a fresh
verified cache and loads it read-only, with five-second memory readings and an
immediate stop below 2 GB available. The owner alone logs in and starts the
prepared 10,000-completed-hand launchers. No Slumbot play, credentials,
release, tag or default changes are authorized here.

Research-Cloud ZIPs, complete member readback, upload acceptance and restoration
index remain outstanding. The draft PR stays for the owner's merge decision.

Validation: nine clock/recipe/key-order/guard tests and repository artifact
checks pass. Initial setup found two M4 environments without psutil before the
clock/build; the existing PR207 research environment supplies the monitoring
and audit dependencies. The earlier Python 3.9 artifact-check failure is
preserved; supported Python 3.11 passes. No scientific retry occurred.

## Continuation and archive preparation

A completion webhook resumes this same PR task after the detached seed-1
supervisor exits. It requests no new research, retries or extra budget and
will be removed at closeout. The owner additionally requested 30-minute T3
wake-ups on October 14; that interval is enabled for health/continuation checks
in this thread, with its first next wake at **16:51:06 UTC /18:51:06 CEST**.
Both the interval and completion webhook will be removed at closeout. Its private routing file is excluded from archives
and Git; the observer's five-second process/RSS readings and delivery receipt
are retained separately from the scientific family readings. The observer is
about 24 MB RSS and does no training. Seed 2 will receive its own completion
observation after admission. The owner is clearing additional disk headroom
while seed 1 runs; all resource thresholds and the original deadline remain fixed.

Canonical archive folder: [PR278 Research-Cloud](https://drive.google.com/drive/folders/1-0y8ETC4xqzfGzSMOnt9BWMJAUkqpNiP),
`~/Local/Research-Cloud/PR-278-hu200-v055-confirmation/`. It is confirmed by
connector ID/name/direct parent beneath project folder
`188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`; no archive uploads exist yet. Closed
per-seed evidence will stream into full and separate final-average ZIPs on M1,
with member hashing during transfer and complete local archive readback. This
avoids an extra payload copy on the training disk. M4 originals remain.

First completed save: seed 2026101421's 500M target contains **500,000,695
actual nodes /320,364 iterations /69,180,237 entries**, 2,730,179,385 bytes,
236.14 write seconds. Full closeout verification and statistics are pending.
No guard stop was recorded through 16:05:43 UTC; this is a progress snapshot.

At **16:20:53 UTC /18:20:53 CEST**, the first seed's growing stage is verified
at **754,080,075 actual nodes /496,692 iterations /85,373,916 entries**,
54 entries beyond the nominal cap. Frozen continuation is active on that seed's
actual terminal key count. [Stage-1 byte/count/key receipt](hu200-v055-confirmation-artifacts/seed-2026101421-stage1-verified.json).
The guard's current snapshot is 8.380 GiB family RSS, 84% free memory,
72.427 GiB free disk and swap 16,777,216 bytes below the original baseline,
with no stop. These are instantaneous readings, not campaign maxima.

## Latest periodic check

The **October 14 17:51 UTC /19:51 CEST** check confirms the original
supervisor identity is live and healthy in frozen training. The latest fully
verified save is **10,000,000,162 actual nodes /8,053,874 iterations**,
retaining **85,373,916 entries** and the exact stage-1 sorted serialized
key digest. Mean traverser visits/key is **19.0021**
and zero-visit share **21.19%**. The 20B save is not yet
available; no progress between saves is inferred.
Current family RSS is **8.68 GiB**, free memory
**80%**, disk **64.07 GiB**, and swap is below the
original baseline. No guard stop or failure latch exists. Seed 2 and server
preparation remain unstarted. Fixed deadline and reserves are unchanged.
[Health receipt](hu200-v055-confirmation-artifacts/health/20261014T1751Z.json),
[verified learning curve](hu200-v055-confirmation-artifacts/seed-2026101421-learning-curve.json).
