# HU200 table capacity on cygnus-x1

Owner-authorized October 15, 2026. Status: preflight; no large run admitted.
Base: current main `f949b930`. This is an experiment, without a milestone gate.
The owner decides merge and any later external play. No trainer changes, release,
tag, default change, Slumbot, credentials or compute on M4/M1/db-ocean.

October17's owner-approved [export recovery](hu200-capacity-export-recovery.md)
supersedes older storage and deadline wording only for one unfinished1302
export/audit: move inactive owned models to the2TB drive, keep active compute on
NVMe, reserve the complete23GB output plus50GB floor and20GB extra physical
headroom, and credit the exact latest stopped interval once. All native20B
training is already complete; no training or completed audits are repeated.

The owner approved the [accidental power-off recovery](hu200-capacity-power-loss-recovery.md)
on October16: one common5B rollback for all three seeds, fresh output paths,
exact rebuilt10B identity checks, and **exactly+3600s once** to the main deadline.
The original clock stays immutable; a separate ledger sets effective deadline
October19 03:29:18UTC /05:29:18Madrid. Evaluation stays24h. All other guards,
science, resource budgets and no-further-auto-retry restrictions remain intact.

## Question and fixed comparison

Does increasing the retained table from approximately 85.37M to a planned
301,153,954 entries produce a stronger HU200 player at the same 20B nodes?
Train fresh seeds **2026101302, 2026101421, 2026101422**, each paired with its
existing small-cap twin. HU200, K50 #163 cards, menu v1, R=1, linear CFR,
opponent-sampled lifetime average and uniform missing/zero-mass play stay fixed.
Use the existing qualified Linux HU executable, record its source/hash and prove
cross-platform checkpoint identity; the twins' macOS executable cannot execute
on Debian. No native trainer source is changed. Any required trainer change or
unexplained parity difference stops the experiment and is reported.

The motivating #268 contrast +0.09 [-8.36,+8.54] BB/100 is the paired difference
between **5B and 20B returns against the 752M parent**. The 20B model itself
beats that parent +76.95 [+71.93,+81.96]. Capacity is a hypothesis, not an
established explanation of the later inconclusive learning curve.

## Admission and exact identity

A bounded preliminary measurement may re-save the qualified seed1302 500M
checkpoint with zero training steps:300M-cap header only, all rows identical,
full post-write verification and every-row current/average audit. Limit this
operation to5400s, two CPUs,12GiB RSS, the same heat/swap guards and50GB floor.
Then, before large training, measure per-entry memory and menu lengths in guarded
500M prefixes, including saved table RSS and sampled train/save peaks. Account
for fixed card tables, aging of initially regret-free entries, shard index
growth/sort workspaces, three simultaneous residents, one export/audit, the OS
and margin. Keep the planned cap only if the measured and conservative aging
budgets fit; otherwise reduce it **equally for all seeds**, publishing the
measurement, arithmetic and selected cap in this PR before launch. Do not raise
it after seeing training or evaluation outcomes.

For each seed, fresh large-cap 500M is before the old entry ceiling. Re-save it
without any training step, using the unchanged native resume CLI and old
`--max-entries 85373862`; its full compressed checkpoint must match the twin's
recorded 500M SHA256 below. Independently compare every serialized row and all
header fields except `config.max_entries` between the two saves. This binds all
keys, menus, regrets, average accumulators, visits, counters and RNG lineage.
No other header exclusion is permitted. Failure stops all dependent work.

| Seed | Actual 500M nodes | Entries | Old-cap checkpoint SHA256 |
| --- | ---: | ---: | --- |
| 2026101302 | 500000140 | 69084815 | `2c4c1484b7e4ebefca2bb364223d6cd12e886f98b20784b51ff6ac6678f4312d` |
| 2026101421 | 500000695 | 69180237 | `251ab69b5f69785722d2442e628d65e41d578ab20751164d482561cca16f954f` |
| 2026101422 | 500000820 | 69055433 | `45144a01a799b81d185c20e15772fe70793518a3d35b03fa84eb6db0bfa46966` |

Each seed additionally proves a fresh 20M checkpoint equals 10M-resumed-to-20M
at the chosen cap, including exact compressed bytes/SHA256. All checkpoint
writes are independently size/hash checked after native telemetry, fully
streamed through EOF/CRC, and current/average exported and audited on every row.
All gate saves are covered too. #163 full table hashes must match before use.

## Execution, heat and memory

Run all three lineages concurrently on separate physical core pairs, with own
output roots/stop files/seeds: six reserved CPUs, two-worker traversal per
trainer, variable utilization. Total busy CPU reservations <=8 at every phase,
including helpers. The owner approved the October16
[scheduling amendment](hu200-capacity-scheduling-amendment.md): begin with two
concurrent one-CPU audit/export workers, admitting three only after measured
throughput, RAM and disk justification. With all three trainers reserved, at
most two one-CPU helpers fit. Each streaming helper has512MiB family RSS cap,
inside the existing2GiB helper allowance. Cache construction remains serialized.
Affinity and OMP/MKL/OpenBLAS/Rayon limits are explicit. No 16-thread
probe. Record the new heatsink's observed CPU/VRM/fan behavior and compare with
the qualified pre-heatsink context without claiming a controlled causal test.

Use #292's sensor/thermal semantics in a campaign supervisor that owns the
exclusive cygnus compute lock, with separate monitored process groups per run.
Sample at least each second, fail closed on missing/unreadable CPU, Thermistor
15 or fan1. Admit only below CPU70/VRM85 degrees C. CPU >=85 C terminates load
immediately; VRM >=95 C continuously for30s requests every affected run's clean
stop. Continue supervision through saving; allow up to3600s clean-save grace
for these larger checkpoints, then terminate. A guard stop is a failed attempt, never
an automatic restart. The owner-requested October15 BIOS shutdown is recorded
as an administrative interruption; October16 authorization permits continuation
of unfinished verification from the completed checkpoint, preserving all partials. Keep all partials. No automatic scientific retry.

Publish per-run/aggregate RSS caps from measured admission. Available RAM must
remain >=6GiB; swap growth >256MiB from the one campaign baseline stops;
free disk <50,000,000,000 bytes stops. Resource/sensor/deadline/CPU breaches
terminate immediately; thermal VRM uses the clean-stop path. Log full CPU,
VRM, fan RPM, memory, swap and disk traces, with sample gaps disclosed.

The existing **open #292 root, archives, upload processes and shared inputs
are protected**. Do not signal, copy, move, archive or remove them. Count their
current disk occupancy and possible upload staging growth in admission.
No campaign cleanup may depend on reclaiming their space.

Campaign controller runs fresh prefixes and then the four fixed node targets through
separate hash-pinned native resumes. After all global prefix/memory gates pass,
independent seed queues overlap training and audits; each seed's save must pass
every audit before that seed continues. This preserves RNG/table lineage,
proven by the resume gates. No seed enters5B before all three twins restore.
Native continuation command per seed at each TARGET (2B/5B/10B/20B):

```text
QUALIFIED_LINUX_BINARY train --stack-bb 200 --menu v1 --seed SEED
 --roots-per-seat 1 --average-rule opponent-sampled --card-buckets TABLES
 --max-entries 301153954 --freeze-at-cap --recovery
 --resume VERIFIED_PREVIOUS_LARGE_CHECKPOINT --resume-sha256 VERIFIED_SHA256
 --nodes TARGET --out SEED_DIRECTORY/TARGET/checkpoint.gz
 --telemetry SEED_DIRECTORY/TARGET/telemetry.jsonl
 --stop-file SEED_DIRECTORY/TARGET/stop.request --max-seconds REMAINING_FIXED_BUDGET
```

Input archive transfers may run during the small identity gates and2B stage.
Before5B, all three twin archives must be fully hash-verified and restored, and
verified transport replicas retired with receipts. A guarded input wait counts
against the unchanged72h campaign ceiling. Incomplete input transport cannot
admit5B or later stages. Source/executable/table hashes, memory arithmetic,
current disk and swap baseline are published in the admission receipt.

The native loader requires an entry limit at least the saved row count. Once
frozen, complete-iteration overshoot can put that count slightly above the
requested301,153,954. Subsequent resumes pass max(selected cap, saved frozen
entries) only to admit those existing rows; the persisted frozen state continues
to forbid new keys. Record the exact per-save CLI value and overshoot.

Checkpoints are at first complete iteration crossing 2B/5B/10B/20B. No target
filename is evidence of reached nodes. All saves and final exports receive full
every-row current/average audits; owner-approved concurrent workers and
cross-seed training overlap follow the amendment and one global watchdog.
Record per-save street/role missing training shares, zero visits and visits/key.

Initial runtime allowance: at most72h for gates/training/export/audit, followed
by at most24h for cache construction, pilots, final evaluation/replay/reproduction.
These are ceilings, not throughput claims. Publish measured projections and
an updated ETA before admitting each phase. Archive/upload ETA is separate and
bandwidth-dependent. No paid compute. No outcome-driven budget extension.

## Disk and retention

Initial host available353,836,523,520 bytes includes all existing files.
#292 root occupies58,415,280,128 bytes at the read-only October15 inventory;
its uploads and archives remain untouched. Plan a fresh campaign root under
`/home/dberweger/hu200-capacity-20261015` and ignored local research paths.

At the planned cap, reserve18GB per checkpoint,11.5GB per retained large
average and18.5GB per large cache. Reserve10GB for the restored twins,10GB for
traces/receipts,20GB for staging/other-host growth and the50GB floor. Treat these
as estimates and check actual bytes before each stage; stop before any admitted
write could cross the floor. No reclamation of #292 is assumed.

The owner's “keep what evaluation and recovery need” instruction authorizes
retiring this campaign's redundant 2B/10B checkpoints after full audits and an
audited successor. Preserve their complete diagnostics, hashes, commands and
failure records. Keep5B averages for evaluation; retain the last audited recovery
checkpoint until its successor audits. After20B audits, retain20B recovery and
both5B/20B averages;5B recovery can then retire. This applies only to this new
campaign, never another open PR. Keep all incomplete/failed attempts.

The October16 restart has263.760GB free on NVMe; the original339GB plan is
superseded. The added2TB drive is excluded from training/evaluation/cache/scratch paths and
scientific budgets. The owner subsequently authorized only the final unpacked
retention copy described below. Existing
#292 occupancy remains58.415GB; unrelated preservation data remains untouched.

Full audits compare all three strictly sorted streams directly, reusing the
existing validators, exact-number checks and normalization math. Strict key
order proves uniqueness, and all three streams must reach validated EOF/CRC.
This removes the table-sized SQLite workspace; unordered files fail rather
than selecting a weaker check. Differential and corruption tests must pass
before this auditor is used. Native trainer/export source and binary stay fixed.

Retire each old seed recovery and disposable current export immediately after
that seed's successor audits. Independent seed queues may overlap within the
same259GB budget; reserve all prospective outputs globally before writers start,
reducing concurrency when they do not fit. A conservative serialized
training peak is six20GB checkpoints120GB + three retained5B averages34.5GB +
one new20B average11.5GB + one current10GB =176GB. Add restored twins10GB,
training traces3GB, staging/other-host growth20GB and the50GB floor:
**259GB**, within263.760GB at restart. Lower phases have fewer retained averages.
Actual output sizes and remaining free space are admitted before every write;
a budget miss stops the campaign. The cap is not lowered to fit disk.

Evaluation initially admits **two seeds in parallel**, then the third, preserving
the owner's seed parallelism within available disk. Only two large caches37GB
and two twin caches10.4GB coexist; finish each20B direct/panel pair before
retiring its rebuildable cache and creating the5B cache. Keep all final recovery
checkpoints and both5B/20B averages. Worst resident model files129GB, twins10GB,
caches47.4GB, traces7GB, staging/growth15GB and floor50GB total**258.4GB**.
Use actual file/cache manifests to tighten or reject this admission after training.
Every policy cache is built once, and compatible matches reuse its loaded registry.

Seal/archive one seed at a time, up to41GB additional per seed; never duplicate
all three simultaneously. Archive all retained recovery/evaluation models and
all raw receipts/traces/failures; intermediate retired checkpoint bytes are not
required for evaluation or recovery. Manifests explicitly mark these retirements
and the retained replacement recovery. Full local member readback and separate
Drive upload acceptance precede disposal of transport copies. Record receipts
and indexed actual archive/model/member hashes and restoration commands.

## Restored twin inputs

Restore from Drive to ignored bb128 inputs, verify whole archive/manifest/member
hashes and match audited headers/parents. Existing source archives stay intact.

| Seed | 20B average SHA256 |
| --- | --- |
| 2026101302 | `9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6` |
| 2026101421 | `c08d86a9615022df7015d34e5dc3de2705bbb8346ed6976bd72284e71d91f444` |
| 2026101422 | `b191c5ed8b943d138e6076fb4a1b0117c98964d92dd2d6253e157f696f691f6f` |

Restoration records are #265's model index and #278's per-seed model indexes.
The Drive connector rejected the3,162,672,661-byte seed1421 archive before
invocation: its raw download limit is268,435,456 bytes. Preserve that failure.
Use the established read-only synced Drive archive to SSH transfer route for
file I/O only; all hashes, extraction and model/scientific work run on bb128.
No other campaign working originals or upload processes are involved.

## Evaluation fixed before final hands

Per seed: large20B versus own85M20B, translation512states/128events on **both**,
**131,072 independent direct duplicate blocks** (two rotated hands/block).
Also large5B versus own85M20B, descriptive, **131,072 blocks**.
Seven scripts: random, check_call, tight_aggressive, loose_aggressive,
pot_pressure, half_pot_pressure, matched_pot_pressure; **16,384 blocks each**,
large20B and twin20B on the same deals/role streams (four hands/block).
Per seed983,040 primary hands; total2,949,120, plus complete reproduction.

Proposed fresh roots: each seed in listed order gets pilot direct/panel roots
202610160301/302,311/312,321/322; final20B/5B/panel roots
202610160303/304/305,313/314/315,323/324/325 respectively.
Reserve against live main and every open PR before use. Pilots:1024 direct
blocks and64 panel blocks per script, cost/variance only, disjoint and unpooled.
Commit model/cache/source hashes, final roots, measured costs and memory/disk
admission before any final hands; counts never change because of outcomes.

Build each model cache once, read-only afterwards. Reuse the existing harness's
registry parameter for as many compatible matches per load as it allows.
Parallelize seeds within <=8 busy threads and actual RAM/disk admission; serialize
cache construction. No new trainer code or alternate inference implementation.

Independent pinned-engine replay and exact full-match reproduction of **every
final hand** precede a global all-results readout barrier. Withhold outcomes
until all fixed contrasts pass. Failures retain all partial data and roots.
Report per-seed conditional paired Student-t95% intervals over independent
blocks, absolute and paired panel scores, and played missing/zero-mass shares
by street/model. Report three training seeds individually and their dispersion;
do not present millions of hands as millions of training-seed observations.
No pass/fail strength gate or model selection. Owner decides later deployment.

## Evidence and closeout

Owner amendment, October16: retain an additional **unpacked** copy of every
final uploaded evidence bundle at
`/mnt/data2tb/research/PR-301-hu200-capacity-cygnus/` on bb128. Preserve the
archive member hierarchy and manifests, using distinct bundle subdirectories
where necessary; native checkpoint/export formats remain unchanged. This is a
final retention destination only, additional to Drive. Recheck PR status,
mounted device identity, space and guards before copying; never write through
an absent mount or overwrite existing evidence. Verify every retained member's
size/SHA256 against the sealed archive manifest and index the actual paths and
verification receipt. Keep the copy at closeout. All scientific budgets,
deadlines, guards and protected dependencies remain unchanged.
[Declared retention plan](reports/hu200-capacity-cygnus-artifacts/local-retention-plan.json).

New PR holds protocol, report, compact resource/thermal summaries, every failure,
input/retrieval pins and archive receipts. Full raw evidence/checkpoints/exports
belong in the designated Research-Cloud folder with embedded member manifests,
complete local readback, whole hashes and separate current cloud metadata/upload
acceptance. Index actual URLs/IDs and retrieval commands in RESULTS_INDEX.md.
Record cleanup decisions/removed paths/restoration links; no unsupported cleanup,
release, tag, merge or default change. The report must distinguish measured
results, unstarted work, incomplete targets and pending upload acceptance.

## Owner interruption and restarted hardware

The owner stopped bb128 for BIOS/fan work on October15 and explicitly resumed
this task on October16. Preserve the old resource stream, including its truncated
last JSONL row, and both partial export scratch trees and archive transfers.
The completed500M checkpoint's every-row identity receipt remains evidence;
exports/audit were not completed and no large training started.

The preliminary5400s allowance carries forward recorded pre-stop work; only its
unused active allowance is available for resumed export/audit. No training is
rerun for this administrative stop. The72h gates/training/audit clock had not
started, so it still begins at first admitted campaign gate, followed by the
unchanged24h evaluation ceiling. All owner-stop downtime is excluded; publish
new absolute deadlines and ETA when those clocks actually begin. A new boot's
observed preexisting swap is recorded as its baseline, not charged as growth.

The owner added five fans, changed the fan curve and enabled XMP3200. Record
owner-reported settings separately from readback; verify unchanged executable,
K50 tables and checkpoint bytes and rerun the declared fresh/resume identity
gates under XMP before dependent large continuation. Keep <=8 busy threads,
CPU85C/VRM95C limits and the owner's accepted uncompleted memtest86+ status.
No writes, mounts, formatting or experiment use of the new hard drive.

Current campaign continuation uses only stage-completion webhook events into
this existing thread: verified prefix/memory gates, restored twins, audited
milestones and terminal status. The observer is metadata-only, affined6–7,
bounded by the training deadline+1h; it cannot start or restart scientific work.
No recurring timer or future research is scheduled. Private event endpoint files
are excluded from all evidence archives/Git; nonsecret delivery/failure receipts
are retained. Delete the task and private files after this experiment closes.
