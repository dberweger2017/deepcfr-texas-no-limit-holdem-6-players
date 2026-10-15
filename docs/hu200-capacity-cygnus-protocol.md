# HU200 table capacity on cygnus-x1

Owner-authorized October 15, 2026. Status: preflight; no large run admitted.
Base: current main `f949b930`. This is an experiment, without a milestone gate.
The owner decides merge and any later external play. No trainer changes, release,
tag, default change, Slumbot, credentials or compute on M4/M1/db-ocean.

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

Before large training, measure per-entry memory and menu lengths in guarded
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

Run all three lineages concurrently, each two busy threads, separate physical
core pairs, own output root/stop file/seed. Total busy threads <=8 at every phase,
including helper work. Training contributes six; only one export/audit/cache
builder or verifier is admitted at once, bounded to at most two additional
threads. Affinity and OMP/MKL/OpenBLAS/Rayon limits are explicit. No 16-thread
probe. Record the new heatsink's observed CPU/VRM/fan behavior and compare with
the qualified pre-heatsink context without claiming a controlled causal test.

Use #292's sensor/thermal semantics in a campaign supervisor that owns the
exclusive cygnus compute lock, with separate monitored process groups per run.
Sample at least each second, fail closed on missing/unreadable CPU, Thermistor
15 or fan1. Admit only below CPU70/VRM85 degrees C. CPU >=85 C terminates load
immediately; VRM >=95 C continuously for30s requests every affected run's clean
stop. Continue supervision through saving; allow up to3600s clean-save grace
for these larger checkpoints, then terminate. A stop is a failed attempt, never
an automatic restart. Keep all partials. No automatic scientific retry.

Publish per-run/aggregate RSS caps from measured admission. Available RAM must
remain >=6GiB; swap growth >256MiB from the one campaign baseline stops;
free disk <50,000,000,000 bytes stops. Resource/sensor/deadline/CPU breaches
terminate immediately; thermal VRM uses the clean-stop path. Log full CPU,
VRM, fan RPM, memory, swap and disk traces, with sample gaps disclosed.

The existing **open #292 root, archives, upload processes and shared inputs
are protected**. Do not signal, copy, move, archive or remove them. Count their
current disk occupancy and possible upload staging growth in admission.
No campaign cleanup may depend on reclaiming their space.

Large continuation command per seed, after parity and capacity admission:

```text
QUALIFIED_LINUX_BINARY train --stack-bb 200 --menu v1 --seed SEED
 --roots-per-seat 1 --average-rule opponent-sampled --card-buckets TABLES
 --max-entries SELECTED_EQUAL_CAP --freeze-at-cap --recovery
 --resume VERIFIED_LARGE_CAP_500M --resume-sha256 VERIFIED_SHA256
 --nodes 20000000000 --milestones 2000000000,5000000000,10000000000
 --out SEED_DIRECTORY/{nodes}.gz --telemetry SEED_DIRECTORY/telemetry.jsonl
 --stop-file SEED_DIRECTORY/stop.request --max-seconds REMAINING_FIXED_BUDGET
```

Checkpoints are at first complete iteration crossing 2B/5B/10B/20B. No target
filename is evidence of reached nodes. All saves and final exports receive full
every-row current/average audits; exports/audits never overlap one another.
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

At the planned cap, reserve18GB per checkpoint: twelve milestone checkpoints
alone consume216GB. Six needed large averages (5B/20B) can add~66GB, nine policy
caches~124GB, restored twins~9.5GB and traces/archives more. Keeping all these
simultaneously **does not fit** above the50GB floor. Admission must publish a
staged budget: full audits and manifested archive acceptance before retiring
any unique checkpoint/export; retain20B recovery and5B/20B evaluation inputs;
create caches only after training; process seeds in parallel only while their
combined resident/backing storage fits. Rebuildable inactive campaign caches
may be retired with regeneration receipts. Open PR originals cannot be deleted
under the standing merged-evidence permission. If safe retention cannot fit,
stop and report rather than borrowing #292 space or silently dropping saves.

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
invocation: its raw download limit is268,435,456 bytes. Preserve that failure;
do not claim restoration or use excluded machines to work around it.

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

New PR holds protocol, report, compact resource/thermal summaries, every failure,
input/retrieval pins and archive receipts. Full raw evidence/checkpoints/exports
belong in the designated Research-Cloud folder with embedded member manifests,
complete local readback, whole hashes and separate current cloud metadata/upload
acceptance. Index actual URLs/IDs and retrieval commands in RESULTS_INDEX.md.
Record cleanup decisions/removed paths/restoration links; no unsupported cleanup,
release, tag, merge or default change. The report must distinguish measured
results, unstarted work, incomplete targets and pending upload acceptance.
