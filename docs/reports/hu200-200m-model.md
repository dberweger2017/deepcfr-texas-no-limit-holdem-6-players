# HU200 200M pipeline model

One fresh M1 run reached **200,001,087 actual nodes /111,844 iterations /
31,582,921 entries**, with recoverable 100M/200M saves, byte-identical recovery,
full checkpoint/export audits and independently replayed internal play.
A **200M-node model is far below any release budget and exists to exercise the
pipeline** for a first live Slumbot baseline in a later PR. This PR makes no
Slumbot requests, runs no external matches and makes no strength, convergence,
release, adoption or default-policy claim.

## Source and recipe

The trainer and readers are unchanged from current main
`d5c113f76efa0b4c63dc5956b278a4acbe53f718` (#247). Execution started at the
documentation-only descendant `16d762a569933abf0411dba787e2e3a2b26004a7`
(the authoritative full revision is in the archived `admission.json`).
The native source tree is `25aafbce93956aa54cf885c7275cbe73581810e0`.
`python -m scripts.build_native_trainer` completed in 14.77s; its
[receipt](hu200-200m-model-artifacts/build-receipt.json) records Rust, linker,
lockfile and binary SHA256
`8c50f67b6628a33f3fddd02a692a15ff06c9bfde82b9e6db0ba594095d32b94a`.
All four focused HU200 tests passed in 5.14s; no game, information-access or
regret-math source changed.

The recipe is **seed 2026101301**, **200 BB**, **opponent-sampled average**,
**linear weighting**, **one root per seat**, **v1 cards**, and the existing
min/pot/conditional-jam menu. The command used `--nodes 200000000`,
`--milestones 100000000`, `--recovery` and `--out HU200-{nodes}.gz`.
Complete iterations account for the 3,185-node /1,087-node milestone overshoots.
The exact command, environment, helper and native runtime are archived.

## Recovery and measured training

A fresh uninterrupted 20M run and a fresh 10M run resumed to 20M were compared
chunk by chunk: all **227,890,921 compressed bytes are identical**, SHA256
`c60977f8cff235886914377a2c8483a25af1bb894d731aaded4e60ab11c475f5`.
Both end at **20,002,603 nodes /8,687 iterations**.
[Recovery receipt](hu200-200m-model-artifacts/recovery.json).
The production 200M run then started fresh, with no parent checkpoint.

| Endpoint | Actual nodes | Iterations | Entries | Mean traverser visits/key | Zero-visit fraction |
|---|---:|---:|---:|---:|---:|
| #216 100M, seed 2026100905 | 100,000,034 | 52,136 | 20,575,288 | 0.9616 | 53.3652% |
| Fresh 100M, seed 2026101301 | 100,003,185 | 51,354 | 20,495,916 | 0.9658 | 53.3542% |
| Fresh 200M, seed 2026101301 | 200,001,087 | 111,844 | 31,582,921 | 1.2576 | 50.7704% |

The [#216 row](hu200-feasibility.md) is its recorded 100M measurement; its model
was not retrieved or re-audited here. These are deterministic stored-key counts,
not playing-strength measurements. At 200M, **16,034,762** keys have zero
traverser visits, while **10,671,093** keys have zero average mass; the two
quantities have different meanings. Positive average mass exists on
**20,911,828** keys. [Training and visit receipt](hu200-200m-model-artifacts/training.json).

| Run/timer scope | Native elapsed including saves (s) | Native save (s) | Elapsed outside saves (s) | Supervised train/save (s) | Sampled family peak (GiB) | Kernel command peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| #216 final 20M→100M resume segment | 143.51 | 89.72 | 53.79 | 156.39 | 1.69 | 1.67 |
| Fresh 0→200M, both milestone saves | 462.15 | 249.12 | 213.03 | 470.47 | 2.64 | 2.62 |

The new saves took **102.76s at 100M** and **146.36s at 200M**. Elapsed outside
save timers includes traversal and intermediate hashing/diagnostics; supervised
wall time also includes startup and final bookkeeping. The #216 timer covers
its last resumed segment, while the fresh timer covers the entire 200M run.
No equal-scope performance comparison is inferred from these rows.

## Export, full audit and compact inference

`export --zero-mass uniform` wrote both current and average in **496.30s**.
The existing native-export auditor, `src.diagnostics.cfr_average.audit`, then
verified **all 31,582,921 rows of each checkpoint/current/average file** in
**866.94s**, with zero mismatches. It checks finite stored totals, normalization,
current regret-matched probabilities, complete/unique key coverage and lineage;
historical accumulator increments are not reconstructed.
[Full audit](hu200-200m-model-artifacts/full-audit.json).

The hash-verified average loaded with the existing
`AveragePolicy(..., expected_schema=HU200_SCHEMA, sorted_rows=True)` compact
path in **177.25s**. Its game is
`hu200-native-reopening-200bb-52card-no-ante-rake-v1`, schema
`hu200-native-reopening-ordered-history-card-v1`, table stacks
`[20000, 20000]` with a 100-chip BB (**200 BB**), two seats and translation off.
The sorted-row reader verifies strict key ordering before exposing compact views.

Exactly **32 duplicate blocks /64 internal hands** against `check_call` were
played. Every action, engine event and settlement was replayed, all selected
policy actions reproduced, and every hand conserved 40,000 chips. Independent
native replay checked **452 decisions /64 hands**, including bounds, menus,
keys and final stacks, with **zero mismatched hands**. Journal digests, unique
block/seat coordinates and action/fixture counts also agree. The **211 candidate
decisions** all hit positive-mass keys in this selected small sample; this does
not establish broad history support. Play plus Python replay took **0.118s**,
the supervised load/play operation **179.08s**. No winrate is reported.
[Smoke receipt](hu200-200m-model-artifacts/smoke.json).

## M1 guards and closeout

The owner's 16 GiB Apple M1 ran at **nice 15**, under the shared exclusive M1
research lock. Process inventory used `psutil.process_iter`; unrelated macOS
`AccessDenied` results were retained and skipped, with no `/bin/ps` use. Owned
process identity/access failures stop the guard. No competing research worker
was found; the M4 and other PR work roots were untouched.

Soft/hard owned-family RSS limits were **6/7 GiB**; normal/warning pressure was
allowed, critical/unknown pressure stopped work, free memory had to remain
**≥15%** and free disk **≥16 GiB**. The soft training stop uses a stop file at a
complete iteration and a terminal latch blocks dependent science. Hard breaches
terminate only the owned process group and retain partial files.

Across **7,456** samples through local archival, the maximum sampled
family RSS was **2.64 GiB** and maximum
kernel command peak **2.62 GiB**.
Minimum free memory was **30%**, pressure states
were **normal/warning only**, and minimum disk **38,702,837,760
bytes /36.04 GiB**. Target sampling was 250ms plus
host queries; the largest observed gap was **1.757s**.
Sampled family and kernel command peaks have distinct scopes and do not give an
exact family supremum. [Resources](hu200-200m-model-artifacts/resources.json).

The initial 5.9 GiB disk observation refused build/training; the owner freed
space and execution admitted at 41,858,174,976 bytes free. The saved initial
refusal and a rejected exploratory configuration probe are preserved. All
science/archive operations completed without a guard stop, failed science stage
or retry. Science closed in **35.88min**, the local
closeout receipt at **39.66min**, inside the 60-minute
cap. ZIP creation/readback took **20.03s**. No originals were deleted or evicted.

## Evidence, model index and retention

Both archives are locally verified under
`~/Local/Research-Cloud/PR-248-hu200-200m-model/`.

| Archive | Bytes | Whole SHA256 |
|---|---:|---|
| `HU200-200M-M1-evidence-20261013.zip` | 4,132,696,783 | `f3f169bd93e67d6b72d195747bff5989789ebd4a0eaafdaf4984af0481aef794` |
| `HU200-200M-average-only-20261013.zip` | 800,092,256 | `6930374dc65ac59f7167c62e5501275926732cec257d19a04bcebf8af3cc72d2` |

All **73 evidence members** and the average-only ZIP's **single member** passed
local byte-count/SHA256 readback. Embedded `ARCHIVE-MANIFEST.json` SHA256 is
`6da6a8b48e65f838d5cdcdeb61e1f556d4a89a843f3dae8dd013a31469a0ab12`;
its source paths/mtime identities preserve restoration and cleanup provenance.
The evidence ZIP includes both milestones, all recovery files, both exports,
audit, smoke journals/fixtures, raw resources, build receipt, source snapshot and
exact helpers/runtime. Later archive/upload/report receipts live in Git outside
the frozen science bundle.

**Native Drive upload acceptance and independent cloud metadata are pending.**
Final `RESULTS_INDEX.md` and model-index integration will record actual Drive
locators, archive/member hashes and retrieval commands before PR readiness.
No remote archive bytes have been downloaded or verified.

The local average remains in the ignored directory
`/Users/dberweger/Local/hu200-200m-model-20261013/results/hu200-200m-20261013/HU200-200000000.gz.average.gz`:
**800,092,100 bytes**, SHA256
`b7c15eb642f2edec82df04983b28a8f80b8c3db310f3d6ea53e21aab6d120402`.
The later PR must verify that hash before use. All other own originals are also
retained. No binaries enter Git, no synced files are removed, and no live run,
unattended continuation or cleanup is scheduled. Merge requires a reviewed diff,
completed storage records and green current-head checks.
