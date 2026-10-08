# HU100 export and audit memory

[PR #202](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/202)
removes policy-sized JSON collections from native export and full accumulator
audit. Every native current/average gzip and Python extraction gzip is **byte
identical** to its frozen baseline for all five HU100 inputs and the uninterrupted
HU20 1B reference; full audit receipts match exactly. No training was launched.

Native export streams averages in checkpoint order and externally sorts current
rows in bounded 65,536-row runs with 32-way, equal-sized merge tiers. Duplicate
keys are checked through the sort. The audit streams the current JSON object and
matches every key through a disposable SQLite index with a 4-MiB cache; extraction
uses the same disk ledger instead of a Python set. Floats retain exact f64/Python
values, normalization, menus and fallback; signed-zero mismatches now reject.
Atomic output staging remains on each output's filesystem. Memory still depends
on the largest individual record and retained metadata, not the entry count;
this is not a hard size cap for arbitrary JSON values.

## Identical-input measurements

M4, 16 GiB, one sequential job, fresh processes, fixed before/after order. Native
export writes current and average together; audit checks every row of the paired
canonical policies; extraction measures the Python path. Family GiB includes the
supervisor and coordinator. All kernel command high-water values and raw 200-ms
family samples are retained in the archive; compact [measurements](hu100-export-audit-memory-artifacts/final-measurements.json)
retain both metrics. One run/case, warm caches and measurement overhead; RSS can
count shared pages repeatedly, miss brief peaks/children, and does not give the
exact simultaneous family peak. Timing is descriptive, not a statistical speedup.

| Actual HU100 nodes / HU20 reference | Operation | Sampled family GiB: before → final | Seconds: before → final |
| --- | --- | ---: | ---: |
| 100691 | export | 0.075 → 0.098 | 0.22 → 0.66 |
| 100691 | audit | 0.184 → 0.137 | 0.70 → 1.40 |
| 100691 | extract | 0.121 → 0.132 | 0.70 → 0.71 |
| 1001382 | export | 0.579 → 0.098 | 2.00 → 4.64 |
| 1001382 | audit | 0.697 → 0.137 | 4.26 → 8.88 |
| 1001382 | extract | 0.179 → 0.135 | 4.03 → 4.23 |
| 5001210 | export | 1.933 → 0.098 | 7.32 → 17.47 |
| 5001210 | audit | 2.519 → 0.138 | 16.52 → 33.44 |
| 5001210 | extract | 0.391 → 0.136 | 14.66 → 15.72 |
| 10001922 | export | 3.227 → 0.100 | 12.41 → 30.09 |
| 10001922 | audit | 4.156 → 0.136 | 27.34 → 56.63 |
| 10001922 | extract | 0.628 → 0.136 | 24.60 → 26.76 |
| 11042440 | export | 3.172 → 0.101 | 13.09 → 31.13 |
| 11042440 | audit | 4.668 → 0.141 | 28.63 → 61.25 |
| 11042440 | extract | 0.643 → 0.136 | 26.13 → 27.82 |
| hu20-reference-1b | export | 4.530 → 0.098 | 19.73 → 42.39 |
| hu20-reference-1b | audit | 5.610 → 0.140 | 40.01 → 80.47 |
| hu20-reference-1b | extract | 0.742 → 0.135 | 39.89 → 42.64 |

The largest HU100 checkpoint has **3,255,387 entries at 11,042,440 actual nodes**;
its requested-1B filename is not a reached milestone. HU20 has 4,319,080 entries.
The original pinned HU20 average remains 142,677,367 bytes / SHA256
`571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`.
The memory reduction trades for extra sort/index I/O and slower export/audit.
The general `FrozenBlueprint` current-policy loader remains eager; average-policy
inference remains compact and unchanged.

## Budget, provenance and validation

[Protocol](../hu100-export-audit-memory.md). #197 was merged before retrieval;
#200 was complete/open on the initial read, then merged at 11:33:04 UTC. Its latest
protocol/report were read at `acef171`; merged source was integrated and relevant
export/audit sources were unchanged. No other agent's roots were used or altered.
All 18 input members (1,179,654,205 bytes) were restored from accepted #197 ZIPs
and matched whole archive, embedded manifest and member hashes. [Exact archive
URLs, member paths/hashes and local retrieval](hu100-export-audit-memory-artifacts/retrieval.json).

Baseline source `7dca996`; initial implementation `a5e63fb`; definitive production
source **`506743412fdbf7ec2cb3f4a1069b68d0effcb538`**. Native binary was compiled
from `ede5aae`; native sources/Cargo lock are byte identical at the definitive
source. Each source tar and binary hash is retained. The common measurement
harness was copied into the isolated baseline source, which imports unchanged
baseline production modules. Later admission logging changes affect only the
harness; production code remains identical to the definitive execution source.

The original persisted **1,800-second clock** includes retrieval, hash verification,
M4 qualification, cost-only timing pilot, all initial/final measurements, complete
equivalence checks and guarded archive/member verification. The 53,743-entry pilot
cost 4.17 seconds; its conservative aggregate quote exceeded the remaining budget,
so each input was admitted separately with 2× cost plus a 60-second reserve,
prioritizing largest HU100 and HU20. No promise to finish all cases was made.
All science/verification finished after **1,550.88 seconds (25.85 minutes)**;
archive verification/copy completed after **1,601.74 seconds (26.70 minutes)**,
within the same clock. Asynchronous cloud acceptance was checked later without
restarting science. Whole-family 10 GiB, pressure level 1 /free ≥15%, full
ceiling +2 GiB at admission, original 440.81-MiB swap baseline/+0.5 GiB, disk floor
15.5 GiB and AC guards were unchanged. [Resources and clocks](hu100-export-audit-memory-artifacts/resources.json).

Two execution corrections remain visible. Initial supervisor setup failed before
any child because the tar checkout lacked Git administration; adding it to this
owned checkout kept the original clock. The first final-source attempt terminated
on fresh system-headroom admission before largest-HU100 extraction. Its failed
receipt/traceback remain; the refused instant was not saved and cannot be rebuilt.
A subsequent recorded snapshot was 82%/normal. A one-use continuation filled only
never-started operations, with the same inputs, source, counts, deadline, baseline
and guards. No failed/partial measurement was retried or guard relaxed. The
measurement helper now persists rejected admission snapshots; a regression checks
that it never spawns the refused child.

M1: 113 focused tests plus the later admission-snapshot regression and nine Rust
library tests passed; M4: 66 final-source
export/audit frozen-fixture tests passed. Corruption tests cover malformed tokens,
chunk boundaries, truncation/CRC, duplicate checkpoint/current keys, missing/extra
keys, wrong probabilities/metadata/lineage/caps/bounds and signed zero. Native
merge tests cover cross-run duplicates and bounded fan-in. Independent source
review round 4 closed all prior findings at `5067434`, ran 67 tests (one collection
fixture excluded under its no-training constraint) plus 955 JSON/chunk combinations.
[Reviews and limitations](hu100-export-audit-memory-artifacts/source-reviews.json).
Independent evidence review at `6933530` reconciled all table rows, equivalence
summaries, clocks/guards and archive acceptance, verified source-tar/production
identity and reproduced capacity arithmetic exactly, with no actionable findings.
It assessed large gzip/ZIP/raw telemetry through receipts rather than rerunning them.
[Evidence review](hu100-export-audit-memory-artifacts/evidence-review.json).
Final-head CI is the PR's live checks; no merge was performed.

## Recommended next growth budget

[Advisory capacity receipt](hu100-export-audit-memory-artifacts/capacity.json),
generated by `scripts.estimate_hu_export_capacity` ([exact inputs/hashes and command](hu100-export-audit-memory-artifacts/capacity-provenance.json)), preserves conservative
**2× plus 10% memory headroom**, 2× time costs and an explicit **2× largest measured
HU100 entry limit**. Historical training/save receipts remain separate from new
serialization measurements; historical after-save RSS is not a training-family
peak. The old dated #197 launch protocol is not reopened by this advisory tool.

Recommend a separately approved **30-minute free-M4 experiment**, resuming the
verified 11.04M parent toward **20M total nodes**, with **6,510,774 entries maximum**,
one terminal checkpoint/audit set, and the same 10-GiB/pressure/swap/disk/AC guards.
A fresh source-bound pilot must fit training, retrieval, save, full export/audit
and closeout before admission. A complete-iteration capacity stop is valid; neither
20M nodes nor an exact table size is guaranteed. Reserve at least
**5.63 GiB for training/save**, **0.62 GiB for serialization/audit**,
**670 seconds for one complete tool set**, **39 seconds for save**,
**9.63 GiB additional disk**, plus retained inputs, the 15.5-GiB
floor and a 180-second closeout reserve. Refuse a fresh quote that cannot fit.

The next memory bottleneck is the native table and its reallocations/save key sort;
audit/index and sort scratch I/O remain time/disk costs. Node-to-entry ratios,
allocator jumps, future menu/record distributions and host pressure limit
extrapolation. These measurements support no 1B/10B budget or completion promise.

## Evidence

Large inputs, every generated export/extraction, raw per-operation/guard telemetry,
all failures/continuations, source tars and binaries are archived under the
[PR202 Research-Cloud folder](https://drive.google.com/drive/folders/10d6DPp8i6uQrW5QaEaJEW364Dd7Vt7nv).
All 309 members and the local/native ZIP hash verify; native uploaded/not-uploading/no-conflict
and connector ID/name/size/parent acceptance match. No remote-byte redownload is claimed.
Later report/review/admission-fix metadata has its own 31-member accepted closeout
ZIP; this M1 seal occurred after the compute cap and restarted no M4 computation.
The final archive receipts and exact restoration command are in [RESULTS_INDEX](../../RESULTS_INDEX.md).
Originals remain on M4 and both isolated M1 checkouts remain; no cleanup, release,
paid compute or further work is scheduled.
