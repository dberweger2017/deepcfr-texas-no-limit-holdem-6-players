# HU200 K50 frozen learning curve

[PR265](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/265)
· [Protocol](../hu200-k50-frozen-protocol.md)

**20B training complete; export audits and archival pending.** Current-main M4 training
exactly reproduces #250's fresh and resumed 20M checkpoints, its 500M save and
its 752,593,175-node cap terminal. Frozen continuation started October 13 at
17:38:41 UTC on the same 85,373,907-key table and saved its final checkpoint at
20:46:01 UTC (22:46:01 Madrid), reaching **20,000,002,032 nodes /16,670,844
iterations** with all keys unchanged. The undertraining hypothesis
remains open until the owner's separate playing evaluation. Tonight measures
stored-key visit density, averaging and missing-decision diagnostics.

## Fixed recipe and reproduction

HU200, K50 cards, menu v1, seed 2026101302, one root per seat, opponent-sampled
averaging and linear CFR. Build source is current main
`c53f7935ebbc45180e179e5a3aea74af48bff783`, including merged frozen-table
support #262. The path-independent M4 binary is 1,492,512 bytes, SHA256
`4f6abcbb954c3c744448c2a3542ab3164e0e1a015cb05ecebeda077525991cd4`.
[Build receipt](hu200-k50-frozen-artifacts/build-receipt.json).

#250's tables were copied read-only into this campaign's ignored, unsynced
root. Source and destination sizes and all three full SHA256s match #163.
Live owners163/250/260/262 were merged; no campaign processes remained from
#260. [Inputs and table retrieval provenance](hu200-k50-frozen-artifacts/inputs.json).
#250's checkpoint was regenerated locally on M4.

Fresh20M and 10M-resumed20M both reproduce **20,002,695 nodes /8,728 iterations
/7,883,678 entries**, **270,143,830 bytes**, SHA256
`4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`.
[Recovery receipt](hu200-k50-frozen-artifacts/gates-complete.json).

The fresh main run used #250's exact 3B target and 500M/1B/1.5B/2B/2.5B saves,
the nominal **85,373,862-entry** ceiling and no freeze flag. Its terminal is
**752,593,175 nodes /493,496 iterations /85,373,907 entries**, exactly
**3,442,723,146 bytes**, SHA256
`4b25e8029ed0ebfbeeb42ac02dee3179303e7f2300c23c21d32d27382f390da4`.
The 500M checkpoint also matches #250's full SHA256
`2c4c1484b7e4ebefca2bb364223d6cd12e886f98b20784b51ff6ac6678f4312d`.
[Byte/count and sorted-key verification](hu200-k50-frozen-artifacts/stage1-verified.json).

SHA256 of the concatenated sorted 32-character key list is
`0318ed5894960151e5e55f8fde538bdfe1eda1d3ec496877eaf735f34ef4bf3c`.
All **85,373,907** keys were scanned, with strict sorted uniqueness, in 42.06 s.

## Learning curve

The frozen 2B, 5B, 10B and 20B saves pass full byte/count and sorted-key checks.
Every frozen save retains exactly the stage-1 sorted key list and 85,373,907
entries. The native run returned zero with no stop request; this is a reached
20B result. [Terminal receipt](hu200-k50-frozen-artifacts/frozen-complete.json).
[Learning-curve receipts](hu200-k50-frozen-artifacts/learning-curve.json).

| Save | Actual nodes | Iterations | Entries | Mean traverser visits/key | Zero-visit share | RSS after save GiB | Write s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reproduced 500M | 500,000,140 | 321,139 | 69,084,815 | 1.4335 | 46.45% | 3.340 | 242.62 |
| Reproduced cap | 752,593,175 | 493,496 | 85,373,907 | 1.7446 | 44.29% | 2.865 | 299.37 |
| Frozen 2B | 2,000,001,632 | 1,401,734 | 85,373,907 | 4.1448 | 34.90% | 2.440 | 284.55 |
| Frozen 5B | 5,000,000,031 | 3,748,464 | 85,373,907 | 9.7324 | 26.41% | 2.730 | 299.92 |
| Frozen 10B | 10,000,002,915 | 7,924,855 | 85,373,907 | 18.9536 | 21.00% | 2.107 | 314.41 |
| Frozen 20B | 20,000,002,032 | 16,670,844 | 85,373,907 | 37.3753 | 16.57% | 1.704 | 328.63 |

At 2B, cumulative discarded traverser visits are **38,421,899** and discarded
merged updates **87,931,122**; the freeze began at iteration 493,496. The 2B
checkpoint is 3,905,686,824 bytes, SHA256
`cd77408951361bffd5a0df992db4bc9fd2b21cc1282f6fb3978730a9df8fc02d`.

At 5B, cumulative discarded traverser visits reach **134,328,929** and merged
updates **304,586,889**; this interval adds **95,907,030 visits /216,655,767
updates**. The checkpoint is **4,357,232,664 bytes**, SHA256
`56244a81d90575eb4c7f73992ccdf36a304d778923076850cadc1531afe21721`.

At 10B, cumulative discarded traverser visits reach **284,862,525** and merged
updates **630,701,771**; this interval adds **150,533,596 visits /326,114,882
updates**. The checkpoint is **4,674,655,846 bytes**, SHA256
`da85e75ecd12c316d0de86d670220e2e228de349185bb4ce129ffb579fab6551`.

At 20B, cumulative discarded traverser visits reach **566,739,555** and merged
updates **1,223,130,425**; this interval adds **281,877,030 visits /592,428,654
updates**. The checkpoint is **4,960,844,586 bytes**, SHA256
`c5929bda17457911347e47cd77a83dd8b34618a5542731bed96ece1fb4288b6f`.

Interval missing-decision shares rise in every street/role in the first frozen
segment compared with the final growing-table segment. At 5B, all street/role
shares remain above that pre-cap baseline. Across the first two frozen
intervals, all traverser shares rise; opponent preflop/flop/turn
shares rise while river falls slightly from 14.679% to 14.442%. From 5B to 10B,
every street/role share falls. Traverser shares remain above the pre-cap
baseline; opponent river falls below it (12.416% versus 13.050%). Every
street/role share falls again in the 10B-to-20B interval. Final traverser
preflop and river shares are below the pre-cap baseline; traverser flop/turn
and opponent preflop/flop/turn remain above it. Opponent river remains below
baseline. The shares thus rise immediately after freezing, then decline as
the policy evolves; they do not rise monotonically. These are different
policy/time intervals, not paired playing evaluations.
Pre-freeze misses are recorded before new keys receive their
complete-iteration updates.

| Save interval | Role | Preflop | Flop | Turn | River |
|---|---|---:|---:|---:|---:|
| 500M to cap | Traverser | 0.937% | 2.151% | 8.076% | 19.965% |
| Cap to 2B frozen | Traverser | 1.111% | 2.462% | 8.757% | 20.177% |
| 500M to cap | Opponent | 0.085% | 0.630% | 3.568% | 13.050% |
| Cap to 2B frozen | Opponent | 0.581% | 1.289% | 4.995% | 14.679% |
| 2B to 5B frozen | Traverser | 1.132% | 2.841% | 9.804% | 21.212% |
| 2B to 5B frozen | Opponent | 0.643% | 1.394% | 5.167% | 14.442% |
| 5B to 10B frozen | Traverser | 1.021% | 2.759% | 9.724% | 20.265% |
| 5B to 10B frozen | Opponent | 0.484% | 1.096% | 4.400% | 12.416% |
| 10B to 20B frozen | Traverser | 0.669% | 2.501% | 9.383% | 19.180% |
| 10B to 20B frozen | Opponent | 0.222% | 0.811% | 3.751% | 10.916% |

The first frozen segment runs at **2,023,708 nodes/s** excluding the write,
from native elapsed 616.3975 s. The first 2B write began 10m16.4s after loading;
the native CLI emits counters only at saves, so no exact 10-minute node count
was available. The retained first-window observation and first completed
segment establish the early rate. Native continuation completed in
**11,153.1061 s including its four writes**, inside the admitted training
budget. The fixed recipe and milestones remained unchanged.
[Early throughput](hu200-k50-frozen-artifacts/early-throughput.json),
[first-window timing](hu200-k50-frozen-artifacts/first-ten-minute-throughput.json).

Missing-key decisions use uniform play for both players forever under this
restricted-game continuation. Visit density describes retained keys; tomorrow's
evaluation is needed to judge the effect on play.

## Frozen admission, time and guards

Resume the pinned terminal with `--freeze-at-cap --max-entries 85373907`,
recovery, a fresh stop file and telemetry. The cumulative node target is 20B,
with intermediate 2B/5B/10B saves. [Exact command and reserve](hu200-k50-frozen-artifacts/frozen-command.json),
[launch admission](hu200-k50-frozen-artifacts/stage2-admission.json).

The fixed nine-hour clock starts at **17:18:59 UTC** with the detached build
and ends October 14 **02:18:59 UTC /04:18:59 Madrid**. Native stage-2
`--max-seconds` is **23,117.93487 s**; the M4 wall guard also bounds loading and
requests a clean training stop at **00:03:59 UTC**, reserving 15 minutes for the
terminal write and two hours for exports/audits. A time-stopped result will
remain incomplete with its last complete milestone retained.

All scientific work and five-second supervision run detached on M4. RSS above
13 GiB, critical pressure, free memory below 8% or swap growth above 4 GB must
persist for 60 continuous observed seconds. Warning pressure is allowed; disk
below 16 GiB immediately requests a clean native stop. Creation-time checks
protect process ownership before signaling. Retention hardlinks preserve the
last complete milestone across an incomplete terminal; they add no model bytes
on disk and are removed when a verified 20B terminal is also the final complete
milestone.

Stage1 sampled **8.268 GiB family peak**, minimum free **52%**, maximum swap
growth **0 bytes**, normal pressure and no guard stop over **212 samples**.
Six existing guard regressions pass. [Stage1 resource summary](hu200-k50-frozen-artifacts/stage1-resource-summary.json).
These are sampled values; after-save process RSS is distinct from family peak.

## Exports and evidence

The **5B current and average exports pass all 85,373,907 rows** through
`src.diagnostics.cfr_average.audit`, with `--zero-mass uniform`. The audit
checks all stored totals and current regrets; it does not reconstruct historical
increments. The sorted-key digest remains the stage-1 baseline. The average has
**62,094,574 positive-mass rows /23,279,333 uniform-fallback rows**. Audit time
was **1,641.72 s**; unweighted mean current/average total variation was 0.3038193.
[Every-row 5B audit](hu200-k50-frozen-artifacts/5000000000.audit.json).

| 5B export | Bytes | SHA256 |
|---|---:|---|
| Average | 2,756,501,598 | `43aa3b3cd3c7365c5be1b177dcc8a7c04e01c8470403c38327aa845922726512` |
| Current | 2,289,968,280 | `b9c4961b6de9b214950d4d272a220312b5b8445a433d4046e1b6ee37b786ebf7` |

Final exports/audit and archive/upload acceptance remain pending. The detached
M4 worker will seal full and average-only ZIPs with
embedded manifests and complete local readback under
`~/Local/Research-Cloud/PR-265-hu200-k50-frozen/`.

Owner-authorized inactive evidence cleanup removed 38 unchanged manifested
files /28,373,295,553 bytes after current merged-owner, native upload, independent
Drive and dependency checks. Two uncertain files were retained. [Exact cleanup
and restoration receipt](../artifacts/pr265-m4-storage-20261013.md).
All tables, environments, shared Git and synced files remain.
