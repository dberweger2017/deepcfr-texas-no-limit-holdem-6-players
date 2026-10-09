# HU200 M1 feasibility and Slumbot preparation

PR216 adds a distinct, correctly trained 200BB game for v0.5.5. v0.5.0 remains
HU100; released HU20 policies/defaults are unchanged. The isolated branch starts
at current-main `6e18043317817080fd38f400c5366fbf18fc6b53`. No M4 input, worker or
campaign was used, and no live match, release, paid compute or merge occurred.

## Correctness before training

Native/Python game, schema, menu and file-format identities now distinguish HU200.
Keys require matching initial stacks; checkpoint/recovery, streaming current and
average export, independent accumulator audit, compact inference and arena
registration preserve that identity. HU100 relabeling/resume and mismatched
models/tables are rejected. The recipe remains linear CFR, one root per seat,
opponent-sampled averaging, v1 cards, the existing min/pot/conditional-jam menu,
uniform missing/zero-mass fallback and translation off.

Qualification covers legal bounds/menus/keys, all-in runout, short raises,
uncalled refunds, ties and settlement conservation. Native/Python parity passes
1,000 HU200 hands /7,432 decisions with zero mismatches. Small reference training
rows and exact uninterrupted-versus-split native checkpoint bytes agree at 200BB;
HU20/HU100 compatibility and released behavior have regression coverage.

Independent reviews found guard-latch, archive-reserve/exception and kernel-RSS
handling defects, then an export/audit soft-latch gap and smoke dataclass
comparison defect. All were fixed before dependent training. Round 3 is clear at
execution source `02e4ddebf37674d2fa32cb494d440347fd5b59a9`; receipts preserve the
findings and responses. The pinned M1 binary SHA256 is
`c72e900bc6e47e1c2ae78f7b16d87afbd2e9397adb946a86149e6ab4160a6878`.
The sole subsequent executable-source change corrects the CLI help's stack list;
execution source/runtime remain pinned separately.

## Measured pilot

The [predeclared protocol](../hu200-feasibility.md) used fresh seed **2026100905**
on the actual 16GiB Apple M1. Ownership checks found no competing campaign and an
exclusive lock was held. Initial available memory was 6,267,584,512 bytes, swap
1,597,180,477 bytes, free disk 34,326,618,112 bytes, normal pressure and AC power.
Only this fresh HU200 lineage resumed between the 1M, 20M and 100M saves.

All three endpoint sets completed and audited; **100,000,034 actual nodes /20,575,288 entries** are retained at the terminal checkpoint. The single clock closed in **1,433.43s =23.89min**, including local archive/readback. No stop request, guard breach, failed pilot stage or retry occurred.

| Target | Actual nodes | Entries | Native traversal M nodes/s | Supervised train incl save (s) | Native save (s) | Both exports (s) | Full audit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1M | 1,001,231 | 502,315 | 2.161 | 2.69 | 1.94 | 6.96 | 13.37 |
| 20M | 20,002,716 | 6,521,284 | 1.753 | 39.58 | 26.95 | 89.74 | 170.53 |
| 100M | 100,000,034 | 20,575,288 | 1.487 | 156.39 | 89.72 | 289.07 | 539.30 |

The 1M measurement admitted 20M; the completed 20M tools admitted 100M with a 3,099.46s save/tool/closeout reserve and 28,374,777-entry capacity ceiling. Entry/node scaling is a bounded capacity forecast, not a promise of reaching the node target. Stored visit counts at 20M are 3,803,827 zero /2,242,504 one /456,435 two–nine /18,343 ten–99 /175 hundred-plus. At 100M they are 10,980,042 /6,879,854 /2,498,719 /212,546 /4,127 respectively. Thus 53.4% of terminal stored keys have zero traverser visits and 1.05% have ten-plus. Positive average mass exists on 13,613,304 keys; visits and average mass are distinct diagnostics.


Native traversal and save timers have different scopes from supervised operation
wall time: the latter includes startup/resume, monitoring and exit. The reported
pure traversal rate subtracts the native save duration, rather than treating a
fast traversal quote as end-to-end completion. Full audits verify every stored
regret/current-policy row and average accumulator total. They do not reconstruct
all historical accumulator increments or prove convergence.

Across **2,680 samples**, maximum sampled family RSS was **1.69GiB** (100M train/save). The kernel command peak was **1.96GiB** (smoke), above its sampled family peak: brief peaks can be missed. These are different scopes; neither is claimed as an exact family supremum. Largest within-operation sample gap was **0.590s**; source snapshot had only one sample. Pressure stayed normal, free percentage stayed >=65%, AC remained connected, and swap growth from the original 1,597,180,477-byte baseline was **zero**. Minimum sampled disk was **27.06GB**, above the unchanged 15.5GiB floor. Local archive/readback took **8.60s**; 28.20GB was free at report integration.

See [operation resources](hu200-feasibility-artifacts/resources.json), [costs](hu200-feasibility-artifacts/costs.json), [science](hu200-feasibility-artifacts/science.json), [baseline](hu200-feasibility-artifacts/baseline.json) and [closeout](hu200-feasibility-artifacts/closeout.json). Raw samples and native tool logs are archived.

## Scripted smoke

Exactly **320 hands**, 32 swapped-seat blocks per each of five unchanged opponents, used the terminal average with translation off. All actions were legal, all hands conserved 40,000 chips, full Python-engine actions/events/settlements replayed and all selected policy actions reproduced. This is offline engine replay, not live-service terminal verification.

| Opponent | BB/100 | Descriptive 95% block interval |
|---|---:|---:|
| random | 664.06 | [-1156.30, 2484.43] |
| check_call | 40.62 | [-107.02, 188.27] |
| tight_aggressive | 360.16 | [-275.92, 996.23] |
| loose_aggressive | -499.22 | [-1163.87, 165.43] |
| pot_pressure | -104.69 | [-780.34, 570.96] |

There were **529 candidate decisions: 496 positive-mass known keys (93.76%) and 33 missing keys (6.24%)**, with zero zero-mass fallbacks in this small sample. Known/total decisions by street were preflop 240/260, flop 109/120, turn 72/74, river 75/75. Among known keys, visits were 2 zero /21 one–nine /473 ten-plus; river alone had 2 zero /17 one–nine /56 ten-plus. This selected small schedule does not measure broad history coverage. Every winrate interval crosses zero; **no strength, growth comparison or milestone qualification** is claimed.

The entire load/play/replay operation took **115.98s**, sampled family peak 1.58GiB and kernel command peak 1.96GiB. Load alone was not timed: 115.98s is an upper bound, not a measured isolated load cost. [Smoke receipt](hu200-feasibility-artifacts/smoke.json) preserves per-opponent coverage and visits.

## Next budgets and external integration

Recommend a **separate 60-minute M1 feasibility pilot**, provisionally resuming this exact hash-verified HU200 100M checkpoint toward **200M**, with an unchanged **28,374,777-entry ceiling**, 3/4GiB family soft/hard guards, original-baseline swap/pressure/AC checks and 15.5GiB disk floor. The measured 2x quote is **3,298.89s =54.98min** including saves, both exports, full audit and 600s closeout. Optional evaluation is not quoted; preserve a valid partial if entry/time capacity stops first. This is a recommendation for a reviewed future protocol, not authorization or a 1B feasibility claim.

Fresh admission requires **>29,066,650,896 bytes free disk** and approximately **4GiB available RAM** (forecast family plus 1GiB margin). Current **28.20GB free fails the disk quote** by about 0.87GB; reclaim headroom only through separately authorized storage work, preserving open-PR roots. No cleanup or follow-on runs are performed here. [Exact budget](hu200-feasibility-artifacts/next-budget.json) records scaling, parent hash and the snapshot admission refusal.


[The Slumbot readiness note](../hu200-slumbot-readiness.md) builds on #213's public
prefix codec, information boundary, injected connection and hash-chained journals.
Remaining work is full terminal/fold/refund/all-in/tie/disclosure/net-winnings
handling, independent verifiable settlement accounting, bounded HTTP transport,
service permission/rates/build identity and terminal ambiguous POST timeouts.
Hidden-card limitations must be reported. No service request was made here.

Recommend a separate **20-hand balanced-seat protocol pilot**, at most **1,020
requests**, one in flight, 10-second timeout, no retry, **30-minute total cap**
including measured model load, replay/accounting and closeout. Reaching 20 hands
is provisional: the timeout worst-case exceeds the wall cap. Network latency,
requests/hand and terminal semantics remain unmeasured; #213's single 0.675s
new-hand check cannot quote a match. Retain the actual M1 resource guards and
fresh admission. Owner approval and reviewed lifecycle implementation precede
live use. Larger matches and milestone acceptance remain owner decisions.

## Evidence and review

All nine checkpoint/current/average files, raw resources, smoke hands, qualifications, resolved reviews, selected source snapshot and runtime are in **HU200-M1-feasibility-20261009.zip**: **2,414,034,193 bytes**, SHA256 `643e064ec7effd35b86251dd345d9292017d38c537f8bd90a5ca12c95df23574`. All **447 members** passed local size/SHA256 readback; embedded `ARCHIVE-MANIFEST.json` SHA256 `33377337a080b66a4ecd4e276082782c855169f2f10cb38f0088772d655a1c60`. [Archive receipt](hu200-feasibility-artifacts/archive-receipt.json) and [model/dependency index](hu200-feasibility-artifacts/model-index.json) give exact restoration identities. The post-archive closeout/upload/final-review lifecycle records live in compact Git receipts rather than a second science ZIP.

Native Drive upload is **pending** at initial post-closeout inspection (`isUploaded=0`, `isUploading=1`); an actual archive cloud ID and independent cloud acceptance are not yet claimed. No remote archive bytes were downloaded. The accepted cloud receipt must supersede this dated snapshot before handback.

[RESULTS_INDEX](../../RESULTS_INDEX.md#pr216-hu200-m1-feasibility--october-9-2026) contains archive/folder IDs, model member hashes and retrieval commands. All originals remain; no deletion or forced offload occurred. Final review and PR readiness do not change the frozen scientific source.

Local qualification observed 129 focused Python tests, 69 compatibility tests,
12 Rust tests, a release build, game/session checks and the 1,000-hand parity sweep;
55 tests then covered the corrected guards and an actual synthetic 320-hand smoke
fixture. These overlapping sets are not an aggregate test count. Full raw local
build/test logs were not retained; the archive contains the candid summary and
independent reviewers' exact validations. Execution-head full CI passed both
shards, native builds/tests, arena reproduction, solver and end-to-end checks.
Final-head CI and the closeout evidence review are separate handback checks.
