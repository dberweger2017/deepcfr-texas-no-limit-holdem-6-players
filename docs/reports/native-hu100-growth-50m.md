# HU100 growth toward 50M: capacity stop and continued loose-aggressive gain

Unchanged-recipe training from the verified 20M parent reached **39,438,279 actual nodes**, then stopped cleanly at the fresh entry ceiling. **50M was not reached.** On a fixed fresh paired sample, loose_aggressive improves **+105.68 [21.62, 189.73] BB/100** with the predeclared multiplicity adjustment; tight_aggressive is **+22.90 [-21.05, 66.85]**, inconclusive. Loose/pot still lose. More training helps one covered primary in this lineage; it does not establish general poker strength.

[Protocol](../native-hu100-growth-50m.md) · [PR204](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/204) · [training receipt](native-hu100-growth-50m-artifacts/training-result.json) · [paired readout](native-hu100-growth-50m-artifacts/playing-result.json). PR203 merged only after independent review and all five exact-head checks passed on `d93c16d`; merge `8fcde7767d0febe34e9fae9d530b5ba6136d17a3` is this branch's base. PR204 remains unmerged; no release, cleanup, recipe change or automatic continuation.

## Execution and integration

Frozen execution source **`d3033822d865e077093809e82633a1ad7fad19ba`**, protocol SHA256 `8aabe90aa07e894c2277220b5d8e76416d3e308cf4e900675fcfbc3cf351371f`, independently reviewed before either clock. Native runtime SHA256 `b547173b838d896888973f4cf283ba66a60b61781a29a7ed719ebcbadb448f2c` remains in ignored `results/runtime/hu20-trainer`. Current main guidance and #197/#200/#201/#202/#203 protocols/evidence were read. M4 availability, normal pressure, AC, disk and original swap baseline were admitted before launch; isolated root `~/Local/hu100-growth-50m-20261008` retains all outputs.

The normal path now derives model entries from a full verified audit; positive/zero counts remain recorded in that audit, binding model bytes, checkpoint lineage, nodes and iteration. Native headers need no invented `entries`. The pinned ignored runtime is hash-checked before operations, while tracked/untracked source cleanliness remains strict. Both final play and reproduction use clean manifests; **all 40 pilot/final/reproduction manifests** match frozen revision and source fingerprint ([proof](native-hu100-growth-50m-artifacts/source-clean-proof.json)). Normal strict reporting completed; no exceptional postprocessing path, failed setup, dirty-source override or hand retry occurred.

Six focused regressions cover audited metadata/tampering, fresh/pilot/disk capacity, timing-only admission, complete two-average/five-opponent play/replay/reproduction/report integration, pinned-runtime mutation, clean native capacity exit, and capacity receipt binding. Local/M4 qualification: **44 focused Python tests**, **9 Rust tests**, native release build and repository artifact checks pass. Independent execution review ran 28 focused tests and cleared the final seal-only amendment ([source review](native-hu100-growth-50m-artifacts/source-review.json)). No native trainer, game, regret math, abstraction, opponent or fallback change.

Seed **2026100601**, exact RNG/recovery coordinates, regrets, cumulative iteration-weighted opponent-sampled averaging, counters, one root/seat, uncapped action menu, HU100 v 1 abstraction and uniform missing/zero fallback remain unchanged. Parent checkpoint/average restored from accepted merged #203 Stage 1 ZIP: whole ZIP, embedded manifest, selected member sizes/hashes and indexed full audit agree ([retrieval](native-hu100-growth-50m-artifacts/retrieval.json)).

## Stage 1: measured growth and stop

One 1,800-second absolute cap began **15:15:18.122 UTC**, before retrieval, and closed **15:23:13.833 UTC** after terminal audits and complete primary archive readback/native copy: **475.71 seconds**. Deadline **15:45:18.122 UTC** was never reset. [Resources and all operation costs](native-hu100-growth-50m-artifacts/stage1-resources.json).

Fresh capacity used #203's 1,710,161,920-byte sampled family peak /4,937,867 entries, 2.2× memory allowance within the unchanged **5.63-GiB soft allowance**, at most twice parent entries, and a new disk quote with 15.5-GiB floor/four fixed evidence GiB/12 combined-asset equivalents. Initial ceiling **7,933,918**, not the old 6,510,774. The 200-ms pilot peak **1,780,940,800 bytes** lowered it to **7,642,767**. Disk/extrapolation were nonbinding; neither ceiling nor guard was increased after pilot.

Pilot saved **20,102,306 nodes /4,953,527 entries** (836-node complete-iteration target overshoot), streamed/audited both exports, then resumed the exact recovery bytes SHA256 `0d61085b47faa3fb72c8ee9d6236dafed09fa49a26289d7a58441a745d065375`. Its progress was preserved. Timing-only full-run quote reserved doubled measured costs, tool/save floors and 180-second closeout: **926.59 seconds** required versus **1,619.85** remaining. No outcomes or sample sizes influenced training admission.

| Quantity | Fixed 20M parent | Audited terminal | Increase |
|---|---:|---:|---:|
| Actual nodes | 20,001,470 | 39,438,279 | 19,436,809 |
| Complete iteration | 14,561 | 30,080 | 15,519 |
| Entries | 4,937,867 | 7,643,261 | 2,705,394 |
| Traverser visits | 3,987,494 | 7,908,886 | 3,921,392 |
| Positive average mass keys | 3,286,182 | 5,105,598 | 1,819,416 |
| Zero average mass keys | 1,651,685 | 2,537,663 | 885,978 |

Terminal native exit 3 is the clean complete-iteration capacity result: **494-entry overshoot** over the frozen applied cap, no node-target overshoot, no stop-file request. Native nonsave time **7.57 seconds** was below the admitted 32.19-second native time stop; entry count and native log establish entry stopping. Original telemetry's `unaudited` save field is preserved; subsequent full audit verifies every **7,643,261** stored node, current regrets/totals and both export bytes ([audit](native-hu100-growth-50m-artifacts/terminal-audit.json)). Historical increments are not reconstructed by that audit; exact recovery preserves them. Terminal traverser street visits: preflop 271,144 /flop 839,984 /turn 2,094,094 /river 4,703,664. Entries by visits: zero 3,991,528 /one 2,549,515 /2–9:1,009,008 /10–99:91,516 /100+:1,694.

| Terminal operation | Seconds | Sampled whole-family peak |
|---|---:|---:|
| Train/load/save supervised operation | 38.74 | 3.026 GiB at 200 ms |
| Native atomic checkpoint write (within above) | 24.68 | Same retained samples |
| Streaming current and average exports | 77.34 | 0.058 GiB at 5 s |
| Full current/average integrity audits | 170.72 | 0.096 GiB at 5 s |
| Archive/local readback/native copy | 7.27 | 0.120 GiB at 5 s |

Kernel command high water **2.938 GiB**, 200-ms family peak **3,248,914,432 bytes**, five-second family peak 3.012 GiB; 169 terminal training/save samples plus 86 pilot samples and 97 full-resource samples preserve timing/allocation/save behavior. Sampled coefficient rose from pilot **359.53 to 425.07 bytes/entry (+18.23%)**. The original 2.2× forecast allowance applied to terminal measurements would require **6.657 GiB**, exceeding the 5.63-GiB *planning allowance*, despite actual 3.026 GiB staying below unchanged soft/hard guards. This is a forecast miss, not measured RAM exhaustion; allocator/table/save contributions require separate diagnosis. No hard guard failed. All samples normal pressure/AC, minimum system-free 82%, disk 51.41 GiB, maximum swap growth 114.75 MiB from the retained 448.81-MiB baseline (limit+0.5 GiB).

## Stage 2: fixed fresh comparison

Separate 1,800-second clock **15:24:59.246–15:31:51.474 UTC**, **412.23 seconds** including all play, replay, full reproduction, strict report and complete archive readback/native copy; original deadline **15:54:59.246 UTC**. [Resources](native-hu100-growth-50m-artifacts/stage2-resources.json). Sixteen cost-only pilot blocks/opponent were excluded. Fresh roots **2026100850411 /2026100850412** and physical-deal checks cover #197/#200/#203 and current pilot. Timing-only freeze admitted **2,048 blocks/opponent**, with 522.54-second doubled quote plus 240-second closeout against 1,683.26 seconds remaining ([freeze](native-hu100-growth-50m-artifacts/frozen-final.json)). Final schedule SHA256 `94d8e95fa3f668c6cee00ffc13bf5ecbb4ec2a0b234cacc873dbabfc64a4ead3`.

Fixed two averages, five unchanged scripted opponents, two swapped seats per block, 100-BB stacks and uniform reference. **61,440 distinct hand evaluations**, **81,920 stored rows /317,417 actions including reference copies**; reference played once then reused. Every final action/settlement replays, every hand/probability/classification reproduces exactly except measured latency, and private streams/menus/keys/coverage/raw-chip statistics recount. No extension, checkpoint selection, failure, admission refusal or hand rerun. Source-clean normal reporter passes throughout.

| Opponent | Parent BB/100 | Terminal BB/100 | Paired terminal−parent interval | Interpretation |
|---|---:|---:|---|---|
| random | +123.16 | +117.21 | -5.94 [-54.86, +42.97] | descriptive |
| check_call | +89.71 | +93.99 | +4.28 [-21.65, +30.22] | descriptive |
| tight_aggressive | -42.33 | -19.43 | +22.90 [-21.05, +66.85] | inconclusive |
| loose_aggressive | -276.55 | -170.87 | +105.68 [+21.62, +189.73] | improvement |
| pot_pressure | -109.35 | -101.42 | +7.93 [-29.91, +45.77] | descriptive |

Tight/loose are the only primary family: two-sided **97.5% Student-t intervals**, Bonferroni familywiseα0.05 over independent swapped-seat block means. Remaining 95% intervals are descriptive and all include zero. Absolute terminal ordinary 95% intervals: tight **−19.43 [−56.52,17.65]**, loose **−170.87 [−246.76,−94.99]**, pot **−101.42 [−156.67,−46.16]**. Gains do not establish profit against loose/pot. Parent is evaluated on these same fresh deals; do not compare its absolute mean to #203's different sample as a learning effect.

### Coverage and visits alongside winnings

Reached target-decision coverage is policy-dependent, descriptive, and includes all streets. The percentages/counts below include zero-mass and missing decisions in the denominator ([aggregates](native-hu100-growth-50m-artifacts/coverage-summary.json)); fallback remains uniform.

| Opponent | Positive-mass% parent→terminal | Missing count parent→terminal | Zero-mass count parent→terminal | Decisions parent→terminal |
|---|---|---|---|---|
| random | 83.06→83.45 | 1009→993 | 7→5 | 5998→6030 |
| check_call | 99.67→99.83 | 40→18 | 8→7 | 14491→14446 |
| tight_aggressive | 99.51→99.89 | 10→1 | 8→3 | 3640→3553 |
| loose_aggressive | 98.91→99.49 | 57→27 | 20→9 | 7074→7056 |
| pot_pressure | 80.38→80.14 | 820→824 | 1→0 | 4184→4149 |

Known decisions with fewer than 10 traverser visits (including known zero-mass keys):

| Primary | Preflop% parent→terminal | Flop% | Turn% | River% |
|---|---|---|---|---|
| tight_aggressive | 3.23→1.16 | 9.14→4.77 | 26.61→12.68 | 39.44→28.77 |
| loose_aggressive | 8.81→3.88 | 10.47→5.69 | 37.00→25.63 | 51.58→38.76 |

Both aggressive primaries already mostly reach positive-mass keys; additional training increases covered visits, with substantial remaining late-street sparsity. Loose all-positive hands' full-sample payoff contribution improves descriptively **−277.08→−177.80 BB/100**, tight **−58.70→−37.49**; remaining zero/missing/no-target hand categories sum exactly to each absolute policy result in the readout. These categories vary with policy and cannot assign causal branch gains. Pot missing coverage remains about 20%; #201 demonstrated action/history support limits on its earlier sample, but this campaign does not relabel each new missing decision without a new exhaustive support audit.

## What limits growth and what next measurements support

The **frozen conservative entry ceiling** limited this run, while export/audit scans dominated its terminal tool time. It did not hit 10-GiB hard family memory or 5.63-GiB soft stop. The measured RSS/entry increase defeats a blind linear projection: applying the unchanged 2.2×/5.63-GiB formula afresh yields **6,464,366 planning entries**, below the existing 7,643,261-entry terminal, so this admission rule supports no further growth from this parent. It neither proves 50M cannot fit nor authorizes raising any guard. Next useful work is bounded native table/allocation/save-memory measurement and engineering, preserving exact recovery and recipe, before considering another larger budget. Late-street sparse visits and unchanged support remain playing limits. An owner-defined external benchmark and independent training seeds are still needed for broader strength claims. No automatic 100M continuation.

Both stages closed successfully with zero guard failures/refusals and permanent phase claims; admission snapshot/never-started one-use continuation protection remains available for future authorized campaigns within their original clocks, never as failed-science retries. Independent source/evidence review and green final-head checks are required for handoff; receipts below retain scope/limitations.

## Archives and restoration

Accepted [Research-Cloud folder](https://drive.google.com/drive/folders/10Dn15X5l9nK3h9TWYGEPZR9hKI7shfLc), parent `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. Stage 1 [ZIP](https://drive.google.com/file/d/12LEcZWKJCwtTCCYcMxW578Gm3rDy9emK/view): **1,641,001,686 bytes /93 verified members**, SHA256 `9f01443ef9008aa112112ca25bee71048f74bd07e5ebb9e364ce243efac22f4b`; `ARCHIVE-MANIFEST.json` SHA256 `681ca26c9f6b7fe4c25525786595ebc6423b809f4f84b87ac3fde2f9705d27ef`. Stage 2 [ZIP](https://drive.google.com/file/d/1D8BRCrV2BDAgKB8tipRYptABERZv88xt/view): **2,032,348,910 bytes /447 verified members**, SHA256 `f6f4bec282c59af1a1ba25a6be1be1a9b89daecbc5fd94ecd9ff37f0e1467d22`; manifest SHA256 `2f96b81cf76b86a428d677c23cc9dc1ec0193a85099c7b678f6f6b07a169e4fd`.

Whole ZIP/all members read back locally and native copy hashes agree inside each scientific deadline. Later native uploaded 1/uploading 0/conflicts 0 and independent connector ID/name/size/parent acceptance agree ([Stage1](native-hu100-growth-50m-artifacts/stage1-cloud-acceptance.json), [Stage2](native-hu100-growth-50m-artifacts/stage2-cloud-acceptance.json)); no remote archive bytes redownloaded. Live merged #203 was checked before parent copying, and live open/unmerged #204 before each primary seal; archival of this open owned campaign is explicitly owner-authorized. Large models/raw hands/source/binary/guards remain outside Git. Own mutable archive guard/state plus later review/report/upload/qualification metadata seal separately; primary ZIPs are immutable. [Index with exact asset/member hashes and restoration](../../RESULTS_INDEX.md#pr204-hu100-growth-toward-50m-and-paired-playing-gains--october-8). Restore by pinned Drive ID into a new ignored nonsynced root, require whole ZIP and manifest hashes, then selected member sizes/hashes before use. Retain all originals and other PR dependencies; restoring does not restart any campaign.
