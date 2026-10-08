# HU100 growth and fresh playing comparison

**Training reached 20,001,470 actual nodes and 4,937,867 entries with the unchanged recipe.** The full current/average audit passed. Additional unchanged-recipe training improved both predeclared aggressive comparisons: **tight +65.31 [23.04, 107.58] and loose +162.73 [80.93, 244.54] BB/100**, with Bonferroni-adjusted intervals. The terminal policy is near break-even against tight and still loses against loose; this is one lineage against scripted opponents, not a general-strength or release claim.

[Prospective protocol](../native-hu100-growth.md) · [PR203](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/203).

## Training and capacity

The #197 native recovery parent was restored from its accepted archive after a live merged-PR check, with whole ZIP, manifest and selected checkpoint/average hashes verified. Pilot training continued the original seed/RNG coordinates, regrets, visits and iteration-weighted opponent-sampled accumulators; its audited 11,143,450-node checkpoint fed the continuation. No averaging reset, menu/abstraction/fallback or game change occurred.

| Quantity | #197 parent | Terminal |
|---|---:|---:|
| Actual nodes | 11,042,440 | 20,001,470 |
| Iteration | 7,722 | 14,561 |
| Entries | 3,255,387 | 4,937,867 |
| Positive average-mass keys | 2,158,927 | 3,286,182 |
| Traverser visits | 2,191,268 | 3,987,494 |
| Checkpoint bytes | 114,633,184 | 177,935,258 |

Terminal target overshoot: **1,470 nodes**, zero entry-threshold overshoot. The target stopped training; neither the 6,510,774 entry threshold nor a resource guard fired. The full continuation took 22.49 seconds including loading, save and guard overhead; atomic serialization alone took 14.81 seconds. Streaming combined exports took 49.88 seconds; the complete current/average audit and visitation scan took 109.78 seconds. Sampled training-family peak was **1.593 GiB**, export **0.045 GiB**, audit **0.082 GiB**. Five-second samples can miss transient peaks and RSS can double-count shared pages.

Stage 1 finished retrieval, pilot, continuation, full audits and local archive readback/copy in **305.18 seconds**, within its original 1,800-second deadline. All 64 guard samples stayed normal/AC, system-free ≥82%, disk ≥64.50 GiB. Swap remained within the original #197 448.81-MiB baseline/+0.5-GiB guard; maximum measured growth from that historical baseline was 114.75 MiB. There were no admission refusals or hard guard failures. [Resource and cost receipts](native-hu100-growth-artifacts/stage1-resources.json).

The cost-only pilot quote reserved 901.98 seconds for continuation/save/tools/closeout with 1,682.16 seconds remaining. Its ceiling projection was 3.94 GiB sampled family RSS, with 2.2× memory and 2× timing allowance. This is a forecast, not an observed capacity ceiling. At 20M the bounded tools stay small; native table allocations/save sorting remain the plausible next memory cost, while full audit scans dominate this run's elapsed cost. Actual RAM exhaustion, 1B feasibility or convergence were not measured. No further training follows.

## Fresh playing comparison

The parent and terminal averages were fixed before evaluation. A 16-block/opponent pilot measured cost only; **2,048 duplicate blocks/opponent** then froze before final play, with **61,440 distinct final hands** (20,480 per candidate and one shared 20,480-hand uniform reference). Pilot hands are excluded. Fresh roots 2026100820411/2026100820412 passed physical-deal collision checks against #197/#200 and each other. Seats swap within every block, scripted opponents and inference settings are unchanged. No sample extension or outcome-driven selection occurred.

| Opponent | Parent BB/100 | Terminal BB/100 | Paired gain BB/100 | Interval | Status |
|---|---:|---:|---:|---|---|
| tight_aggressive | −65.15 | +0.16 | +65.31 | [23.04, 107.58] | Primary improvement |
| loose_aggressive | −376.79 | −214.06 | +162.73 | [80.93, 244.54] | Primary improvement |
| random | +66.56 | +87.59 | +21.02 | [−21.11, 63.16] | Descriptive |
| check_call | +105.85 | +117.22 | +11.38 | [−16.79, 39.54] | Descriptive |
| pot_pressure | −140.08 | −126.09 | +13.99 | [−31.84, 59.82] | Descriptive |

Primary intervals are paired-block Student-t **97.5%**, Bonferroni familywise α=0.05 for the two predeclared comparisons. Remaining intervals are ordinary descriptive 95%; none clears zero. Absolute terminal ordinary 95% intervals are tight [−36.16, 36.48], loose [−296.09, −132.04] and pot [−180.26, −71.91]. Improvement does not establish profit against the aggressive pool.

### Coverage and visits alongside winnings

| Opponent | Parent decisions: positive / zero / missing | Terminal decisions: positive / zero / missing | Positive-mass coverage |
|---|---|---|---|
| tight_aggressive | 3,644 /7 /13 (3,664 total) | 3,534 /4 /11 (3,549 total) | 99.45% →99.58% |
| loose_aggressive | 6,746 /26 /103 (6,875 total) | 6,691 /37 /51 (6,779 total) | 98.12% →98.70% |
| pot_pressure | 3,389 /2 /897 (4,288 total) | 3,367 /3 /887 (4,257 total) | 79.03% →79.09% |

Among **known-key decisions**, including zero-average-mass keys, the fraction with fewer than ten traverser visits fell: tight turn 40.1%→26.8%, river 57.3%→37.3%; loose turn 54.1%→39.7%, river 74.2%→56.6%. Covered late-street decisions remain sparse. Globally 2,706,536 of 4,937,867 entries have zero traverser visits; average mass and own-seat traverser visits are different measurements.

Disjoint hand-payoff contributions use the full 4,096 candidate hands/opponent as denominator. All-positive-mass tight hands contribute −91.49→−21.84 BB/100; loose −348.23→−206.67. Most observed gain is associated with these covered hands. Policy-dependent exposure differs, so this is **descriptive association, not a causal branch effect**. Pot-pressure ever-missing hands still contribute −160.47→−151.76 BB/100. #201's unsupported action/history branches remain an unchanged-recipe limitation; the new missing decisions have not been individually reclassified as unsupported. [Complete statistics, coverage, visit bands and hand partitions](native-hu100-growth-artifacts/playing-result.json).

### Preserved operational failures and deadlines

Stage 2 first stopped constructing model specs because native average headers lack `entries`. **No operation intent, pilot/config/freeze or hand existed.** Independently reviewed source `ed9be57` takes counts from the verified parent index and terminal audit, binds exact bytes, and allows one continuation of this never-started stage with the **original start/deadline**. The failed state/log are retained. [Amendment review](native-hu100-growth-artifacts/source-review-amendment.json).

All play, full replay and full reproduction then passed at `ed9be57`. The summary child refused `Changed/dirty source or model`: the evaluator had recorded only the pinned untracked runtime binary as dirty. The original launcher remains **failed**, with its complete raw outputs and failure guard retained. A separately qualified, independently reviewed postprocessing controller at `ae8d9b8` verified every output/guard hash, fixed model order/sample/settings/schedule, and all 40 manifests against the frozen Git source fingerprint `a8915e6a867eab543f0482d509acbe55225f35938d1034c2d12d76eca85ffec9`. Its exception admits exactly that binary; ordinary reporter strictness remains. It writes separate summaries and archives only, with no hand rerun path. [Source proof](native-hu100-growth-artifacts/source-provenance.json) · [independent controller review](native-hu100-growth-artifacts/source-review-postprocessing.json).

**81,920 stored rows /316,158 stored actions** replay, including copied reference rows; the distinct hand count is 61,440. All settlements, keys/menus, legal probability streams, coverage and arithmetic match full reproduction. Stage 2 verified postprocessing/archive closeout finished **914.25 seconds** after its original start, within 1,800 seconds, including both corrections. Sampled family peak **1.105 GiB**, disk ≥57.23 GiB, system-free ≥80%, normal pressure/AC, historical-baseline swap growth ≤114.75 MiB. No admission refusal or hard guard failure. [Closeout](native-hu100-growth-artifacts/playing-closeout.json) · [resources and operation costs](native-hu100-growth-artifacts/stage2-resources.json).

## Verification and storage

Training source `f5791b56f445a6f6081884a4b1074fd626568133`; native binary SHA256 `b547173b838d896888973f4cf283ba66a60b61781a29a7ed719ebcbadb448f2c`. M4 qualification: **49 Python tests, nine Rust tests, release build and repository artifact check** passed. Initial independent execution review passed with 19 focused tests; the setup amendment passed with 20 and an independent original-deadline/one-use continuation fixture. The final controller qualification passed **34 focused M4 tests** and the artifact check; its independent review passed 22 focused tests and reconstructed the frozen Git fingerprint. Full playing replay and reproduction pass. Final independent evidence review is recorded separately below.

Stage 1 [Research-Cloud archive](https://drive.google.com/file/d/1hC98K7KSU-pFEdPD7lfFMT17EcxpW_Re/view): **1,143,009,755 bytes /78 members**, local archive and every member verified; native uploaded/not-uploading/no-conflicts and cloud ID/name/size/parent acceptance agree. SHA256 `bb42f78802d552e4d790e491ca23bbf3a6970f1a233230bbcf409fd116b2a896`; manifest SHA256 `4424b8066fbf7638db09ddc8936fceae94676080e628124c05ec20da117d9a9b`. Exact checkpoint/export hashes and restoration are in [RESULTS_INDEX](../../RESULTS_INDEX.md). Remote archive bytes were not redownloaded. Originals remain; PR stays unmerged.


Stage 2 [Research-Cloud archive](https://drive.google.com/file/d/1Qs_8_7YUCMccw6A44jNld_s0ksKfTM-b/view): **2,078,090,291 bytes /456 verified members**, SHA256 `752e045b67261f630e0c1be0331659a60c708ba91e6052d1548524fb7932c59c`, manifest SHA256 `be3ef6bce2416f0b34f6c84211f1127dae5c7ba33dddc65212799fa8a9335ab2`. Local whole/member/native-copy hashes and native uploaded/not-uploading/no-conflicts agree with connector ID/name/size/parent acceptance. It preserves original failed states, raw hands/replays/reproductions, model snapshots, frozen training/evaluation/controller sources and separate verified summaries. [Native status](native-hu100-growth-artifacts/stage2-native-upload.json) · [cloud acceptance](native-hu100-growth-artifacts/stage2-cloud-acceptance.json). No remote-byte redownload, cleanup, release, merge or additional training.
