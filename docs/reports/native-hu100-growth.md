# HU100 growth and fresh playing comparison

**Training reached 20,001,470 actual nodes and 4,937,867 entries with the unchanged recipe.** The full current/average audit passed. The fresh, prospectively paired comparison is running; no playing conclusion is available yet.

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

## Playing comparison and setup correction

The two averages are fixed: parent at 11,042,440 and terminal at 20,001,470 nodes. Two primary comparisons—tight_aggressive and loose_aggressive—use paired duplicate-block means and Bonferroni-adjusted 97.5% intervals. Other opponents are descriptive. Fresh roots 2026100820411/2026100820412 are separate from #197/#200; final sample size will freeze using pilot costs only.

Stage 2 initially stopped while constructing model specs: native average headers contain no `entries` field. **No operation intent, pilot/config/freeze or hand was created.** The failure/log/state are retained. Independent-reviewed amendment `ed9be57` takes counts from the verified parent index and terminal full audit, binds exact bytes, and admits one continuation of only this never-started stage under its **original start/deadline**. The recipe, opponents, comparisons, guards and eventual count rule are unchanged. No scientific child or partial result was retried. [Independent amendment review](native-hu100-growth-artifacts/source-review-amendment.json).

## Verification and storage

Training source `f5791b56f445a6f6081884a4b1074fd626568133`; native binary SHA256 `b547173b838d896888973f4cf283ba66a60b61781a29a7ed719ebcbadb448f2c`. M4 qualification: **49 Python tests, nine Rust tests, release build and repository artifact check** passed. Initial independent execution review passed with 19 focused tests; the setup amendment passed with 20 and an independent original-deadline/one-use continuation fixture. Full playing replay, reproduction and evidence review remain pending.

Stage 1 [Research-Cloud archive](https://drive.google.com/file/d/1hC98K7KSU-pFEdPD7lfFMT17EcxpW_Re/view): **1,143,009,755 bytes /78 members**, local archive and every member verified; native uploaded/not-uploading/no-conflicts and cloud ID/name/size/parent acceptance agree. SHA256 `bb42f78802d552e4d790e491ca23bbf3a6970f1a233230bbcf409fd116b2a896`; manifest SHA256 `4424b8066fbf7638db09ddc8936fceae94676080e628124c05ec20da117d9a9b`. Exact checkpoint/export hashes and restoration are in [RESULTS_INDEX](../../RESULTS_INDEX.md). Remote archive bytes were not redownloaded. Originals remain; PR stays unmerged.
