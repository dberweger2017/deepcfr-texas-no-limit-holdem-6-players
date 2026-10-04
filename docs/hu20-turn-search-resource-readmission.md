# HU20 calibration resource readmission proposal

**Owner-approved October 3, 2026; calibration-02 has since stopped cleanly under a later owner-approved budget amendment.** Calibration-01 stopped at the family RSS guard. Its [complete attempted curve and stop evidence](reports/hu20-turn-search-calibration.md) stay published. One replacement worker (18889) and sidecar (18890) are recorded in the [launch evidence](reports/hu20-turn-search-artifacts/calibration-02-launch.json); the recurring continuation is active. This proposal does not claim a quality qualifier or authorize paid compute.

The [approved plan](hu20-turn-search-approved-plan.md) says: “No duplicate worker, automatic restart after a guard failure, budget reset or paid allocation is authorized by the recurring check.” The owner’s continuation instruction additionally requires retaining partial evidence and asking the owner after a guard stop.

## Concrete correction

Before each external play or quality request, measure current owned family RSS and cap the native allocation ceiling to:

```
min(configured ceiling, floor_to_MiB(max(0,
    admitted family RSS limit - current owned family RSS - 256 MiB)))
```

The configured search ceiling remains 5 GiB. The actual request records both configured and admitted bytes, and its receipt retains that admission. A zero cap returns a legal base distribution without launching a native process. The native harness's pre-allocation tree estimate rejects larger trees as `memory_refusal`; it does not silently switch menus or compression. This correction is wired into calibration, sampled river validation and arena workers, including paid workers. It leaves the caller's configuration and cache identity unchanged.

The 256 MiB reserve covers native startup/input/tree/transport overhead. It is a declared precaution, not a proof that every tree fits: the family RSS, swap, disk, cumulative time and decision watchdogs remain mandatory. A further guard stop again requires owner input. Focused tests cover measured family headroom, rounding, no-headroom refusal without process launch, native oversize output and retained request/receipt identity; 51 campaign/search/protocol/reference tests passed locally.

## Prospective restart conditions

After explicit owner approval, first verify that no worker or new ownership claim exists. Measure reclaimable memory and swap again, then freeze and push a fresh admission and settings file before new outcomes. Worker family plus sidecar reservation must fit 80% of measured reclaimable memory, with owned family RSS below 10 GiB. Do not increase the former 4.25 GiB family cap unless fresh measured headroom permits it. Retain the 0.5 GiB sidecar reservation, 1 GiB maximum swap growth and 8 GiB free-disk guard. No change to unrelated services is proposed.

Start a new output directory and complete the prospective staged calibration under the corrected allocation law. Preserve calibration-01, its 17 timeout rows and interrupted eighteenth request as a separate failed attempt. Do not stitch its timings into the revised configuration selection, select only favorable roots, reset its clock, or rerun Part A/reference preparation/native build already completed.

The existing [frozen scientific settings](../configs/diagnostics/hu20-turn-search-calibration-frozen.json) and original #145 references remain unchanged: native/cap2 menus; 1/2/4/6 threads; compression on/off; 25/50/100/200/400 iterations; epsilon 0/0.01. Keep the staged screen and timing-only finalist rule, all 48 roots and three average lineages, both seats and 288 final coordinates per finalist. Record the new allocation law/reserve in revised settings and admission provenance. The new settings file must not overwrite the original frozen file.

Select the fastest native qualifier by cold p95, then median, then epsilon=0.01. Keep cold decision p95 ≤30 seconds and strict weighted mean/p95 residual ≤0.5% pot in the full native tree. Relaxed qualification applies only if no strict native qualifier exists: mean ≤1%, p95 ≤2%, and residual below 10% of blueprint `e_bp` on every eligible root. Publish every attempted root, exclusion, support gap, failure and curve; reduced menus remain unqualified without full-native quality and additional LBR solve costs. No claim is made that the correction will satisfy either gate or the latency budget.

Separately validate the original 32 outcome-blind river slots, eight strata, both seats and three lineages. Arena counts remain 2,048 LBR/native-pressure blocks and 256 per other panel, or 82,944 hands. The measured RunPod quote follows completed calibration/river evidence; paid compute still needs a separately approved quote and actual-pod x86_64 Linux runtime parity. No emulation on M1 is allowed.

## Remaining budget and evidence

The append-preserving M4 journal has used **5740.473981583312 seconds (1.595 hours)**, including the stopped calibration's **599.949789542 seconds**. **22.41 hours remain** of the original 24-hour allowance. Builds, pilots, failures and verification remain charged; neither an extension nor a journal reset is proposed. Part A used 5113.35 seconds within its six-hour cap and remains complete with average selected.

M4 evidence stays at `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-01`, with the cumulative `budget.json` in its parent. The complete M1 copy is `/Users/dberweger/Local/hu20-turn-search-20261002/m4-calibration-01`; all 94 manifest members (46,619,321 bytes) passed independent size/SHA verification. Preserve the original #145 binary, external AGPL sources, stopped requests/responses/receipts, TensorBoard events and all prior evidence. No cleanup, training, promotion, merge or paid allocation is authorized by this proposal.

## Approved modifications for calibration-02

The owner [approved restart with modifications on PR #148](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/148#issuecomment-5966496087), then directly instructed this chat to continue with those modifications. This section supersedes the former resource cap and mandatory 30-second arena deadline above; the other conditions remain.

Measure fresh reclaimable memory as **free + inactive + speculative + file-backed**, recording each component and macOS page size. Admit worker family RSS at **min(8 GiB, 80% × reclaimable − 0.5 GiB sidecar reserve)**. Keep the 10 GiB ceiling, 1 GiB swap-growth stop, 8 GiB free-disk guard and stop on a further guard failure. The native configured allocation ceiling remains 5 GiB, bounded by measured current family headroom minus 256 MiB. The owner's cleanup is an admission input; this task does not stop unrelated services.

The [new prospective settings](../configs/diagnostics/hu20-turn-search-calibration-readmission-02.json) retain all candidate axes, roots, original-law quality, 32 river slots and arena counts. Run and report the original 30-second strict-first/relaxed selection first. If it produces no native qualifier, run a separate 120-second screen/final campaign with the same timing-only staging. Select the fastest **strict-quality native** qualifier with cold p95 ≤120 seconds; no relaxed research-quality gate or reduced-menu selection. Apply the same p95/median/epsilon tie-breaks. Report the 30-second result and chosen deadline explicitly. This is research latency; 30 seconds remains the engineering target. Carry the selected deadline into river validation, research arena and the measured RunPod quote. Ask the owner if nothing qualifies even at 120 seconds, or on another guard stop. The fallback is frozen before any calibration-02 outcomes; all thirty-second rows and prior failed evidence remain separate and visible.

The corrected scientific implementation passes 55 focused tests, including engineering-before-research selection, strict-only bounded 120-second selection and cache-inclusive memory admission with sidecar accounting. Full CI at the allocation-fix source 349951e passes. Record the new pushed source and its checks before launch. No budget reset, extension or paid-compute approval is implied.

## Later owner-approved final-stage budget amendment

[Owner-approved comment 5967372814](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/148#issuecomment-5967372814) required a clean stop, timing-only iteration pruning, exact seat-request sharing only where valid, one finalist/floor if two cannot fit, and a forecast covering conditional 120-second work while reserving at least three hours for sampled rivers. It explicitly requires asking the owner if the forecast still cannot fit. This supersedes the running status and automatic continuation above.

The worker stopped at 238 retained screen rows and zero final rows; no guard failure. The [published forecast](reports/hu20-turn-search-calibration-02-budget.md) cannot fit even one finalist/floor (60.410 hours versus 20.652 remaining). The new settings proposal is forecast-blocked, and the heartbeat is PAUSED. No restart or new outcomes are authorized until an additional owner decision and pushed amended freeze/admission. The suggested 30-second-only phase followed by a separate fallback decision is **not yet approved**. Prior readmission, scientific gates, originals, clock and stop evidence remain retained.

## Owner-approved thirty-second-only final — October 3

The owner [approved the bounded final](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/148#issuecomment-5970668869): one timing-selected native finalist per floor (six threads, uncompressed), 25/50/100 iterations and all 48 roots, three lineages and both seats. Resume from the retained calibration-02 screen without rerunning it, use exact unlocked request sharing only at epsilon zero, and keep epsilon=0.01 separate. If no thirty-second qualifier results, stop and ask; a 120-second campaign needs a separate approval. The owner's preference for future paid parallel research is not permission to rent.

The [new prospective settings](../configs/diagnostics/hu20-turn-search-calibration-final-30-only-03.json) and [approved-scope forecast](reports/hu20-turn-search-artifacts/calibration-03-approved-forecast.json) retain all 1,728 final coordinates across six configurations. Estimated play and quality plus the three-hour river reserve is 13.576 hours. Failed native requests are never shared; extra attempts and overhead consume the remaining headroom and remain guarded/charged. The calibration watchdog reserves 10,800 seconds of cumulative allowance for river validation.

Quality uses the existing second native request. Successful full matrices and quality results are shared only when the complete request data match and neither request has locks. For the second seat, cold latency includes remeasured preparation/completion plus the retained native receipt duration, and is at least the first seat's measured cold duration. A reconstructed deadline overrun remains a base fallback evaluated under the original reference law. Retain actual elapsed and reconstructed latency separately; cached wall time cannot qualify as cold latency. The external harness/binary stays unchanged.

Before launch, push settings and tested code, confirm CI, refresh source-bound resource/ownership admission, and push admission evidence. Keep the heartbeat paused until the single worker is confirmed. No completed Part A/build/reference work repeats, no original evidence changes, and no calibration-01 timings enter selection.

## Resumed final launch

The bounded owner-approved final is running as calibration-03, worker 87830 and sidecar 87831. Tested source 93d55eb passed full CI, and fresh admission 70e445d was pushed before final outcomes. [Launch evidence](reports/hu20-turn-search-artifacts/calibration-03-launch.json) and the active status supersede preparation/paused statements above. The recurring task is ACTIVE for quiet checks; all guards, separate 120-second approval and three-hour river reserve remain.

## Completed thirty-second final — October 4

Calibration-03 completed all 1,728 final coordinates without a guard failure. [The complete report](reports/hu20-turn-search-calibration-03.md) records no strict or relaxed thirty-second qualifier. All worker/native/sidecar processes exited; the heartbeat is PAUSED as required by owner-approved comment 5970668869. No new worker or automatic 120-second campaign is authorized. Recommend preparing a separate native 120-second RunPod calibration proposal/quote; the strict quality gate, original corpus and independent paid-compute approval remain. Verification/retrieval are charged to the original append-preserving journal. Earlier launch/running sections above are historical.
