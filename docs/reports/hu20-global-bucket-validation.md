# Global equity buckets on 40 held-out turn roots

**PASS:** #163's global K=50 turn/river tables reproduce #149's fitted equity witness advantage. Held-out loss is **0.3917 BB [0.3614, 0.4234]**, versus **0.3874 [0.3582, 0.4190]** for the corpus-fitted equity-50 witness and **0.6567 [0.6273, 0.6868]** for v1. The tables are suitable for the separately labeled native trainer schema in step 2 of [the abstraction lessons](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/189). This is a constructed witness result; training and production keys are unchanged.

## Frozen measurements

All 40 limped, checked-through turn roots, three frozen B500M lineages and both seats are retained: 120 equilibrium collections, six opposite-half policies, 120 zero-CFR held-out lock evaluations, 18 deterministic replays and 240 seat/lineage observations. Policies are pooled from the other frozen 20-board half. Loss E is BB per spot, with lower values better. The original board weights and 2,000 paired-board bootstrap draws use seed **202610030304** and sorted spot IDs. Seat and lineage pairing is preserved. Intervals condition on the frozen fitted witnesses; they omit fitting uncertainty.

| Witness | E, BB per spot [95% interval] |
|---|---:|
| Recorded v1 | 0.6567 [0.6273, 0.6868] |
| Recorded fitted equity-50 | 0.3874 [0.3582, 0.4190] |
| Global K=50 | **0.3917 [0.3614, 0.4234]** |
| Global K=200 | 0.4233 [0.3795, 0.4740] |
| Recorded per-root v1 solution | 0.4728 [0.4365, 0.5100] |
| Recorded blueprint | 1.4194 [1.3310, 1.5161] |

The primary absolute gap from the predeclared fitted50 reference 0.3874 is **0.0043 BB**, below 0.10; the global50 interval upper bound **0.4234** is below v1's predeclared lower bound **0.6273**. All 24 K/fold/lineage/street coverage cells pass the inherited 5% ceiling. Both primary requirements therefore pass. Turn-only/river-only failure sensitivities are not triggered.

Paired global50 minus fitted50 is **+0.0043 BB [−0.0017, +0.0104]**. Global200 minus global50 is **+0.0316 [0.0054, 0.0683]**: K200 is descriptively worse in this witness construction, despite its finer partition. It does not provide an additional pass opportunity. This comparison does not choose K for a trained model; that decision still requires the matched-visits or plateau bench.

| Group | Global K50 E [95% interval] | Global K200 E [95% interval] |
|---|---:|---:|
| Lineage 2026093001 | 0.3939 [0.3633, 0.4268] | 0.4195 [0.3778, 0.4686] |
| Lineage 2026093002 | 0.4042 [0.3690, 0.4421] | 0.4404 [0.3901, 0.5030] |
| Lineage 2026093003 | 0.3770 [0.3444, 0.4132] | 0.4100 [0.3670, 0.4576] |
| Seat 0 | 0.3306 [0.3024, 0.3592] | 0.3780 [0.3286, 0.4396] |
| Seat 1 | 0.4529 [0.3979, 0.5096] | 0.4686 [0.4131, 0.5277] |

## Keys and coverage

Private-card labels occupied across these 40 roots are **49 turn /50 river** for K50 and **188 turn /200 river** for K200. The table dimensions remain 50 and 200 on each street; unoccupied labels are not removed. A policy key also includes the unchanged public history/menu template. Counts below are per opposite-half policy, with positive-training-mass ranges across the three lineages. Evaluation half 0 trains on half 1, and vice versa.

| K | Evaluation half | Turn keys | Positive turn keys | River keys | Positive river keys |
|---|---:|---:|---:|---:|---:|
| 50 | 0 | 7,350 | 7,023–7,226 | 133,500 | 133,500 |
| 50 | 1 | 7,350 | 6,937–6,983 | 133,500 | 133,480–133,497 |
| 200 | 0 | 27,900 | 22,358–23,222 | 534,000 | 533,913–533,975 |
| 200 | 1 | 26,850 | 20,616–20,921 | 534,000 | 533,007–533,372 |

Available evaluation-half turn keys are 7,350/7,350 for K50 and 26,850/27,900 for K200; river counts are 133,500/133,500 and 534,000/534,000 respectively. Training and evaluation counts need not match. Uniform fallback remains frozen for absent or zero-mass keys.

| K | Maximum missing turn decision reach | Maximum missing river decision reach |
|---|---:|---:|
| 50 | 0.002992% | <0.000001% |
| 200 | 4.019391% | 0.002335% |

Coverage uses board-weighted own-policy decision reach, separately for every evaluation half, lineage and street, with both seats retained. K200 uses four times the river keys and has more turn fallback, but these counts alone do not establish a causal explanation for its higher loss. Exact cell values and every policy count are in the archived readout and [compact evidence](../artifacts/pr190-validation-readout.json).

## Design and independent checks

The [prospective design](../hu20-global-bucket-validation.md) uses an MIT data adapter and the unchanged external AGPL lock-only tool. Its SHA256 is `fb32974d9d211fa66001d1efb330ec4af2d825005d24b287dc6cf3c37fa8812f`. No native source was changed. Existing transport slots `eq50-fit0` and `eq50-fit1` carry global50/global200, with explicit alias metadata. #163's unchanged class-key/table reader supplies each legal holding/runout label; blocked holdings retain the frozen sentinel. Only diagnostic card labels/maps change; history, menus, ranges, blueprint and v1 maps remain frozen.

All 120 fresh collections were regenerated, as required by the binding #163 PR comment; old grouped statistics were not reprojected. The legacy board rerun exactly reproduced its recorded equilibrium, v1 and fitted equity-50 numeric measurements for both seats. Each fresh collection passes its original equilibrium/blueprint/per-root reference checks. Every held-out job passes lock-only reference and regenerated-v1 reproduction checks; all 18 frozen replays pass global-statistics and best-response parity. No outcome-dependent roots, exclusions, budgets or K selection were used.

The separate auditor reopened raw lock-only ZIP members and recomputed all **1,440 seat metrics** from native float32 chip subtraction and scaling. It independently recomputed every aggregate, lineage, seat and paired contrast with scalar `fsum` weighting and bootstrap multiplicities. Maximum absolute disagreement with the reporter was **4.44 × 10⁻¹⁶ BB**. Archive/member identities, input provenance and qualification receipts are retained. The full readout keeps all 240 rows; the compact evidence preserves summary, coverage and counts.

## Resources, stop and continuation

Free M1 only: one serial native worker, six Rayon threads, nice 10, 4-GiB arena, 7-GiB worker/8-GiB owned-family RSS ceilings, at least 15 GiB free disk and at most 1 GiB system-wide swap growth from the original baseline. The fixed budget began October 7 at 18:06 CEST; the original scientific stop October 9 at 17:06 and hard closeout deadline at 18:06 were not extended.

The first storage admission failed its forecast and is retained. Existing owner-approved merged-evidence cleanup removed only 83 verified inactive #149 local copies, reclaiming 3.021 GiB; [restoration/dependency receipt](../artifacts/pr190-merged-pr149-cleanup.json). No open-PR root, active input, shared Git or synced file was removed. Earlier setup failures, including an executable-permission failure before scientific computation, remain archived.

The initial main run stopped during collection 20 on October 7 after 19 reference-qualified collections because the system-wide swap guard fired. Its baseline was 1,236,533,248 bytes; the absolute ceiling was 2,310,275,072. The owner reported concurrent computer use, but the telemetry cannot attribute the increase to a specific process. The old guard did not save the exact failing sample; its traceback establishes which guard fired. Logging was corrected before continuation, without changing any threshold or scientific behavior.

After the owner reported the M1 free and explicitly authorized resumption, [fresh admission](../artifacts/pr190-continuation-admission.json) passed under the original baseline/deadline. All 19 completed payloads, requests, identities and gates were reverified. The interrupted twentieth output was preserved and excluded; its original request was retried. Exactly one continuation launched October 7 at **22:32 CEST** and completed all remaining frozen work October 8 at **10:39 CEST**, with no new failure or recovery attempt. The owner later authorized resource-crash recovery, but none was needed. All original stop evidence remains in its [252-member ZIP receipt](../artifacts/pr190-resource-stop-archive.json).

Recorded passing telemetry from the original budget through main completion reached **6.24 GiB peak owned-family RSS**, **2.073 GiB maximum system swap**, and **20.45 GiB minimum free disk**. These are sampled values, not a claim to know the unlogged initial stop peak. Budget-to-main completion was 16.55 wall hours, including preparation and the resource interruption. The guarded serial independent readout followed; sustained native computation is finished. No M4 computation or paid resource was used.

## Interpretation and restoration

The global K50 tables recover the fitted50 witness advantage without fitting their buckets to these boards. Proceeding to a separately labeled card-key schema is supported. Before any full-game claim, step 3 still requires trained turn/river comparisons at matched visits per key or plateau with opponent-sampled averages; step 4 still requires a fresh direct match and the arena gate. This PR performs no training, changes no production schema and makes no strength or release claim.

All research belongs to `~/Local/Research-Cloud/PR-190-HU20-bucket-validation/`, as immutable member-hashed ZIPs. Exact hashes, Drive IDs, native/cloud acceptance and restoration commands are indexed in [RESULTS_INDEX](../../RESULTS_INDEX.md). Raw native responses use 258 separate ZIPs (120 collection, 120 lock, 18 replay); fitted policies and the full readout belong to the final evidence ZIP. Inputs/preparation/pilots and the original stop have separate archives. Local originals remain; synced files are never deleted or evicted. The 1,551-member final evidence ZIP and all 258 native payloads have passed fresh archive/member and native/cloud acceptance checks; [exact receipts and readout hashes](../artifacts/pr190-final-archive-receipt.json). The [29-member publication ZIP](../artifacts/pr190-publication-archive-receipt.json) also verifies and is uploaded; terminal archive receipts are checked into this PR. The first final-archive helper encountered a macOS extended-attribute API compatibility error before creating a ZIP; its failed source, lock and receipt remain retained. A separate corrected helper uses the native `xattr` command, with identical scientific data and guards.
