# v0.4.1 confirmation for O

## Background

[#165](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/165) failed its frozen O−R rule only on native-pressure precision: **+0.44 [−18.47, +19.34] BB/100** at 2,048 blocks. This is inconclusive, not a measured regression. Fresh [#175](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/175) then measured O beating shipped R1 directly **+11.45 [+8.83, +14.07]**. #175 is merged at `88c6c9f6d4543ed77fa587405416aeb049e05ac3` with green checks and owner acceptance of [Claude’s analysis](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/175#issuecomment-6013778852) as its review.

That analysis separates the floor-related direct loss from the traverser-reach native-pressure drop: T ≈ shield ≈104 versus O ≈R ≈128 BB/100 on the earlier panels. Shield−T is a direct fixed-budget/extraction control. T/shield pressure absolutes come from different runs and roots; do not infer equivalence from their proximity. No shield−O direct comparison was run. The present campaign confirms O alone, without substitute arms or an automatic release.

## Predeclared release rule

Load **exact #165 O exports**, seeds **2026100601/02/03**, 1B-node **opponent-sampled average**, and all three **R** v0.4.0 B100M current lineages by manifest size/SHA256 and header/checkpoint provenance. R1 (2026093001; SHA256 `4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf`) is shipped v0.4.0. No training, checkpoint selection, policy re-extraction or new opponent.

O is eligible for an **unpublished v0.4.1 release preparation** only if ALL hold, using paired nominal Student-t **95%** intervals in BB/100:

1. **Fresh direct O vs shipped R1:** three-lineage aggregate lower bound **>0**. Keep #173/#175 direct runner, duplicate deals/seats, reporter and auditor unchanged. Report all three individual lineages as well.
2. **Bounded LBR O−R:** lower bound **>0**.
3. **Native-pressure O−R:** lower bound **>−10**.
4. **All other eleven panels:** no O−R upper bound **<−20**.

The panel play/loader/opponents/configuration are #165’s, with **R and O only**, each on the same paired blocks, both seats and three lineages. Generalize the reporting/audit enumeration to those declared arms; preserve every action, observation, settlement and raw-chip arithmetic check. No phantom C/T policies or silent substitutions. Intervals average lineages and seats within independent deal blocks, conditional on the three retained lineages. Nominal intervals and the unchanged original thresholds define this rule; no retrospective multiplicity adjustment or strength claim beyond these fixed opponents.

## Pilot, sizing and admission

Reserve and search tracked configurations/reports and readable retained plans before launch:

- Arena pilot **202610062001**; arena final **202610062101**.
- Direct pilot **202610062201**; direct final **202610062301**.

Pilot arena: **256 LBR**, **2,048 native-pressure**, **64 blocks per other panel** for each of R/O’s three lineages. Direct pilot: **2,048 blocks** for each O versus R1 pair. Inspect only paired block SDs, runtime, memory, completeness/failures; never pilot means, confidence intervals or labels. Pilot deals are excluded from final poker estimates. The lock-only cost pilot below uses the first sorted fixed board and exposes cost only.

Outcome-blind final sizing from pilot SDs, solving Student-t(0.975,n−1) × SD / sqrt(n) and rounding upward:

- **Native pressure:** target **7 BB/100** projected half-width for headroom against requested ≤8; minimum **12,288**, multiples of **4,096**. Size the three-lineage O−R aggregate.
- **LBR:** target **9 BB/100** for headroom against ≤10; minimum **2,048**, multiples of **512**. Size the three-lineage O−R aggregate.
- **Other eleven:** freeze #165’s **256 blocks each**.
- **Direct:** target **3.5 BB/100** for headroom against ≤4; maximum SD across the three individual pair series and their aggregate, minimum **32,768**, multiples of **4,096**.

These are projections; report achieved widths with the fixed sample and no outcome-driven extension. Post all counts, distinct roots, startup-inclusive costs, exact plans and canonical bundled hash **before final play**. Use all observed lineage costs for a three-worker schedule estimate, add independent replay/report and full turn/river costs, plus **25% headroom**. If that projection exceeds **3 hours**, post it and **wait for owner instruction before any final play**. Otherwise freeze an aggregate **3-hour final compute cap**; retain partials if any guard fails. Pilot has a separate 90-minute cap.

All compute uses the **M4**, free local only, **no RunPod**. At most **three workers**, **6 GiB RSS per worker**, **≥15 GiB free disk**, with guarded aggregate deadlines and no concurrent competing measurements. A sequential native turn/river worker uses the same 6-GiB job ceiling, replacing the old 7-GiB admission guard while keeping the native requests, trees, ranges and arithmetic unchanged. Preserve every failure and partial; never bypass a check.

## Independent verification and progress report

Replay and audit **every pilot and final action and settlement** with the #175/#173 checks. Independently recompute all aggregate/lineage/position statistics, panel absolute values, direct labels and rule checks from raw chips, without loading policies or invoking the production reporter. Audit pilot arithmetic only after final counts are frozen. Record all missing-key/zero-mass counts and bounded-LBR incomplete/soft-budget telemetry.

For O seed **2026100601** versus shipped **R1**, use the identical **#149 lock-only pipeline #173 used for shield**: all 40 frozen turn/river boards, both seats, common original B500M ranges/tree/reference, qualified binary SHA256 `fb32974d9d211fa66001d1efb330ec4af2d825005d24b287dc6cf3c37fa8812f`, no new solving. Pin prepared inputs and reference inventory. Replace only the candidate’s locked probabilities with exact O’s export. Legacy native metric label `cfrplus` means O in this run and must be displayed as O. E is best-response gain over the retained approximate reference; Q=(E−P)/(B−P) uses the same B/P anchors, with the identical paired 2,000 board-bootstrap draws. Lower is better; this is conditional turn/river distance, not full-game exploitability. Independently audit all 160 metrics and bootstrap arithmetic, pinning O’s actual hash explicitly instead of the shield-specific auditor constant.

Final report: rule table with pass/fail per check, all 13 panels with O/R absolute and O−R intervals, direct aggregate/per-lineage results, readable head-to-head progress, bounded-LBR attacker win rates (negative target profit; lower-bound weakness, lower is better), E/Q and common B/P anchors, and a short O/shield/T comparison distinguishing direct controls from different-root panel descriptions.

If all four checks pass, prepare artifacts/hashes/manifests/release notes per ROADMAP **without publishing**. The owner must review the numbers and explicitly approve in chat before any tag, GitHub release, default-model change or publication; **v0.4.0 stays stable**. If any check fails, retain the descriptive result, with no release preparation or substitute arms. No further experiment is automatically authorized.

Archive all exact inputs, source, environment, plans, pilot/final raw traces, logs, failures, independent audits, reports and any unpublished release preparation into a member-hashed ZIP at `~/Local/Research-Cloud/PR-<n>-HU20-v041-O-confirmation/`; verify every member by readback. Keep originals; **never delete or evict synced files**. Owner-approved PR comments are binding.
