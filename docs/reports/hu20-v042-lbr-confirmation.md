# v0.4.2 LBR confirmation: safeguard passes narrowly

**Fresh bounded LBR passes the unchanged safeguard:** matched-seed equal-three-lineage 10B−1B target-profit difference **−1.091512 [−4.965223, +2.782198] BB/100**. Lower >−5 passes by **0.034777 BB/100**; half-width **3.873710** meets the requested ≤5 and projected-sizing target ≤4.5. This is a narrow non-regression safeguard pass; the point estimate favors 1B, and LBR improvement is not established. Combined with #185's already-passed checks, all four candidate checks pass. The fixed first-seed inference package is prepared for owner review; **publication is not authorized**.

Final play started **October 7, 18:07:32 Madrid (CEST)** and all scientific reports/replay audits completed **October 8, 01:57:25**, in **7 hours 49 minutes 52 seconds**, within the owner-approved 12-hour cap and **06:05:45** hard deadline. Play finished at **01:55:02**. The original conservative 04:13 ETA and updated throughput forecast 02:15 are retained; actual completion preceded both. The detached M4 notifier posted [verified completion](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/188#issuecomment-6049257177) at 01:57 without M1. [Completion receipt](hu20-v042-lbr-artifacts/final-complete.json), [independent final audit](hu20-v042-lbr-artifacts/final-audit.json).

The original eight-hour admission stop is preserved; the owner amended only runtime admission/cap before final play. Counts, roots, source, checks and resource guards remained fixed. [Runtime freeze](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/188#issuecomment-6041799667), [final plan](hu20-v042-lbr-artifacts/final-plan.json), [owner approval](hu20-v042-lbr-artifacts/owner-runtime-approval.json), [actual start](hu20-v042-lbr-artifacts/final-start.json).

[Protocol](../hu20-v042-lbr-confirmation.md), [blind quote](hu20-v042-lbr-artifacts/quote.json), [frozen sizing proposal](hu20-v042-lbr-artifacts/frozen-sizing-proposal.json), [canonical hash](hu20-v042-lbr-artifacts/frozen-sizing-proposal-hash.json), [admission stop](hu20-v042-lbr-artifacts/admission-stop.json).

## Verified inputs and unchanged science

Every one of **34,826,546 keys** across the six original #185 average exports matches an independent direct JSON parse for `entries`, `visits` and `zero_mass`, with zero mismatches. Exact bytes/SHA256 and #185 original manifest references agree. Current-main source `e74316fb8f919c77e52e82a5c6b0b4d0833a838c` was used for verification; all 433 archived Python source files separately match its checked checkout. Six guarded workers, at most three concurrent, finished in 136.4 seconds; peak RSS 0.627–1.043 GiB. [Per-model receipts](hu20-v042-lbr-artifacts/loader-verification-complete.json), [preflight and all scientific source hashes](hu20-v042-lbr-artifacts/preflight.json).

Execution science is pristine #176/#185 `523e4a347d012beb8cb523497d2e9a4cbb90035f` plus **only #186 compact storage**, snapshot `0621302417589fc3823765e939e8ad05ff6d8391`. Original uniform validation, play, probabilities, information boundary, bounded LBR opponent, reporter and independent auditor are unchanged. Current main's unrelated zero-mass-current and search changes are excluded. The unchanged reporter's legacy >0 LBR fields remain tool output; the owner's **lower >−5** safeguard is evaluated separately from the same unchanged interval.

Both Macs were searched for roots **202610077101 /202610077201**: 160,636 M1 and 77,545 M4 readable retained files, zero matches; current-main tracked configs/reports also have zero matches. Unreadable and oversized historical paths are retained in full private receipts. [Search summaries](hu20-v042-lbr-artifacts/root-freshness-m4-summary.json). Every #185 root and hand is excluded.

## Blind pilot and fixed final

Pilot root 202610077101: 512 duplicate LBR blocks/model plus 32 blocks for each of twelve non-LBR panels, **10,752 hands**, six complete workers, **637.38 seconds** wall time. The small non-LBR cells serve unchanged reporter/auditor compatibility only and do not replace #185's passed gates. Zero incomplete LBR decisions; peak policy RSS **1.034 GiB**. Only SDs, timing, memory and completeness were inspected. All pilot/final arithmetic and action audits ran after final play; outcomes stayed uninspected during execution and sizing. Pilot means/intervals were first available for closeout after completion and are excluded from the final estimate.

Equal-three-lineage paired pilot SD **357.933482** BB/100; lineage SDs **542.435095 /494.227385 /473.671043**. Student-t sizing, minimum 8,192 and 512-block rounding, selects **24,576 LBR blocks/model**, projected half-width **4.475239**, requested achieved ≤5. Final root is **202610077201**, with **294,912 LBR hands /299,520 total hands**. No historical/pilot pooling or outcome-driven extension.

Measured startup-inclusive three-worker play projects **7.7269 hours**; conservative every-pilot/final-action replay/report allowance **20.6848 minutes**; nominal **8.0716 hours**, **10.0895 hours with 25% headroom**. The eight-hour admission rule initially stopped final launch. The owner then approved this exact fixed evaluation with a 12-hour final cap. Counts, margin, source, resources and fresh roots stay fixed.

## Updated v0.4.2 decision table

| Check | Evidence, BB/100 | Rule | Status |
| --- | --- | --- | --- |
| Direct three-lineage 10B−1B | #185 +3.50 [+1.63, +5.37] | Lower >0 | PASS |
| Fresh bounded LBR 10B−1B | #188 −1.091512 [−4.965223, +2.782198]; half-width 3.873710 | Lower >−5; requested half-width ≤5 | PASS |
| Native pressure 10B−1B | #185 +4.49 [−1.91, +10.90] | Lower >−10 | PASS |
| Severe panel regression | #185 all other upper bounds ≥−20 | No upper <−20 | PASS |

#185's LBR +2.79 [−6.25, +11.83] remains the failed precision-limited historical safeguard; it is not pooled into this confirmation. [Full #185 report](hu20-o-10b.md). The 32-block non-LBR compatibility cells are not new decision evidence. The unchanged reporter/auditor preserve their legacy `lbr_lower_above_0=false`, `native_pressure_lower_above_minus_10=false` and `release_rule_passed=false` fields. These refer to the reporter's original >0 rule and the tiny compatibility cells. The predeclared #188 rule uses the unchanged LBR interval with the owner's unchanged −5 margin; pressure and severe-regression evidence remain #185. No failing field was rewritten or waived.

## Fresh LBR detail and interpretation

| Matched training seed | 10B−1B target profit, BB/100 (95% interval) |
| --- | --- |
| 2026100601 (fixed package seed) | −3.439331 [−9.403488, +2.524826] |
| 2026100602 | +4.900106 [−1.069029, +10.869241] |
| 2026100603 | −4.735311 [−10.786142, +1.315520] |
| Equal-weight three-lineage aggregate | **−1.091512 [−4.965223, +2.782198]** |

Aggregate by seat: button **+2.397325 [−3.270176, +8.064827]**; big blind **−4.580349 [−9.922874, +0.762175]**. Seed 603 big blind is **−10.091146 [−18.367937, −1.814354]**. These descriptive strata are retained, not additional release gates or a reason to change seed/counts. The declared rule concerns the equal-weight aggregate only; it does not establish each lineage's non-regression.

Attacker return is **26.188829 [21.083223, 31.294436]** against 10B and **25.097317 [20.211917, 29.982717] BB/100** against 1B. Lower is better. Target-profit difference reverses the attacker difference. Intervals are nominal paired Student-t 95% intervals over independent duplicate deal blocks, averaging two seats and all three matched lineages equally. They are conditional on the three saved lineages and fixed bounded attacker, not population uncertainty over training seeds or a full-game exploitability certificate.

## Audit, resources and retained failures

Final **299,520 hands /1,404,549 actions**, including **294,912 LBR hands**; excluded pilot **10,752 hands /52,008 actions**. All **310,272 hands /1,456,557 actions and settlements** independently replay, with raw arithmetic/interval agreement and pinned plan/source hashes. Every final worker completes its frozen count. Final LBR coverage has **644,848 attacker decisions**, zero incomplete or over-soft-budget decisions; **740,145 target decisions**, including five missing keys and six zero-mass keys using the unchanged uniform fallback.

At most three detached M4 players; worker peak **1.024323 GiB**, report/replay peak **0.213 GiB**, all below the **3-GiB per-process guard**. The ≥15-GiB disk guard and the aggregate cap passed throughout checked execution. No RunPod or paid compute. The first launch stopped before any final hand on generated pilot caches; its failure and correction are retained in [launch receipt](hu20-v042-lbr-artifacts/final-launch-attempt-01.json). Only generated `__pycache__/` files were locally ignored; all pinned scientific source/input hashes reverified before the same fixed plan successfully launched. Prior transfer/probe failures and the original admission stop are retained in the complete archive.

Subsequent coordination merges from main never enter the frozen M4 scientific snapshot. Independent source-hash checks preserve all 403 scientific Python files. No scientific restart, historical pooling, extra samples, substitute check or outcome-driven extension occurred.

## Unpublished inference package and archival closeout

The first-seed **2026100601** exact 10B average export is packaged with checksums, preparation manifest, standalone verifier, [model card](../releases/v0.4.2/MODEL_CARD.md) and [release notes](../releases/v0.4.2/RELEASE_NOTES.md), following #176/#181 preparation. The model bytes/SHA256 remain #185's pinned export; only its external filename is descriptive. No model binary enters Git. Preparation retains `owner_publication_approval=false`, null approved release source and `unpublished-owner-review`. There is no tag, release, publication or runtime default change; v0.4.1 remains Latest.

Complete member-hashed archives, native upload acceptance, independent cloud metadata and exact retrieval instructions are recorded in [RESULTS_INDEX](../../RESULTS_INDEX.md). All local originals and other PR dependencies are retained. No deletion or eviction. The owner authorized merge after report/analysis and green CI, while publication still requires a separate explicit go.
