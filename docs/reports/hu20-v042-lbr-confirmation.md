# v0.4.2 LBR confirmation: waiting for runtime approval

**Final evaluation has not started.** The fresh outcome-blind pilot requires **24,576 LBR blocks per model** for projected half-width **4.475 BB/100**. Its measured quote is **10.09 hours**, including all replay/report costs and 25% headroom, above the owner's predeclared eight-hour admission limit. [Posted sizing and stop](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/188#issuecomment-6041747764). The proposed counts and roots are frozen; no pilot means, intervals or labels have been inspected. Owner approval for a 12-hour final cap is requested; it is not assumed.

[Protocol](../hu20-v042-lbr-confirmation.md), [blind quote](hu20-v042-lbr-artifacts/quote.json), [frozen sizing proposal](hu20-v042-lbr-artifacts/frozen-sizing-proposal.json), [canonical hash](hu20-v042-lbr-artifacts/frozen-sizing-proposal-hash.json), [admission stop](hu20-v042-lbr-artifacts/admission-stop.json).

## Verified inputs and unchanged science

Every one of **34,826,546 keys** across the six original #185 average exports matches an independent direct JSON parse for `entries`, `visits` and `zero_mass`, with zero mismatches. Exact bytes/SHA256 and #185 original manifest references agree. Current-main source `e74316fb8f919c77e52e82a5c6b0b4d0833a838c` was used for verification; all 433 archived Python source files separately match its checked checkout. Six guarded workers, at most three concurrent, finished in 136.4 seconds; peak RSS 0.627–1.043 GiB. [Per-model receipts](hu20-v042-lbr-artifacts/loader-verification-complete.json), [preflight and all scientific source hashes](hu20-v042-lbr-artifacts/preflight.json).

Execution science is pristine #176/#185 `523e4a347d012beb8cb523497d2e9a4cbb90035f` plus **only #186 compact storage**, snapshot `0621302417589fc3823765e939e8ad05ff6d8391`. Original uniform validation, play, probabilities, information boundary, bounded LBR opponent, reporter and independent auditor are unchanged. Current main's unrelated zero-mass-current and search changes are excluded. The unchanged reporter's legacy >0 LBR fields remain tool output; the owner's **lower >−5** safeguard is evaluated separately from the same unchanged interval.

Both Macs were searched for roots **202610077101 /202610077201**: 160,636 M1 and 77,545 M4 readable retained files, zero matches; current-main tracked configs/reports also have zero matches. Unreadable and oversized historical paths are retained in full private receipts. [Search summaries](hu20-v042-lbr-artifacts/root-freshness-m4-summary.json). Every #185 root and hand is excluded.

## Blind pilot and proposed final

Pilot root 202610077101: 512 duplicate LBR blocks/model plus 32 blocks for each of twelve non-LBR panels, **10,752 hands**, six complete workers, **637.38 seconds** wall time. The small non-LBR cells serve unchanged reporter/auditor compatibility only and do not replace #185's passed gates. Zero incomplete LBR decisions; peak policy RSS **1.034 GiB**. Only SDs, timing, memory and completeness were inspected. Pilot arithmetic/action audit follows the final count freeze; outcomes remain uninspected while admission is pending.

Equal-three-lineage paired pilot SD **357.933482** BB/100; lineage SDs **542.435095 /494.227385 /473.671043**. Student-t sizing, minimum 8,192 and 512-block rounding, selects **24,576 LBR blocks/model**, projected half-width **4.475239**, requested achieved ≤5. Final root is **202610077201**, with **294,912 LBR hands /299,520 total hands**. No historical/pilot pooling or outcome-driven extension.

Measured startup-inclusive three-worker play projects **7.7269 hours**; conservative every-pilot/final-action replay/report allowance **20.6848 minutes**; nominal **8.0716 hours**, **10.0895 hours with 25% headroom**. The eight-hour admission rule stopped final launch. There is no final finish timestamp while waiting for owner approval. Counts, margin, source, resources and fresh roots stay fixed.

## Current decision table

| Check | Evidence, BB/100 | Rule | Status |
| --- | --- | --- | --- |
| Direct three-lineage 10B−1B | #185 +3.50 [+1.63, +5.37] | Lower >0 | PASS |
| Fresh bounded LBR 10B−1B | Final not played | Lower >−5; requested width ≤5 | PENDING |
| Native pressure 10B−1B | #185 +4.49 [−1.91, +10.90] | Lower >−10 | PASS |
| Severe panel regression | #185 all other upper bounds ≥−20 | No upper <−20 | PASS |

#185's LBR +2.79 [−6.25, +11.83] remains the failed precision-limited historical safeguard; it is not pooled into this confirmation. [Full #185 report](hu20-o-10b.md). No new LBR estimate, inference package, release, tag, publication or default change. v0.4.1 remains Latest.

M4 is idle after pilot completion. Work and archive roots are indexed in [RESULTS_INDEX](../../RESULTS_INDEX.md); all originals are retained. The PR remains open because final runtime approval and dependent science/closeout are unresolved.
