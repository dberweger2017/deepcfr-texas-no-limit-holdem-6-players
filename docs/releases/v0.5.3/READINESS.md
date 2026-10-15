# v0.5.3 package readiness

**Science, final assets and real M1 runtime verification complete. Archive upload acceptance is pending; this is not ready to merge or publish.**

The owner declared PR272’s original seed 2026100601 without selection. The verified average is 3,196,291,551 bytes, SHA256 `c3a8edc4cc78098f07ac3e4ecbbb3dacae298a266b9fcd5fc788d0e2b6fcfc8e`; the task’s initial byte count belonged to its earlier parent. All input archive/manifest/model/table pins and all seven published v0.5.2 assets verify. All three real v0.5.0–v0.5.2 trace hashes match, every compatibility hand replays and old manifests rebuild byte-identically. [Trace receipts](verification/real-release-traces.json).

All 720,896 original final hands independently replay and exactly reproduce before readout. Seven paired translation on-minus-off contrasts are 0.00 [0.00,0.00] BB/100, passing the declared upper-bound safeguard. Fresh direct against translated v0.5.2 is descriptively +37.10 [30.16,44.04] BB/100; independent statistics agree exactly. No v2 witness was selected in the scripted experiment; random alone made off-menu raises, 8,231 conditional-menu all-ins per arm, with 8,222 unresolved lookup attempts. [Full science report](../../reports/hu100-v053-translation.md).

PR296’s fixed-seed direct gains were +47.04 [40.13,53.94], +40.95 [34.02,47.87], +45.42 [38.49,52.36] BB/100. Scripted opponents/internal model matches only; no external HU100 benchmark. Half-pot and matched-pot margins are smaller; intervals condition on fixed policies/seeds. The original matched-pot contrast −26.57 [−40.79,−12.34] does not establish 20-BB/100 non-inferiority. Human smoke is functionality, not a strength test.

## Final assets

The 12 assets are built from clean source `ffa9ff7d9b68cd96c9cec218088f5a62fc6dcca2` in:

`/Users/dberweger/Local/hu100-v053-release-package/results/v053-release-package/assets/`

Every asset’s bytes/SHA256 is in the [inventory](verification/asset-inventory.json) and PR description. Three deterministic model parts are below GitHub’s 2 GiB limit. Standalone `python -I -S` verifies parts/tables and freshly reassembles the pinned whole model outside the immutable asset directory. [Bundle verification](verification/bundle-verification.json). Later integration/docs retain byte-identical runtime and bundled source; [source comparison](verification/package-source-equivalence.json). All five integration-head CI checks pass; [receipt](verification/package-source-ci.json).

## Real M1 verification

| Fresh CLI load | Seconds | Cumulative peak CLI child RSS |
|---|---:|---:|
| 1 | 587.862 | 1,735,606,272 bytes (1.62 GiB) |
| 2 | 581.969 | 1,735,606,272 bytes (1.62 GiB) |

Real restart before a pending 550-chip off-menu bot decision preserved the exact idempotent reply in a new PID. The HTTP driver completed 20 hands, including eight spectator hands with postflop play. Four native-browser hands completed restricted v2 400-chip sizing, free 800-chip off-menu sizing, a 10,000-chip all-in and spectator stepping. Wrong model, table and adapter identity are refused. [Runtime](verification/runtime-summary.json), [real refusals](verification/runtime-refusals.json).

Every retained runtime journal independently replays and reproduces decisions: 37 completed hands plus the retained partial spectator hand, across both failed and successful attempts. Verified snapshots protect the original journals, whose bytes and modification times remain unchanged. The successful run’s human-facing bot decisions were 37 exact, five translated and three uniform; the earlier attempt had 27 exact, two translated and two uniform. [All journal audits](verification/all-runtime-journals-audited.json). The fresh independent reader took 565.241 s; its OS peak RSS was 1,790,656,512 bytes. Each journal reconstructs its own RNG.

The first runtime attempt failed at its second spectator hand’s first postflop decision after both loads, restart and 12 human hands completed. Spectator classification and its independent audit omitted the policy-owned K50 tables. The regression fails before correction; all 188 relevant tests pass after the fix. Original failed journals, logs and initial bundle remain retained; verified hardlinks reconstitute the old bundle’s identical payloads. [Failure/correction](verification/runtime-failure-000.json).

The journal-audit reporting helper subsequently collided with an existing summary after the old attempt verified. Separate output paths and a model-free contract check corrected it; the full real audit was repeated successfully. The archive preflight’s unsupported macOS Python xattr API was replaced with native xattr. Native provider fields omitted while uploading are treated as unknown, never accepted. All original failure/source records remain. No scientific hand rerun. [Earlier failures](verification/failures.json), [closeout failures](verification/closeout-failures.json).

## Archive and resources

The evidence ZIP is locally verified: 603,730,754 bytes, SHA256 `8704a092b928de163bc8955dbd5fa2d4b2b965f4f0455996a1f857c420418599`. All 630 member sizes/hashes read back; embedded `ARCHIVE-MANIFEST.json` SHA256 `c11cd598f329e905d4f63c3f7ab3100d01e74d10796645321edfe8060569184b`. [Local receipt](verification/archive-evidence-local-pending.json).

It is in the designated [PR298 Research-Cloud folder](https://drive.google.com/drive/folders/1bK_JcDKPW16zIvwRGZyKli3Mp690ZQH-). The native client still reports uploading, with no completed Drive file ID. Its UI shows “Loading your account…”; its error list reports no current sync error. Independent cloud search has no completed file yet. **No upload acceptance is claimed.** Native and independent cloud ID/name/size/parent confirmation, the lifecycle receipt and final RESULTS_INDEX locators remain required. Remote archive bytes have not been downloaded. All local originals remain.

All model/science/runtime/build work ran on M1 under `caffeinate -i`, the shared lock and the declared 60-second sustained guards. Before archiving, 2,161 observations show no sustained breach; peak sampled family RSS 5,919,965,184 bytes. [Closed-guard resource summary](verification/resource-summary-before-archive.json). Upload monitoring continues with the same host guards; battery operation is permitted, stopping cleanly at 10% or below.

Inactive reproducible caches/build payloads were retired only after unchanged inventories, input/dependency and live-handle checks, with exact rebuild receipts. The primary caches reclaimed 10,063,863,808 allocated bytes. All canonical models/tables and unique evidence remain. Initial package payloads were temporarily retired for the corrected rebuild, then restored through verified hardlinks; no net package-space reclamation is claimed. No synced deletion or forced offload. [Cache cleanup](verification/cleanup-primary-receipt.json), [package rebuild](verification/package-rebuild-retirement-receipt.json).

HU20’s default remains v0.4.2. No release, draft release, tag, publication or merge. The owner decides merge and separate release approval after readiness is complete.
