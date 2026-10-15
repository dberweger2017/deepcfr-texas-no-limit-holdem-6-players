# v0.5.3 package readiness

**Package prepared for owner review. No release, tag or publication; the owner decides merge and separate release approval.**

The owner declared PR272’s original seed 2026100601 without selection. The verified average is 3,196,291,551 bytes, SHA256 `c3a8edc4cc78098f07ac3e4ecbbb3dacae298a266b9fcd5fc788d0e2b6fcfc8e`; the task’s earlier byte count belonged to its parent. Restored archive, manifest, model and K50 table pins verify. All seven published v0.5.2 assets verify. Real v0.5.0–v0.5.2 main/branch trace hashes match and every compatibility hand replays; old manifests rebuild byte-identically.

All 720,896 original final hands independently replay and exactly reproduce before readout. Seven paired translation on-minus-off contrasts are 0.00 [0.00,0.00] BB/100, passing the declared safeguard. Fresh direct against translated v0.5.2 is descriptively +37.10 [30.16,44.04] BB/100. Independent statistics agree exactly. Random alone made off-menu raises: 8,231 all-ins per arm; 8,222 v2 attempts selected no witness. The other six styles were on-menu. No v2 translation strength benefit is measured. [Full report](../../reports/hu100-v053-translation.md).

PR296’s fixed-seed direct gains were +47.04 [40.13,53.94], +40.95 [34.02,47.87] and +45.42 [38.49,52.36] BB/100. Scripted opponents only; no external HU100 benchmark. Half-pot and matched-pot margins are smaller, and intervals condition on fixed policies/seeds. The original matched-pot contrast −26.57 [−40.79,−12.34] does not establish 20-BB/100 non-inferiority. Human smoke checks establish functionality, not strength.

## Final package

The 12 assets are built from clean source `ffa9ff7d9b68cd96c9cec218088f5a62fc6dcca2` in:

`/Users/dberweger/Local/hu100-v053-release-package/results/v053-release-package/assets/`

Every asset’s exact bytes/SHA256 is in the [inventory](verification/asset-inventory.json) and PR description. Three deterministic parts remain below GitHub’s 2 GiB limit; standalone `python -I -S` checks the parts/tables and freshly reassembles the pinned whole model outside the immutable asset directory. Later integration/readiness commits preserve tested runtime and bundled source bytes. [Bundle verification](verification/bundle-verification.json).

## Real M1 runtime

| Fresh CLI load | Seconds | Cumulative peak CLI child RSS |
|---|---:|---:|
| 1 | 587.862 | 1,735,606,272 bytes |
| 2 | 581.969 | 1,735,606,272 bytes |

The real CLI loaded the complete pinned bundle with translation on. Restart before a pending bot decision preserved the session and exact idempotent reply. Restricted v2 sizing, free off-menu wagers, all-in play and spectator hands were exercised through HTTP and the browser. The successful run’s human-facing bot decisions were 37 exact, five translated and three uniform; the earlier attempt had 27 exact, two translated and two uniform. These are functional checks. The browser record and screenshots remain in the accepted evidence archive. Wrong model, table and adapter identity are refused. [Runtime](verification/runtime-summary.json), [refusals](verification/runtime-refusals.json).

Every retained journal from the failed and successful runtime attempts independently replays and reproduces decisions, including partial hands. A fresh real policy reader is shared only for these independent audits; each journal reconstructs its own RNG. Verified snapshots protect the originals, whose bytes and modification times remain unchanged. [All journal audits](verification/all-runtime-journals-audited.json).

| Runtime attempt | Human completed | Spectator completed | Partial | Sessions |
|---|---:|---:|---:|---:|
| Successful | 15 | 9 | 0 | 8 |
| Original failed | 12 | 1 | 1 | 4 |

Total: **37 completed hands plus one partial**, all independently audited.

## Failures and resources

The first runtime attempt completed both CLI loads, restart recovery and 12 human hands, then failed at the second spectator hand’s first postflop decision. Spectator classification and independent audit omitted the policy-owned K50 tables. The regression fails before the correction; all 188 relevant tests pass after both call sites receive the tables. The full functional runtime was repeated after the fix; no scientific hands were rerun. The first journal-audit reporting helper also hit an exclusive-output file collision after the old attempt verified. Separate raw/enhanced report paths and a model-free output-contract check correct that administrative failure; the complete real audit is repeated. Original failed journals, logs, partial hand and initial package metadata remain. The initial package is byte-identically reconstituted through verified payload hardlinks. [Failure/correction](verification/runtime-failure-000.json), [all prior engineering failures](verification/failures.json). [Closeout failures and recovery](verification/closeout-failures.json) include the macOS xattr preflight, initially incomplete native upload metadata and the upload-observer window.

Model restoration, science, builds and real runtime checks ran on M1 under `caffeinate -i` and the shared lock. Battery operation was permitted. Across 2,534 observations, peak sampled family RSS was 5,919,965,184 bytes, minimum free disk 21,205,139,456 bytes and minimum battery 20%. No sustained memory, disk, swap or pressure threshold was breached. The initial 30-minute upload-observer window expired during Drive recovery; metadata-only monitoring continued under the same resource/battery guards. Scientific counts and budget stayed fixed. [Resource summary](verification/resource-summary.json) retains every guard status and links the raw observations through the science archives.

Completed regenerable policy caches were retired only after input/inventory/dependency and live-handle checks; canonical models/tables and unique evidence remain. Inactive package payloads were temporarily retired for the corrected rebuild, then the initial package was restored with verified hardlinks and no duplicate payload allocation. No synced file deletion or forced offload. [Cache receipts](verification/cleanup-primary-receipt.json), [package receipt](verification/package-rebuild-retirement-receipt.json).

## Accepted evidence

- [evidence archive](https://drive.google.com/file/d/1V_ma54XVi0PTAFUDLe8nJJt1L5ADfGGd/view): `hu100-v053-package-evidence-M1-20261015.zip`, 603,730,754 bytes, SHA256 `8704a092b928de163bc8955dbd5fa2d4b2b965f4f0455996a1f857c420418599`; embedded `ARCHIVE-MANIFEST.json` SHA256 `c11cd598f329e905d4f63c3f7ab3100d01e74d10796645321edfe8060569184b`.
- [lifecycle archive](https://drive.google.com/file/d/1BQUpP4dlG-mrPPF1H3P59rMC3bzOBrIN/view): `hu100-v053-package-lifecycle-M1-20261015.zip`, 262,131 bytes, SHA256 `2670a634a6a2d4e958bf61c4876f289b3a146584bc268149e2de3d1805e1fc23`; embedded `ARCHIVE-MANIFEST.json` SHA256 `8ce1793af80459502c9ac566f5c219e81148d1f2a9960b7bfce5ec1d667ab58c`.

The native Drive client remained at “Loading your account…” until a single normal quit/reopen recovered it, following [Google’s basic restart procedure](https://support.google.com/drive/answer/2565956?hl=en). The exact sealed ZIP was first backed up outside the synced folder. No account, permission or credential change was made; two preexisting Lost and Found items were untouched. All local member sizes and hashes read back. Native uploaded/not-uploading/no-conflicts state and independent Drive ID/name/size/parent confirmation separately establish upload acceptance. Remote archive bytes were not downloaded. Exact restoration commands and dependencies are in [RESULTS_INDEX](../../../RESULTS_INDEX.md). All local unique evidence and active inputs remain retained. The lifecycle archive’s own complete final guard observations and upload proof are in the [Git closing receipt](verification/archive-lifecycle-closing.json); its temporary idle-sleep inhibitor had already expired normally.

HU20’s default remains v0.4.2. No GitHub release, draft release, tag, publication or merge was performed.
