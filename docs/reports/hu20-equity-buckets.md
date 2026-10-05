# HU20 full-deck equity-distribution bucket build

The frozen 52-card build at commit `4ade72cc18e512ed9f11fa6afa83c701a55069c8` completed successfully. All six tables were retrieved and hash-verified on M4 before the only active rental was terminated. Every table loads, every sampled situation is found, and no bucket is empty. This is build and retrieval evidence; playing-strength validation and training remain unrun. [Draft PR #163](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/163) remains a draft.

## Build and resources

The reviewed builder and reader were unchanged. `pod_build.sh` ran the five release Rust tests, including exhaustive enumeration of all 133,784,560 seven-card hands; [the retained log](hu20-equity-buckets-artifacts/build.log) records all five passing. The prior pod Python checks also passed, as recorded in the takeover state; closeout independently exercises the exact frozen Python reader on M4.

Settings: 52 cards, 13 ranks, 50 histogram bins, K=50/200, seed `202610050003`, 100 maximum Lloyd iterations, 23 cgroup-sized CPU threads. The RTX 3090 was unused. [Method](../hu20-equity-buckets.md), [complete summary](hu20-equity-buckets-artifacts/summary.json).

| Stage | Elapsed seconds |
|---|---:|
| River equities and classes | 8.4 |
| Turn features | 12.3 |
| Flop features | 11.7 |
| River K=50 clustering and table write | 35.7 |
| River K=200 clustering and table write | 37.4 |
| Turn K=50 clustering and table write | 125.1 |
| Turn K=200 clustering and table write | 408.7 |
| Flop K=50 clustering and table write | 12.0 |
| Flop K=200 clustering, table write and final summary | 38.5 |
| Table generation total | 689.8 |

Durations are differences between the builder's one-decimal progress timestamps, including output work between markers. The river stages finish at cumulative 105.5 seconds, including feature construction; river clustering/write itself takes 73.1 seconds. Release compilation began at 07:07:01 UTC, table generation at 07:07:04, checksum generation at 07:18:34, and the log ends `BUILD COMPLETE` at **07:18:44 UTC, October 5, 2026**. Table launch through final checksums is 700 seconds.

[Peak RSS](hu20-equity-buckets-artifacts/peak-rss.txt): **10,822,724 KiB / 10.321 GiB**, sampled every five seconds by the existing builder wrapper. This is an observed builder peak, not an exact continuous maximum or total host memory.

| Street | Suit-isomorphism classes | Raw situation orbit weight |
|---|---:|---:|
| Flop | 1,286,792 | 25,989,600 |
| Turn | 13,960,050 | 305,377,800 |
| River | 123,156,254 | 2,809,475,760 |

## Bucket balance and objectives

Masses below are each bucket's orbit-weighted fraction of all raw situations, read from `summary.json`; they are not unweighted class counts. Mean equities and objectives also come from that file. Objectives use scalar equity distance on river and the sum of absolute cumulative-histogram differences on flop/turn, so compare K within a street rather than objective magnitudes across streets. Rounded mass fractions sum to one within 5×10⁻⁸ in the retrieved summary.

| Table | Smallest mass % | Largest mass % | Bucket mean equity range | First → final objective | Iterations recorded |
|---|---:|---:|---:|---:|---:|
| flop-k50 | 0.502078 | 3.778419 | 0.176843–0.943486 | 1.624189956 → 1.279221833 | 100 |
| flop-k200 | 0.089667 | 1.497815 | 0.152153–0.959101 | 1.135196136 → 0.911069185 | 100 |
| turn-k50 | 0.507094 | 4.061798 | 0.106366–0.960453 | 1.750486871 → 1.524016969 | 100 |
| turn-k200 | 0.057737 | 1.830054 | 0.081936–0.978763 | 1.200746986 → 1.067384533 | 100 |
| river-k50 | 0.963216 | 3.758274 | 0.004667–0.988918 | 0.006167766 → 0.005129313 | 21 |
| river-k200 | 0.105605 | 1.556025 | 0.004377–1.000000 | 0.001358701 → 0.001136975 | 12 |

All six have positive summary mass in every bucket. A separate full scan of the binary bucket IDs confirms every ID is in range and every bucket contains at least one class. Turn and flop reach the configured 100-iteration limit; this report does not claim optimization to convergence. River K=200's final recorded objective has a tiny increase (~3.28×10⁻⁹) over the previous iteration; the complete objective sequences are retained, with no filtering or rerun.

## M4 retrieval and sanity checks

The whole `/workspace/hu20-buckets-out/` folder and `/workspace/build.log` were copied by `scp` **initiated on M4**, using direct port 8884, into:

`/Users/dberweger/Local/hu20-equity-buckets-20261005/`

The builder's ten `SHA256SUMS` entries all pass `shasum -a 256 -c SHA256SUMS`. An additional SHA256 inventory generated on the pod covers every output file, the original checksum file itself, and `build.log`. On M4, **all 16 files / 2,768,097,096 bytes match with zero mismatches**, verified at 2026-10-05T07:27:41.432939+00:00. [Retrieval receipt](hu20-equity-buckets-artifacts/retrieval.json), [additional inventory](hu20-equity-buckets-artifacts/retrieval-SHA256SUMS). M4 had **34.362 GiB free** after verification, above the required 20-GiB floor.

The sanity run uses the exact `src/blueprint/equity_buckets.py` reader from the build commit, with SHA256 recorded in [the receipt](hu20-equity-buckets-artifacts/sanity.json), NumPy 1.26.4, and seed `202610050004`:

- Loads all six memory-mapped tables and checks their header street, K, class count and exact binary length.
- Samples 5,000 distinct-card deals per street and looks up each in both K variants: **30,000 successful lookups, zero missing situations**. This random sampling is a sanity check, not an exhaustive lookup proof.
- Scans every stored bucket ID for validity and occupancy: **zero empty buckets**. The smallest unweighted buckets have 6,927/1,499 flop classes, 71,734/8,868 turn classes and 1,173,998/142,849 river classes for K=50/200.
- River bucket means span **0.004667–0.988918 (K=50)** and **0.004377–1.000000 (K=200)**, meeting the near-zero/near-one check.

## Rental closeout and cost

RunPod MCP confirmed pod `90gceq0t9mqz6q`, community RTX 3090, $0.22/hour, started at **06:47:47.739 UTC**. The verified-retrieval receipt preceded deletion. MCP deletion returned 204 at **07:28:24.103 UTC**, follow-up get returned 404, and the complete unpaginated pod list no longer contained the target. The four September 24–25 pods remain `EXITED`, with unchanged IDs and statuses. [Closeout receipt](hu20-equity-buckets-artifacts/termination.json); complete MCP responses remain in M4 `verification/`.

The start-to-delete-response lifetime upper estimate is **2,436.364 seconds / 40 minutes 36 seconds / 0.676768 hours**. **$0.22 × 0.676768 = $0.148889 estimated compute cost**. This rate-times-lifetime estimate is not a settled billing total and does not add unobserved storage charges. No other rental was created or changed.

## SHA256

The exact original manifest is [retained here](hu20-equity-buckets-artifacts/SHA256SUMS). Table and summary hashes:

| File | SHA256 |
|---|---|
| `flop-k200.bin` | `4fb24d2c738b4e8913eb4524648c75f815c2fb2f1f7bd3e1be8dd4133722c975` |
| `flop-k50.bin` | `084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111` |
| `river-k200.bin` | `b0591710df149248b2feb89f0a9c808f9745be1f7e5c4eb63aa8a2d93b78864e` |
| `river-k50.bin` | `70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629` |
| `summary.json` | `4417a6f119b1c6b416c030f4984751dc951edbc9db3fd9436e6fe17cca82946a` |
| `turn-k200.bin` | `ccec40009426fd64cac53816e6a286adec159eb2a22c479bb273029faa1b7f6c` |
| `turn-k50.bin` | `c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f` |

## Archive staging

A whole lossless archive contains **29 file members / 2,768,143,080 logical bytes**, including the frozen reader, verification scripts, original build logs and complete MCP closeout responses. Every archived member and every original source reverified against the archive manifest. The compressed archive is **1,991,949,530 bytes**, SHA256 `03d4828f030082f213c8ce82e29fa0fa8fd97288bd8035ad0056de85bc0816ca`; manifest SHA256 `a5d64a7882da34840dad53905e4a03284f822487732ce691ecc82cdf1585bdef`.

Originals remain at the M4 folder above. A separate local archive remains at `/Users/dberweger/Local/hu20-equity-buckets-closeout-20261005/hu20-equity-buckets-20261005.tar.gz`. A separate APFS copy was SHA256-verified and staged through native Drive desktop at `~/Local/Research-Cloud/PR-163-equity-buckets/hu20-equity-buckets-20261005.tar.gz`, in [the designated PR163 folder](https://drive.google.com/drive/folders/1gwungsUq-b9uey8InGy0yOD2GbaMRhp-). The archive includes `ARCHIVE-MANIFEST.json` and `RESTORE-README.txt`; extract into a fresh directory and verify member hashes before use. [Staging receipt](hu20-equity-buckets-artifacts/drive-staging.json). **Cloud upload completion remains unconfirmed**; staging is not upload acceptance. No local copy was deleted. M4 retained **32.505 GiB free** after archive staging.

## Proposed validation — awaits owner approval

Use #149's same forty limped/check-through turn roots, frozen 20/20 halves, three B500M lineages, public ranges, native action tree, board weights and both seats. Keep the full-deck tables and their hashes frozen. K=50 is primary; K=200 is a declared secondary comparison, with no choice between them based on outcomes. The full-deck abstraction is fixed without using #149's policies or fitting on its forty boards; the projected policy still fits only the opposite twenty-board half.

Generate global turn/river labels through the frozen reader for every compatible holding/runout. Preserve the existing factored public-history/action-menu key and change only the equity label. Prepare these labels in a short-lived process so table mappings are released before native tree allocation. Reuse the existing external evaluator; no builder modification or table rebuild is needed.

#149 retained sufficient statistics already aggregated by its corpus-fitted labels and v1 keys, and its production requests did not dump full equilibrium profiles. Those aggregates cannot be exactly regrouped under different global labels. Therefore regenerate the 120 deterministic equilibrium/collection jobs at the same frozen settings to obtain global-label sufficient statistics; verify original equilibrium values/residuals, then pool each lineage on the opposite half. This is diagnostic witness construction, not training. Do not present an evaluation-only timing quote that silently omits this preparation.

Score the 120 root/lineage coordinates with the **existing lock-only evaluator, `max_iterations=0`**, using hash-linked reference equilibrium EVs. Evaluate both global K variants, both seats, and retain uniform missing/zero-mass fallback and per-street reach coverage. Compare paired losses against held-out v1 **P=0.6567 BB (~0.66)** and #149's opposite-half corpus-fitted equity witness **0.3874 BB (~0.39)**. Include lineage/seat rows and paired-board conditional intervals using #149's frozen 2,000 bootstrap draws/seed; report K=200 as secondary. Preserve every row, missing-key fraction, failed attempt and exclusion. The existing 5% per-fold/lineage/street missing-reach rule governs whether a primary interpretation is admitted. A useful primary signal is a negative global-K50-minus-v1 mean with its paired conditional interval below zero; any training decision is separate.

**Expected M4 time: approximately 10–12 hours on one worker, six native threads.** #149 measured 120 collects at mean 161.587 seconds (5.386 hours) and 120 lock-only jobs at mean 93.883 seconds (3.129 hours): 8.515 hours of native work before label preparation, fitting, orchestration, extra K comparisons, sampled deterministic checks and closeout. Proposed total wall cap: **14 hours, including one hour reserved for verification/retrieval**, subject to a timing-only preflight after approval; stop if the revised forecast cannot fit. Keep #149's 8-GiB family/7-GiB native worker limits, swap-growth check and 20-GiB free-disk floor, releasing reader/fit processes before native allocation. Global-label statistics may change memory use, so admission must precede production. No rental, training, root expansion, PR merge or automatic continuation is proposed.

This plan has not been run. Await explicit owner approval before any validation preparation, preflight, solve or lock-only evaluation.
