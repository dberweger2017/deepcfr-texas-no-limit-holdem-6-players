# HU100 4B and independent 2B direct ladders

All three predeclared primary contrasts improve: seed2026100601 **4B vs2B +17.90 [14.98,20.83] BB/100**; independent seeds2026100901 and2026100902 **2B vs1B +28.14 [24.87,31.41]** and **+27.90 [24.65,31.15]**. Each interval is two-sided98.333333%, Bonferroni-adjusted over the three primary contrasts.

This establishes conditional progress past2B for one fixed seed and repetition of the2B-over1B gain in two further seeds. It does not estimate a training-seed population effect, external strength or scripted-opponent qualification. No default, recipe, menu, abstraction, trainer, release or publication change was made by this campaign. [Prospective protocol](../hu100-4b-seed-ladder.md).

## Fresh training and exact reference gates

Each seed starts fresh with100BB, roots-per-seat1 and opponent-sampled average. Later segments continue only this campaign's own newly generated state; no archived resume. Training runs sequentially. All seven checkpoints are streamed to current/average exports and fully audited. All four historical checkpoint gates and all twelve indexed model byte/hash identities match exactly; no header-only exception was needed.

Reference entry caps57,658,644 at1B and67,419,934 at original-seed2B retain byte-exact headers. Later segments use72,437,552 entries, floor((7GiB−200MB)/101bytes per entry). Original seed reaches4B; no entry-cap substitution occurs.

| Seed | Requested nodes | Actual nodes | Entries | Checkpoint SHA256 |
| --- | ---: | ---: | ---: | --- |
| 2026100601 | 1,000,000,000 | 1,000,002,065 | 41,010,014 | cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec |
| 2026100601 | 2,000,000,000 | 2,000,000,460 | 54,626,283 | e84039c7a934a966a2c01f237748a23951124a8b665edc2585ef4809e5681676 |
| 2026100601 | 4,000,000,000 | 4,000,000,174 | 70,466,972 | 65fd61d6999d7a06f36fc1c718aea20e7c23856aaa491df7d517e312c1c33368 |
| 2026100901 | 1,000,000,000 | 1,000,000,506 | 40,752,103 | 5f4898d839011d62fda20e3df7e1546150ca1d72dd666654a6d4bfcaaf63a6e7 |
| 2026100901 | 2,000,000,000 | 2,000,001,279 | 54,579,477 | a8d97ae543ff44673312916a4150070f7ecb753c315fa6ffa9b35d093395a4c0 |
| 2026100902 | 1,000,000,000 | 1,000,001,141 | 41,047,344 | c21e6dd51694b379b28cb5dcc2ab595fa154092f1b69a4cfd1a9149fc85b23e4 |
| 2026100902 | 2,000,000,000 | 2,000,000,390 | 54,210,037 | 361032d94f109f5251a1680106a5bd09c24c0f93a2a02ab14c22e4b73145bc0e |

The [model index](hu100-4b-seed-ladder-artifacts/model-index.json) pins all checkpoint/current/average files and restoration commands. [Milestones](hu100-4b-seed-ladder-artifacts/milestones.json) retain full-audit counts, native state and stored traverser-visit histograms. Audits verify all stored totals/current regrets; historical increments are not reconstructed.

## Fixed direct duplicate evaluation

100BB reset per hand,50/100 blinds, no rake, seat-swapped duplicate deals, private action streams, translation off and menu-only actions. Final root202610104402 and timing-only root202610104401 pass physical-deal disjointness against each other and every declared earlier HU100 schedule. Every final hand/action/key/probability/menu/settlement replayed and deterministically reproduced.

| Contrast | BB/100 [interval] | Half-width | Duplicate blocks | Replayed hands | Replayed actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026100601: 4B vs 2B | +17.90 [+14.98, +20.83] | 2.92 | 524,288 | 1,048,576 | 5,348,563 |
| 2026100901: 2B vs 1B | +28.14 [+24.87, +31.41] | 3.27 | 524,288 | 1,048,576 | 5,114,916 |
| 2026100902: 2B vs 1B | +27.90 [+24.65, +31.15] | 3.25 | 524,288 | 1,048,576 | 5,064,665 |
| 2B: 2026100601 vs 2026100901 | -0.21 [-2.76, +2.33] | 2.54 | 524,288 | 1,048,576 | 5,209,888 |
| 2B: 2026100601 vs 2026100902 | -0.37 [-3.94, +3.21] | 3.58 | 262,144 | 524,288 | 2,598,153 |

First three rows use98.333333% adjusted primary intervals, with lower bound above0 required. Last two use ordinary95% descriptive intervals. The cross-seed intervals crossing0 do not establish equal strength. Total **4,718,592 hands and23,336,185 actions**, with full replay and reproduction; no information leak, invalid action or accounting mismatch was found.

Historical #223 seed2026100601 2B−1B is **+29.51 [26.82,32.20]**, ordinary95%, on a different schedule. The new gains +28.14 and+27.90 are shown alongside it, without pooling or treating the nominal historical interval as an adjusted member of this campaign's family.

## Coverage and visit bands

Primary-only reached decisions by policy/street are below. Positive-mass coverage is exact-key lookup coverage; missing or zero mass uses uniform fallback. Visit counts describe traverser visits at the reached stored key and need not correspond to positive average mass. These are decision-weighted observations, not independent key counts.

| Seed / actual nodes / street | Decisions | Positive mass | Visits0 | Visits1–9 | Visits10–99 | Visits100+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026100601 / 2000000460/flop | 721,463 | 99.9992% | 83 | 934 | 7,910 | 712,536 |
| 2026100601 / 2000000460/preflop | 1,090,829 | 99.9996% | 2 | 84 | 2,972 | 1,087,771 |
| 2026100601 / 2000000460/river | 342,062 | 99.9722% | 598 | 4,968 | 24,303 | 312,193 |
| 2026100601 / 2000000460/turn | 516,129 | 99.9853% | 587 | 4,949 | 26,545 | 484,048 |
| 2026100601 / 4000000174/flop | 724,773 | 99.9997% | 43 | 578 | 5,120 | 719,032 |
| 2026100601 / 4000000174/preflop | 1,089,437 | 99.9998% | 1 | 31 | 2,086 | 1,087,319 |
| 2026100601 / 4000000174/river | 344,741 | 99.9901% | 204 | 2,655 | 17,078 | 324,804 |
| 2026100601 / 4000000174/turn | 519,129 | 99.9923% | 279 | 2,950 | 18,437 | 497,463 |
| 2026100901 / 1000000506/flop | 679,697 | 99.9950% | 166 | 1,987 | 13,520 | 664,024 |
| 2026100901 / 1000000506/preflop | 1,083,163 | 99.9995% | 12 | 351 | 10,858 | 1,071,942 |
| 2026100901 / 1000000506/river | 313,067 | 99.9368% | 1,196 | 8,947 | 36,006 | 266,918 |
| 2026100901 / 1000000506/turn | 477,551 | 99.9464% | 1,254 | 8,684 | 40,905 | 426,708 |
| 2026100901 / 2000001279/flop | 685,539 | 99.9982% | 89 | 1,113 | 9,086 | 675,251 |
| 2026100901 / 2000001279/preflop | 1,080,394 | 99.9999% | 3 | 116 | 4,914 | 1,075,361 |
| 2026100901 / 2000001279/river | 316,356 | 99.9697% | 452 | 5,205 | 26,414 | 284,285 |
| 2026100901 / 2000001279/turn | 479,149 | 99.9758% | 545 | 5,377 | 28,645 | 444,582 |
| 2026100902 / 1000001141/flop | 676,279 | 99.9954% | 204 | 2,075 | 13,610 | 660,390 |
| 2026100902 / 1000001141/preflop | 1,078,376 | 99.9998% | 5 | 226 | 9,151 | 1,068,994 |
| 2026100902 / 1000001141/river | 305,399 | 99.9208% | 1,303 | 8,851 | 35,546 | 259,699 |
| 2026100902 / 1000001141/turn | 471,991 | 99.9449% | 1,256 | 8,714 | 40,473 | 421,548 |
| 2026100902 / 2000000390/flop | 673,645 | 99.9984% | 71 | 1,136 | 9,067 | 663,371 |
| 2026100902 / 2000000390/preflop | 1,077,898 | 100.0000% | 1 | 99 | 4,751 | 1,073,047 |
| 2026100902 / 2000000390/river | 307,042 | 99.9658% | 385 | 5,207 | 26,071 | 275,379 |
| 2026100902 / 2000000390/turn | 474,035 | 99.9785% | 484 | 5,382 | 28,435 | 439,734 |

[Coverage CSV](hu100-4b-seed-ladder-artifacts/coverage.csv) and [summary](hu100-4b-seed-ladder-artifacts/summary.json) retain separate missing/zero-mass counts and visit bands for both policies and all streets in every primary/descriptive contrast. Direct menu-only coverage does not quantify support against translated off-menu opponents.

## Timing, admission and resource evidence

Five32-block pilots measured only load/play/replay/reproduction timings before outcomes. Samples froze before final hands. All primaries use524,288 blocks; descriptives524,288 and262,144. The1,048,576-block choice failed preventive memory admission. Historical-SD projected adjusted primary half-width3.68BB/100 was disclosed above the approximately3 target; realized half-widths2.92/3.27/3.25. No extension, pooling or checkpoint selection.

Measured final projection5.93h, with11.87h2x guard allowance, was posted in the PR body before any final hand with SHA-bound posting receipt. Completed guard operations through calibration total5.65h, checkout wall at freeze6.68h; full prospective conservative scope18.57h. Primaries ran first because full scope exceeds10h. Optional scripted opponents were omitted prospectively. Actual five final matches including verification took4.319h. [Frozen quote](hu100-4b-seed-ladder-artifacts/frozen-final.json) pins sizes, timing, memory/disk admission and excluded indexed models.

All52 science/preparation guards passed. The monitor-qualification fixture records earlier preparation source3fd71759; actual training, pilots and final scientific controls use frozen d3120c2. Scientific samples: 179,841; peak whole-family 6.965GiB; maximum total swap 1.589GB; minimum free 48%; minimum disk 51.18GiB; normal pressure/AC throughout. Whole-family limits remained7GiB soft/9GiB hard, total swap3GB, normal pressure/system free>=15%, disk>=16GiB and AC. [Resource receipts/extrema](hu100-4b-seed-ladder-artifacts/campaign-resources.json) include every operation and both archival attempts. No scientific operation was retried.

## Archival guard stop and owner-authorized recovery

The first86.644682-second archive allowance expired at87.126210s. Its19,314,987,621-byte ZIP has no end-of-central-directory record and is retained unchanged, unaccepted. The worker returncode field0 did not make the failed guard successful. The campaign stopped, reported an issue/SOMA and disabled its timer; M4 was released forPR232.

The owner then doubled archival time and explicitly approved a separately named replacement, retaining the interrupted ZIP and rerunning no science. The initial recovery launcher was blocked by the original campaign stop marker before any worker. A separate archival guard retained that stop record and enforced unchanged resource limits. Recovery completed in **112.413s**, below **173.289365s**, with no failure or surviving child. ZIP source hashing, write, all-member readback and whole hash took105.143s. [Recovery authorization/provenance](hu100-4b-seed-ladder-artifacts/archival-recovery.json) preserves worker/controller bytes, identities and original failure.

The replacement is **22,545,560,259 bytes**, SHA256 **ea58899645688f3fb391c12b501dc0941754878e96eec165b835e1847b4105b1**, with **460 member hashes** plus embeddedARCHIVE-MANIFEST.json SHA256 **5dad61abfe120225592fb3b29f11bb812febc6ca438635ccb1d333b9ea4dd664**. Every member size/hash passed local readback. [Native upload](hu100-4b-seed-ladder-artifacts/native-upload-status.json) confirms uploaded/no pending upload/no conflicts/not trashed/sync active/not excluded. [Independent cloud metadata acceptance](hu100-4b-seed-ladder-artifacts/cloud-acceptance.json) confirms archiveID/name/22,545,560,259bytes/designated parent. [Accepted ZIP](https://drive.google.com/file/d/1qfgsSf9CodCtYE6rjH57zI1ovbT0t23E/view?usp=drivesdk) and [folder](https://drive.google.com/drive/folders/1YP4RR_c6HlPjGyGSUoZQaVI33kgF-TfW). Local integrity, native upload and independent metadata are separate claims. No remote bytes were downloaded.

Scientific source **d3120c2c195b5eca136b7f0d9969e32ba5159fff** remains reachable inGit; archivedsource controls and nativebinarySHA **de8dcee4b8a6592439b3def04e2d745e061e5df52c37d4085df4f08ebcfbda6b** restore execution. New4B and two2B model sets are archived; twelve indexed #207/#215/#223 model files are references, not duplicated. All originals remain. Closed archival receipts are compactGit evidence; original/recovery raw guard streams and administrative consoles remain in the isolated M4 planning/results paths, outside the scienceZIP. No synced deletion/offloading or other-PR cleanup occurred.

## Validation and review

One independent source review cleared scientificsource d3120c2; fourteen focused tests and all five exact-source CI checks passed. The single [end evidence review](hu100-4b-seed-ladder-artifacts/evidence-review.json) is clear with no open findings. It independently reconstructs all five paired results, verifies full replay/reproduction counts, exact gates, coverage/visit bands, frozen schedules/source/quote posting, resource extrema and restoration/native/cloud acceptance. Its two presentation findings (blank CSV zeros and preparation-guard labeling) were corrected and verified. Eight campaign tests and the diff check pass on the M4 after current-main integration; staged artifact guard and green final-head CI precede merge.
