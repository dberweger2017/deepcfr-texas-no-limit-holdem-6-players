# HU100 near-4B seed replication and translated safeguard

Both new capacity terminals improve over their own 2B averages: seed 2026100901 **+18.67 [15.94,21.40] BB/100**, seed 2026100902 **+19.42 [15.58,23.25]**, each two-sided 97.5% with Bonferroni adjustment over the two primaries. The original seed 2026100601 full-4B candidate triggers **no severe scripted regression** against v0.5.1's 2B under the predeclared 95%-upper-bound-below −20 rule. Pot pressure still beats both models in absolute terms.

These new lineages are **3.885B and 3.967B clean capacity terminals, not nominal4B**. This repeats the near-4B-over-2B gain on two additional fixed seeds; it is not a three-seed nominal4B qualification, training-seed population inference or external strength claim. Evidence only: no release, tag, publication or default change.

Science source 9516eecfef73ecbaf4e0a9e4c683f27cf1be4c28 on the isolated AppleM4/10-core/16GiB checkout; training source 8817a4732eebb05256da0eca544ef9b49840bfbd. Native trainer tree 92e86805edba980febe17076f7a9082dfa6707d9 and binary SHA c04aef9db4b3f53bcfbb8a27638ee8da5956b3e397c3c80d4ba140bf79dccc94 are unchanged. [Prospective protocol](../hu100-4b-seed-replication.md), [prior#226 ladder](hu100-4b-seed-ladder.md), [#223 ladder](hu100-3b-ladder.md), [#215 independent stages](hu100-independent-stages.md).

## Exact training and restoration

Fresh seeds2026100901/0902 ran sequentially with 100 BB, one root per seat and opponent-sampled averaging. All four 1B/2B gates reproduce indexed checkpoint/current/average files byte for byte: **12 exact files**, no declared-max-entries header exception needed. Every new checkpoint was stream-exported and fully audited, including all rows. Frozen growth cap 69,977,876entries was not raised.

| Seed | 1B actual nodes | 2B actual nodes | Growth terminal nodes | Entries / complete-iteration overshoot |
| --- | ---: | ---: | ---: | ---: |
| 2026100901 | 1,000,000,506 | 2,000,001,279 | 3,885,473,598 | 69,977,886 / 10 |
| 2026100902 | 1,000,001,141 | 2,000,000,390 | 3,967,101,196 | 69,977,900 / 24 |

Both capacity stops are clean complete iterations, with no soft-stop request. Growth train/save peaks were 6.35 GiB and 6.41 GiB. Max-entries for byte-exact1B/2B gates reused#226's57,658,644/72,437,552 values; growth ceiling was derived from #226 measured bytes/entry with 10% headroom plus controller/iteration reserves.

The scripted candidate is the original #226 seed 0601 average at 4,000,000,174nodes (2,121,415,396bytes; SHA 0ed7f9512996bb3e9535a7258a5d0e4bc8c00f22734d511c642c0e5b95cbd3c0), compared with the 2,000,000,460-node v0.5.1 average (1,603,422,956bytes; SHA e6f79ccac39a651352424e05382681f4d1994bb570b025f692556d170f87e9ae). Both retained averages were freshly hash/size/header-verified; reviewed#226 full export/audit receipts were reused with all three indexed asset pins/counts/lineage bound. Owner cleanup#237 had removed historical checkpoint/current originals; we do not claim a new audit of absent files or retrain them.

## Fixed final comparisons

Unchanged#226 direct harness: 100 BB reset each hand, blinds 50/100, no rake, seat-swapped duplicates, private action streams, translation off. Direct root 202610114502 and scripted root 202610114504 are frozen and physically disjoint from earlier inventoried schedules; scripted arms deliberately share deals. Samples were fixed using timing-only 32/512-block pilots before outcomes; no extension, pooling or checkpoint selection. Memory planning reduced the second primary and last descriptive to 262,144 blocks, so the roughly3BB/100 precision aim is not met for the second primary.

| Comparison | Blocks | Confidence | BB/100 difference | Half-width |
| --- | ---: | ---: | ---: | ---: |
| seed-2026100901-terminal-vs-2b | 524,288 | 97.5% | +18.67 [+15.94, +21.40] | 2.73 |
| seed-2026100902-terminal-vs-2b | 262,144 | 97.5% | +19.42 [+15.58, +23.25] | 3.83 |
| terminal-seed-2026100601-vs-2026100901 | 524,288 | 95.0% | +0.38 [-1.92, +2.68] | 2.30 |
| terminal-seed-2026100601-vs-2026100902 | 524,288 | 95.0% | +0.43 [-1.86, +2.71] | 2.28 |
| terminal-seed-2026100901-vs-2026100902 | 262,144 | 95.0% | +1.21 [-2.03, +4.46] | 3.25 |

Both primary lower bounds are above 0. Cross-seed descriptive intervals include 0; they do not establish equality. Full final freeze SHA ef7760791186aad61f6e34d111c31e492a60eeb855fb8e75c92ede276260bc40 is unchanged after the owner-authorized resource amendment.

## Translated scripted safeguard

Original seed 0601 full 4B and 2B on shared deals against all five opponents: 4,096 duplicate blocks/opponent, translation 512 states/128 events. Absolute and paired-difference intervals below are ordinary 95%, as predeclared. Severe regression requires difference upper bound below −20BB/100.

| Opponent | 4B absolute BB/100 [95%] | 2B absolute BB/100 [95%] | 4B −2B [95%] | Severe |
| --- | ---: | ---: | ---: | --- |
| check_call | +140.41 [+119.47, +161.34] | +134.38 [+112.21, +156.56] | +6.02 [-11.62, +23.67] | no |
| loose_aggressive | +82.15 [+49.55, +114.74] | +47.07 [+13.14, +81.00] | +35.08 [+7.77, +62.38] | no |
| pot_pressure | -49.44 [-82.28, -16.61] | -45.59 [-78.41, -12.76] | -3.86 [-30.06, +22.35] | no |
| random | +138.42 [+66.43, +210.41] | +136.19 [+64.69, +207.69] | +2.23 [-20.91, +25.36] | no |
| tight_aggressive | +49.49 [+32.41, +66.57] | +49.64 [+32.40, +66.88] | -0.15 [-13.71, +13.40] | no |

None meets the severe-regression rule. The loose-aggressive difference is positive on its individual 95% interval; this is not an additional adjusted primary claim. Pot pressure remains negative for both averages. Absence of the predeclared severe flag does not rule out every 20 BB/100-sized loss: e.g. pot-pressure and random difference intervals extend below −20. Legacy raw arm labels terminal/1b mean candidate 4B/v0.5.1-2B here, not4B/1B.

## Ladder beside earlier evidence

| Seed | Earlier 2B −1B | Earlier 4B −2B / new capacity-terminal −2B |
| --- | --- | --- |
| 2026100601 | #223 +29.51 [26.82,32.20], ordinary 95% | #226 +17.90 [14.98,20.83],98.333333% |
| 2026100901 | #226 +28.14 [24.87,31.41],98.333333% | #236 +18.67 [15.94,21.40],97.5%; actual3.885B |
| 2026100902 | #226 +27.90 [24.65,31.15],98.333333% | #236 +19.42 [15.58,23.25],97.5%; actual3.967B |

Historical intervals retain their original confidence/family labels. The new campaign reruns terminal-vs2B only; historical2B-vs1B results are neither rerun nor pooled. Byte-exact lower checkpoints support the historical policy identities.

## Verification, coverage and visits

All **4,358,144 final hands and 22,185,262 policy actions** replayed and reproduced completely (4,194,304 direct hands plus 163,840 scripted; reproductions are additional passes of those same hands). The final readout rehashed all raw hand/decision files, reconciled replay/reproduction counts and recomputed intervals from paired raw block chips. No information leak, invalid action or accounting mismatch was found. The preserved interrupted descriptive trace is not counted as accepted final evidence.

Full direct and scripted coverage/visit cells are in [coverage-visits.csv](hu100-4b-seed-replication-artifacts/coverage-visits.csv) and [summary.json](hu100-4b-seed-replication-artifacts/summary.json). Direct visit-band 0 includes missing keys; the table separates missing and stored-zero visits. Zero-mass lookup is a different quantity. Scripted exact/selected bands retain missing separately.

| Primary seed/model/street | Decisions | Missing | Stored visits 0 | Visits 1–9 | Visits 10–99 | Visits 100+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026100901-average-2000001279/flop | 709,288 | 1 | 73 | 1,006 | 7,901 | 700,307 |
| 2026100901-average-2000001279/preflop | 1,084,984 | 0 | 4 | 81 | 3,454 | 1,081,445 |
| 2026100901-average-2000001279/river | 337,092 | 65 | 588 | 4,967 | 24,316 | 307,156 |
| 2026100901-average-2000001279/turn | 510,038 | 51 | 557 | 4,951 | 26,427 | 478,052 |
| 2026100901-average-3885473598/flop | 710,250 | 1 | 35 | 556 | 5,218 | 704,440 |
| 2026100901-average-3885473598/preflop | 1,084,360 | 0 | 1 | 24 | 2,196 | 1,082,139 |
| 2026100901-average-3885473598/river | 339,445 | 29 | 211 | 2,822 | 17,731 | 318,652 |
| 2026100901-average-3885473598/turn | 512,619 | 30 | 262 | 2,984 | 18,282 | 491,061 |
| 2026100902-average-2000000390/flop | 357,502 | 0 | 40 | 523 | 3,923 | 353,016 |
| 2026100902-average-2000000390/preflop | 546,215 | 0 | 2 | 44 | 1,617 | 544,552 |
| 2026100902-average-2000000390/river | 167,954 | 28 | 304 | 2,505 | 11,895 | 153,222 |
| 2026100902-average-2000000390/turn | 254,768 | 25 | 233 | 2,522 | 13,386 | 238,602 |
| 2026100902-average-3967101196/flop | 358,655 | 1 | 14 | 303 | 2,552 | 355,785 |
| 2026100902-average-3967101196/preflop | 541,903 | 0 | 0 | 8 | 1,001 | 540,894 |
| 2026100902-average-3967101196/river | 168,812 | 5 | 88 | 1,285 | 8,602 | 158,832 |
| 2026100902-average-3967101196/turn | 256,081 | 6 | 124 | 1,472 | 9,243 | 245,236 |

Translated pot-pressure selected-key missing rates (preflop/flop/turn/river) remain a useful limitation:
- 4B: preflop 1.50%, flop 3.66%, turn 10.63%, river 18.48%.
- 2B: preflop 1.77%, flop 4.41%, turn 10.53%, river 17.29%.

## Resource stop, owner amendment and preserved failures

The initial whole-family guard was 7 GiB soft / 9 GiB hard, 3 GB total swap, normal pressure/free >=15%,16 GiB disk floor and AC. On October 11 at 23:19 CEST, the first descriptive play operation exceeded 7 GiB by 262,144 bytes (0.25MiB), observed7,516,454,912bytes. It was marked failed and the campaign stopped. The child exit code 0 does not override the failed guard receipt. All original failure/partial bytes remain preserved.

The owner then instructed: “New soft ceiling is 9 gb, continue where it stopped”. At 02:39 CEST October 12 the reviewed continuation resumed with **9,000,000,000bytes soft and unchanged9 GiB hard**. Only the interrupted descriptive match reran, from the start on the identical frozen schedule in an exclusive directory; there is no hand-position resume. The completed primaries/scripted panel were reused untouched. This is an explicitly disclosed owner-authorized retry after the original terminal stop, not an uninterrupted original-protocol run. No partial outcomes were inspected before continuation or sample selection.

All three continued descriptive guards and final readout completed; maximum accepted family RSS 8,235,925,504bytes (8.236GB/7.670GiB). Across recorded training/evaluation operations, max total swap 1,327,696,445.44bytes, minimum free memory 48%, minimum disk 43,749,228,544bytes, normal pressure and AC throughout. Archive resources are recorded separately in the closeout receipts.

Other retained failures/admissions: initial GitHub 401 and 27.3 GiB / 48 GiB preventive storage refusals; owner provisioning admitted 90.02 GiB against 77.57 GiB. A resolved-Python-symlink seed 0902 bootstrap import failure occurred before any native training; the corrected absolute-venv launch was its first science attempt. The continuation helper's initial qualification had a receipt-path FileExistsError before worker/science; corrected exclusive-path qualification-v2 passed. No native trainer edits, paid compute, M1 science, other-PR cleanup or synced deletion/eviction.

## Timing and closeout

Training/export/full audits took 4.690 h actual guarded time. The final required primary/scripted operations took 58.79 min, 33.75 min and 15.33 min. Continued descriptive operations took 61.10 min, 61.40 min and 35.48 min, followed by 59.01 s final readout. The unsuccessful descriptive attempt's 22.47 min and the 23:19→02:39 stopped interval are retained separately. Final science completed around 05:18 CEST.

Pilot costs fixed samples before any outcomes: 32/512 direct blocks and 32 scripted blocks/opponent. Original expected final play/replay/reproduction 4.87 h and 16.64 h conservative future guard allowances were not ETAs; the continuation reused fixed per-stage allowances, with expected 2.60 h remaining descriptive work. Exact pre-play PR-body quote/readback receipts are archived. The final actual archive allowance uses 3x measured throughput from 14.318 GB / 542 members / 50.080 s calibration plus fresh actual bytes/member/disk admission.

Main member-hashed archival at exact science HEAD 9516, native and independent Drive upload acceptance, model restoration index and the single end evidence review are recorded below when closeout completes. Models already indexed by#207/#215/#223/#226 are referenced rather than duplicated. All working originals remain retained; no cleanup is part of this campaign.

**Accepted main archive:** [hu100-4b-seed-replication-M4-20261011.zip](https://drive.google.com/file/d/1SsCcSHSsXNPKW0q9rq7rqlXv8jzXPGK4/view?usp=drivesdk), Drive ID `1SsCcSHSsXNPKW0q9rq7rqlXv8jzXPGK4`, **19,777,801,324 bytes / 878 members**, whole SHA256 `7cce272f16d292c62c62cdb0ac6b27e007fb51ef1460bb8e04fd9934f686ecb5`. Embedded `ARCHIVE-MANIFEST.json` SHA256 `3bfe24115e1ee069dc0f146812bbc3ac93d2f72c97dc21ffe05bbf096c45400c`; every member size/hash was read back locally. Hash/pack/member-readback/whole-hash took 96.334s; guard 116.013 s stayed within 207.526 s actual-byte allowance.

**Accepted administrative sidecar:** [hu100-4b-seed-replication-M4-20261012-closeout.zip](https://drive.google.com/file/d/1_QHe6UZCyYp4P8oDP1uBgMPzpfJOsx0T/view?usp=drivesdk), ID `1_QHe6UZCyYp4P8oDP1uBgMPzpfJOsx0T`, 208,091 bytes / 17 members, whole SHA256 `582591cc5cfd867112e707bddbccd3e762dc4ce875006f47b16daff09a2bebf7`; embedded manifest SHA256 `290d8dbafe51c56c2b721ab4753522d889fad35c30692bd616c5ba38ee9e5894`. Every member was read back locally. It retains the closed archive guard/resources/log, upload evidence, integration CI and independent-review calculation sources/logs. Compact final review/document binding is retained in Git.

[Designated folder](https://drive.google.com/drive/folders/1OUNJcDqZLFRXsffKrtjDfliumj_DmdaI), ID `1OUNJcDqZLFRXsffKrtjDfliumj_DmdaI`. Separate [native upload](hu100-4b-seed-replication-artifacts/native-upload-status.json), [independent cloud metadata](hu100-4b-seed-replication-artifacts/cloud-acceptance.json), and corresponding administrative receipts confirm uploaded/no-pending/no-conflicts and actual IDs/names/sizes/parent for both ZIPs. **No remote archive bytes were downloaded or verified**; local content integrity and remote upload acceptance are distinct.

The [20-asset restoration index](hu100-4b-seed-replication-artifacts/model-index.json) pins six new terminal checkpoint/current/average members and references the fourteen already indexed required files from #207/#215/#223/#226. It includes exact byte counts/SHA256, archive/member identities and retrieval commands. Restore only into a fresh ignored destination, verify the whole archive, extract the named member and verify its SHA256. [RESULTS_INDEX](../../RESULTS_INDEX.md) records source/run/archives/restoration and failures. All working originals are retained.

Source validation: 14 native and 19 focused Python tests, then 11 promoted controls tests; staged artifact guard and source HEAD 9516 CI passed. Final integration/evidence-review receipts follow after accepted archival. No release or publication follows from this report.

Current-main integration preserves additive #235/#237 evidence and roadmap entries. Focused integration checks: **21 passed, 3 optional retained-model K50 tests skipped**. All five integration-HEAD427cb70 CI checks passed. Final evidence-review/document binding and exact final-head CI remain required before merge.
