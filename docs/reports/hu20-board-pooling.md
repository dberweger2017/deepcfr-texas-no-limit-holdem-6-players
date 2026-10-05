# HU20 board-blind pooling — completed held-out diagnostic

**The frozen primary readout is trainer/coverage consistent: D=0.1943 [0.1550, 0.2359].** All 40 boards qualify across all three B500M stored-average exports, with 20 boards in each frozen half and 100% retained weight. All 120 collect solves, 120 lock-only evaluations and 18 sampled replays completed. No main support, convergence or replay exclusions occurred. [PR #149](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/149) has owner approval to merge after checks; paid compute cost is **$0**.

## Result and next 0.4.x step

The held-out board-blind v1 witness loses **0.6567 BB [0.6273, 0.6868]**, compared with **1.4194 [1.3310, 1.5161]** for the actual blueprint and **0.4728 [0.4365, 0.5100]** for the per-root witness. Board pooling adds a signed 0.1839 BB to the per-root mean, while 0.7627 BB separates the blueprint from the held-out witness. D=(P−L)/(B−L)=0.1943 is below the frozen 0.3 trainer/coverage threshold, and its entire conditional 95% interval lies below 0.3. These differences describe constructed witnesses; they are not an identified causal decomposition.

**Prioritize trainer changes for the next 0.4.x step.** First instrument v1 key visits, board/context occupancy, signed regret updates and lifetime-average accumulation, then check the sampler and averaging implementation against native small-game references. Propose a separately frozen, owner-approved comparison of the current trainer and the validated change with equal work and fresh held-out boards. This report authorizes no new training. The companion found visits from 18 to 12,905 and retained-average TV up to 0.8786 across lineages, but historical board occupancy was not retained; it cannot identify a specific trainer defect.

The equity-50 held-out witness reaches **0.3874 BB [0.3582, 0.4190]**, or 0.5900 [0.5525, 0.6290] of held-out v1 loss. Abstraction improvement therefore remains promising, but this experiment supports investigating trainer behavior before replacing v1 solely to explain the blueprint gap. The classification is conditional on the seed-1-selected **limped/check-through public line**, forty sampled boards and frozen ranges. It does not decide raised-pot behavior, all-board/full-game strength, inherited aliases across arbitrary roots, preflop errors or flop strategy. Projections are feasible witnesses, not abstraction equilibria or lower bounds.

## Frozen report and intervals

[Frozen reporter output](hu20-board-pooling-artifacts/m4-main-06/report.md), [all estimates, seat rows and raw inventory](hu20-board-pooling-artifacts/m4-main-06/summary.json), [resource and replay checks](hu20-board-pooling-artifacts/m4-main-06/resources.json), and [M1 reporter replay](hu20-board-pooling-artifacts/m4-main-06/reporter-replay.json). The unchanged `scripts.report_board_pooling` ran at runtime source e0c91d25. M1 replay of all 240 hash-verified atomic records reproduces every non-inventory summary field exactly. Its compact input inventory differs from the full M4 raw inventory by design.

Intervals use the frozen 2,000 paired-board bootstrap draws and seed 202610030304. They preserve lineage/seat pairing and weights, condition on the fitted opposite-half policies/codebooks and omit fitting uncertainty. No bootstrap D draw was undefined. Root pots are 2 BB, so a loss of 1 BB equals 50% pot.

### BB loss — mean [95% interval]

| Group | Blueprint | Per-root v1 | Held-out v1 | Held-out equity50 | In-sample v1 | In-sample equity50 |
|---|---:|---:|---:|---:|---:|---:|
| 2026093001 | 1.4313 [1.3220, 1.5472] | 0.4733 [0.4360, 0.5120] | 0.6670 [0.6375, 0.6972] | 0.3863 [0.3563, 0.4188] | 0.6071 [0.5812, 0.6340] | 0.3587 [0.3289, 0.3908] |
| 2026093002 | 1.3248 [1.2436, 1.4148] | 0.4667 [0.4298, 0.5030] | 0.6496 [0.6169, 0.6833] | 0.3967 [0.3633, 0.4340] | 0.6018 [0.5731, 0.6287] | 0.3693 [0.3365, 0.4041] |
| 2026093003 | 1.5019 [1.4061, 1.6098] | 0.4783 [0.4411, 0.5168] | 0.6533 [0.6179, 0.6898] | 0.3792 [0.3474, 0.4151] | 0.6100 [0.5773, 0.6431] | 0.3547 [0.3236, 0.3908] |
| pooled | 1.4194 [1.3310, 1.5161] | 0.4728 [0.4365, 0.5100] | 0.6567 [0.6273, 0.6868] | 0.3874 [0.3582, 0.4190] | 0.6063 [0.5787, 0.6331] | 0.3609 [0.3318, 0.3918] |
| seat-0 | 1.2307 [1.1335, 1.3326] | 0.4584 [0.4162, 0.4999] | 0.6133 [0.5789, 0.6492] | 0.3287 [0.3008, 0.3581] | 0.5713 [0.5390, 0.6043] | 0.2930 [0.2727, 0.3136] |
| seat-1 | 1.6080 [1.4862, 1.7491] | 0.4871 [0.4553, 0.5202] | 0.7000 [0.6451, 0.7588] | 0.4462 [0.3915, 0.5056] | 0.6413 [0.5901, 0.6942] | 0.4288 [0.3748, 0.4855] |

### Loss as percent root pot — mean [95% interval]

| Group | Blueprint | Per-root v1 | Held-out v1 | Held-out equity50 | In-sample v1 | In-sample equity50 |
|---|---:|---:|---:|---:|---:|---:|
| 2026093001 | 71.5659 [66.1007, 77.3604] | 23.6648 [21.8022, 25.6008] | 33.3513 [31.8737, 34.8610] | 19.3166 [17.8161, 20.9387] | 30.3566 [29.0586, 31.6976] | 17.9348 [16.4466, 19.5386] |
| 2026093002 | 66.2408 [62.1808, 70.7403] | 23.3344 [21.4905, 25.1493] | 32.4825 [30.8429, 34.1670] | 19.8374 [18.1674, 21.6997] | 30.0879 [28.6570, 31.4339] | 18.4650 [16.8264, 20.2053] |
| 2026093003 | 75.0974 [70.3047, 80.4918] | 23.9164 [22.0535, 25.8425] | 32.6663 [30.8953, 34.4915] | 18.9622 [17.3716, 20.7574] | 30.5021 [28.8636, 32.1552] | 17.7358 [16.1794, 19.5402] |
| pooled | 70.9680 [66.5515, 75.8058] | 23.6385 [21.8229, 25.4994] | 32.8333 [31.3634, 34.3417] | 19.3721 [17.9084, 20.9510] | 30.3155 [28.9344, 31.6556] | 18.0452 [16.5910, 19.5900] |
| seat-0 | 61.5365 [56.6730, 66.6311] | 22.9197 [20.8099, 24.9963] | 30.6665 [28.9463, 32.4604] | 16.4332 [15.0406, 17.9040] | 28.5674 [26.9517, 30.2163] | 14.6498 [13.6326, 15.6818] |
| seat-1 | 80.3996 [74.3079, 87.4532] | 24.3574 [22.7673, 26.0118] | 35.0002 [32.2564, 37.9381] | 22.3109 [19.5729, 25.2805] | 32.0637 [29.5025, 34.7100] | 21.4405 [18.7386, 24.2742] |

| Group | Primary D [95% interval] | Covered-context D [95% interval] |
|---|---:|---:|
| Pooled | 0.1943 [0.1550, 0.2359] | 0.1887 [0.1493, 0.2308] |
| 2026093001 | 0.2022 [0.1548, 0.2545] | 0.1971 [0.1500, 0.2497] |
| 2026093002 | 0.2132 [0.1696, 0.2573] | 0.2068 [0.1653, 0.2505] |
| 2026093003 | 0.1710 [0.1338, 0.2066] | 0.1655 [0.1279, 0.2017] |
| seat-0 | 0.2006 [0.1625, 0.2376] | Not separately summarized by frozen reporter |
| seat-1 | 0.1899 [0.1358, 0.2451] | Not separately summarized by frozen reporter |

Covered-context held-out v1 loss is 0.6514 [0.6209, 0.6818] BB and 32.5683 [31.0468, 34.0895]% pot. This hybrid applies the held-out strategy on covered keys and the per-root witness on absent/zero-mass keys. It is a supplementary sensitivity, not conditional EV, an implementable globally board-blind policy or a causal attribution.

## Missing-key coverage

**The 5% coverage rule passes in all twelve fold/lineage/street cells.** Maximum missing-key own-policy decision reach is 0.80493% (fold 1, lineage 2026093003, turn); D is admitted as the primary readout, not merely descriptive. These are per-street own-decision reach fractions, not joint/chance-weighted hand frequencies. Uniform fallback on absent/zero-training-mass keys remains exactly as frozen.

| Evaluation fold | Lineage | Turn missing reach % | River missing reach % |
|---|---|---:|---:|
| 0 | 2026093001 | 0.128350 | 0.000401 |
| 0 | 2026093002 | 0.123509 | 0.001114 |
| 0 | 2026093003 | 0.137361 | 0.000658 |
| 1 | 2026093001 | 0.785350 | 0.008936 |
| 1 | 2026093002 | 0.796106 | 0.003688 |
| 1 | 2026093003 | 0.804926 | 0.022922 |

There are **zero main exclusions and zero main failures**. No failed or interrupted prior attempt contributes a value. The in-sample pooled policies are secondary only; primary policies and equity codebooks fit the opposite frozen half. Empirical board weights reweight the evaluated equilibrium witnesses rather than solving a new empirically weighted equilibrium.

## Completion, resources and validation

Main-06 finished October 5 at 04:58:05 UTC / 06:58:05 CEST. Guard exit code is zero and owned campaign processes exited; no resume was used. Main guarded use is **36,740.534 seconds / 10.206 hours**; cumulative use including all eight prior stages is **42,238.400 seconds / 11.733 hours**. The owner takeover supplied a fresh main wall deadline of October 5 at 18:45:44 UTC / 20:45:44 CEST; it did not reset the append-only 86,400-second guarded clock. The one-hour closeout reserve was retained.

| Stage | Count | Mean native seconds | Maximum native seconds | Maximum worker GiB |
|---|---:|---:|---:|---:|
| collect | 120 | 161.587 | 261.308 | 5.562 |
| relock | 120 | 93.883 | 98.054 | 5.552 |
| Replay solve | 18 | 219.090 | 251.543 | 6.622 |

Native lock timings omit covered-policy construction, orchestration and any sampled replay. The guard recorded **6.680 GiB peak family RSS**, below the 8-GiB cap. Native worker peaks remain below 7 GiB; the 4-GiB arena, one worker/six threads/nice 10, swap-growth guard and 20-GiB disk floor were unchanged. The last live sample retained about 53 GiB free disk and swap below the initial baseline. The first three relock native peaks were 5.368, 5.371 and 5.358 GiB; their gates passed, including the first sampled replay at 6.622 GiB.

All 120 solve residuals range **0.154310–0.199842% pot** (mean 0.183155), at or below the frozen 0.2% target. Every lock-only pass records zero CFR iterations and inherits its hash-linked collect equilibrium. All eighteen deterministic solve/statistics replays and lock-only versus solved-tree best-response checks pass. All 258 native response hashes agree with the frozen reporter inventory. Qualification-05 remains 61/61 passing, including the three fixed 20,000-deal V4 checks, memory estimates, fixtures and singleton gates. Linux parity remains **not-run**; this was a same-host macOS campaign.

**36 pooling tests pass** on M1 after reporting; no M1 solve was run. External AGPL solver/harness sources and binaries remain outside the MIT repository. Native fingerprint `fb32974d9d211fa66001d1efb330ec4af2d825005d24b287dc6cf3c37fa8812f`; qualification `a1914726d352a179e55e75706b245eb867060bb3a5ae9242889664e224b128c0`; frozen plan `754eac947757ecef9638e665007e8dbf7975b2e02a876267fb6e5e623f462db1`; final summary `6cdcefce0a55f9ac361966b4d298c4e9e750a9b668053054b4b2e3e09b94e91f`.

## Retrieval and archive staging

The entire campaign is losslessly packed as a **16,802,192,861-byte tar.gz**, with 5,740 file members / 56,262,193,754 logical bytes and member hashes, hard-link and symlink provenance. Archive SHA256 is `20ad0f67df71464e7c06bdb1cf63451c243944c14e732b83650861f8f9bbcaed`; manifest SHA256 is `d5608ba75a785dfc5633f96dc27246d1b0c7781e478e67bc31972163565fca65`. A separate APFS clone is locally hash-verified and [staged](hu20-board-pooling-artifacts/m4-main-06/drive-staging.json) in the existing PR149 Drive folder under `M1-board-pooling/M4-main-06-20261005/`. **Cloud upload completion is pending.** **Full M1 retrieval now verifies all 5,740 members with zero mismatches**, including 82 hard-link groups; [closeout receipt](hu20-board-pooling-artifacts/m4-main-06/retrieval.json). Archive and manifest hashes match M4 exactly. Verification streamed every recovered member without expanding the 56-GB campaign, so the complete raw evidence fits current M1 disk space. There are no symlink members in this snapshot. All Git administration directories are excluded; all working sources, external source archives and exact runtime commit remain retained. The compact frozen reporter replay also matches exactly, and all five earlier immutable M1 copies reverify with zero mismatches. All M4 originals, prior stop receipts, preparation hard-link dependencies and isolated restored average inputs remain preserved. No deletion or eviction is authorized.

The initial SCP transfer disconnected at 9,137,894,400 bytes (exit 255); an attempted rsync sender path was absent (exit 12). The existing partial resumed using the installed `/usr/bin/rsync` with append verification (exit 0). Full archive/member SHA256 verification then passed; no solver was restarted or value recomputed. These transport failures are separate from the zero main failures. The closeout receipt records conservative guarded-plus-closeout wall accounting below the cumulative 24-hour ceiling, while preserving the original append-only guarded journal.

## Prior attempt chronology (historical, superseded by main-06 completion)

The following prior closeout records are retained verbatim. Their references to stopped work, old deadlines or pending owner readmission describe their respective October 4 stages. The owner’s [takeover](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/149#issuecomment-5983208898) and source e0c91d25 introduced per-lineage policy files and a short-lived covered-policy child, preserving scientific values, then admitted main-06 under the new wall allowance. Every earlier failure stays charged in the clock; no partial result was stitched into this readout.

## Earlier qualification, measured admission and main stop


[Qualification, stage resources, clock and stop receipt](hu20-board-pooling-artifacts/m4-qualified-stop-05.json)
records 61/61 passing gates: K, retained river fixtures, singleton projection,
three fixed real-export V4 checks with 20,000 independent native deals each,
three converged equilibria and lock/replay comparisons. First-pilot recovery
pins every old/fresh hash and requires identical fresh blueprint EV before
retaining its fixed V4 Monte Carlo. Middle/last pilots and all previously
unfinished replay work are fresh. Linux parity remains not-run.

| Fixed pilot | Solve s / owned GiB | Lock-only s / owned GiB | Replay s / owned GiB | Residual % pot | Native memory estimate, plain / compressed GiB |
|---|---:|---:|---:|---:|---:|
| First | 173.617 / 5.171 | 96.686 / 5.502 | 239.650 / 5.035 | 0.183158 | 3.640 / 1.833 |
| Middle | 138.209 / 4.156 | 93.338 / 4.278 | 204.704 / 4.557 | 0.170967 | 2.656 / 1.338 |
| Last | 138.674 / 4.581 | 93.378 / 5.070 | 208.343 / 4.587 | 0.167133 | 2.656 / 1.338 |

V4 solver EVs 0.368446 / 0.639241 / 0.736151 BB lie inside the respective
95% native Monte Carlo intervals [0.294008, 0.407592], [0.516404, 0.676996]
and [0.589207, 0.769393]. All residuals are below the fixed 0.2% pot target.
The conservative final forecast was **55,952.809 seconds / 15.542 hours**,
`1.5 × 138 × (173.617301 + 96.686125)`, versus **60,058.807 seconds /
16.683 hours** before the one-hour reserve at admission. It was
[posted before main values](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/149#issuecomment-5982659835).
Main used source f2ad9e7 and the hash-pinned engineered native binary.

At 17:43:28 UTC on October 4, the owned guard was deliberately signalled after
finding that the Python phase transition retained all 120 parsed statistics
records. This was a preventable driver capacity defect, distinct from the
successful native optimization. **One atomic collect result and one partial
are preserved; no relock or common board is complete.** The full clock now
charges **5,497.866452 guarded seconds**, including every failed attempt.
The original deadline remains October 5, 11:19:43 UTC, with a one-hour
retrieval reserve; idle time also consumes this absolute allowance. Owned
processes are verified absent, the heartbeat is paused, swap stayed at
34,015,805 bytes, and the stop inventory recorded 92.354 GiB free disk.
No RSS, disk, swap, convergence or scientific parity gate was bypassed.

## Bounded-memory pooling engineering

The [data-only M1 preflight](hu20-board-pooling-artifacts/engineering-pooling-memory-05.json)
uses three real pilot statistics with synthetic identities/folds for byte
parity and capacity inspection only. Eager fitting peaks at **2.565 GiB**;
lineage-streamed fitting peaks at **1.662 GiB**, taking 32.394 seconds versus
26.043 seconds. All-board and both cross-fit policy files are **byte identical**,
including probabilities, floating addition order, menus, keys and metadata.
Root containers alone occupy 182–188 MB each: extrapolating the smallest to
120 roots gives **20.379 GiB**, before strings/numbers. That is a capacity
extrapolation, not a measured 120-root peak.

The repair processes one lineage and one root at a time with the unchanged
numeric pooling function, then writes each file atomically. A single sequential
owned Python child releases the fitting heap before native workers resume.
The existing family/disk/swap/clock guard remains active and the child keeps
the same 7-GiB worker cap. This changes representation/lifetime only; requests,
4-GiB arena, six threads, forty boards, three averages, halves, codebooks,
coverage, bootstrap, eighteen replays and every estimator remain frozen.
**35 Python tests pass.** No M1 solve was run. The driver repair is prepared
on M1 and has not been deployed or qualified for a new M4 attempt. Native
engineering froze after about 24 minutes; the later data-only repair remained
inside the owner's approximately two-hour engineering box.

The [closeout scheduling warning](hu20-board-pooling-artifacts/engineering-closeout-forecast-05.json)
uses the same conservative multiplier/count but the slower completed main
collect (195.462 seconds): **16.799 hours**, versus about **16.0 hours** before
reserve at closeout, excluding fit overhead. The fixed-pilot forecast remains
15.542 hours; this new observation explains why its earlier admission is not
fresh admission now. No outcome threshold or inference changed. Continuing
requires an owner budget/host decision and fresh admission; neither a longer
allowance nor a rental is authorized.

## Incomplete scientific readout and verified closeout

The [frozen-plan reporter](hu20-board-pooling-artifacts/m4-stop-05/report.md)
and [summary](hu20-board-pooling-artifacts/m4-stop-05/summary.json) admit
**zero common boards**. Pooled/lineage/seat BB and pot-percent bootstrap
intervals, missing-key coverage by fold/lineage, D and covered-context sensitivity
are unavailable. Every one of the forty planned boards remains excluded from
primary inference because complete three-lineage collect/relock coverage is
absent. No trainer-versus-abstraction classification is permitted.

The one completed phase-1 root is descriptive only (seed-1-selected limped,
check-through line, lineage 2026093001). Seat-0/1 blueprint losses are
0.887957 / 1.126317 BB (44.397850 / 56.315834% pot); per-root v1 losses are
0.310904 / 0.437559 BB (15.545183 / 21.877939% pot). Its residual is
0.196663% pot at 275 iterations, elapsed 195.462 seconds and owned peak
5.086 GiB. These points have no spot-level interval and are excluded from the
pooled readout; the interrupted second job never counts. Raw hashes and all
metric/coverage records are in the stop receipt and retained atomic result.

The [fifth retrieval verification](hu20-board-pooling-artifacts/m4-stop-05-retrieval.json)
checks **3,476 files / 12,124,413,687 logical bytes / 9,295,160,304 unique-inode
bytes**, including 82 hard-link groups, with zero mismatches. All four prior
immutable copies were independently reverified. Original M4 evidence, source,
binaries, exports, isolated inputs and receipts remain untouched. Nothing was
deleted or sent to Drive; cloud completion is not claimed. No paid allocation,
training, promotion or merge occurred.

**Next 0.4.x step:** finish this frozen diagnostic only after explicit owner
readmission and a fresh forecast against the shrinking original deadline.
The streaming repair is concrete and reviewable; restarting requires owner
input under the protocol's no-automatic-restart rule. These incomplete outcomes
cannot justify choosing longer v1 training, changing the board key or promoting
a policy. Any eventual conclusion is restricted to this seed-1-selected
limped/check-through line. Projections are feasible witnesses rather than
abstraction equilibria; covered-context hybrid sensitivity cannot replace the
primary missing-key coverage requirement.

## Engineering amendment and fresh parity

The owner authorized about two hours of M1 development without M4 solves
during engineering, followed by exact M4 pilot-0 solve/lock reruns. No longer
allowance or rental was approved. Development finished in about 24 minutes;
M1 tests used zero CFR iterations. The external build caches a pooled file
through its final use, clears the same native locks directly and copies
interpreter state for depth-first traversal. Native lock normalization, solver
arithmetic/storage, original request files, tree, arena and science are fixed.
[Source/build fingerprint](hu20-board-pooling-artifacts/engineering-amendment-05.json).

Both fresh M4 responses reproduce **every scientific field exactly** against
qualification-04, including all action masses, coverage, gains, EVs, iterations,
residual and both memory estimates. Only top-level time/RSS are excluded;
float values and signed zero are retained. Solve scientific SHA-256
`70c43785081be85d57764914c03b44bc051415fdc7a304653b284922910124f0`;
[full exact comparison and evidence pins](hu20-board-pooling-artifacts/engineering-parity-05.json).
This engineering parity does not create main outcomes.

| Pilot-0 pipeline | Qualification-04 | Engineered | Owned peak RSS |
|---|---:|---:|---:|
| Solve | 256.901 s | 173.617 s | 5.171 GiB |
| Lock-only | 285.396 s | 96.686 s | 5.502 GiB |

[Timings and provisional forecast](hu20-board-pooling-artifacts/engineering-forecast-05.json):
1.5 × 138 × (173.617301 + 96.686125) = 55,952.809 seconds / **15.542 hours**,
versus 61,565.835 seconds / **17.102 hours** remaining before the one-hour
reserve at measurement. This lower bound permits remaining qualification;
only the final slowest-pilot forecast may admit main.

The [solve profile](hu20-board-pooling-artifacts/engineering-05-solve-profile.json)
records CFR 105.550 seconds, sufficient-statistics collection 37.555 seconds,
node-policy construction 15.972 seconds and locked BR 2.041 seconds.
The [lock profile](hu20-board-pooling-artifacts/engineering-05-lock-profile.json)
records policy construction 72.405 seconds, actual lock calls 2.622 seconds,
BR 6.114 seconds, one pooled-file load 0.539 seconds, blueprint EV 0.699
seconds and twelve bulk unlocks 0.095 seconds. Traversal totals **include**
policy/lock time; do not add nested timers. Tree construction/validation,
compact parsing and writing are each under a second. A data-only M1 benchmark
measured five repeated loads at 4.472 seconds versus one cached load at 0.677
seconds. Repeated root replay/unlock work, rather than native locking or CFR,
explains the removed overhead. Remaining policy construction/aggregation is
measured wrapper work, not all inherent solver cost.

Qualification-05 subsequently completed retained fixtures and the remaining fixed roots afresh.
The new first solve/lock can be reused only through pinned exact comparison;
first V4's fixed 20k native MC additionally requires its blueprint EV to match
the fresh value exactly. Its previously unfinished deterministic replay runs
fresh. Every pilot retains separate solve/lock RSS and timings. All old
failures, clock charges, science, 7/8/4-GiB limits and reserve remain unchanged.
**32 Python tests passed at native admission; the subsequent driver repair raises this to 35.** Full raw retrieval is now verified above; all M4 originals remain intact.

## Qualification-04 resource amendment and completed work

The owner explicitly approved changing M4 worker RSS from 5 to 7 GiB before
any main values. The 8-GiB family cap, 4-GiB solver arena, one worker/six
threads, nice 10, 20-GiB disk floor, original start/deadline and all failed
time stayed fixed. Corpus, halves, policies, codebooks, native menu, thresholds,
coverage rule, bootstrap and eighteen replay jobs did not change. The
[amendment](../hu20-board-pooling-protocol.md#owner-approved-m4-rss-amendment-before-main-values)
was posted on the PR before execution. No AGPL binary/source change occurred.

Readmission-04 reverified 253 preparation/source members and every one of the
120 native request/compact pairs before reusing `prepared-03`. No prior solver
completion was reused. K100k was inherited from the hash-pinned unchanged
preparation; 34 retained river checks and thirteen singleton checks ran fresh
and passed. The first fixed seed-1 root's fresh 20,000-deal V4 and V5 passed.
Its unchanged equilibrium request SHA-256 is
`02aa2b9869cefdf43e067c73d484e29f9892ffa5f45f0800e770a9314eb3bc25`.
Residual remained 0.1831579% pot in 225 iterations. Both memory estimates
remained 3,908,266,992 bytes uncompressed / 1,968,657,456 compressed, below the
unchanged 4-GiB arena; compression was used. Native-policy V4 solver EV
0.3684459 BB remains inside MC CI [0.2940080, 0.4075920] BB.

| First fresh pilot stage | Completed seconds | Peak worker RSS | Outcome |
|---|---:|---:|---|
| V4 locked EV | 32.577 | 4.465 GiB | Passed, then independent 20,000-deal native MC |
| Solve pipeline | 256.901 | 4.902 GiB | Passed V1/V5, metrics and statistics completed |
| Lock-only pipeline | 285.396 | 5.568 GiB | Completed ten measurements; reference-lock gate passed |
| Real replay | Incomplete | 2.818 GiB observed before interruption | Deliberately stopped when forecast admission became impossible |

Completed stages remained below 7 GiB. Swap baseline and peak stayed at
34,015,805 bytes. The outer family sampled peak was 5.559 GiB and the nested
lock-only worker recorded 5.568 GiB; these separate sampling observations are
below their respective limits. Interrupted replay observations are not a
completed peak or timing. Middle/last pilots were unattempted, so full
qualification and the eighteen main replays are not certified. Linux parity
remains not-run. No 7-GiB breach or fresh convergence failure occurred.

The completed first lock was checked again offline against the hashed fresh
equilibrium response after shutdown; `check_lock_only` passed. This recheck
performed no native solve and did not admit qualification or main values.
[Compact latest qualification evidence](hu20-board-pooling-artifacts/m4-qualification-stop-04.json).

## Measured forecast and stop

The predeclared formula is 1.5 × (slowest completed solve pipeline × 138 +
slowest lock-only pipeline × 138), for one worker. Already with the first
completed pilot, **1.5 × (256.900991 + 285.396349) × 138 = 112,255.549 seconds
(31.182097 hours)**. The original absolute deadline left **64,437.759 seconds
(17.899378 hours)** for main after reserving 3,600 seconds for retrieval.
This is a measured **lower bound** on the final three-pilot forecast, not an
assertion that unattempted pilots are equally fast. Adding pilots cannot lower
either maximum. The full three-pilot timing distribution is unavailable.

The owner's instruction explicitly says to stop if the forecast plus reserve
cannot fit. Continuing the remaining qualification could not restore admission,
so the owned guard received SIGTERM on October 4 at 16:25:45 UTC
(18:25:45 CEST). Its interruption record is preserved; it is an intentional
forecast stop, not a spontaneous RSS/solver gate failure. No main directory,
production approval or main solver value was created. No original-clock reset,
resource increase or scientific change was inferred. The measured forecast was
[posted before main](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/149#issuecomment-5982075615).

Qualification-04 consumed **693.705460 guarded seconds** including the
interrupted replay. The append-only total is **3,773.578918 seconds
(62.893 minutes / 1.048216 guarded hours)** across all five stages. The original
absolute start 1791112783.505676 and deadline 1791199183.505676 stay unchanged;
the latter is October 5 at 11:19:43 UTC (13:19:43 CEST). Guarded time does not
replace wall allowance: idle, administration and retrieval also consume the
absolute deadline. Closeout free disk was 93.52 GiB, above the floor.

## Outcomes and intervals

The [fresh frozen-plan reporter](hu20-board-pooling-artifacts/m4-stop-04/report.md)
and [summary](hu20-board-pooling-artifacts/m4-stop-04/summary.json) record
**0/40 common three-export boards, zero common weight and zero in each half**.
All forty main roots are unattempted, not observed solver/support exclusions.
Preparation has zero support exclusions. All pooled, lineage and target-seat
BB/pot-percent campaign estimates and bootstrap intervals are unavailable.
There is no D, covered-context D, held-out coverage by fold/lineage, equity
headroom or classification. Incomplete pilot values never count in the primary
mask. No point estimate is silently treated as a confidence interval.

For completeness, these are **first-pilot qualification points only**. The
pilot pool uses that single root; labels such as `e_cross_v1` here do not mean
that the frozen opposite twenty-board half has been fitted. They are engineering
comparators, not forty-board held-out results or an inferred abstraction gap.
There is no board-bootstrap interval for these points.

| Native pilot metric | Target solver seat 0 BB (% pot) | Target solver seat 1 BB (% pot) |
|---|---:|---:|
| e_bp | 0.777066 (38.8533%) | 1.145156 (57.2578%) |
| e_root_v1 | 0.284480 (14.2240%) | 0.418938 (20.9469%) |
| e_board_v1 | 0.284480 (14.2240%) | 0.418938 (20.9469%) |
| e_board_eq50 | 0.184433 (9.2217%) | 0.260130 (13.0065%) |
| e_cross_v1 | 0.284480 (14.2240%) | 0.418938 (20.9469%) |
| e_cross_eq50 | 0.186205 (9.3103%) | 0.257971 (12.8985%) |
| e_cross_v1_covered | 0.284480 (14.2240%) | 0.418938 (20.9469%) |

## Latest verified retrieval and next 0.4.x step

The fresh fourth M1 copy is
`/Users/dberweger/Local/hu20-board-pooling-m4-retrieval-04-20261004`.
All **3,198 members / 10,473,469,109 logical bytes / 7,644,215,726 unique-inode
bytes**, and **82 hard-link groups** verify with zero mismatches. Manifest
SHA-256 is `8089e51629c5358f34530af062a3fcfd2cf9e814043d496e63e0c484b15b56d7`.
[Fourth verification receipt](hu20-board-pooling-artifacts/m4-stop-04-retrieval.json).
All three older copies' manifest members were independently reverified too.
LAN SSH used the pinned M4 key. The fresh retrieval shares unchanged files via
`--link-dest` with the third copy to avoid another physical duplicate; old
member contents and immutable receipts are verified, and nothing was deleted.
All M4 originals, isolated inputs, prepared recovery dependencies, failed
attempts, partial replay and external tools remain preserved. No cloud upload
completion is claimed. The existing RESULTS_INDEX archive destination remains
unchanged. Source 2dfde352bd2d229146fc894a0694ab4db552b724; plan and both native
binary fingerprints below remain fixed. Twenty-five Python diagnostic tests
pass, including resource amendment rejection and separate solve/lock RSS
retention on failure; publication checks confirm no main rows or inference.

For the next 0.4.x step, **this run supplies no new evidence to choose trainer
changes over board-key changes**. It shows that the first fresh lock fits the
approved worker allowance, while the one-worker M4 campaign cannot fit the
frozen conservative forecast within the original clock. Keep the forty-board
science and seek an owner-approved longer M4 allowance or a faster-compute
quote/qualification plan. Other pilots may require more time or fail resource
gates; first-pilot success is not a whole-campaign guarantee. Bounded-memory
engineering remains the fallback for an actual future 7-GiB breach, not a
conclusion forced by this forecast stop. No training, promotion, rental or
merge occurred; PR stays draft, paid cost $0, monitoring paused.

## Preserved 5-GiB stop, preparation history and original scope

**Preparation completed 120/120 exports. Qualification stopped at the frozen
5-GiB worker RSS ceiling; main execution never started. No hypothesis decision
is possible.** The first real-export V4 check and first equilibrium solve passed,
but the fresh lock-only evaluation exceeded its RSS budget. All owned experiment
processes exited; the heartbeat is paused and no automatic restart occurred.
[PR #149](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/149)
remains open. Paid cost is **$0**, with the $5 RunPod allowance unused.

## Frozen question and inference limits

The October 4 owner request released M4 after #148 merged and replaced the
RunPod plan. The [protocol](../hu20-board-pooling-protocol.md) retains forty
boards on the seed-1-occupancy-selected limped, BB-check, flop-check-through
line, three B500M stored-average exports, the frozen 20/20 halves, seeds,
cross-fitting, classification thresholds and 2,000 paired-board bootstrap draws.
Ranges are taken as given. Target convergence is at most 0.2% pot, using the
native reopening menu without raise caps or substitutions.

D = (held-out board-blind v1 loss − per-root v1 loss) / (blueprint loss −
per-root v1 loss), with the predeclared 0.1-BB gap floor. D <=0.3 supports the
trainer/coverage interpretation; D >=0.7 supports board pooling; otherwise the
readout is mixed. Classification additionally requires a complete campaign,
32 common three-export boards, 80% common weight and sixteen boards in each
half. Missing own-target-policy decision reach above 5% on either street in any
fold/lineage makes D descriptive only. The covered-context sensitivity uses
the held-out policy at covered keys and the per-root witness at missing or
zero-mass keys; it is a hybrid full-game policy, not conditional EV or a causal
loss decomposition. Uniform fallback remains the primary rule.

All these fields were frozen before main outcomes. Projections are feasible
witnesses, not abstraction equilibria or lower bounds. Any eventual readout
applies to this selected public line; it cannot settle raised pots, flop
strategy, preflop range error or full-game strength. Bootstrap intervals would
condition on fixed fitted policies/codebooks and omit fitting uncertainty.
The [solver-free companion](hu20-board-pooling-preparation.md) reports retained
diagnostic board diversity, not historical training board occupancy.

## Completed preparation and partial qualification

| Work | Recorded outcome |
|---|---|
| Three stored-average sources, engine and external tools | Frozen hashes verified; sources isolated in `inputs-restored-02` |
| Corpus, card features, codebooks and native trees | All forty boards exported for all three lineages |
| K | 100,000 comparisons, zero mismatches, passed |
| Compact exports | 120/120 complete, zero support exclusions |
| Immutable recovery | Eighty prior exports accepted only after pinned hashes, regenerated features/codebooks, freshly recomputed ranges and exact complete requests matched; third lineage newly exported |
| River fixture parity | 34 checks over eight retained fixtures passed; 600 terminal payoff queries, zero chip error |
| Singleton qualification | Thirteen checks passed, including lock-only BR parity and missing-key fallback |
| First real-export V4 | Passed, 20,000 independent native-policy deals |
| First real V1 and V5 | 9,369 native-tree nodes matched; 225 iterations, 0.1831579% pot residual |
| First fresh lock-only evaluation | RSS guard failure; no completion or loss values |
| First real replay and middle/last pilots | Unattempted after the stop |
| Full qualification and production forecast | Not admitted; no completed qualification receipt |
| Linux parity | Not run; M4 same-host evidence does not constitute Linux parity |
| Main collect / locked / eighteen replay jobs | 0/120 main solves; all main work unattempted |
| Common three-export boards | 0/40, zero common weight, zero in each half |

The `real-v4/gates.json` partial file says `passed: true` for the first V4 and
V5 checks accumulated so far. It does **not** certify all qualification gates,
lock checks, three pilots or a production forecast. The V4 stage uses a
one-iteration locked EV check; its large residual is not a failed equilibrium
solve. Only the later 225-iteration solve is convergence-qualified.

The [generated frozen-plan report](hu20-board-pooling-artifacts/m4-stop-03/report.md)
and [summary](hu20-board-pooling-artifacts/m4-stop-03/summary.json) contain no
campaign rows. Pooled, each lineage and each target-seat BB/pot-percent losses,
bootstrap intervals, D, covered-context D, equity50 headroom and held-out
missing-key coverage are **unavailable**. Forty scheduled roots are recorded
as missing all policy indices; they are unattempted, not observed solver
exclusions. Incomplete pilot values never count toward the campaign mask.

## First fixed real pilot — descriptive qualification evidence only

Spot `cd8511f851948a2ea410adeb133b13d1cfcbdd9f8336cecf050fa303a13209a9`,
seed 2026093001, root pot 200 chips (2 BB). This fixed pilot was selected by the
qualification protocol. It is not a main outcome or cross-fit board sample.

The real-policy locked solver EV was **0.3684459 BB**, inside the native
Monte Carlo 95% interval **[0.2940080, 0.4075920] BB** (mean 0.3508 BB,
20,000 independent deals, seed 202610030310, weighted holdings with blocker
rejection). V4 passed. Both solver memory estimates were called before
allocation: **3,908,266,992 bytes uncompressed (3.640 GiB)** and
**1,968,657,456 compressed (1.833 GiB)**, below the 4-GiB arena budget.
Compression was used.

| Target solver seat | Blueprint loss BB (% pot) | Per-root v1 loss BB (% pot) |
|---|---:|---:|
| 0 | 0.777066 (38.8533%) | 0.284480 (14.2240%) |
| 1 | 1.145156 (57.2578%) | 0.418938 (20.9469%) |

These are first-pilot points only; no board-bootstrap interval can be estimated
from them. They do not measure board-blind loss, coverage, D or headroom.
The completed equilibrium pipeline took **255.950 seconds** by watchdog
(255.677 seconds emitted by native completion), with **4,956,536,832 bytes
(4.616 GiB)** peak guarded worker RSS. Equilibrium EVs were
[-35.8770638, 35.8770638] chips; achieved residual was 0.1831579% pot.

The subsequent fresh lock-only job stopped after **22.322 seconds** at
**5,372,346,368 bytes (5.003388 GiB)** against **5,368,709,120 bytes (5 GiB)**.
No locked measurements completed. Its swap baseline and peak were unchanged
at 34,015,805 bytes. The outer family guard sampled a 4.998-GiB peak; the
nested worker sample caught the RSS breach. These different samples do not
imply the 8-GiB aggregate ceiling was exceeded. Arena estimates fitting does
not establish that decoded policies, statistics and strategy locks fit the
complete worker RSS budget.

[Compact qualification evidence](hu20-board-pooling-artifacts/m4-qualification-stop-03.json)
retains gate details, runtimes, point metrics and raw response hashes. The
failed lock response contains tree/memory evidence only, no fabricated losses.

## Every attempt and resource stop

One six-thread worker, nice 10 for native work, 5-GiB worker / at most 8-GiB
owned family RSS, 4-GiB arena and a 20-GiB free-disk floor were preserved.
Unrelated jobs were left untouched; no new TensorBoard or M1 solve ran.

| Stage | Guarded seconds | Peak owned RSS | Outcome |
|---|---:|---:|---|
| Prepare 01 | 267.468 | 1.654 GiB | Free-disk guard; nine exports complete, tenth partial |
| Prepare 02 | 1,603.875 | 1.670 GiB | Missing third source after shared input alias archived; eighty exports complete |
| Prepare 03 | 879.817 | 2.016 GiB | All 120 exports completed under verified recovery |
| Qualify 03 | 328.714 | 4.998 GiB outer sampled; 5.003 GiB nested worker | First fresh real lock exceeded RSS; outer stage exited 1 |

The first disk stop occurred October 4 at 11:25:19 UTC; the missing-input
stop at 13:34:34 UTC; the qualification stop at 14:47:56 UTC. Both resumptions
followed explicit owner instructions, with fresh approvals and output folders.
No automatic restart followed a guard failure. The append-only journal totals
**3,079.873 seconds (51.331 minutes / 0.8555 guarded hours)**, including all
failed attempts. Original start 1791112783.505676 and absolute deadline
1791199183.505676 (October 5 at 11:19:43 UTC) remain unchanged. The absolute
deadline also bounds idle, administration and retrieval; guarded hours alone
are not remaining wall allowance. The 3,600-second closeout reserve was not
converted into production time. There is no complete slowest-lock timing, so
no conservative production forecast can be admitted.

The stop snapshot recorded 94.21 GiB free disk, above the floor. The latest
failure was neither disk nor swap growth. The first guard stopped correctly
as unrelated archive work reduced disk; future promised free capacity never
waived the current floor. Early low-throughput Tailscale retrievals were
interrupted, then transferred over LAN using the existing pinned host key.
A first reporting helper invocation lacked PYTHONPATH and was rerun before
verification; it changed no experimental evidence.

## Provenance, recovery and retained evidence

The missing-source stop retained the original shared restoration pointer.
All three original average exports were restored from the retained M1
`/Users/dberweger/Local/hu20-m4-archive-20261002/hu20-exact-flop-check-inputs`
copy into the isolated M4 `inputs-restored-02`; frozen hashes matched on both
hosts. Drive and the archived shared alias were untouched. Source
`fd5ebcd653c325e4d54cfa40f056f49c6a528ed7` prechecks all inputs and implements
pinned exporter recovery. Twenty-three Python board-pooling tests pass,
including corruption, range/feature changes, missing later sources and exact
JSON-normalized request equality. Immutable hard links preserve accepted
index/compact files; old attempts and receipts remain intact. This is exporter
recovery, not reused solver results or a scientific change.

The full third retrieval verifies **3,029 files / 10,125,291,134 logical bytes**,
**7,296,037,751 unique-inode bytes**, with **82 hard-link groups** preserved
and zero member mismatches. Its manifest excludes itself and has separately
verified SHA-256 `cd970496c4a1cff3564410de51916d8902aa4c52c4a81a521eec4c8c9365b944`.
[Third verification receipt](hu20-board-pooling-artifacts/m4-stop-03-retrieval.json).
The first immutable [receipt](hu20-board-pooling-artifacts/m4-stop-retrieval.json)
retains 2,418 files / 1,084,490,993 bytes, manifest
`05c82c25f050958894658cb732b1e187844e383a690f76d4a5443bcdd897122f`.
The second [receipt](hu20-board-pooling-artifacts/m4-stop-02-retrieval.json)
retains 2,623 files / 4,697,713,515 bytes, manifest
`38710bfd86d9d18c73b93e059ef95bc67b0d501597a276406957507c7595d8f5`.
Both prior copies remain unchanged.

Engine commit `5db20e3d5d6862b32a7402035c1340b622d3b005`; AGPL upstream
`9d1509fe5077d019825f833eed04b16d342dfda1` remains outside the MIT repo.
Mac v2 binary SHA-256
`4f22f58bd677c6fb46e980c69fcfd14f13cadd850417c0aca711983aaacc7792`;
reference Mac binary
`b48284303f120acaf6694b60ff344cb466f08aaabe4cc5d9bf1af64c7acd10eb`.
Frozen plan SHA-256
`754eac947757ecef9638e665007e8dbf7975b2e02a876267fb6e5e623f462db1`.
Every member hash, source export and codebook/corpus/half hash is retained in
the generated inventory and linked receipts. The equilibrium response hash is
`b4a0c29d2be4ac7050af15f44e9ff9575f9c25a5ec5ec11a628eb0ebec876998`;
the failed lock response hash is
`a1b2c9895f6dea99c0e011cd28482525b7dd806ee8b7a18da3fa35e72d2808e0`.

M4 originals remain at `/Users/dberweger/Local/hu20-board-pooling-20261004`.
The latest fresh M1 retrieval is
`/Users/dberweger/Local/hu20-board-pooling-m4-retrieval-03-20261004`; both
prior dated retrievals remain. Sources, prepared-02 recovery dependencies,
prepared-03 and all external tools/failed attempts are retained for the
existing PR149 Drive destination in RESULTS_INDEX. No cloud completion,
cleanup or deletion authorization is implied. **Paid cost $0; no rental,
training, model promotion or merge.**

## Next implementation step

This stop gives no new basis to choose trainer or abstraction changes. First
reduce peak memory in the lock pipeline, then seek explicit owner readmission
with fresh resource qualification under the frozen science and original clock.
A source inspection suggests streaming/narrowing decoded statistics and
releasing large Python reference responses before fresh lock calls; native
pooled-file loading and dense locks also deserve measurement. This is an
engineering hypothesis, not a measured attribution of the peak. Raising the
ceiling or repeating the same failed job is not automatic continuation.
All real locked/replay checks, remaining fixed V4 pilots and the posted
conservative forecast must still pass before main values.

## Owner-approved resource-only readmission

After this immutable stop report was published, the owner raised the M4 worker
RSS ceiling from 5 to 7 GiB and authorized fresh qualification. The 8-GiB family
cap, 4-GiB solver requests, all frozen science and original clock stay unchanged.
Fresh first/middle/last solve, lock-only and replay resource records are required.
No main value existed when the amendment was approved. The earlier failure and
all three verified retrievals remain intact. Bounded-memory engineering is now
the fallback if the amended pilot fails, rather than a prerequisite to this
explicit owner-approved resumption. Forecast admission remains mandatory.

Fresh [readmission receipt](hu20-board-pooling-artifacts/m4-resume-04.json)
verifies 253 preparation/source members plus all 120 request/compact pairs,
with 93.88 GiB free disk and the unchanged 8-GiB family cap admitted.
`qualification-04` runs alone from source 2dfde352bd2d229146fc894a0694ab4db552b724;
monitoring is active again. Twenty-five Python tests pass. Main still requires
complete fresh gates and posted forecast. These are continuation facts, not
new campaign results or a rewrite of the earlier stop receipts.
