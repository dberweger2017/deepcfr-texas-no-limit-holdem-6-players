# HU20 board-blind pooling — engineered qualification in progress

**The owner-authorized wrapper engineering passed exact pilot-0 scientific
parity and reduced the provisional forecast to 15.542 hours, versus 17.102
remaining before reserve. Remaining qualification is running; main has not
started and no hypothesis decision is available.** The original 31.182-hour
forecast stop and every prior attempt remain preserved below. Draft
[PR #149](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/149)
retains all evidence; paid cost is $0.

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

Qualification-05 runs retained fixtures and the remaining fixed roots afresh.
The new first solve/lock can be reused only through pinned exact comparison;
first V4's fixed 20k native MC additionally requires its blueprint EV to match
the fresh value exactly. Its previously unfinished deterministic replay runs
fresh. Every pilot retains separate solve/lock RSS and timings. All old
failures, clock charges, science, 7/8/4-GiB limits and reserve remain unchanged.
**32 Python tests pass.** Full raw retrieval awaits terminal closeout; compact
receipts above are copied locally, while all raw M4 originals remain intact.

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
Draft [PR #149](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/149)
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
