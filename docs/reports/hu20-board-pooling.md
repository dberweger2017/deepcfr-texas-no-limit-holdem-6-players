# HU20 board-blind pooling — revision 3 qualification stop

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
