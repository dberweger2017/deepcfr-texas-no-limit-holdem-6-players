# Exact diagnostic LBR ranker: frozen engineering protocol

This bounded experiment starts from merged main `a2053bbeea8a5e5170a1d83bf5d440684f82283d`.
The completed #123 profile and draft #124 report identify ranking as a cost
hypothesis, not a speed forecast. No posterior or target conditional values,
training, paid host, release change or automatic merge is authorized here.

## One candidate and independent reference

Implement one original, pure Python exact seven-card evaluator using rank
counts and suit bit masks, with a generated 8,192-entry straight-high table.
It returns the same best-five lexicographic tuple as `src/game/showdown.py`
`hand_value`. It has the same 8,192-entry tuple-keyed LRU limit. Five/six-card
inputs delegate to the unchanged reference; no independent fast-arity claim.
No third-party code, tables or dependency is introduced.

Select it only in a new diagnostic cached-LBR executor. Bind the native
`choose_action` code object into private globals differing only in
`hand_value`; do not mutate native globals or engine settlement. This retains
the native utility/reduction, menus, RNG draws, range support, first-index
argmax and five-second soft checks verbatim. A private clock injection in
tests checks the native batch boundary; production uses `perf_counter`.

## Frozen rank checks

Use deterministic category/kicker/wheel/straight-flush/two-trips/three-pairs/
quads/full-house/best-five/board-tie fixtures, with explicit expected tuples.
Generate **100,000 distinct** seven-card sets by `random.Random(202610040101)`
sampling the canonical `search.DECK`, sorting each set in canonical deck
order, rejecting duplicate sets, and retaining generation order. Compare
every tuple with the independent five-subset reference. On every fixture and
the first 128 generated sets, check all 24 global suit permutations. On the
first 512 sets, check reversed and one-card-rotated input order. Keep the
generated corpus digest, counts, and the first minimal mismatch reproduction.
These are bounded checks, not exhaustive enumeration of all seven-card hands.

Run controlled timer fixtures with preparation already over five seconds,
and with the clock crossing five seconds after the second full comparison
batch. Both executors must attempt at least one full batch, stop only between
batches, retain requested/completed counts and identical RNG progression.
Also check zero-evidence preservation and first-index exact ties.

## Frozen real cases and inputs

Retain all **336** #121 cases, case digest
`1559b5aac31016a9db92cfc2b319c28abf110446866d50dba0f45126eacadc4e`,
and #119 selection digest
`578b8b67f21a15d2961e88cf498aeeb64831c9827e578e5ce1db62f85d43e322`.
Use the actual #117 curve raw directory
`/Users/dberweger/Local/hu20-b100-diagnosis-pr117/results/hu20-b100-diagnosis-m4-20260930/curve`
and the retained #116 recovery `models.json`, not its differently named hand
archive. Verify raw hashes, public-prefix digests, model/checkpoint hashes and
the sealed #121 and #123 retained manifests before execution.

For every real case compare original shared-cache and candidate shared-cache
executors with identical seed, ordered menu, weights/support, zero-evidence
records, exact chosen action/raise-to, requested/completed batches and final
RNG state. Require per-action values within absolute **1e-10 chip**, no relative
tolerance. Repeat candidate, and compare both under #121's coupled cyclic
suit/deck permutation. Preserve mismatches without weakening the gate.
Run the full corpus again in separate fresh original/candidate processes;
compare output digests with validation and each other.

Fixed-work equivalence uses an injected constant clock. Real-clock benchmark
runs retain the production five-second rule. Any completion/action difference
is reported separately and blocks an unconditional bounded-attacker drop-in
claim, even when fixed-work ranks are correct.

## Frozen uninstrumented performance measurements

Only after correctness passes, use separate fresh processes in this order:
original 336, candidate 336; original larger workload, candidate larger
workload. The larger set is exactly #121's hash-ranked **3,591 first calls +
672 incremental independent calls**, retaining its root `202610030101`,
selection, order and seed derivation. Do not select by timing or output.
Record actual action/value/RNG/range digests for every timed call, first versus
additional samples, all four streets and full inventory weights.

Separately time the 100,000 distinct rank sets, original then candidate,
clearing each LRU before its pass. Time the last 4,096 sets again immediately
after each pass to quantify controlled LRU hits. These microbenchmarks and
natural large-workload LRU statistics are distinct from saved-query caching.
No cProfile totals enter speed estimates. Report load/preprocessing, call
CPU/wall, startup-inclusive costs, repeated/cold behavior, peak aggregate
owned RSS, swap/free disk, and observed real-clock limitations.

Project all 58,047 one-sample likelihood calls using street-weighted first
and incremental means, conservatively using at least the first-sample mean
for additional samples. Show four/eight-sample likelihood, 24 decisions ×
**96 worlds per range** (48 selection + 48 held-out evaluation, two ranges),
historical 1,728-second value-work proxy, 1,200-second controls/report reserve
and **1.25× headroom**. Add proposed posterior-stability work explicitly;
no generated posterior or conditional values enter this engineering task.
Label subset extrapolation, unmeasured controls and memory scaling limits.
There is no arbitrary speed threshold and no ten-hour veto on the future audit.

## Resources, attempts and stopping

Register one immutable engineering clock on M4 immediately before the first
heavy validation/test, with deadline **start + 14,400 seconds**. Never reset it.
Before every phase read `/tmp/DR_RESEARCH_M4_COORDINATION.txt`, inspect actual
processes and claim task/PID/worktree/start; release promptly in coding gaps.
One heavy M4 process, AC/caffeinate, aggregate owned RSS ≤10.5 GiB, swap growth
≤0.5 GiB and ≥8 GiB free disk. External supervision checks these guards and
the deadline. M1 performs only lightweight edits/Git/compact transfer/status.
Retain every failure/partial and stop on rank/equivalence failure; reporting
repairs may preserve existing outputs but never rerun to obtain a favorable
timing. No candidate/parameter sweep follows a disappointing result.

Publish a draft PR with compact verified records and exact large-artifact
retrieval commands. Prepare a NEW posterior-audit attempt and concrete
host/safety/recovery proposal for owner review; preserve #119's stopped clock.
Claude v7 is advisory: resolve `a2053bb`, record absent temporary scripts and
unreplicated M1 training observations, and supply a separate observation-reuse
handoff. Its quoted cost, balance, worker count and training suggestions have
no authority here.

## Implementation correction within the original clock

Attempt 2 at `645bb86` passed independent ranking and native-code-bound
fixed-work tests, but review found that the production optional executor
captured `DECK` at import time. The fixed-clock helper rebound globals inside
the suit context and therefore did not test this production-context boundary.
Bind the active globals when each actual optional executor is constructed,
and have fixed-work validation change only that instance's clock. Add an
actual-executor cyclic-suit regression. This corrects the stated control,
without changing rank computation, cases, random seeds, menus or work counts.

The first attempt's repeated-rank timing also included output-digest
bookkeeping. Stop the timer before bookkeeping in the corrected run; retain
the original measurement and exclude it from performance claims. No original
attempt is overwritten. The corrected full protocol uses a fresh output
directory with `--clock` pointing to attempt 2's immutable start/deadline,
**1790769277.5442078 / 1790783677.5442078**. Only this corrected attempt's
measurements enter the final candidate performance claim. There is still one
candidate implementation and no parameter or outcome-dependent selection.
