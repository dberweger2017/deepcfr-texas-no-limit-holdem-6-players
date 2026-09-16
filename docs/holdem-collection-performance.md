# Profiling Hold’em collection

Profile the saved policy and random streams that exposed a collection limit before
choosing a larger training budget. This is a diagnostic replay, not resumed
training: no replay admission, fitting, strategy publication or model selection.
The original incomplete report and its limits remain unchanged.

## Replay a checkpoint's next collection root

```bash
python -m scripts.profile_holdem_collection \
  --run results/holdem-baseline-v1 --job scenario-3-seed-103 \
  --iteration 1 --traverser 4 --max-nodes 10000 --max-seconds 60 \
  --out results/collection-prefix
```

The checkpoint's SHA-256 and stored manifest are checked before loading. The
report records that historical manifest separately from the current source and
environment: loading an old policy for a diagnostic does not certify that the
current checkout could resume its training byte for byte. `--iteration` names the
completed checkpoint; the collector reconstructs the next iteration's button,
physical-seat mapping, deal stream and opponent-action stream. `--traverser`
is a compact participant index, as in the collector.

Use a new output directory. A node/time limit produces `status=collection_limit`
and preserves the exception and measurements; it does not return partial samples.
For completed roots, the report includes node, terminal, target and execution
counts, net BB value and a digest of the full public records and learning targets.
`collection_seconds` excludes setup and digest construction; `total_seconds` adds digest construction but also excludes setup. Neither root value nor
collection speed is an arena win rate.

Add `--instrument` to save `collection.prof`, readable with Python's `pstats`.
Instrumented timings include profiler overhead; use separate uninstrumented runs
for before/after comparisons. Peak RSS is the whole process's high-water mark,
including imports and checkpoint loading, measured before output serialization.
Fresh processes give comparable cold-cache measurements. Cache warm-up and
contention from other processes can affect wall time.

The time limit is checked cooperatively at each visited node, not a hard process
kill. Setup, an in-flight operation, failure cleanup and result hashing can extend
elapsed time beyond it. The CLI limits each collection request to 300 seconds.

## Measurements and acceptance

- Retain the original unequal-stack seed 103 timeout as the motivating failure.
- Compare identical fixed node prefixes in fresh processes, with the same saved
  weights, deal, branch order and action random stream. Report repeated timings
  and process peak memory, not only the fastest run.
- Profile call costs separately. Cumulative timings overlap; do not sum them as
  independent fractions of total runtime.
- Compare full traversal digests on completed four-/five-/six-player fixtures.
  Preserve every target, legal execution and zero-probability traverser branch.
- Keep observation leakage, symmetry, side-pot/showdown, transactional failure
  and fresh-process recovery checks passing.
- Preserve timeouts in larger bounded probes. A faster prefix does not establish
  the total tree size or an end-to-end training budget.

## Implementation

Public replay accumulates per-player values in local scratch arrays and creates
immutable `Player` records once at the observation boundary. Nothing mutable
escapes or persists between replays. Validation and public disclosure behavior
are unchanged.

Three bounded caches retain pure calculations on immutable inputs: visible-card
canonicalization (4,096 entries), event feature rows (16,384) and showdown hand
values (8,192). Event keys retain the relative seat, betting street, pot, big blind
and canonical revealed cards. A board reveal or a different owner's suit mapping
therefore cannot reuse an incompatible row. They cache neither model predictions
nor opponent observations. They are process-local performance state, not training
state, and do not need checkpoint recovery. Bounds limit retention across hands;
longer campaigns still need memory measurements.

Shared event rows also avoid retaining a fresh copy of every prefix's numeric
history for every branch. The policy still receives the full history, and source
records remain attached to each decision. No history truncation, bet-menu change,
branch pruning or sampling correction is introduced.
