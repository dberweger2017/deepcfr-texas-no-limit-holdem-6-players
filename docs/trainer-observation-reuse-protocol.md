# Acting-observation reuse: frozen engineering protocol

September 30, 2026. Owner-authorized task 1; no poker outcomes, strategy change,
automatic merge or substantial training campaign. Task 2 is a separate PR and
requires approval of its live-priced total rental cap before any rental.

## Candidate and exact control

Base `7d74b6c` was pulled from merged main before branching. The only candidate
is one private acting-observation cache on each immutable concrete `Hand`.
Only the acting seat with empty prior-hand history may use it. A new Hand,
transition, replaced event history, other seat or prior-hand context cannot
reuse that snapshot. `Hand.apply` still calls full `LegalActions.validate` and
the unchanged engine. No key/history/card/menu/traversal mathematics changes.
The original control executes the literal `Hand.observe` method from the
merged base, extracted from Git, in the same candidate checkout/process setup.

## Work frozen before execution

Python 3.11.14, pinned engine `5db20e3`, uncapped HU20, K1, two roots per
iteration (one per seat), one process. The small-table stream starts from zero
with seed 2026093011 and the #129 configuration (250k per-iteration node
ceiling, 4M entry ceiling, 300-second iteration ceiling). The mature stream
loads the original first fixed B100M seed 2026093001, checkpoint SHA-256
`b560669df702057b9df72c90495195a60741e1c6c4b42603cc7c330e2615f64a`:
`/Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-m4-recovery-20260929-1008/training/B-2026093001/checkpoint-100000000.json.gz`.
Its existing configuration/iteration/seed/weights are retained exactly.
Original inputs are read-only; engineering outputs are not promoted models.

For each starting state:

1. Original then candidate, 100k added complete nodes, instrumented correctness
   trace. Hash every returned observation, ordered menu/key and action RNG
   seed/state/choice; preserve deterministic iteration telemetry and final,
   current-policy and next-iteration checkpoint artifacts. Count actual replay
   calls separately; duplicate replay count is expected to differ.
2. Three fresh-process performance pairs, each 1M added complete nodes from the
   same original state, with order original/candidate, candidate/original,
   original/candidate. No tracing in timed runs. Save a midpoint at >=500k
   complete nodes, final checkpoint/current export, then one next iteration.
3. Fresh-process original and candidate resumes from their first performance
   midpoint, finishing that exact 1M target and next iteration. Compare complete
   final/next checkpoint and export bytes with direct runs and each other.

The first complete iteration crossing a target is retained; report overshoot,
completed iterations/nodes and any discarded partial work. Never shorten a
stream after inspecting speed or state. A failed job stops the sequence and is
retained; no blind retry or normalization of differences.

## Mandatory gates and measurement

All observations, keys, menus, action-RNG states, iteration work, regrets,
strategy sums, visits, keys/entry ordering and next-root RNG states must match
exactly. All deterministic artifact bytes must match on M4, including reload,
fresh resume and next iteration. No floating-point tolerance. Timestamp,
elapsed/replay seconds, RSS/PID, host provenance and process duration are
explicit nonsemantic metadata; do not remove any other difference.
Focused boundary/hidden-information/reopening/checkpoint tests run on M4.
Report all three throughput ratios (no minimum speedup target), load, midpoint
save, final save/export/reload, startup-inclusive duration, table growth and
peak resources. Adoption is conditional on correctness and actual utility.

## Resources and artifacts

One heavy M4 child at a time; claim/release the established `/tmp` coordination
note. M1 only edits/Git/compact transfers/status. Four-hour absolute engineering
safety ceiling from the first heavy child, never reset; this is a safety cap,
not a workload target. Existing supervisor enforces AC, aggregate owned-job
RSS <10.5 GiB, swap growth <=0.5 GiB, free disk >=8 GiB, and phase deadline.
Every attempt/log and original input hash is retained. Final global inventory
is created and independently checked after all logs close. Large archives stay
M4 with exact hash/retrieval commands. No evaluation/strength claim.
