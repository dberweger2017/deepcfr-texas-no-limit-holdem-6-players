# Approved stability-gated posterior audit v2

The owner authorized this execution after merging #126. The merged base is
`7d31a39c80cc72deb772336ce251f3c4bdcd46c9`, containing reviewed head
`b49c1d9b231bd3f298e89a89f7d848a26fef4de1`. Both `test` and GitGuardian
passed; the sealed report SHA-256 is
`ea9ea795e68a30e13bea81a9020347ed15380324af54b5ea43e91449db19f52c`
and manifest SHA-256 is
`c82f6808163e23b7588f6753147d4933b56bb3b3249abec11356f41b18e70fe6`.
All 64 archived files were reverified without rerunning the benchmark.

## Frozen scientific settings

The [merged proposal](hu20-posterior-audit-v2-proposal.md) and
[fixed selection record](reports/hu20-posterior-audit-v2-proposed-selection.json)
remain unchanged. The exact 24 coordinates and digest, five stability cases,
four suit cases, three river references, four likelihood samples, three
independent four-sample stability repetitions, one 16-sample comparison,
96 worlds per range and 48/48 split are retained. No new scientific outcome
has been inspected at this implementation freeze.

Main root: `202610050126`; stability: `202610050127/128/129`; higher-count:
`202610050130`; worlds: `202610050131`. Likelihood streams use the existing
`stream_seed(root, "test", "opponent", "reverse-lbr", rank, public_event_index,
holding, sample_index)` derivation. Repetition identity is its independent root.
The coupled suit estimate uses the original holding label for its seed and
maps both holding and ordered deck through the fixed `c→d→h→s→c` permutation.

Timer holdings are lowest/median/highest SHA-256 of
`repr((rank, public_event_index, holding))`, as in the retained preflight.
Two independent timer seeds use
`stream_seed(main_root, "validation", "opponent", "timing-v2", rank,
public_event_index, holding, repetition_index)`. Compare original-ranker
shared-cache and ranked shared-cache executors with actual five-second clocks,
including completed batches, action, value vector, posterior and RNG state.
Every recorded attacker prefix must also have complete requested batches.

For paired range worlds, holding uniforms are
`stream_seed(world_root, "test", "deal", "holding", rank, world_index)`;
runouts use the corresponding `"runout"` stream. Target policy and LBR
continuation streams use `"action", "target"` and `"opponent", "continuation"`
with the same rank/world coordinates. Range/control labels are journal IDs;
they deliberately do not separate the paired randomness. One-world candidate
actions share cards and continuation streams. No played hidden cards, original
deal seed or recorded future deck enters these calculations.

The real-clock gate runs first. Three four-sample and one 16-sample stability
estimate run for all five cases. Any failed prospective gate stops before
main likelihoods/values. If they pass, compute all main likelihoods and apply
the same gate again with the main estimate on the five cases. Then run coupled
suit likelihoods, primary/suit values and river references. Every holding is
sampled at every prior action, including previous finite-sample zeros.
Zero evidence or limited simulated LBR work prohibits inference; no smoothing,
extra samples, coordinate replacement or favorable retries are permitted.

River reference boundaries are native terminal fold/call and matched
checkdown, over all compatible holdings, compared with an independent chip
ledger/ranker. No strategic raises are solved. Count reconstruction and action
transitions toward the fixed 100,000-node ceiling per coordinate. An incomplete
reference remains incomplete; it is never exact full-game response evidence.

## Durable supervision

`scripts.run_posterior_audit_v2` starts the immutable 9.5-hour clock immediately
before input verification/focused tests, the first heavy work. New science
stops at +9 hours; reporting/sealing retains the final 30 minutes. One heavy
child, AC/caffeinate, aggregate owned RSS ≤10.5 GiB, swap growth ≤0.5 GiB,
and free disk ≥8 GiB are externally checked every two seconds. All poker
tests, loads, evaluation and hash/audit scans run on M4. M1 only edits/Git/
compact transfers/status. No training or paid host belongs to this audit.

Likelihood journals persist raw deterministic sample IDs with count equality,
holding, public-prefix metadata, exact observed/predicted action, requested/
completed batches and timer status. Flush each row; fsync every holding; seal
hash indexes every 64 holdings and every completed coordinate. Worlds are
fsynced/hash-indexed individually. These checkpoints are well inside the
15-minute ceiling. A torn uncommitted tail stays in its original segment;
recovery opens a new segment and skips every valid persisted ID. Failed rows
also remain completed records and cannot be retried blindly. The immutable
clock and frozen source must match on `--resume`; an already running or stopped
coordinator cannot be resumed. All large archives remain M4-only.

The detached wrapper closes coordinator/phase logs before the separate final
seal. The independent reporter recomputes count-based posteriors and the
held-out mixture gaps/intervals, verifies durable IDs and all parent/model/input
hashes, and writes a readable report even for a scientific stop. The final
inventory is verified again after log closure. No experiment setting changes
in response to results. Local gaps remain conditional on one estimated range;
their intervals do not include all posterior-estimation uncertainty.
