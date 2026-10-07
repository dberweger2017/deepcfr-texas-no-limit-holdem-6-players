# Native HU100 preparation validation

Prepared on October 7, 2026 from main `cf8428e`, in the isolated
`feature/native-hu100-preparation` checkout. [Protocol and operator commands](../native-hu100-preparation.md).
This report contains engineering evidence only; there is no trained HU100
model, measured HU100 resource estimate or poker-strength result.

## Changes and checks

Native equal-stack hands now carry explicit HU20/HU100 identity. Keys select a
separate HU100 schema and payoff targets subtract the selected initial stack.
Traversal order, regret updates, linear weighting, opponent-sampled averaging
and the existing menu/card descriptor remain unchanged. HU20 default checkpoint
headers and released format IDs remain unchanged. New HU100 and opt-in HU20
recovery checkpoints preserve counters and coverage provenance; legacy recovery
requires independently verified completed nodes.

Research current/average readers, extraction audits and arena registration bind
the full versioned game. The HU100 schema is explicit at average loading and a
wrong starting-stack scenario fails before dealing. Release pins and the web
catalog are untouched. The preparation tool only writes plans/commands; the
existing macOS supervisor accepts explicit per-campaign resource limits.

Local validation used a new ignored Python 3.11 environment in this worktree
with the pinned engine, NumPy and pytest. An initial isolated Python 3.14 setup
was rejected by the engine's Python requirement; it did not change a shared
environment or run any fixture. Final checks:

- `cargo test --locked --manifest-path native/hu20-trainer/Cargo.toml --lib`:
  **6 passed**, including exact recovery on tiny legal river roots in both games,
  legacy recovery provenance and malformed row rejection.
- `cargo build --locked --manifest-path native/hu20-trainer/Cargo.toml`:
  debug library/binary compile successfully.
- Python preparation/serialization/rules fixtures plus compact-policy checks:
  **14 passed, 2 skipped**. Skips require external retained large policies;
  no policy was downloaded. The new preparation and HU20/HU100 fixtures all pass.
- Native/Python parity includes 24 deterministic generated correctness hands,
  plus six explicit refund, short-all-in and split-board hands. It checks legal
  bounds, menu targets, byte-identical keys and final chip accounting. Tiny
  manually populated checkpoint rows verify exact export normalization,
  independent row audits, compact loading, cross-game rejection and snapshots.
- `bash -n` accepts generated operator commands; manifest tests verify that
  preparation does not invoke a trainer, supervisor or research process.
- Python compilation and staged repository-artifact/diff checks pass.

Normal PR CI builds the release native binary and runs the previously optional
native parity/export regressions with the existing full test suite. This local
report does not claim those heavier checks passed before CI returns. The pinned
1B HU20 reference regeneration is deferred to Stage 1; tiny fixtures cannot
establish full campaign hash equality.

## Independent review

An independent agent reviewed implementation commit `38eefbd`, focused on
native rules, hidden-information keys, CFR payoff/regret/average math, resume,
exports and research evaluation loading. It found no blocking correctness issue
and independently repeated **6 Rust /8 Python fixture tests** without research
runs. Its hardening recommendation was implemented: coverage counters are
validated against completed nodes and stored visits, and a baseline identifies
coverage collected after legacy recovery. Final recovery/protocol review found two campaign gate issues: the generated
shell could continue after a failed phase, and an incomplete pilot could be
admitted for extension. Both were fixed with fail-fast shell commands,
target-bound audits and explicit 10M/nonzero-street/full-baseline parent checks.
Success and rejection fixtures pass. The reviewer independently repeated
**6 Rust /12 new Python fixtures**, with no remaining blocking findings.

## Deferred work and decisions

No `train` command, pilot, arena/LBR run, benchmark, large artifact retrieval,
M1/M4 worker operation, background scheduling, merge or release was performed.
No research checkpoint, export, archive or cleanup receipt was produced; test
fixtures were transient. #188/#190 were checked as open, with research-only
protocol/evidence changes, and their active inputs remained untouched.

The current abstraction/menu is a limited baseline, with unknown HU100 key
sparsity, RAM and throughput. The staged future protocol requires HU20 reference
regression, an audited 10M resource/coverage pilot, then a measured quote and
explicit owner authorization before approximately 1B total nodes. The proposed
budget is not a convergence claim. Later multi-seed strength confirmation and
the external benchmark require separate protocols. The owner still chooses the
benchmark opponent/acceptance criteria, idle worker and any revised compute plan.
