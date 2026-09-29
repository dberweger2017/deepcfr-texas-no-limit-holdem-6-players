# Frozen blueprint lookup and free-fold audit

This PR tests one repaired lookup boundary without changing checkpoint bytes,
training, the action menu, raise sizes, search solver or river solver. The
legacy lookup remains available for every historical report. The explicit
`button-zero-compatible-v1` lookup is permitted only for the three
hash-pinned fixed-button checkpoints described in
[the lookup contract](blueprint-seat-symmetry.md). The no-free-fold wrapper
postprocesses a distribution after either lookup while retaining the exact
original menu for key hashing.

## Preflight and chosen work

The [resource-only preflight](reports/blueprint-seat-preflight-m4.md)
completed 504 hands with each of the 5.83M and 12M checkpoints, without
saving or inspecting profits. It measured 4.89/6.47 GiB peak process RSS,
22.77/61.44 seconds including load, zero illegal actions and unchanged swap.
The [5.83M plan](../configs/blueprint/seat-symmetry-confirmation-5m-m4.json)
and [12M plan](../configs/blueprint/seat-symmetry-confirmation-12m-m4.json)
therefore freeze the same two fresh schedules: 1,024 independent six-rotation
blocks against `tight_passive`, `loose_aggressive`, `pot_pressure`, and 256
blocks against `random`. Every block rotates the evaluated identity through
all six seats. All arms share the exact deal seed, opponent assignments,
button schedule, public hand identifier and per-identity action seeds; each
arm gets a separate policy instance. Different actions can lead to different
later public histories despite these common streams. The scripted pool and
random suite are separately named. These seeds have not been used in earlier
blueprint validation or in the resource preflight.

Each checkpoint loads in its own sequential process. Each process has a
4-hour-50-minute wall cap, 10.5-GiB lifetime process RSS cap and 30-GiB
free-disk guard; the two caps sum to less than the ten-hour M4 experiment
ceiling. Record system memory pressure and swap separately. The 58.02M
checkpoint receives the [streamed whole-table audit](reports/blueprint-seat-density-m4.md)
but is not loaded for play under the M4 memory ceiling. No paid host is used.

## Arms and fixed contrasts

- `U`: uniform original menu.
- `U_safe`: uniform original menu after no-free-fold processing.
- `B_legacy`: current regret-matched checkpoint policy, legacy lookup.
- `B_legacy_safe`: legacy policy after no-free-fold processing.
- `B_canonical`: the same checkpoint policy through the button-zero
  compatibility lookup.
- `B_canonical_safe`: compatibility lookup followed by no-free-fold processing.
- `TAG`: the existing `tight_aggressive` scripted hero.

The **primary learning contrast** is `B_canonical_safe − U_safe` against the
scripted pool for each checkpoint. Estimate each block's BB/100 from the six
rotations, then form a paired mean and a Student-t **97.5% two-sided interval**
over independent blocks. Bonferroni across the two checkpoint primary claims
gives at least 95% family coverage. A positive lower bound is evidence of an
additional learned-policy contribution under this declared comparison; a
nonpositive bound is inconclusive or adverse, not proof of zero value.

Predeclared additional contrasts are `B_canonical − B_legacy`,
`B_canonical_safe − B_legacy_safe`, `U_safe − U`, and
`B_canonical_safe − TAG`. They, the random suite, absolute arm returns,
checkpoint scaling and subgroup views use unadjusted 95% intervals and are
exploratory. We will not increase block counts based on a near-positive
interval. An incomplete or invalid run has no confirmatory contrast.

## Coverage and artifacts

At every evaluated hero decision the audit computes both legacy and
compatible keys on the **same observation**, without drawing an action or
consuming RNG. It records trained/missing matches by street and button,
found-entry visit histograms, found-but-single-update counts, and both
action-frequency-weighted and distinct-key-within-group summaries.
Separate same-decision categories (`both`, `legacy_only`, `canonical_only`,
`neither`) isolate coverage effects from policy-induced histories. The
no-free-fold wrappers count eligible decisions, changed distributions,
all-fold-mass cases and total removed probability. Selected action counts
are retained by street and button. Whole-table density statistics remain a
separate analysis: hashed keys cannot be assigned to streets after the fact.

The runner retains a hand row for every attempt, failures with public event
traces, per-block progress, reached-decision counters, manifest, source and
checkpoint hashes, resource measurements and artifact checksums. It checks
chip conservation and legal actions through the native engine. The analysis
requires every planned arm/rotation/block row before reporting paired
intervals. We will retain negative and inconclusive results and leave PR #109
draft for review; this audit cannot promote a default player or establish
six-player professional strength.
