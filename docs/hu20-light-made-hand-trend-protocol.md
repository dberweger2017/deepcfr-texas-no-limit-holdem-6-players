# Completed light-panel made-hand trends

I analyze only the closed selective-stackoff light records from #136 on M1,
without model loading, new deals or changes to the active campaign. This is a
post-hoc descriptive extension of the frozen #134 first-large-raise analysis.

## Inputs and checks

Doctor Research owns M4 and the originals. Transfer only closed compressed
hand files, their result/task identity metadata and the frozen campaign plan,
under the existing coordination note. Verify every transferred file's size and
SHA-256 before decoding. Preserve model/checkpoint hashes, original scientific
source and plan identity, transfer authorization and retrieval paths. Keep large
raw inputs under ignored `results/`; publish compact selected events and hashes.

Use every available planned 100M–500M light checkpoint for all three original
lineages, including intermediate 150M/250M/350M/450M saves. Do not choose
checkpoints or hands by return. Require complete paired blocks, exact panel,
seed/model/checkpoint identity, native-replay evidence, chip conservation and
recomputed tail counters. Missing tasks remain explicit; a three-lineage table
requires all three matching checkpoints. Do not replace it with fewer seeds.

## Analysis

Reuse `first_large_raise` from `src/diagnostics/stackoff_made_hands.py` unchanged.
Large means the rival owes at least 800 integer chips, accounting for its street
commitment and remaining stack. Each hand enters at its first large target
raise only, split by whether the rival previously raised on that street.
Compare both made hands on the board at that action; preflop is separate.

For every checkpoint and lineage retain counts, folds/continuations/no-response,
ahead/behind/tied/preflop, one-pair postflop continuations, street denominators,
whole-hand chips and exact full-stack wins/losses. Report full-stack counts both
for all panel hands and within the selected events, with separate denominators.
Include no-large-raise hands and all original whole-hand profit in panel totals.

Both cards are joined only offline in generated simulator data. The playing
policies' information boundaries and actions stay unchanged. Made-hand order is
not equity; draws, future runouts and alternative-action values are unmeasured.
Whole-hand returns are not bet EV. Shared deals and selected events are
correlated, so these counts are descriptive. No independent Luna confirmation,
strength claim, abstraction intervention or promotion follows.

## Scope

One sequential M1 analysis worker; no M4 parsing, model load, training or paid
compute. Tests use small generated records. Keep all original transferred bytes
and report input/output hashes, source revision, time and peak RSS. Stop after
the scoped report and PR; the other two requested diagnostics have separate PRs.
