# Faithful reverse-LBR likelihood acceleration: frozen protocol

This is a new, dependent research task after the #119 feasibility stop. It
measures implementation equivalence and speed only. It must not produce a
posterior-conditioned range, target action value, poker profit, or training
result. The native `LocalBestResponse` remains unchanged and is the oracle.

## Fixed source and equivalence corpus

Use #119's frozen 24 trained B100M decision coordinates and the same #117
fresh-curve raw hands. Verify their saved selection digest and raw-file hashes.
For each selected target-visible observation, take the first and last **prior
attacker decision** in its public history (one if identical; none if no prior
attacker action), plus the first subsequent attacker decision after the
selected target action when one exists. The attacker-visible prefix is the
native public history immediately before that action. This includes a
preflop/button case even when the target acted first. For each
prefix, enumerate holdings compatible with the target's current cards and
the board visible at that prefix and choose the lowest, median and highest
holdings by SHA-256 of
`reverse-lbr-corpus-v1|selected-rank|event-index|holding`. Use two independent
internal RNG seeds per holding, derived from root `202610030101`, the selected
rank, event index, holding and sample index. Keep all cases, including calls
whose original observed attacker action differs from the simulated action.

Add deterministic synthetic heads-up preflop and river fixtures with legal
check/call/raise/facing-wager states, fixed decks and RNG seeds. Add a
zero-likelihood Bayes update unit fixture and an exact first-index tie unit
fixture. Cases with fold/check/call/raise in the *legal menu*, and any chosen
instances of those actions, are reported by category; absent categories are
reported as corpus limits, not filled after inspecting results. Include the
same global cyclic suit permutation of selected corpus cases as a separate
paired control. For the control, permute both players' cards, all public board
cards and the deck enumeration order used by the native LBR and world sampler,
while preserving the same RNG draws. This couples corresponding worlds rather
than comparing independent finite-Monte-Carlo samples.

Generate and commit the corpus manifest, its hashes, source hashes and case
identifiers **before** running the optimized implementation or reading its
timings. The corpus generator reads no terminal payoff or LBR numerical value.
The existing #119 outcome-blind selector is not changed.

## Equivalence criteria

For each fixed public prefix, hypothetical attacker holding and internal RNG
seed, run native and optimized LBR from fresh instances. Require exact chosen
action including raise-to, identical ordered menu, requested/completed chance
sample counts, zero-likelihood event record, compatible/positive target-range
support, and deterministic repeat. Compare the returned per-action chip-value
vectors elementwise at absolute tolerance `1e-10` chip; this is far below the
integer chip unit and should normally be bit-identical because caching changes
no arithmetic order. An exact action mismatch is a failure regardless of
vector tolerance. Retain all mismatches and near-tie margins. Compare the
native and optimized implementations separately on the suit-permuted corpus;
also check native original versus native permuted under the coupled deck/RNG.
If suit coupling itself fails, report that as a validation failure rather than
relaxing the test.

Stage A uses deterministic small fixtures to check source-query cache keys,
Bayes zero-evidence behavior and tie-breaking. Stage B runs the fixed real
corpus above. No performance claim is accepted before both stages pass.

## Candidate and cache safety

The first candidate is a **separate selectable executor** that shares only
immutable `source.distribution(replay(history, seat, pair))` results across
short-lived LBR instances for one saved source. The key is the complete public
event tuple, acting seat and hypothetical private pair; the cache is bound to
one source object/model identity. The cached function has no RNG input and no
access to the actual opponent cards or future deck. Every LBR still owns its
Bayesian weights, sampled future-chance RNG, timer, values and decision.
Reuse across a different saved source or across a changed public event is
forbidden. Keep the native class and its default call path untouched. A cache
that changes support, sample completion or action fails validation.

## Timing and resources

After correctness passes, time native and optimized executors on the same
frozen real corpus in separate fresh processes with the same model-loading
path. Measure cold model/checkpoint load, native/cached call wall time,
per-street times, cache build/hits/misses/bytes, CPU time, RSS, swap and disk.
Report single-process algorithmic call speedup separately from whole-job
startup-inclusive speedup. The optimized executor may process a larger
hash-ranked prefix of #119's 58,047 one-sample calls, or all if a conservative
timing check shows it fits; do not launch native over the full workload. Keep
partial attempts and every failure. Extrapolations must label the measured
fraction and variation across streets.

Recompute the complete future #119 cost for at least 4×96, 4×192, 4×384
and 8×96: 24 selected decisions, all compatible holdings, both uniform and
posterior values, suit controls, independent river references, report/audit
reserve and at least 1.25× headroom. Use #119's measured 1,728-second
two-range 96-world allowance as a *historical planning proxy* until a better
outcome-free measurement exists. Do not run those worlds or references.

Use one heavy M4 process at a time, 10.5-GiB RSS, ≤0.5-GiB swap growth and
≥8-GiB free disk. Before every heavy phase, read `/tmp/DR_RESEARCH_M4_COORDINATION.txt`,
inspect real processes, claim the M4 with task/PID/worktree/start, and release
it when the phase ends. Permit other agents' bounded work during coding and
reporting gaps. M1 must not run poker/model benchmarks. This attempt uses a
three-hour absolute heavy-work ceiling from the first M4 validation; it is a
ceiling, not a workload target. Stop earlier on a correctness failure, unsafe
resource use or plainly unhelpful throughput. No paid host or training.

One final report and machine-readable set must state exact equivalence,
algorithmic and end-to-end speedup, measured or defensible complete-workload
time, 4-sample time, total future experiment time, M4 feasibility and exactly
one next recommendation. If feasible, do **not** launch #119's scientific
value experiment without separate owner authorization.
