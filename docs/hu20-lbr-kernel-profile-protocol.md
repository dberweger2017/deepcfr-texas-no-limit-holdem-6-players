# B100M faithful LBR kernel profile: frozen pilot

This is an outcome-free engineering pilot after PR #121. It does not form
posterior-conditioned ranges, target action values, playing returns, or a new
training policy. The native `LocalBestResponse` and saved B100M policies are
unchanged. A larger or paid run is a separate decision.

## Fixed input and selection

Use PR #121's 336-case corpus with `case_digest`
`1559b5aac31016a9db92cfc2b319c28abf110446866d50dba0f45126eacadc4e`
and #119 selection digest
`578b8b67f21a15d2961e88cf498aeeb64831c9827e578e5ce1db62f85d43e322`.
Verify the raw-hand source hashes and B100M seed roster as in the existing
validation runner. Within each of the 24 seed × street × position strata,
choose the case with the lowest SHA-256 of
`lbr-kernel-profile-v1|<case_id>`. The resulting IDs are committed in
`docs/reports/hu20-lbr-kernel-profile-selection.json` before profiling.
Selection uses no LBR value, chosen action, terminal payoff, or timing.

For each seed, load its unchanged model once. For each selected case, run a
fresh cached LBR instance once with a new shared query cache (cold), then a
fresh instance with the same cache, public prefix, hypothetical holding and
internal RNG seed (warm). Profile the warm call with Python `cProfile`; report
its requested and completed batches and the cold and warm wall/CPU times.
This deliberately measures an upper bound on reusable saved-policy queries;
it is not a complete-workload speed benchmark. The exact `LBRConfig(4, 5)`
and native five-second soft-time check remain unchanged. Profile overhead can
change batch completion near that soft limit; record the difference and do
not claim equivalence from this pilot.

## Measurements and decision rule

Report per-call and per-street `cProfile` self time and call counts for
`hand_value`, `LocalBestResponse.update`, `_fold_probabilities`,
`_sample_world`, random chance sampling, card/deck construction and NumPy
utility work where profiler attribution permits. Also retain the top 25
functions by self time, cold/warm timing, model-load time, sampled RSS, swap
growth and free disk. Interpret profiler self time and cumulative time
separately; do not sum overlapping cumulative times.

The largest aggregated non-cache self-time category across the balanced 24
warm calls is the **only candidate** for a separate exact-kernel optimization.
If the profile lacks one clear dominant category or the cache is not warm in
these cases, stop with the measured diagnosis. Do not swap candidate targets
after seeing an implementation's speed. No poker outcome, new training or
paid machine is part of this pilot.

## Execution limits

Run on the refrigerated, AC-powered M4 only, after checking processes and
`/tmp/DR_RESEARCH_M4_COORDINATION.txt` and claiming a short exclusive window.
Use one heavy process, at most 10.5 GiB owned-job RSS, at most 0.5 GiB swap
growth, and at least 8 GiB free disk. The profile has an absolute one-hour
ceiling from its first M4 process, which is a safety bound rather than a work
target. Preserve failures and partial attempts. M1 may only edit, transfer,
perform light Git work, and inspect compact status. Release the M4 immediately
after the profile exits.
