# First-decision local CFR construction smoke

This bounded M4 check ran draft PR #106 at clean source revision
`765610c0d543f233185077fe0f7d3bff6a5ee9fa` with the saved 12M-entry
blueprint (`c6649a3e8ab061c70c5f6192d91f1f4090f15d68f4821cf3b50f2746f816c845`).
The [frozen smoke plan](../../configs/blueprint/local-cfr-frontier-smoke.json)
used new validation roots 2026093001 and 2026093002, eight scripted blocks,
four random blocks, a 15-minute wall limit, 10.5-GiB process RSS limit and
five-second per-decision limit. No paid host or larger checkpoint was used.

## Result

Both comparisons were valid. All **144 hands** completed with zero invalid
actions. The scripted suite had **3/3** eligible first-hero-flop local solves;
all completed **128** full per-player cycles with no fallback. Its maximum
solver time was **3.00 seconds** and peak process RSS across the run was
**6.55 GiB**. The four random blocks happened to contain **no** eligible
local solve. Unsupported decisions delegated to corrected rollout search.
The small smoke passed the runner's operational check, but three solves do
not establish a 95% completion rate for the changed implementation. The
previous 84/84 feasibility result belongs to the earlier source revision.

The [machine-readable result](blueprint-local-cfr-frontier-smoke.json) retains
all telemetry. Raw hand rows, solver-attempt rows, manifests, checksums and
TensorBoard events are on the M4 in
`~/Local/blueprint-local-cfr/results/local-cfr-frontier-smoke-20260925/`;
a same-revision replay is in the matching `-replay` directory. All eight
checksummed files verified in each directory. The 96 scripted hand rows were
byte-identical between runs (SHA-256
`cc968279b7fd8222ab238c62f9fd6c2fa0ab1f8b927f6745ef8d3cf4998647ec`),
as were the 48 random hand rows (SHA-256
`a71a0fd09073b90f6385f5fe5ca1bc87c1907ae09ac587c7da827402d9e57df9`).

## Construction checks and limits

The production turn-boundary leaf traversal was tested with two identical
flop observations and different internally sampled turn cards: continuation
regret updates used the same pre-turn information set. A separate native-game
test checks the public key directly. Eligibility tests show a later hero flop
decision delegates to corrected search. An exhaustive three-seat game with
nonuniform private ranges checks the ordinary and forced-hero compatible-deal
proposal weights and regret increments against independent enumeration. It
also demonstrates that adding the targeted pass reweights the actual-hand
stratum, so the hybrid update remains a documented heuristic rather than a
standard unbiased external-sampling estimator. The old average accumulator
was removed because sampled regret visits did not yield a correct own-reach
average and its posterior ranges did not affect play.

The M4 focused local-CFR suite passed **17 tests**, and the complete suite
passed **792 tests**. A full local M1 suite attempt was invalidated by a
nearly full disk during temporary-fixture creation; the M4 suite passed with
ample space. This smoke is a resource and reproducibility check, **not** a
paired playing-strength retest. The [earlier full-arena report](blueprint-local-cfr-corrected-m4.md)
still shows no established gain, but its numerical outcome must not be
attributed to this changed solver.

**Decision:** keep #106 draft for review. The changed first-decision pilot is
legal and reproducible in this short M4 check. A larger fresh-schedule run
would be needed to re-establish the 95% feasibility gate and paired effect at
this revision; low continuation coverage and sparse useful range traversal
remain measured concerns. No 58M or paid-host comparison follows from this
smoke.
