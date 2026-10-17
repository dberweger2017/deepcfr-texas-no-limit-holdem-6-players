# HU200 river search on the frozen v0.5.5 blend

Owner-requested study, October 16, 2026. This document records the prospective
sequence; an unfinished gate is not an evaluation result. One draft PR holds
implementation, correctness, cost, the committed strength declaration and readout.
The owner decides merge. No Slumbot play, training, release or default change.

## Inputs and scientific boundary

Use #306's audited average, 6,282,968,150 bytes, SHA256
`ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485`,
156,208,331 entries and 482,975,338 action slots, with its pinned K50 tables.
Verify the model and a read-only play cache before use. Blueprint play outside
river search uses translation 512/128. All builds, search, cache verification
and evaluation compute run on the 16 GiB Apple M4. Shared research locks and
resource guards apply. bb128 supplies files and stores final evidence only.

Policies receive public observations and their own cards. Enumerated holdings
are synthetic possibilities, never the actual opponent cards. Both ranges use
blueprint action likelihoods over the public history, with structural card
exclusion and compatibility normalization. No turn search. Preserve complete
hero matrices at earlier river decisions, including fallback matrices, and
reuse on-tree river solutions within a hand. Insert exact observed wagers
when rebuilding the river tree, including 1.5 times pot.

The external solver starts at the equal-contribution river round root. Lock
every observed opponent river prefix to the blueprint likelihood model and
every earlier hero prefix to the matrix actually used. Conditioning those
fixed path factors gives the current-decision compatible product law. Check
current joint reach and the queried holding's marginal support. Reuse a prior
solution only if all earlier complete lock matrices agree by action/holding.
Future opposing actions are still solved equilibrium responses.

For an exact menu action use its blueprint probability. For an off-menu raise
the inherited kernel is `max(.01, sum(p_raise * exp(-abs(log(actual/menu)))))`.
This .01 kernel floor is distinct from the zero opponent-likelihood floor.
At an observed off-menu river prefix insert its exact action with that
likelihood and scale the old menu by one minus that likelihood. This is a
declared augmented action model, not literal blueprint probability for an
action absent from its menu. It remains a range-model limitation.

The existing external solver supports fixed hero strategy locks but exposes no
opponent counterfactual-value gadget for safe re-solving. The initial design is
therefore ordinary joint-range river CFR with hero locks. It has no safety
or improvement guarantee: conditioning on an incorrect blueprint opponent can
misread value callers, and equilibrium continuation can predict folds that a
scripted opponent never makes. The HU20 diagnosis is a required diagnostic,
not a source of opponent-specific policy rules or hidden information.

## Ordered gates

1. Independently evaluate small river profiles and best responses; verify
   exact wagers, locks, no hidden information and repeated searched decisions.
   Correctness failures remain findings and block dependent strength play.
2. Freeze an outcome-blind real-spot sample before timing. Include #304
   journals and the blend exploratory journals through read-only copies.
   Measure complete decision preparation, solve and output cost, with p50,
   p95, maximum and peak process-family RSS. Target p95 at most 3 seconds
   and at least 3 GB physical headroom. Report failures in the denominator.
   A missed target is a design/plumbing finding for the owner; do not reduce
   iterations, narrow ranges, cut menus or accept partial profiles silently.
3. Before strength play, commit exact solver work/budget, fresh reserved roots,
   counts, stopping rules and analysis. Direct duplicate blend+search versus
   blend is primary. Run all seven scripts with both arms on common deals
   and independent paired blocks; emphasize pot_pressure. Use pilot cost and
   independent-block variance to size for +10 BB/100 at approximately 80%
   two-sided 5% power. Include reproduction, replay and 25% time headroom.
   If more than about 24 M4 hours are needed, propose counts before running.
4. Read outcomes only after complete replay/reproduction. Publish all nominal
   95% block intervals and seven-opponent adjusted contrast intervals, changed
   actions, model EV deltas, fallbacks/timeouts, per-opponent effects and a
   range-model diagnosis of losses. Model EV is not realized causal gain.

Use fixed iteration work; never play a partial timed profile. Timeout/error
returns blueprint and records its cause. Reproduction must retain the failure
ledger as well as full successful matrices; timing-dependent availability must
be reported separately from deterministic completed solver work. The precise
reproduction contract is a correctness gate, not assumed from fixed RNG seeds.

Persist the complete current public policy matrix, support mask and fallback
cause with a checksum keyed by blueprint/configuration, public history and
prior hero locks. Fresh replay reads this ledger, refuses missing/changed
entries and performs no solve. Native completed-work repetition is checked
separately: seeds alone do not determine whether a wall-clock timeout occurs
under a different machine load. No partial timed strategy is played.

## Fixed cost pilot

Use 24 distinct-hand river decisions: six free and six facing a wager from
each of #304's indexed archive and the blend's verified exploratory prefix.
Select the lowest public-decision SHA256 in each stratum, before timing,
without terminal outcomes or revealed opposing cards. The full frozen sample
stays in research storage; the compact declaration records its hash.

Work is 50 iterations, six threads, uncompressed full native menu, all 1,081
river-compatible holdings per player before history conditioning, zero global
opponent floor, 2 GiB native allocation ceiling, 300,000 public-node limit and
120-second search-work deadline. The long deadline measures the unweakened
design against the 3-second target; it does not change that target. A new
policy instance per spot measures cold decision work. Earlier actual hero
river actions in these blueprint journals are locked to blueprint matrices.
Include preparation, fallback, ledger, comparison-action and output costs.
The deadline rejects incomplete/late search work; fallback policy construction,
ledger writes and comparison/output work can extend total service time. They
remain in the measured latency, so a fallback cannot make this design appear
to meet the 3-second target. Per-spot timing receipts include the decision
output write and exclude only their own small timing-receipt write.

Per decision record sampled simultaneous family RSS (20 ms), tree size,
fallback cause, same-uniform-draw action change and conditional action EVs.
EV differences compare search and blueprint mixtures against the same solved
continuation; they are not guarantees of realized gain. Native normalization
can leave tiny off-path conditional EVs unavailable. Preserve that mask;
never replace it with a numeric estimate. A missing EV on a successful actual
decision blocks qualification. Native root EV and constrained best response
must match the independent reference within the predeclared .01-chip limit;
also report unrestricted best response separately from lock-constrained gains.

Research failures are retained without rerun, retuning or sample extension.
Fix and continue infrastructure failures with append-preserving evidence.
Keep the PR draft until all authorized research has actually run, unless an
explicit gate requires an owner decision. Post ETA at start, after cost pilot
and after strength declaration. Store final uncompressed evidence on bb128 and
upload its manifest-bearing ZIP with research_cache.py; record restoration
provenance in RESULTS_INDEX. Mac synced Drive folders remain read-only.

## Status

Small-fixture correctness passes on M4: native/reference EV and constrained BR
agree within .0004 chips at all three roots, including exact 1.5 times pot;
completed native matrix repeats have zero difference. A complete searched river
continuation replays with identical actions and settlement in fresh policy/player
state without invoking the solver. Expanded regressions pass 118 checks, with
one further post-review allocation-fallback check. Independent Python review
findings are addressed. [Compact evidence](reports/hu200-river-search-artifacts/)
retains the original plumbing failures, reference definitions and source pins.

The fixed 24-spot cost pilot completes: p50 15.03 seconds, p95 36.79 seconds,
maximum 39.94 seconds, peak family RSS 4.41 GB and minimum host available
memory 5.58 GB. All 24 searches complete with no fallback, timeout or missing
actual-decision EV. Memory passes; the 3-second latency target fails.
[Readout and owner options](reports/hu200-river-search.md). No strength test is
declared and no strength hands have been played. The original gate does not
qualify a search setting. The owner amendment and completed continuation below
record the later 10-second qualification. The PR remains draft pending approved
strength counts and completion. The prospective sample and work above stay fixed.

## Owner amendment after the original cost failure

On October 16 the owner selected transport/adapter engineering in this same PR.
Keep the original 3-second target and failed result unchanged. Optimize matrix
transport, startup, parsing and evidence I/O, and eliminate ancestor work only
while preserving search outputs: the same ranges, full menu, 50 iterations,
six threads and complete prior-policy locks. Preserve reusable earlier policies.
An uncertified change in resolver reuse is not a transport optimization.

Before another timed gate, prove output parity with the current implementation
on the identical 24 frozen spots. Compare final scientific request identities,
all public-node holding/action strategies after the same normalization, EVs and
their unavailable masks, prior locks, chosen actions and fresh replay. Timing,
paths, process identifiers and transport metadata are not strategy outputs.
Preserve the baseline evidence and every failure; do not overwrite old runs.

Then rerun the same sample with the amended **p95 at most 10 seconds** and
unchanged **at least 3 GB free** memory gate. The owner justifies 10 seconds as
a threefold margin within Slumbot's 30-second request timeout; this study still
does not authorize any Slumbot play. The 120-second measurement watchdog and
fixed work are unchanged. Report both old and amended gates distinctly.

If the amended gate passes, quote the strength study from measured cost and
independent-block variance, considering at most three concurrent searches only
after measured family/headroom admission. **Propose counts to the owner before
any strength hands**, regardless of whether the estimate is below 24 hours.
No strength play, root consumption or outcome inspection is authorized by this
engineering amendment. Commit the exact validation and cost declaration before
running those stages; post their ETA in the PR.

The first amended engineering run passes exact fixed 24 parity and reduces
native solves 48 to 35, but its p95 **10.766536 s** still misses10 seconds.
This failed engineering measurement remains evidence, distinct from the
original36.794921-second finding. No concurrency or strength follows that miss.
Continue the authorized parser/I/O work by sharing the two validated holding
tables and reading compact numeric payloads through a read-only mapping,
instead of repeating holding conversion/set checks and per-array file reads
at every node. No range, menu, iteration, lock or native executable changes.
Require another full fixed 24 bit-parity gate before the next same 24 timed
measurement, under the unchanged10-second/3GB thresholds. This is a mechanical
transport continuation; no poker strength outcomes have been observed or tuned.

## Completed unchanged-source continuation

The owner approved resumption after an interrupted validation with no surviving
result. Exact pre-disconnect source `214695a3` passes all 24 full-output parity
checks and the amended cost gate: p95 **9.648689 seconds**, peak search-family
RSS **2,912,206,848 bytes**, minimum host available **5,323,390,976 bytes**, no
fallbacks or missing actual EV. The original 3-second and first amended
10-second failures remain recorded. No search setting or scientific gate changes.
The clock excludes only the owner-approved downtime; historical receipts remain
unchanged. [Report](reports/hu200-river-search.md).

The [strength-count proposal](hu200-river-search-strength-proposal.md) is not an
approved final plan. No strength hands or fresh roots are consumed. Storage
admission and owner-approved counts still precede a committed final declaration.

## Owner-approved evidence retention and separate calibration

On October 17 the owner approved the [evidence-I/O and calibration protocol](hu200-river-calibration-protocol.md):
search unchanged, full ledgers/telemetry/checksums for every decision, full native
bodies for a predeclared outcome-independent sample and all failures, then
full24 parity and the same cost gate. Those gates pass at p95 9.834 s. Two/three
workers also pass output/memory checks; three are selected from measured
throughput before calibration outcomes.

The calibration is exactly 128 direct duplicate blocks plus 32 paired blocks
per each of seven scripts, fresh disjoint roots excluded from finals, with a
hard two M4 hours including loading and all verification/readout. A launch
callback plumbing repair precedes all actions and consumes the original clock.
Propose final counts from measured variance with two runtime quotes: the M4 at
its measured admitted concurrency, and cygnus-x1 only after #301 finishes within
its eight-busy-thread heat limit. Any cygnus quote before a search benchmark is
explicitly conditional. The owner chooses before any final hands.
