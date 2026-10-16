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
declared, no strength hands have been played, and no search setting is qualified.
The PR remains draft pending the owner's response to the cost gate. This
status update does not change the prospective sample or work declaration above.
