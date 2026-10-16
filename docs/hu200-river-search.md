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

Research failures are retained without rerun, retuning or sample extension.
Fix and continue infrastructure failures with append-preserving evidence.
Keep the PR draft until all authorized research has actually run, unless an
explicit gate requires an owner decision. Post ETA at start, after cost pilot
and after strength declaration. Store final uncompressed evidence on bb128 and
upload its manifest-bearing ZIP with research_cache.py; record restoration
provenance in RESULTS_INDEX. Mac synced Drive folders remain read-only.

## Status

Implementation and input preparation in progress. No strength test declared,
no strength hands played, and no search setting qualified.
