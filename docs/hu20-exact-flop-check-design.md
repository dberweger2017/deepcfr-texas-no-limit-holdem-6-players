# HU20 exact flop check: implementation and admission

This is the owner-requested diagnostic, not training or a model promotion. Work
is on `feature/hu20-exact-flop-check`; the PR remains draft and is not merged
automatically. The main-run protocol is frozen only after validation and the
three-spot cost preflight, before any Set B playing results are inspected.

## External solver boundary

The AGPL-3.0-or-later solver and its Rust harness live outside this MIT repository
in `~/Local/hu20-exact-flop-tool`. Upstream is pinned to
`9d1509fe5077d019825f833eed04b16d342dfda1`. Only independently authored Python
export/report/monitor code, documentation, and compact JSON evidence belong in
this repository. Requests and responses cross the boundary through files.
Retain the external harness source, Cargo.lock, source hashes and binary hash in
the private artifact inventory; do not vendor it into this repository.

Source inspection confirms `ActionTree::add_line/remove_line`,
`PostFlopGame::memory_usage/allocate_memory`, `solve_step`,
`compute_exploitability`, `lock_current_strategy`, `compute_mes_ev`, and
`compute_current_ev`. MES respects the responding player's own locks: unlock
that player when measuring a best response. Both EV APIs subtract half the root
pot; native validation uses the same centered payoff. For equal-investment HU20
roots it equals net chips from the original stack, conditional on that root.

Bet syntax is not the native action menu. Export the native `choices()` tree,
including exact street raise-to amounts and conditional jams, then add/remove
lines and compare every action at every public node. Solver seat 0 is OOP (the
big blind); physical seats must be mapped explicitly.

## Gates and resources

K requires at least 100,000 full-key/factored-key comparisons across spot types,
streets, runouts and holdings, with no mismatch. V1 checks the complete declared
tree, V2 native terminal payoffs, V3 river information-set best responses, V4
both-locked EV against at least 20,000 independent native deals, and V5 records
the equilibrium residual for each solved spot. Failed or missing gates prevent
main-run admission; excluded results remain visible.

Before M4 work, retain processes/RSS, memory pressure and swap. Use one nice'd
solver, idle-core Rayon allocation, a measured memory budget no greater than
10 GiB, pre-allocation memory estimates, an aggregate-process watchdog and a
stop on swap growth above 1 GiB. Preserve atomic per-spot results and every
attempt. Existing jobs and artifacts are not modified.

Memory fallbacks are compression, removal of zero-weight holdings, then a
declared per-street raise cap of at least three retaining legal jams. A capped
spot cannot enter the decision rule without measuring removed blueprint and
equilibrium target reach and showing at most 2% removed target mass. Oversize
spots stop for a separately owner-approved paid-compute quote.

## Interpretation

All values are conditional on stated root ranges. A projected equilibrium is a
feasible abstract policy and its loss upper-bounds the minimum loss achievable
by that abstraction. A high projection ratio is therefore a heuristic H1
classification, not a proof of an abstraction lower bound. A low ratio provides
a constructive witness that the abstraction can do better on the declared
subgame. Per-flop equity buckets favor the diagnostic over a global abstraction.

The call/pot screen is not an equilibrium fold target. Report range-wide folds
and holding-level discrepancies without interpreting selected hands as an MDF
violation. LBR accounting is descriptive: root ranges, reaching a flop, and
different root occupancies prevent causal whole-hand attribution.

## Requested sequence

1. Exporter, external harness, monitoring and fixture gates.
2. Three real spot preflight, first B500M current lineage: limp, minimum open,
   minimum open/minimum three-bet. Retain both memory modes and all failures.
3. Freeze policies, outcome-blind Set A/Set B selection, ordering, weights,
   thresholds and costs in `docs/hu20-exact-flop-check-protocol.md`.
4. Admit the main run only if all gates pass and projected M4 work fits the
   owner's approximately 24-hour ceiling; otherwise propose a smaller Set B.
5. Complete the report and roadmap entry, with no promotion or automatic merge.

## Owner-requested turn follow-up

The [turn-only extension](hu20-exact-turn-check-design.md) preserves this flop
experiment and its resource failure. It introduces a separately validated and
frozen turn corpus, and a turn-only cap-two fallback after cap three. No turn
result answers the flop-root question.
