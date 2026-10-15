# Native N-player traversal and known-game gate

[PR289](https://github.com/dberweger2017/CFR-NLHE-6max/pull/289), October 15, 2026.
[Predeclared protocol](../nplayer-traversal-protocol.md). This follows the owner's
independent review and merge of [PR288](https://github.com/dberweger2017/CFR-NLHE-6max/pull/288).

**Three-player Kuhn passes; Leduc passes the separately approved fixed
confirmation after failing its first endpoint.** Both games use the same
external-sampling traversal as the N-player cash adapter. Native exact values
and per-player best responses agree with independent OpenSpiel calculations.
These are finite-game correctness and approximation results, not a multiplayer
CFR equilibrium guarantee or an NLHE strength claim. Independent review of
traversal, information access and regret math is still required before size
measurement or the later bounded cash-game smoke.

The first fixed study used `1578d66fe7e47bed1d72ee34651893d76d625824`.
The final implementation and confirmation use
`7900a8019ed6b868891c0b733adfa0da768f1a1e`, native tree
`e3f8ad4e4190fefbdde4f7cdeb68f81e848a0fa0`. After the owner merged PR288,
rebasing onto main produced `f4bc8d2b689060ce56f4fedd6fcb4b375aa19a78`
with that **identical native tree**. Reports retain original run source pins.

## Traversal, information and recovery

`multi_cfr.rs` branches on every traverser action and samples each other
player's current policy. Every complete iteration sweeps all N seats, each
with an independent deal, before applying the accumulated updates. Regret and
average increments have linear iteration weight. Domain-separated root RNG
streams depend only on seed, complete iteration and traverser, so recovery
needs no opaque RNG state. Each branch restores the simulator state.

The requested opponent-sampled average accumulates at non-traverser decisions.
For N>2 its visitation includes other sampled players' reach; it is **not an
unbiased own-reach average**. This limitation is explicit in
[OpenSpiel's external-sampling implementation](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/external_sampling_mccfr.py).
Acceptance evaluates the actual stored average, not a theoretical guarantee.
The street limit omits average arrays beyond the chosen street, including
opponent-only visits. Frozen mode finishes its boundary iteration, fixes key
membership, continues existing-key updates and counts discarded updates.

The cash adapter constructs keys and actions only from the reviewed immutable
observation. Keys include N, button-relative seat, live/all-in masks, blinds,
relative starting/current stacks and contributions, current card class,
and ordered public actions with actor, street and exact amounts. Rotation
preserves key identity; the table owner is relative to the button while the
traverser remains the actual seat. Public street markers do not include raw
board cards that would undo the card abstraction. Preflop classes and #163's
current-street K50 buckets are unchanged; previous card classes are not added,
so this retains the existing card abstraction's recall limitations.

The menu follows HU v1: fold/check/call, minimum raise, pot and its conditional
all-in, with clamping and duplicate removal. As in HU v1, an all-in larger than
twice the pot after calling is omitted unless it is the minimum raise. Blinds,
stack depth and unequal stacks are parameters. No multiway card abstraction
or larger menu is selected here.

N-player checkpoints use `nplayer-cfr-v1`, with N, complete game identity,
seed, iteration, averaging scope, frozen state, telemetry and sorted rows.
Writing is atomic; loading validates metadata, action dimensions, finite
accumulators, scope and telemetry consistency. Legacy HU formats and entry
points remain unchanged. The experimental cash CLI requires explicit node,
entry and wall-time bounds, pinned K50 inputs and a new output path. Resume
requires the expected SHA256 and matching configuration. Recovery includes
full compressed-byte identity, not just equivalent decoded values.

## Known-game rules and exact evaluation

Three-player Kuhn follows [Szafron, Gibson and Sturtevant (2013)](https://poker.cs.ualberta.ca/publications/AAMAS13-3pkuhn.pdf):
four ranks, one-card private hands, one-chip antes and one possible one-chip
bet. Published Tables 2–3 supply an analytical equilibrium subfamily with
beta in [0,1/4] and utilities
`[-(0.5+beta)/24, -0.5/24, (1+beta)/24]`. Native values and best responses
match analytical profiles at beta 0, 1/8 and 1/4 to 1e-12. The final trained
value vectors are also compared with this family; matching values alone is
insufficient, so every player's BR gain has a separate gate.

Leduc uses [OpenSpiel's multiplayer rules](https://github.com/google-deepmind/open_spiel/tree/48401890ee9857e611678302371378175a8e4c6b/open_spiel/games/leduc_poker):
for three players, eight physical cards comprising two of each of four ranks,
one private card each and one public card between two betting rounds. Antes
are one chip, raises are 2/4 chips, with at most two bets/raises per round.
Tied winners split fractionally; integer odd-chip rules apply to cash NLHE,
not these toy-game utilities. This is distinct from six-card heads-up Leduc.

The exact evaluator enumerates chance and the complete legal tree: Kuhn has
600 nodes /48 information sets; Leduc has 228,480 nodes /7,428 information
sets. A best response chooses one action per information set after aggregating
hidden histories using opponents' and chance reach. It does not choose a
different action for each hidden deal. An independent Python bridge maps
public OpenSpiel information strings to exported native keys, verifies complete
information-set coverage, and uses OpenSpiel **2.0.2** C++ expected returns and
tabular best response. Uniform, published analytical and all trained endpoint
profiles agree in values and best responses to 1e-10.

## Fixed outcomes

Gains below are exact `BR_i(average_-i) - value_i(average)`, in chips/hand.
All entries are final stored averages; no earlier iterate or seed was selected.

| Game / study | Seed | Iterations | Seat 0 gain | Seat 1 gain | Seat 2 gain | Limit | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Kuhn initial | 2026101501 | 2,000,000 | 0.000374 | 0.000861 | 0.000347 | 0.005 | pass |
| Kuhn initial | 2026101502 | 2,000,000 | 0.000348 | 0.002171 | 0.000460 | 0.005 | pass |
| Kuhn initial | 2026101503 | 2,000,000 | 0.000815 | 0.001120 | 0.003183 | 0.005 | pass |
| Leduc initial | 2026101501 | 500,000 | 0.031265 | 0.030870 | 0.042425 | 0.025 | **fail** |
| Leduc initial | 2026101502 | 500,000 | 0.032293 | 0.033122 | 0.044560 | 0.025 | **fail** |
| Leduc initial | 2026101503 | 500,000 | 0.031817 | 0.030432 | 0.030623 | 0.025 | **fail** |
| Leduc confirmation | 2026101511 | 2,000,000 | 0.019309 | 0.018398 | 0.021380 | 0.025 | pass |
| Leduc confirmation | 2026101512 | 2,000,000 | 0.014116 | 0.013450 | 0.021521 | 0.025 | pass |
| Leduc confirmation | 2026101513 | 2,000,000 | 0.019345 | 0.014259 | 0.016177 | 0.025 | pass |

Kuhn's maximum utility errors against the published family are 0.0000275,
0.0000880 and 0.0005901 chips/player, below the predeclared 0.005 limit.
[Initial full-precision results](nplayer-traversal-artifacts/known-results--results.json).

The initial six-endpoint job failed its declared quality gate. No threshold
was changed and no automatic continuation was launched. The owner explicitly
approved the separate confirmation with three fresh seeds, a fixed 2M endpoint
and unchanged 0.025 limit, quoted at approximately three minutes with a
600-second cap. It finished in **182.97 seconds**, including build, independent
reference checks and split/resume verification. All three seeds pass.
[Confirmation results](nplayer-traversal-artifacts/leduc-confirmation-results--results.json),
[admission/source/binary pins](nplayer-traversal-artifacts/leduc-confirmation-results--admission.json).

The first seed of each study/game also runs half its target, resumes and
matches both complete checkpoint and average-export bytes against fresh.
The confirmation's fresh and resumed Leduc checkpoint SHA256 is
`5c8e31ae69094f822e9d420afc4d322c3c922100e61b1740dcbc3aa04acc1ed7`.
[Recovery receipt](nplayer-traversal-artifacts/leduc-confirmation-results--leduc-recovery.json).

A post-study sampling edge-case fix ensures a rounded cumulative probability
cannot fall back to a zero-probability last action. A deterministic maximal-RNG
unit test covers it. Reproduction of the original six fixed endpoints under
the final core is a byte-compatibility check, retaining the original failed
Leduc decisions; it is not a new quality confirmation.

## Tests and heads-up preservation

Final Rust checks pass **35 library and 5 CLI tests**. They include:

- An exhaustive expectation check over every three-player Kuhn deal, all
  three traversers and 256 equally probable action-sampling tapes per deal.
  Every expected regret and opponent-average increment matches an independent
  complete-tree calculation; parent state is restored.
- Published analytical Kuhn profiles and an exploitable uniform control;
  hidden-card/future-board invariance; 2–6-player legal play and conservation.
- Full and preflop-only averages, fresh/resumed bytes, frozen membership,
  malformed checkpoint rejection, button rotation and actor-labelled keys.

The 16 new Python cash-CLI checks also pass, including all N=2–6, pinned K50,
20-BB fresh/resume, scoped averages, frozen membership, capacity stops and
configuration-mismatch rejection. These are at most 20,000-node targets and
are correctness checks, not the later Phase 0 sampler or smoke campaign.

Final legacy suite and HU identity receipts are being completed before review.
Existing HU game/key/traversal/checkpoint source files are unchanged; `lib.rs`
only exports the added modules. No legacy format or released model is replaced.

## Resources, failures and remaining work

All load runs on idle bb128 under its on-host **85°C** watchdog, stricter than
the owner's 90°C ceiling, with at most eight assigned CPUs, four build workers,
8-GiB family RSS and 32-GiB disk floor. The confirmation used 445.12 MiB peak
sampled family RSS. M1 performs edits and light checks; M4 is not accessed.
[Commands, guards and log hashes](nplayer-traversal-artifacts/guards.json).

Preserved setup failures include a missing Cargo PATH before compilation and
two incorrect OpenSpiel Python API probes. Corrected probes use its C++
uniform-policy table and tabular-best-response interface. The scientific
failure is the original Leduc quality gate; its endpoints, logs and checkpoints
remain intact. No rule, numerical or reference mismatch was suppressed.

Independent traversal/information/regret review remains the next dependency
gate. Size-growth measurements, the Phase 2 configuration recommendation,
the bounded cash smoke, and N-player inference export/loader remain later work.
No search, release or new NLHE playing-strength claim is made. Protected raw
evidence and shared inputs are indexed in [RESULTS_INDEX](../../RESULTS_INDEX.md);
cloud archival closeout is pending and no originals have been removed.
