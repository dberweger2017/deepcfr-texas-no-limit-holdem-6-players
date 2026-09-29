# Exact-value river CFR pilot

This draft PR adds an opt-in terminal river solver. It targets a six-seat hand
with exactly two active, non-all-in players and one contested pot at the river
root. Other states use the existing corrected rollout search. It does not alter
or train a blueprint checkpoint. A complete river strategy is retained for
later on-tree decisions; a later off-tree action delegates for the rest of the
hand.

## Declared game

The core receives a public river-round root and active-seat private-card
ranges. It never receives the hero's actual holding. For controlled fixtures,
the joint law is

`Q(h0,h1) = r0(h0) r1(h1) 1[h0 and h1 are disjoint] / Z`.

Board-incompatible and zero-mass holdings cannot enter a deal. The adapter
builds active-seat marginals from observed action likelihoods under the saved
blueprint, then applies the same collision mask. This
`active-marginals-ignore-folded-removal-v1` law integrates folded cards out as
an **explicit approximation**; it is not a true posterior and does not use
actual folded or opponent cards. The range law remains fixed within a solve.

One native public betting tree uses the existing `choices` menu with two
river raises. An observed off-menu raise can be inserted at its exact size
before a solve. This is a restricted action game. Private information sets are
`(public node, acting seat, exact hole-card pair)`; the same profile covers
all compatible deals. The root excludes side pots and already-all-in active
players. Legal all-ins reached during the river are included. Terminal fold,
showdown, refunds, tied pots, and odd chips use exact public-ledger settlement,
checked against the native engine. Each seat's incremental return subtracts
half the fixed root pot, producing a zero-sum two-player reference in BB.

The solver performs simultaneous full-range Linear CFR sweeps. Regret updates
use signed, linearly weighted counterfactual regrets under the profile at the
start of the sweep. A sweep is published only after both players finish.
Current policy means the last **played** profile. Average policy uses
linearly weighted own-reach probabilities at each information set; zero
denominators yield a uniform action distribution. No extra actual-hand
traversal, clipping, regret discount, pruning, or continuation leaf is used.
The adapter defaults to average extraction; both extraction rules are measured
in the development preflight and neither is selected as a strength winner yet.

## Independent checks

The frozen [fixtures](../configs/blueprint/river-reference-fixtures.json)
contain 12 two-player and four tiny three-player six-seat river situations.
The scalar oracle enumerates compatible deals, replays native settlement,
and computes information-set best responses by summing hidden histories
**before** choosing an action. Two-player quality is half the sum of individual
best-response gains, reported in BB and root-pot units. Three-player fixtures
report each gain and NashConv; they are correctness checks, not a fast
multiway player or a convergence claim. A scalar one-sweep reference also
checks array regret increments. Tests cover all-ins, ties and odd chips,
off-menu raises, hidden-card invariance, average reach weighting, and
interrupted-sweep atomicity.

The proposed all-fixture quality gate of exploitability at most `1e-3` root
pots has **not** been adopted or claimed. A fixed work budget and extraction
rule must be chosen from development measurements before any fresh
confirmation. The current PR does not include a conditional playing-strength
comparison.

## Frozen short M4 preflight

The [preflight plan](../configs/blueprint/river-preflight-m4.json) runs all 16
fixtures without loading a blueprint. For each two-player case it records
independent and batched quality at 1, 256, 1,024, and 4,096 complete sweeps.
It then loads the immutable 12M-entry checkpoint once, constructs one full
public range on `hu-dry`, and measures setup, up to 30 seconds of solving,
both strategy qualities, process RSS, swap, and memory pressure. The overall
limit is 600 seconds, process RSS limit 10.5 GiB, and minimum free disk 30 GiB.
The script retains rows, failures and partial work, input/source/native hashes,
and artifact checksums. No paid host or additional blueprint training is used.

The preflight measures resources and reference behavior. It does not freeze or
launch the proposed 10-hour conditional campaign. A later protocol would need
fresh public roots, a chosen extraction rule and work budget, calibrated
compute-matched corrected rollout, separate evaluation deals, and uncertainty
clustered by root. No full-game BB/100 or player promotion follows from this
preflight.
