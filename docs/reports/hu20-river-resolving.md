# HU20 B500M fixed-sweep river re-solving

The [protocol](../hu20-river-resolving-protocol.md) and timing-only setting
selection precede playing outcomes. This report tests using exact holdings
at the river with B500M current on earlier streets; it does not change the
policy, card abstraction or #108's six-seat defaults. The
[adapter guide](../hu20-river-resolving.md) defines ranges, cache identities,
prior-action consistency, failure behavior and reproduction commands.

## Completed cost and conditional quality curve

First B500M lineage by fixed order, one hash-verified load; three predeclared
roots with exact pot sizes 2/10/26 BB. Both players have all 1,081 board-compatible
holdings. The later two forced flop raises are off the blueprint menu and
use the declared existing likelihood kernel; each records 1,081 such queries.
These likelihoods are modeled, not observed opponent posteriors.

| Root pot | 250 / 500 / 1,000 / 2,000 sweep exploitability (root-pot fraction) | 250-sweep seconds | 2,000-sweep cumulative seconds |
| --- | --- | ---: | ---: |
| 2 BB | .02252 / .01646 / .00852 / .00451 | 9.85 | 80.81 |
| 10 BB | .02275 / .01382 / .00809 / .00491 | 11.40 | 92.80 |
| 26 BB | .000693 / .000313 / .000159 / .000068 | 6.81 | 54.23 |

Setup is 0.63 seconds/root. Total curve including one model load and quality
checks is **240.48 seconds**, peak **2.597 GiB**. All 12 milestones complete;
zero-sum quality errors stay below 6e−15 BB. Quality is information-set best
response within the specified river menu and range law; it is not global
blueprint exploitability. No exact-equilibrium threshold is passed/claimed.
[Raw curve and identities](hu20-river-artifacts/timing/curve.jsonl),
[summary](hu20-river-artifacts/timing/summary.json) and
[checksums](hu20-river-artifacts/timing/manifest.json) retain every milestone.
Curve source: `a406689` (full identity in summary).

## Frozen playing comparison

I selected **250 sweeps** for the first cost/signal comparison. The
[plan](../../configs/diagnostics/hu20-river-comparison.json) fixes three original
B500M current policies, direct versus river average re-solving, 16 fresh
paired blocks and both positions on each of eight panels: uniform, six native
styles, selective-stackoff. Total **1,536 hands**, 2,000 chips/seat reset per
hand, 100 chips/BB, no rake/ante; root `202610010501` is separate from timing
and earlier reports. The whole M1 watchdog is 2,700 seconds and 6 GiB.
No bounded LBR, training, paid compute, M4 access or changes to #136.

Playing source: `845bd45` (complete source identity captured by runner).
Results/replay/resource tables are pending while the frozen worker runs.

## Interpretation and validation

Both positions and the three fixed lineages are averaged within each deal
block. Panels remain separate. Sixteen independent blocks are below the
arena's existing 30-block inference minimum: that gate remains unchanged and
reports `too_few_blocks`. The separate report adds explicitly exploratory
Student-t intervals, with small-sample/long-tail limitations; they cannot
support model promotion or a strength-qualified release. Whole-hand return
partitions are not individual-action EV. Net profits accumulate despite
displayed stacks resetting each hand.

Generated small-policy/tiny-range tests cover native/scalar quality and
settlement, deterministic sweeps, public ranges/card compatibility, hidden
worlds, exact holdings, full-profile cache reuse, on-tree retention and both
check→bet and bet→raise later off-tree re-solving. Prior hero matrices stay
fixed across all hypothetical holdings. Existing #108 river checks are run
with unchanged six-seat defaults. Actual-model timing is separate from these
fixture checks. No UI or cross-architecture identity claim is made.

Future questions remain range calibration, residual solve error at the chosen
sweep count, broader menus and the cost of later re-solving. Safe/nested
guarantees and turn search are separate work. Nothing here promotes B500M or
claims human strength.
