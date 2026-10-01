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
The run completes **1,536/1,536 hands** in **1,595.77 seconds (26.60 minutes)**,
peak **2.706 GiB**, with one verified load per lineage (10.72/10.07/10.23 seconds).
All 139 fresh solves complete 250 sweeps; no watchdog failures, delegations or
additional hands. The process exits after writing closed evidence.

Raw strategy `current` means direct B500M current; `average` means **that same
current blueprint plus river solver average**, not #141 stored CFR-average.
[Closed summary](hu20-river-artifacts/comparison/summary.json),
[raw lineage 1](hu20-river-artifacts/comparison/B-2026093001-500000000.hands.jsonl.gz),
[lineage 2](hu20-river-artifacts/comparison/B-2026093002-500000000.hands.jsonl.gz),
[lineage 3](hu20-river-artifacts/comparison/B-2026093003-500000000.hands.jsonl.gz)
and [checksums](hu20-river-artifacts/comparison/manifest.json) preserve every
observation, menu/probability, exact action, search record and native payoff.

## Playing signal: mixed and inconclusive

Each cell has 96 hands/arm (three fixed lineages × 16 blocks × two positions).
The delta averages both positions and the three lineages **within each of 16
independent paired deal blocks**, rather than treating 96 hands as independent.
Intervals below are explicitly exploratory, unadjusted Student-t descriptions;
all span zero. The arena's existing inference minimum is unmet. No playing
strength improvement is established.

| Panel | Direct BB/100 | River search BB/100 | Paired change BB/100 | Exploratory 95% t interval |
| --- | ---: | ---: | ---: | --- |
| uniform | 445.83 | 403.12 | -42.71 | [-112.86, 27.44] |
| tight_passive | 71.88 | 73.96 | +2.08 | [-2.36, 6.52] |
| loose_passive | -15.62 | 3.12 | +18.75 | [-45.69, 83.19] |
| tight_aggressive | 80.21 | 68.75 | -11.46 | [-35.88, 12.96] |
| loose_aggressive | -15.10 | -14.06 | +1.04 | [-73.15, 75.24] |
| pot_pressure | -106.25 | -47.92 | +58.33 | [-63.65, 180.32] |
| train_pressure | -11.98 | 15.10 | +27.08 | [-32.17, 86.34] |
| selective-stackoff | 57.81 | 51.56 | -6.25 | [-27.48, 14.98] |

The point estimates suggest a lead under pot/train pressure and a decline
against uniform; neither is reliable from this sample. Selective-stackoff
falls slightly. Do not pool panels or pick the best-looking opponent as a
strength result. Point changes by lineage (seed suffixes):

| Panel | 3001 | 3002 | 3003 |
| --- | ---: | ---: | ---: |
| uniform | -90.62 | +6.25 | -43.75 |
| tight_passive | +0.00 | +0.00 | +6.25 |
| loose_passive | +0.00 | +31.25 | +25.00 |
| tight_aggressive | +0.00 | +0.00 | -34.38 |
| loose_aggressive | +3.12 | +40.62 | -40.62 |
| pot_pressure | +59.38 | +56.25 | +59.38 |
| train_pressure | +6.25 | +21.88 | +53.12 |
| selective-stackoff | -21.88 | +3.12 | +0.00 |

[Every seed/position level](hu20-river-artifacts/derived/panels.csv),
[paired change and interval](hu20-river-artifacts/derived/paired-changes.csv),
[three-lineage block changes](hu20-river-artifacts/derived/aggregate.json)
retain all contrasts. Zero observed variation receives no exploratory interval,
not a claim of certainty. This is a fresh schedule, not #136's broad campaign.

## Tails, intervention denominators and cost

Counts below are direct/search, with 96 hands/arm/panel. Large wager is the
existing diagnostic threshold (the rival would face a call of at least 800
chips / 8 BB after the raise); opportunities
are target decision opportunities, not hands. Fold/continue counts follow such
raises. Full-stack wins **and** losses refer to terminal ±20 BB. Counts can
contain multiple opportunities/actions per hand.

| Panel | Large opportunities | Large raises | Rival folds | Rival continues | Full-stack wins | Full-stack losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 57/57 | 22/22 | 16/15 | 6/7 | 15/15 | 1/3 |
| tight_passive | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| loose_passive | 12/12 | 2/4 | 0/0 | 2/4 | 0/2 | 1/1 |
| tight_aggressive | 3/3 | 1/1 | 1/1 | 0/0 | 1/0 | 0/0 |
| loose_aggressive | 25/23 | 5/9 | 1/1 | 4/8 | 2/4 | 3/3 |
| pot_pressure | 19/20 | 11/9 | 3/3 | 8/6 | 0/0 | 8/5 |
| train_pressure | 13/12 | 5/3 | 1/1 | 4/2 | 1/0 | 3/1 |
| selective-stackoff | 6/6 | 2/1 | 0/0 | 2/1 | 0/0 | 0/0 |

The raw summary also retains whole-hand return partitions by the rival's
response to the first large raise (fold, continue or no large raise). These are outcome-conditioned
**whole-hand returns**, not individual-bet EV or causal error scores. Search's
21 full-stack wins/13 losses versus direct's 19/16 are descriptive totals over
different panels; they do not establish an overall performance improvement.

| Panel | River intervention hands / 96 | River actions | Fresh solves | Cache hits | Later re-solves | Search seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 11 | 14 | 12 | 0 | 1 | 104.71 |
| tight_passive | 5 | 5 | 5 | 0 | 0 | 55.51 |
| loose_passive | 47 | 47 | 44 | 3 | 0 | 502.40 |
| tight_aggressive | 3 | 3 | 2 | 1 | 0 | 22.37 |
| loose_aggressive | 26 | 31 | 25 | 1 | 0 | 269.97 |
| pot_pressure | 5 | 8 | 8 | 0 | 3 | 103.76 |
| train_pressure | 25 | 28 | 23 | 5 | 3 | 263.06 |
| selective-stackoff | 22 | 25 | 20 | 2 | 0 | 238.91 |

Across 768 search hands, **144 (18.75%)** need river action, producing **161
river decisions**, 151 solve requests, **139 fresh solves / 12 cache hits**,
and **7 later off-tree re-solves**. There are 34,750 completed CFR sweeps.
The per-lineage bounded caches also record 24 range hits and 171 retained
profile queries; queries include the runner's distribution/choice queries and
are not decision counts. Eviction totals combine range/profile stores.

Fresh solve+setup latency is **6.18 / 11.82 / 16.58 seconds min/median/max**.
Search calls total 1,560.68 seconds; hand execution totals 1,561.45 seconds
for search versus 0.67 for direct, excluding policy loads and final reporting.
This is substantial latency for a future live table. The modest cache hit rate
on fresh deals is preserved; a whole solution still serves all holdings at
an identical root. The earlier three-root quality curve cannot certify the
quality of every playing or constrained re-solve root.

## Coverage and exact later off-tree handling

All 1,081 holdings per seat are retained in each range. Positive support can
be much smaller (minimum observed is **5 holdings** in one seat), because
on-menu zero blueprint likelihood remains zero. Across the 151 request records,
range counters total 1,085,317 trained / 4,331 missing queries, with 5,405
off-menu likelihood queries. Repeated/cache request counters can refer to the
same range construction; these are not unique independent computations.
Missing keys use the existing uniform fallback and off-menu likelihoods use
the declared existing kernel. This law's calibration remains unresolved.

Direct has 20/1,281 missing target lookups; search trajectories have 23/1,281
counterfactual blueprint lookups marked missing, but only 17/1,120 earlier-street
actions actually use that fallback. The other 161 target actions use river
search; no river delegation occurs. The blueprint lookup recorded alongside
search is exposure telemetry, not its action distribution.

All seven later re-solves are listed, including losses. Raise-to values below
are the exact observed **river** targets inserted in the tree; earlier sizes
remain in public history. Block/rotation locate the raw hand. Each re-solve
freezes one previously used full hero matrix, not only the actual hand row.

| Seed suffix | Panel | Block / rotation | Inserted river raise-to chips | Frozen hero nodes | Whole-hand BB |
| --- | --- | --- | --- | ---: | ---: |
| 3001 | pot_pressure | 11 / 0 | 300 | 1 | -1.0 |
| 3001 | train_pressure | 6 / 0 | 100, 500 | 1 | +8.0 |
| 3001 | train_pressure | 14 / 1 | 800 | 1 | -8.0 |
| 3002 | pot_pressure | 11 / 0 | 900 | 1 | -3.0 |
| 3003 | uniform | 0 / 0 | 400, 1600, 1800 | 1 | -20.0 |
| 3003 | pot_pressure | 11 / 0 | 300 | 1 | -1.0 |
| 3003 | train_pressure | 14 / 1 | 800 | 1 | -8.0 |

The native replay reproduces these exact wagers. A realized hand loss or win
is not expected action value. Generated fixtures independently cover check→bet
and hero bet→opponent off-menu raise, exact holding selection, unchanged prior
matrices and reuse of a solved profile across different hidden worlds.

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
with unchanged six-seat defaults. **61 staged focused/existing tests pass**,
and both full CI shards plus the
required test gate are green at `a004653`. The model-free audit independently
reconstructs all **1,536** native hands, verifies **2,562 target observations**,
exact actions/payoffs/event digests/tail arithmetic, all frozen coordinates and
**768 identical earlier-street pairs**. It loads no policy; 1.32 seconds,
150.2 MiB process peak. [Audit](hu20-river-artifacts/derived/audit.json) and
[derived checksums](hu20-river-artifacts/derived/manifest.json) are retained.
Actual-model timing/playing evidence is separate from generated fixture checks.
No UI or cross-architecture identity claim is made.

Environment: **M1, macOS 27.2 arm64, Python 3.11.15**, NumPy 1.26.4, SciPy
1.17.1; one-thread BLAS environment. The installed native engine's direct VCS
revision is verified as `5db20e3d5d6862b32a7402035c1340b622d3b005`.
[Environment and runtime file hashes](hu20-river-artifacts/environment.json)
identify the unchanged playing files at `845bd458b6e32a22bc6bb17520eafd73b66a104c`;
subsequent commits add tests, reporting and documentation, without changing
those runtime bytes. Input lengths, hashes, schema/extraction and lineage are
in the frozen plans and loaded descriptions. No private session/model binary,
M4 work, training, LBR, paid compute or promotion is included.

Future questions remain range calibration, residual solve error at the chosen
sweep count, broader menus and the cost of later re-solving. Safe/nested
guarantees and turn search are separate work. Nothing here promotes B500M or
claims human strength.
