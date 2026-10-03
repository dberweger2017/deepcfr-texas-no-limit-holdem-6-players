# HU20 exact turn check: completed M4 campaign

The frozen campaign completed all **288 spot-policy jobs** on 2026-10-03 at
00:58 UTC (02:58 Madrid): **284 exact solves and four literal zero-policy-support
exclusions**. All completed V1/V5 gates pass, and every solve meets the
≤0.2%-of-pot convergence target. All **32 Set B roots** qualify across all six
exports, four in each of the eight frozen strata. Set A retains 13 of 16 roots.

The predeclared pooled rule gives **H2-turn: a feasible-v1 witness to substantially
better turn/river play**. Blueprint BR loss is **3.1958 BB [2.9073, 3.5148]**;
the feasible full-v1 projection loses **0.6951 BB [0.6069, 0.7902]**.
**R = 0.2175 [0.1917, 0.2466]**, below the frozen 0.3 threshold.
Signed within-root alias cost is **0.0039 [0.0011, 0.0081]** of blueprint loss.
These are bootstrap 95% intervals over independent roots, not over individual
hands, nodes, target seats or policy exports.

This supports proposing an audit of the native trainer, regret/average updates
and useful key visitation before another isolated-turn abstraction intervention.
It does **not** show that longer v1 training will fix the plateau. Each projected
strategy can depend on its exact turn root; v1 pools boards across roots, so the
experiment does not certify one equally good global v1 blueprint. A cross-root
consistency audit is a separate proposed experiment. No follow-on is launched.
**Turn/river results cannot answer the original flop question or inherited
preflop/flop aliases across different roots.**

Draft [PR #145](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/145)
contains this diagnostic; the separate history-alias audit is complete in draft
[PR #146](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/146).
No training, promotion, rental or automatic merge occurred.

## Frozen design and estimand

The [protocol](../hu20-exact-turn-check-protocol.md) and original config were
pushed in `31030f4270b92f6780937edd90282e058b9a05b4`, before any main-root values.
The [owner-authorized resumption](../hu20-exact-turn-check-resume.md) and
[resource-only readmission](../hu20-exact-turn-check-resource-readmission.md)
preserve roots, six exports, order, native menus, thresholds and cumulative clock.
The final [configuration](../../configs/diagnostics/hu20-exact-turn-check-resource-readmission.json)
uses runtime source `dd89c1cc0e9388df845b7ca074c3742f9d6a3cd0`.

Set A selects 16 roots, two per pot-type × physical-button stratum, from 67
unique turn roots behind 74 LBR-facing decisions in #143's pinned hand archives.
The selected sample contains 17 decisions. It retains all 74 source references
and empirical preflop range groups. Set B selects 32 unselected roots, four per
stratum, from 1,088 live turn roots in 3,000 fresh first-lineage current-policy
self-play deals; 1,912 hands terminate earlier. Sampling stops before the turn
action or outcome. The three fixed cost pilots are excluded from both sets.

Recorded seeds: self-play deal/action 202610020201/202610020202; Set B selection
202610020203; Set A selection 202610020205; interleaved job order 202610020206;
paired root bootstrap 202610020207 (2,000 draws); selected-fold bootstrap
202610020208. Selection was never adapted to losses, convergence or support.

Current and stored CFR-average B500M exports from lineages 2026093001/2/3 each
receive the same 48 roots. Their own `public_ranges` condition on **preflop and
flop** policy likelihoods; exact zeros remain zero and missing-key behavior is
recorded. The betting tree comes from native `choices()` replay, using integer
chips (100 chips = 1 BB), rather than approximating its menu with solver syntax.
Every admitted solve uses the full native compressed tree; no cap or removed
line is admitted. The declared native → cap 3 → cap 2 ladder would require
blueprint/equilibrium removed-reach auditing; no such fallback was needed.

For each target seat, `compute_mes_ev` gives the respondent's best response to
the locked target. Loss is that value minus the respondent's exact-card
equilibrium value from the same primary root ranges. Both-blueprint locked EV,
per-seat values and per-node folds are retained in every complete atomic result.
Full v1 projects by the real information key using own range × own equilibrium
action reach, pooling lines and river runouts **within this turn root**. Per-line
v1 pools river runouts separately at each public line. Equity K=50/200 uses
20-bin exact uniform-opponent equity histograms on the turn, cumulative-L1/EMD
assignment with mean centroids (seeds 202610020254/202610020404), and pooled
equity quantiles on the river. This convention does not claim Euclidean k-means
optimizes EMD, or that these favourable per-root buckets are a blueprint design.

Primary summaries use the same common eligible six-export root intersection
for every lineage, extraction and target seat. Average exports/seats at a root,
then reach-weight roots by multiplicity / stratum inclusion probability.
Bootstrap independent roots within strata, paired across all exports and seats.
R and alias cost are **ratios of pooled means**, with paired ratio intervals.
Percentage-of-pot summaries average each root's normalized loss; they are not
the BB mean divided by a mean pot. Export comparisons condition on different
policy-derived ranges and therefore are not pure strength comparisons under a
single fixed range law. Set B's public-root population comes from the first
current lineage, not a fresh occupancy sample for every export.

## Coverage and exclusions

| Set / pot type | Button 0 eligible / selected | Button 1 eligible / selected |
| --- | --- | --- |
| A / limped | 2 / 2 | 1 / 2 |
| A / min-raised | 2 / 2 | 1 / 2 |
| A / pot-raised | 2 / 2 | 1 / 2 |
| A / 3-bet | 2 / 2 | 2 / 2 |
| B / limped | 4 / 4 | 4 / 4 |
| B / min-raised | 4 / 4 | 4 / 4 |
| B / pot-raised | 4 / 4 | 4 / 4 |
| B / 3-bet | 4 / 4 | 4 / 4 |

All 48 frozen roots have six recorded outcomes, but exclusions are not solved
profiles. Four jobs with zero policy likelihood affect three Set A roots.
The other 14 successful profiles on those roots remain individual evidence and
are excluded from all common-root summaries. Set B has no excluded root and
meets the frozen ≥16-root / ≥2-in-every-stratum decision gate.

| Excluded root (full identifier) | Pot / button | Export with zero support |
| --- | --- | --- |
| 4a9ae0577e469cff4129893b9fa30cfd7a6d5a1a125430c3df2d5be86c6bf1e6 | min-raised / 1 | B-2026093003-500000000 |
| 9b1d2799342bf293c85cd979800cd38ee6f089fdd4ba80912222bd255d5dbbdd | limped / 1 | B-2026093001-500000000 |
| 9b1d2799342bf293c85cd979800cd38ee6f089fdd4ba80912222bd255d5dbbdd | limped / 1 | B-2026093002-500000000 |
| d869dfd2a7c82e0ac2e9b6da50950768abfb9cf2f5227ceffcc4d6b333850986 | pot-raised / 1 | B-2026093003-500000000 |

No completed job failed a convergence gate, no native turn root was oversize,
and no failed attempt's partial values enter the estimates. The original flop
roots remain oversize: cap-three compressed estimates were **97.64 / 56.86 /
25.86 GiB** for limped/min-raised/3-bet, versus measured 6/5-GiB allowances;
native arenas already exceeded admission. The original
[flop report](hu20-exact-flop-check.md) and its failure evidence remain unchanged.

## Set B: primary BB values

Reach-weighted BB [bootstrap 95% interval], all 32 common roots.

| Policy group / target | Roots | e_bp | e_v1proj full | e_v1proj line | e_eq50 | e_eq200 |
| --- | --- | --- | --- | --- | --- | --- |
| pooled | 32 | 3.1958 [2.9073, 3.5148] | 0.6951 [0.6069, 0.7902] | 0.6826 [0.5891, 0.7846] | 0.3205 [0.2983, 0.3455] | 0.2558 [0.2376, 0.2759] |
| 2026093001 | 32 | 3.3191 [3.0020, 3.6650] | 0.7180 [0.6174, 0.8238] | 0.7057 [0.5976, 0.8188] | 0.3288 [0.2912, 0.3758] | 0.2680 [0.2366, 0.3077] |
| 2026093002 | 32 | 3.1012 [2.8057, 3.4362] | 0.6986 [0.6075, 0.8061] | 0.6853 [0.5884, 0.8005] | 0.3220 [0.2955, 0.3491] | 0.2486 [0.2275, 0.2691] |
| 2026093003 | 32 | 3.1672 [2.8571, 3.4981] | 0.6687 [0.5823, 0.7682] | 0.6568 [0.5627, 0.7634] | 0.3108 [0.2898, 0.3319] | 0.2508 [0.2348, 0.2674] |
| current | 32 | 4.3235 [3.9499, 4.7334] | 0.7095 [0.6147, 0.8090] | 0.6972 [0.5971, 0.8014] | 0.3473 [0.3131, 0.3894] | 0.2734 [0.2465, 0.3062] |
| stored-average | 32 | 2.0681 [1.8343, 2.3260] | 0.6807 [0.5937, 0.7754] | 0.6680 [0.5746, 0.7682] | 0.2938 [0.2756, 0.3139] | 0.2382 [0.2230, 0.2536] |
| OOP | 32 | 3.0542 [2.7423, 3.4125] | 0.6956 [0.5966, 0.8051] | 0.6854 [0.5819, 0.7989] | 0.3378 [0.3093, 0.3708] | 0.2737 [0.2503, 0.3006] |
| IP | 32 | 3.3374 [3.0392, 3.6654] | 0.6946 [0.6042, 0.7921] | 0.6798 [0.5820, 0.7812] | 0.3033 [0.2835, 0.3248] | 0.2378 [0.2218, 0.2539] |

## Set B: primary percentages of root pot

Percent of root pot [bootstrap 95% interval], the same 32 common roots.

| Policy group / target | e_bp | e_v1proj full | e_v1proj line | e_eq50 | e_eq200 |
| --- | --- | --- | --- | --- | --- |
| pooled | 60.1360 [54.1965, 66.5215] | 12.4322 [11.1272, 13.7354] | 11.8087 [10.5227, 13.2149] | 6.0575 [5.4282, 6.7098] | 5.0059 [4.3216, 5.7050] |
| 2026093001 | 62.2649 [55.9444, 68.7556] | 12.8238 [11.5784, 14.0135] | 12.2125 [10.9308, 13.5033] | 6.2696 [5.5688, 6.9822] | 5.2310 [4.5564, 5.9339] |
| 2026093002 | 58.4492 [52.2176, 64.5512] | 12.3809 [10.7954, 13.9774] | 11.7152 [10.2298, 13.3995] | 6.0135 [5.0971, 6.8560] | 4.8728 [3.9944, 5.7048] |
| 2026093003 | 59.6940 [53.3334, 66.6971] | 12.0919 [10.7184, 13.5137] | 11.4984 [10.1402, 13.0533] | 5.8894 [5.3192, 6.4777] | 4.9139 [4.2645, 5.5732] |
| current | 82.4550 [73.6205, 91.5992] | 12.6897 [11.4940, 13.9654] | 12.0767 [10.8589, 13.4908] | 6.4895 [5.8252, 7.1223] | 5.3070 [4.6300, 5.9796] |
| stored-average | 37.8170 [34.2674, 41.5373] | 12.1747 [10.7299, 13.5454] | 11.5407 [10.1914, 13.0046] | 5.6255 [4.8290, 6.3814] | 4.7047 [3.9362, 5.4655] |
| OOP | 56.4217 [49.8138, 63.7426] | 12.2871 [10.9294, 13.7435] | 11.7813 [10.3923, 13.3616] | 6.1965 [5.6479, 6.7479] | 5.1947 [4.5923, 5.7863] |
| IP | 63.8503 [57.3566, 70.6848] | 12.5774 [11.2369, 13.9775] | 11.8361 [10.5899, 13.2363] | 5.9185 [5.1310, 6.7260] | 4.8171 [4.0161, 5.6487] |

## Ratios, alias cost and equity headroom

Paired ratios of reach-weighted means [bootstrap 95% interval]. Full v1 is the
denominator for the equity-headroom ratio; blueprint loss is the denominator
for R and signed alias cost. No negative difference is clipped and no undefined
bootstrap draw is discarded. All Set B ratio draws are defined.

| Set B policy group / target | R = v1 / BP | Signed alias cost | eq200 / v1 |
| --- | --- | --- | --- |
| pooled | 0.2175 [0.1917, 0.2466] | 0.0039 [0.0011, 0.0081] | 0.3680 [0.3237, 0.4208] |
| 2026093001 | 0.2163 [0.1921, 0.2418] | 0.0037 [0.0011, 0.0077] | 0.3733 [0.3261, 0.4274] |
| 2026093002 | 0.2253 [0.1921, 0.2637] | 0.0043 [0.0012, 0.0090] | 0.3558 [0.3092, 0.4128] |
| 2026093003 | 0.2111 [0.1836, 0.2431] | 0.0037 [0.0010, 0.0077] | 0.3750 [0.3285, 0.4293] |
| current | 0.1641 [0.1445, 0.1871] | 0.0028 [0.0008, 0.0059] | 0.3853 [0.3362, 0.4409] |
| stored-average | 0.3291 [0.2895, 0.3719] | 0.0061 [0.0017, 0.0128] | 0.3499 [0.3058, 0.4040] |
| OOP | 0.2277 [0.1979, 0.2625] | 0.0033 [0.0009, 0.0068] | 0.3935 [0.3467, 0.4520] |
| IP | 0.2081 [0.1842, 0.2352] | 0.0044 [0.0012, 0.0091] | 0.3424 [0.2971, 0.3962] |

The frozen point rule is R ≥0.7 with descriptive share ≥25% for H1-turn,
R ≤0.3 for H2-turn, otherwise mixed; share <10% takes precedence as H3-turn.
The pooled result meets **H2-turn**. The current-only R is below 0.3; the
stored-average point (0.3291) is in the mixed region and its interval crosses
0.3. This subset does not replace the predeclared pooled decision. No claim
that current/average extraction changes absolute playing strength follows from
these separately conditioned ranges.

Pooled full-v1 minus per-line loss is **0.0125 BB [0.0039, 0.0249]**;
alias cost is **0.3902% [0.1133%, 0.8089%] of blueprint BR loss**, below the
frozen small-difference threshold of 10%. This concerns public lines inside a
single turn root. It cannot exonerate inherited preflop/flop aliasing or pooling
across boards. These signed projection differences are not a causal loss
decomposition.

Per-root equity-200 loss is **0.2558 BB [0.2376, 0.2759]**. Its ratio to full-v1
loss is **0.3680 [0.3237, 0.4208]**, a 63.2% lower point loss for this other
feasible witness. That is useful headroom, not an abstract-equilibrium estimate
or a guaranteed gain from global equity buckets. The immediate proposal is a
native trainer/coverage audit, with a separate cross-root projection check before
declaring global v1 abstraction sufficient. Longer training and any A/B need
their own owner-authorized protocol and budget.

## Set A: selected-root values

Only 13 common roots qualify. Three button-1 strata have a single eligible
root, so the frozen stratified bootstrap does **not** provide valid intervals.
The following are descriptive point estimates; `[CI unavailable]` applies to
every metric and ratio, including lineage, extraction and seat views. Set A
does not drive the primary decision.

BB point estimates:

| Policy group / target | Roots | e_bp | e_v1proj full | e_v1proj line | e_eq50 | e_eq200 |
| --- | --- | --- | --- | --- | --- | --- |
| pooled | 13 | 3.2868 | 0.5387 | 0.5370 | 0.3064 | 0.2351 |
| 2026093001 | 13 | 3.1151 | 0.4689 | 0.4672 | 0.2837 | 0.2157 |
| 2026093002 | 13 | 3.4505 | 0.5723 | 0.5707 | 0.3038 | 0.2353 |
| 2026093003 | 13 | 3.2949 | 0.5750 | 0.5732 | 0.3317 | 0.2542 |
| current | 13 | 4.1245 | 0.5636 | 0.5619 | 0.3253 | 0.2475 |
| stored-average | 13 | 2.4492 | 0.5138 | 0.5121 | 0.2875 | 0.2227 |
| OOP | 13 | 3.0762 | 0.5501 | 0.5486 | 0.3146 | 0.2540 |
| IP | 13 | 3.4975 | 0.5274 | 0.5255 | 0.2982 | 0.2162 |

Percent-of-pot point estimates:

| Policy group / target | e_bp | e_v1proj full | e_v1proj line | e_eq50 | e_eq200 |
| --- | --- | --- | --- | --- | --- |
| pooled | 42.5251 | 8.0260 | 7.9405 | 4.4210 | 3.4211 |
| 2026093001 | 42.0262 | 7.3426 | 7.2579 | 4.2147 | 3.2369 |
| 2026093002 | 43.5959 | 8.3006 | 8.2209 | 4.3798 | 3.3981 |
| 2026093003 | 41.9531 | 8.4347 | 8.3427 | 4.6685 | 3.6283 |
| current | 55.4676 | 8.2319 | 8.1474 | 4.6222 | 3.5631 |
| stored-average | 29.5825 | 7.8200 | 7.7335 | 4.2198 | 3.2791 |
| OOP | 40.5793 | 8.0653 | 7.9883 | 4.4670 | 3.6241 |
| IP | 44.4709 | 7.9866 | 7.8926 | 4.3750 | 3.2181 |

Ratio point estimates, all with CI unavailable:

| Policy group / target | R | Signed alias cost | eq200 / v1 |
| --- | --- | --- | --- |
| pooled | 0.1639 | 0.0005 | 0.4364 |
| 2026093001 | 0.1505 | 0.0005 | 0.4601 |
| 2026093002 | 0.1659 | 0.0005 | 0.4112 |
| 2026093003 | 0.1745 | 0.0006 | 0.4421 |
| current | 0.1367 | 0.0004 | 0.4391 |
| stored-average | 0.2098 | 0.0007 | 0.4335 |
| OOP | 0.1788 | 0.0005 | 0.4617 |
| IP | 0.1508 | 0.0005 | 0.4100 |

## Turn folding and the overfold question

The selected-node estimator matches the stored line and target seat, uses
inverse root-inclusion weights once per observed decision, and requires all six
exports and positive equilibrium reach. Fourteen of 17 selected decisions have
common profiles; only **12 decisions at 11 roots** have usable folds in all six.
Covered decision weight is **48.5 / 70.5 = 68.79%**. Excluded roots and unreachable
nodes stay in the full frozen denominator.

On that covered subset, blueprint fold probability is **29.58%**, equilibrium
**38.75%**, difference **−9.17 percentage points**. The selected-fold bootstrap
also has singleton strata, so its 95% interval is unavailable. **No H0-turn
conclusion is admitted**: the frozen ≥90% coverage requirement fails, and the
covered point difference is outside three points. This neither resolves nor
contradicts #143's selected flop overfold screen; it is a later-street subset
with different ranges and reach. The bot can fold less overall while folding
particular continuing hands too often.

For context, the following descriptive frequencies include every target-facing
turn bet node on common roots. Weight by frozen root reach × equilibrium joint
node reach, average the six exports, and count repeated bet nodes as decisions.
This is a separate all-node description, not the H0 selected-node estimator.

| Set / target | Blueprint fold % | Equilibrium fold % |
| --- | --- | --- |
| A/IP | 36.31 | 42.14 |
| A/OOP | 36.41 | 41.79 |
| B/IP | 29.51 | 36.35 |
| B/OOP | 36.18 | 44.77 |

The equity-decile table covers **hand contexts with positive blueprint excess
fold probability**. It is not the distribution of all hands or decisions.
Equity is against a uniform opponent; decile 0 is lowest. Masses include root
reach weights and repeated nodes. The percentages below normalize each set's
total positive excess-fold mass, not all-hand probability.

| Set | Equity decile | Hand contexts | Weighted reach mass | Weighted excess-fold mass | Share of excess % |
| --- | --- | --- | --- | --- | --- |
| A | 0 | 1117 | 0.0060 | 0.0009 | 0.02 |
| A | 1 | 20030 | 0.1592 | 0.0092 | 0.25 |
| A | 2 | 77411 | 1.5152 | 0.2307 | 6.25 |
| A | 3 | 95858 | 2.8075 | 0.7459 | 20.21 |
| A | 4 | 142265 | 3.6702 | 1.1786 | 31.93 |
| A | 5 | 137581 | 2.3427 | 0.6906 | 18.71 |
| A | 6 | 110225 | 1.7980 | 0.2558 | 6.93 |
| A | 7 | 106062 | 1.3208 | 0.2908 | 7.88 |
| A | 8 | 137834 | 1.2025 | 0.2279 | 6.17 |
| A | 9 | 116373 | 0.4009 | 0.0612 | 1.66 |
| B | 0 | 3068 | 0.3516 | 0.0374 | 0.05 |
| B | 1 | 50053 | 2.0016 | 0.3103 | 0.44 |
| B | 2 | 200679 | 18.2161 | 4.1551 | 5.94 |
| B | 3 | 292526 | 46.2509 | 11.9069 | 17.02 |
| B | 4 | 324474 | 64.8666 | 19.9738 | 28.56 |
| B | 5 | 293906 | 54.0043 | 14.4820 | 20.70 |
| B | 6 | 328214 | 47.1660 | 11.3616 | 16.24 |
| B | 7 | 309617 | 35.1027 | 5.4064 | 7.73 |
| B | 8 | 356619 | 21.6070 | 1.4750 | 2.11 |
| B | 9 | 283604 | 10.9113 | 0.8380 | 1.20 |

Largest v1-key groups by weighted positive excess-fold mass, displayed
descriptively. The complete table has **45,530 (set, target, full-key) groups**
in [main03-overfold-key-totals.jsonl.gz](hu20-exact-turn-check-artifacts/main03-overfold-key-totals.jsonl.gz).
The [full root/export/decile/key table](hu20-exact-turn-check-artifacts/main03-overfold-groups.jsonl.gz)
retains every qualified group; atomic results retain every node and raw solver
responses retain individual holdings. The key column is the real 128-bit v1
information-key hash, including its public-history/legal-menu factors.

| Set | Target | Full v1 key | Hand contexts | Weighted excess-fold mass |
| --- | --- | --- | --- | --- |
| A | IP | e5f1335aeda6400a0551795c7fb436f6 | 45 | 0.1497 |
| A | IP | 2e2d8cb68165498ce0bc1c7190aa5d7d | 1748 | 0.1486 |
| A | OOP | 255588798e9a20f355cc94440f247259 | 140 | 0.1354 |
| A | IP | 0fc492a1a05b19bfc2400c8eecb5e227 | 184 | 0.1262 |
| A | IP | 05430b54d64d80392d6674575edcd9dd | 600 | 0.1114 |
| A | OOP | 31e1ef06a2f3c953659220879546223f | 1156 | 0.0976 |
| A | OOP | 1cd9413371703d4a8fb359318de2d7de | 873 | 0.0940 |
| A | IP | e903c40653c81703d5fc0d170872271e | 1286 | 0.0931 |
| A | IP | 7520f67e6811601a722665acc8e56229 | 146 | 0.0885 |
| A | OOP | b2098d6f2bf91f1907b46b393285e8d3 | 2153 | 0.0833 |
| B | OOP | 90cf7dd54b02b6b0460a72533e3b36c3 | 2124 | 2.6065 |
| B | OOP | eed5335059437f7d2315765304e8d269 | 3762 | 2.4441 |
| B | OOP | c9bdc13238bd56403e64ce998d1b678a | 448 | 2.3115 |
| B | IP | 9e12b2c41446596c2680637b4edaef26 | 844 | 1.9795 |
| B | OOP | e898dd1b340ff5db1fdd649cb8ab7758 | 542 | 1.7753 |
| B | IP | 0807c87eac1192a8166ff96e1ce42ef4 | 3303 | 1.7311 |
| B | IP | f1c5624984e9a67d4b517446b10a4df6 | 1145 | 1.6766 |
| B | IP | 16b5e00f766a4168d121a7707b72109a | 1513 | 1.6734 |
| B | IP | b760fd5b8dfe9f0788588bbe8e5977bd | 868 | 1.5262 |
| B | OOP | 82bce9dc5ac8bc57be4d9a09dc1e2cec | 812 | 1.3760 |

## Secondary empirical-range support and locked EV

Secondary BR values replace only the respondent range with #143's empirical
preflop range on the line/position, filtered for this board; they do not apply
an empirical flop likelihood update. The reference reweights the **primary
equilibrium strategy**, rather than solving a new secondary equilibrium.
An empirical holding outside the primary solver range is unsupported, not
silently smoothed into the solve. All unsupported/partial-support rows remain
in the atomic results, with input and retained mass.

| Set | Common-root target/range pairs | Zero support | Full support | Median retained fraction |
| --- | --- | --- | --- | --- |
| A | 156 | 66 | 45 | 0.1920 |
| B | 384 | 156 | 133 | 0.2727 |

These counts deduplicate the five metrics for each target/range pair. Of 192
Set B jobs, 33 have all ten secondary values with complete support; **zero roots
have complete secondary support across all six exports**. Therefore the frozen
secondary aggregate, matched-primary subset, BB/%-pot intervals and ratios are
unavailable. Individual supported values cannot substitute for that intersection.
The [supplement](hu20-exact-turn-check-artifacts/main03-supplement.json) lists
every support pair; source empirical groups are in the frozen Set A corpus.

Each of the 284 completed jobs also records both-blueprint locked EV for both
solver seats, in chips, with the solver-to-physical-seat map. These are exact
conditional EVs under that export's given ranges, not an arena strength score.
They remain in the [full atomic results](hu20-exact-turn-check-artifacts/main03-results.jsonl.gz).
The real-export V4 validation below checks their end-to-end mapping against
native Monte Carlo before qualification; it is not repeated as hundreds of
independent 95% gate tests.

## Descriptive LBR accounting

The frozen calculation is `(134 / 768) × mean e_bp / 0.65`. It gives
**0.5576 BB per panel hand**, or **85.79%
[78.04%, 94.35%]** of the approximately 0.65-BB/hand
bounded-LBR loss, propagating only the Set B root bootstrap. Occupancy 134/768
comes from #143's B100M v1 hand panel; the exact diagnostic uses six B500M
exports and an exact respondent, rather than that LBR. The unselected self-play
root population and conditional ranges also differ from LBR's selected roots.
This is a **descriptive comparison**, not a measured or causal share of LBR's
loss, not a bound on its turn attribution, and not an estimate of earlier-street
error. It can exceed 100% in other data. Its point clears the frozen 25%
materiality screen; it does not override the pooled R rule or establish H3.


## Qualification, convergence and resources

Qualification passed before main admission:

- K: 100,000 real information-key samples, zero mismatches.
- V1: every native line/action/chip amount agrees in real fixtures and all 284
  solved main trees; zero mismatches.
- V2: 300 independent native terminal samples per real fixture, including
  folds/showdowns; zero integer payout mismatches, floating error ≤0.000101 chips.
- V3: current-binary river BR agrees with `profile_quality` within 6e-7 BB;
  secondary-range river current/BR oracle error ≤7.53e-7 BB.
- V4: the real B500M policy lock EV is inside each independent 20,000-deal
  native Monte Carlo 95% interval; current-tool lock EV reproduces qualification.
- Independent native compact fixture: ten projection/BR comparisons across
  412 aliased nodes agree within 1.53e-7 BB. Artificial four-holding ranges are
  validation evidence, not strength measurements.
- Secondary identity: ten comparisons per cost fixture agree with primary
  values within 1.53e-7 BB.

| Real pilot | Exact BP EV seat 0 BB | Native MC BB [95%] | EQ residual % pot | Full pipeline sec | Plain / compressed GiB | Peak owned RSS GiB |
| --- | --- | --- | --- | --- | --- | --- |
| Limped | +0.211196 | +0.1751 [0.130884, 0.219316] | 0.186956 | 360.58 | 2.100 / 1.063 | 2.600 |
| Min-raised | +1.525193 | +1.5359 [1.433996, 1.637804] | 0.160866 | 128.78 | 0.936 / 0.477 | 1.280 |
| 3-bet | −0.748324 | −0.6987 [−0.822558, −0.574842] | 0.172093 | 89.52 | 0.650 / 0.331 | 0.938 |

V5: all 284 main solves reach the ≤0.2% target; none approach the 0.5%
exclusion boundary. Tiny equilibrium residuals remain numerical uncertainty,
so the values are noise-free enumerations conditional on a finite-tolerance
equilibrium, rather than mathematically exact equilibrium certificates.
The supplement lists convergence/iterations/time/memory for every solve.

| Main solved profiles | Minimum | Median | Maximum |
| --- | --- | --- | --- |
| compressed_estimate_gib | 0.0106 | 0.4607 | 1.8335 |
| iterations | 25.0000 | 150.0000 | 350.0000 |
| peak_owned_rss_gib | 0.1282 | 1.2523 | 4.4928 |
| plain_estimate_gib | 0.0185 | 0.9138 | 3.6399 |
| residual_pct_pot | 0.0405 | 0.1713 | 0.1997 |
| solver_seconds | 2.7359 | 110.4091 | 492.9761 |

`memory_usage()` estimates are recorded before allocation in both modes.
Compressed tree storage excludes several runtime and exporter costs: the
largest compressed estimate is 1.833 GiB while the complete pipeline reaches
4.493 GiB. One external solver at a time, two Rayon threads, nice priority,
native compressed menus and nonzero-range holdings were retained throughout.
No raise-cap removal or oversized turn exclusion was needed.

Measured readmission had 8.781 GiB reclaimable; its 80% floor permits 7 GiB,
so the frozen owned-process limit is 5 GiB within the owner's ≤10-GiB ceiling.
Main-03 swap baseline stays **1,295,777,792 bytes**; maximum recorded swap is
the same, so there is **no growth during main-03**. End-of-run swap is
901,513,216 bytes; reclaimable memory is 7,843,758,080 bytes. Driver/sidecar
exit is verified; TensorBoard 6006 remains running, 16008 remains stopped,
and unrelated jobs are untouched.

Cumulative main time is **43,102.427453 seconds (11.973 hours)** against the
unchanged 86,400-second ceiling, including all **1,132.611673 seconds** of
preceding main execution and failed compute. The owner pause/diagnosis interval
is excluded as declared; no fresh 24-hour allowance is created. The original
outcome-blind admission estimate was 21.0559 hours with twofold contingency.
Main-03 uses the same source/binary/scientific inputs after the resource amendment.

## Retained failures, interruption and resumption

| Attempt | Failure and disposition |
| --- | --- |
| Original flop native/cap-three pilots | Native arena oversize; compressed cap-three estimates 97.64/56.86/25.86 GiB. No main flop values. Original report and every failure remain retained. |
| compact-validation-alias-01 | Python comparison depended on JSON action-field ordering. Canonical action identity fixed and regression-tested. |
| compact-validation-alias-02 | Fixture watchdog exceeded a deliberately small 1-GiB limit; retained. |
| compact-validation-alias-03 | Wrong input-plan path; no solver allocation. |
| compact-validation-alias-04 | Concurrent verbose Python profile and locked solver exceeded 4-GiB aggregate RSS. Separate workers release the profile before solving. |
| compact-validation-alias-05 | Synthetic smaller-stack fixture failed table-roster validation; no solve. |
| compact-validation-alias-06 | Compiler correctly refused non-HU20 initial stack; no solve. Synthetic path removed. |
| compact-validation-alias-07 | Full native fixture passes after worker separation, peak 2.815 GiB. |
| cost-preflight-limp-01 | Worker executable-path typo, exit 127, no allocation. Corrected limp-02 passes. |
| main-01 | Two complete atomic jobs; third refused at 80.70 seconds by measured-headroom guard as an Ollama model appeared. Subsequent swap 2,314.88 MiB versus original 745.38-MiB baseline also exceeds +1 GiB. Owner stopped work and requested cleanup for Ollama. |
| Initial cleanup | M4 staging/input/tool/TensorBoard files archived and verified: 3,103 files, 3,152,458,858 bytes, zero SHA mismatches. Four temporary roots removed. Shared source/venv, original evidence and Ollama retained. Owner subsequently requested both TensorBoards and orphan thermal-monitor processes stopped; verified. |
| main-02 owner-authorized resumption | Two original complete results preserved byte-for-byte, followed by three additional completions. Sixth full-range stored-average limped job exceeded 4-GiB owned RSS at **4.016 GiB**, exit −15, 458.246 seconds; no swap growth. Its ten primary and ten secondary rows are **incomplete and excluded**, because both-blueprint/final completion is missing. |
| Frozen main-03 readmission | Measured resource-only limit changes to 5 GiB before the retry; same binary/roots/exports/order/menus/thresholds, swap baseline unchanged, all five results preserved byte-for-byte and prior 1,132.611673 seconds charged. Retried root completes in 466.05 solver seconds, peak 4.493 GiB; ten primary values match the previous partial rows bit-for-bit, now with both-BP EV and final completion. |
| main-03 completed campaign | 288 atomic outcomes, 284 solves, four zero-support exclusions; all gates/targets pass. No guard bypass, capped/adaptive replacement, restart after a stop or rental. |

The owner pause/cleanup archive remains at
`/Users/dberweger/Local/hu20-m4-archive-20261002` on M1. Cleanup and TensorBoard
shutdown records describe their historical stages, not the current completed
state. On resumption only the task's port-6006 server was restarted; unrelated
port 16008 stays stopped. Previous failure JSON, raw responses and manifests
remain beside this report. The [25%](hu20-exact-turn-check-artifacts/milestone-25.json),
[50%](hu20-exact-turn-check-artifacts/milestone-50.json) and
[100%](hu20-exact-turn-check-artifacts/milestone-100.json) milestones distinguish
recorded jobs, solves, exclusions and then-incomplete six-export coverage.

## Evidence, hashes and reproducibility

Full atomic main-03 results, requests, solver responses, nested progress and
resource/admission/process records are retained on the M4 at
`/Users/dberweger/Local/hu20-exact-flop-check-20261001/results/turn-check-20261002/main-03`
and retrieved to M1 at
`/Users/dberweger/Local/hu20-exact-turn-main-20261002/main-03`.
The raw inventory covers **3,731 files / 4,095,830,672 bytes**; all 288 atomic
results independently reproduce their original per-file SHA-256 values.
The retrieval verification records every archived file's size and hash.
Its inventory excludes itself and the subsequently generated final input
verification record, which have separate hashes in the artifact inventory.

The main report is generated by `scripts.report_turn_check` with the frozen
resource-readmission config. The JSON retains every pooled/lineage/extraction/
seat BB/%-pot interval, paired ratio interval, exclusion and selected-fold
coverage. Additional descriptive resource/support/key tables are recorded
separately in the supplement and do not replace the primary estimator.

```sh
python -m scripts.report_turn_check \
  --protocol configs/diagnostics/hu20-exact-turn-check-resource-readmission.json \
  --run /Users/dberweger/Local/hu20-exact-turn-main-20261002/main-03 \
  --out /Users/dberweger/Local/hu20-exact-turn-report-new
```

The AGPL solver and Rust adapter stay outside the MIT repository at
`/Users/dberweger/Local/hu20-exact-flop-tool`. Upstream pin:
`9d1509fe5077d019825f833eed04b16d342dfda1`; external binary SHA-256:
`cdc46b10d985d64747982ed1e3d40a1697d533cfbc444c0d16ed284cc4148952`.
Native engine commit is `5db20e3d5d6862b32a7402035c1340b622d3b005`.
The [qualified inventory](hu20-exact-turn-check-artifacts/inventory-resource-readmission.json)
includes all external/upstream source hashes, Python/native source, six exports,
three hand archives and toolchain provenance; no AGPL source is committed here.
All six policy bytes and the binary were reverified after completion.

| Policy / hand archive | SHA-256 |
| --- | --- |
| B-2026093001-500000000.policy.json.gz | e0dfb7c0a0ebde1e8a904fce89481919f4e9b32a633b912e88d241d039e69628 |
| B-2026093002-500000000.policy.json.gz | 67427db96fb3d1aba3570e0235bb368a81e129978ff1a3e4c6e083b162563456 |
| B-2026093003-500000000.policy.json.gz | bcf7ca319b95688a0d8fcd15536b04b311310feacefdc6fff81221304662d083 |
| B-2026093001-500000000.average.jsonl.gz | f83d250e27d45d5e12434b90fb270c0217a329bb76e19e7d73bfcefd701c0aaa |
| B-2026093002-500000000.average.jsonl.gz | 8829d26430e6dc0c47d5e550ae1f8fcf6e99a28619c53b54097f51a9478f445d |
| B-2026093003-500000000.average.jsonl.gz | c5fa910a1211c75996773cbf280650c94f47a13f83c0fd36d64be6647d100285 |
| 2026093001/evaluation/hands.jsonl.gz | 81c82e2d914b28c57dc49134a5653562aeaf63e0590e2d3bcc8ecf8094f8a5d2 |
| 2026093002/evaluation/hands.jsonl.gz | 7e5366049b0657a30f1b5b0391daab40c428cc1004947e10109271a9ab43e0bc |
| 2026093003/evaluation/hands.jsonl.gz | 2af3e4bff4ca29da2921ed5e848b2c077fe46468d1fc4f43fd5b1cdce15b1b82 |

Corpus A SHA-256 is
`db6332db5f4d16a55806c13010b6a9edabb4392620157d8ae6d42c4d2075fc99`;
corpus B is
`a273ad29897cb46a232e9a2b7c1651971e155ab5f0b057669c8daf90da374be0`;
final config is
`b9d248f28752587971cc633250e87c24479d3788f797026dd87211ae0d59219f`.
All result/report hashes are listed in
[main03-artifact-inventory.json](hu20-exact-turn-check-artifacts/main03-artifact-inventory.json),
and original raw hashes in
[main03-file-inventory.json](hu20-exact-turn-check-artifacts/main03-file-inventory.json).
Retrieval verification is
[main03-retrieval-verification.json](hu20-exact-turn-check-artifacts/main03-retrieval-verification.json).

Validation: all 110 local diagnostic tests passed at readmission; the nine
turn/report/readmission tests pass again during reporting. Full Linux CI passed
for resource readmission, the report-coverage correction and the 50% milestone.
The [final reporter verification](hu20-exact-turn-check-artifacts/main03-report-verification.json)
reproduces all summary estimates locally and the decompressed full-key table,
and verifies all 284 solved request/response hashes against metadata, the
qualified binary, two threads and native menu. Final CI is recorded with the
reporting commit/PR update. Both PRs remain drafts.

## Limits and next decision

- Given policy ranges exclude preflop/flop errors; this cannot diagnose the
  original flop plateau or infer a causal street decomposition of LBR losses.
- Projections are feasible witnesses, not the abstraction's own equilibria.
  High R would only be an upper-bound heuristic. Low within-root R does not
  establish a globally consistent strategy under cross-board v1 pooling.
- Per-turn equity buckets are more favourable than global blueprint buckets.
  Their headroom is not a promised training or search improvement.
- Signed within-root alias cost cannot measure aliases inherited across roots;
  differences between projections are not a causal decomposition.
- Set A common exclusions change its represented population; singletons block
  its intervals, and selected-fold coverage blocks H0-turn inference.
- Empirical secondary weighting reweights the primary equilibrium; support is
  sparse and no six-export secondary aggregate qualifies.
- These 32 B roots and root bootstrap quantify uncertainty in this frozen
  stratified population. They do not certify full-game playing strength,
  global exploitability, model promotion or a releasable agent.

Review the H2-turn feasible-witness result, #146's proposed versioned menu-name
history A/B and cross-root consistency before choosing the next intervention.
#144's history experiment is not a menu-name A/B and should not be treated as
one. No trainer change, 100M A/B, longer training, paid flop solve or merge starts
from this report alone. The diagnostic is finished; no automatic solver restart
is scheduled.

TensorBoard 6006 remains available with archived run
`hu20-exact-turn-check-20261002/main-03`; the solver and sidecar are stopped.
Owner access remains:

```sh
ssh -N -L 6006:127.0.0.1:6006 m4
```

Open http://localhost:6006 on M1. Unrelated port 16008 stays stopped.
