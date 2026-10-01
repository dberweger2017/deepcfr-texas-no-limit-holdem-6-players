# HU20 card abstraction v2 — 100M paired comparison

I tested the frozen contribution/kicker/draw refinement at the same 100M completed-node target in each of three matching lineages. **It did not improve this measured 100M panel.** All 13 aggregate v2-minus-v1 point estimates are negative. The largest declines are against original-cap2 min-raise, native pressure and bounded LBR. I retain every result and leave v1 as the default; there is no promotion or automatic 500M follow-on.

V2 separates all five highlighted #139 examples, but creates about **7.0× as many learned keys**: 10.52–10.56M versus 1.49–1.50M. Stored traverser visits/key fall from 13.45–13.51 to 1.95–1.96. The reached-decision histograms below show severe postflop sparsity. This controlled A/B measures the combined effect of a finer representation and its lower sampling density at a fixed node budget. It does not establish that private-card detail is irrelevant.

All three from-zero training runs and **38,400/38,400 final hands** completed; every final hand passed native replay. All three archives and all 45 original work-manifest files per archive were verified before teardown. All six owned pod IDs, including the failed setup attempt, are API-confirmed absent. The conservative cumulative compute/disk estimate is **$3.203879**, under the approved $10 cap; settled billing is unverified. No M4/#136 compute or models were used.

## Frozen comparison and interpretation

- Game: two-player native-reopening NLHE, 20BB reset each hand, 100 chips/BB, no rake/ante. Both arms use current extraction and the unchanged min/pot/conditional-jam target menu; no river search. Only the card representation changes. The 20M safety capacity replaces an inactive 3M abort bound; CFR math, training streams, K1, public history and actions stay unchanged.
- Matching lineages: 2026093001/2/3, from zero, stopping at the first complete iteration reaching 100M nodes. Actual overshoot is reported below; these are traversal nodes, not poker hands.
- Fresh final root `202610010701` plus fixed panel index; disjoint outcome-blind timing root `202610010702`. Standard panels: 256 paired deal blocks × two positions × two arms × three seeds; LBR: 128 blocks. Seeds share each deal block, so the aggregate averages the three lineage changes **within** each block before its Student-t interval. Rotations/lineages of the same deal are not independent samples.
- Reported intervals are exploratory, unadjusted 95%, conditional on these three trained policies. They do not quantify generalization over new training seeds; 13 panels/position comparisons are not a confirmatory family. Per-seed and positional results remain visible.
- `uniform`/`passive` use the uncapped menu; `minraise-cap2`/`pressure-cap2` retain the original-cap2 opponent; the six styles, exact native-pressure and frozen post-Luna selective-stackoff retain their existing separate contracts. Never pool sizing contracts. Bounded LBR adapts to each target with the existing four-chance-sample/5-second soft-guard configuration and validated ranked/cache executor. This is not exact exploitability. Per-action LBR sample-completion telemetry was not copied by this wrapper, so I do not claim every action completed four samples.
- Whole-hand returns partitioned after a first large raise are **not individual-bet EV**. Reached-state visit/fallback and wager ratios are descriptive under each arm’s own policy; they are not a matched-state causal decomposition.

## Playing results

Positive changes favor v2. Absolute arm points below average the three lineage means; the interval belongs to the paired change, not the absolute point. Full absolute/positional intervals and raw block changes are in the [independent summary](hu20-card-v2-artifacts/production/independent-summary.json).

| Opponent panel | Blocks | v1 BB/100 | v2 BB/100 | Paired v2−v1 [95% interval] | Seed 1 Δ | Seed 2 Δ | Seed 3 Δ |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| uniform | 256 | +108.17 | +74.74 | -33.43 [-77.49, +10.63] | -47.66 | -30.76 | -21.88 |
| passive | 256 | +109.31 | +105.40 | -3.91 [-49.59, +41.78] | +25.78 | -28.61 | -8.89 |
| minraise-cap2 | 256 | +119.60 | -31.71 | -151.30 [-216.20, -86.40] | -132.62 | -166.02 | -155.27 |
| pressure-cap2 | 256 | +10.97 | -77.08 | -88.05 [-139.41, -36.69] | -98.63 | -28.32 | -137.21 |
| tight_passive | 256 | +51.89 | +29.65 | -22.23 [-33.88, -10.58] | -17.19 | -16.11 | -33.40 |
| loose_passive | 256 | +4.75 | -15.07 | -19.82 [-43.03, +3.38] | -24.71 | -20.41 | -14.36 |
| tight_aggressive | 256 | +43.72 | +26.89 | -16.83 [-36.14, +2.48] | -10.64 | -28.81 | -11.04 |
| loose_aggressive | 256 | +36.26 | -12.08 | -48.34 [-86.24, -10.44] | -77.64 | -11.91 | -55.47 |
| pot_pressure | 256 | +7.49 | +6.38 | -1.11 [-22.26, +20.05] | +4.98 | +4.39 | -12.70 |
| train_pressure | 256 | +1.82 | -1.73 | -3.55 [-30.54, +23.45] | -0.88 | +18.65 | -28.42 |
| native-pressure | 256 | +177.12 | +67.97 | -109.15 [-165.48, -52.81] | -81.74 | -88.09 | -157.62 |
| selective-stackoff | 256 | +36.17 | +20.31 | -15.85 [-30.90, -0.81] | -19.34 | -12.60 | -15.62 |
| lbr | 128 | -72.92 | -142.45 | -69.53 [-130.58, -8.48] | -58.20 | -134.96 | -15.43 |

The min-raise decline is negative in all three lineages (−132.62/−166.02/−155.27). Native-pressure changes are −81.74/−88.09/−157.62. LBR changes are −58.20/−134.96/−15.43 with substantial lineage uncertainty. Seven aggregate unadjusted intervals exclude zero on the negative side; the remaining six are inconclusive. No panel has a positive aggregate point. The small selective-stackoff decline (−15.85, interval −30.90 to −0.81) is exploratory, with only 25 v2 large raises over all three lineages.

### Position and lineage detail

Each cell is the paired v2−v1 BB/100 [unadjusted 95% interval]. Each standard position uses one outcome/arm per 256 deal blocks, LBR 128. Full arm estimates are retained in JSON.

| Panel | Seed | Button Δ [95%] | Big blind Δ [95%] |
| --- | --- | --- | --- |
| uniform | 2026093001 | -41.02 [-137.16, +55.13] | -54.30 [-143.79, +35.20] |
| uniform | 2026093002 | -9.96 [-106.15, +86.23] | -51.56 [-152.57, +49.44] |
| uniform | 2026093003 | -0.78 [-102.14, +100.58] | -42.97 [-133.01, +47.07] |
| passive | 2026093001 | -59.38 [-144.89, +26.14] | +110.94 [+15.65, +206.23] |
| passive | 2026093002 | -0.20 [-91.19, +90.80] | -57.03 [-155.96, +41.89] |
| passive | 2026093003 | -22.07 [-125.44, +81.30] | +4.30 [-91.92, +100.51] |
| minraise-cap2 | 2026093001 | -159.38 [-295.40, -23.35] | -105.86 [-251.98, +40.26] |
| minraise-cap2 | 2026093002 | -96.88 [-211.56, +17.81] | -235.16 [-378.28, -92.04] |
| minraise-cap2 | 2026093003 | -51.17 [-171.17, +68.83] | -259.38 [-393.28, -125.47] |
| pressure-cap2 | 2026093001 | -78.91 [-174.45, +16.64] | -118.36 [-233.40, -3.32] |
| pressure-cap2 | 2026093002 | -63.67 [-153.51, +26.16] | +7.03 [-101.89, +115.95] |
| pressure-cap2 | 2026093003 | -112.70 [-217.77, -7.62] | -161.72 [-270.58, -52.86] |
| tight_passive | 2026093001 | -38.28 [-67.20, -9.36] | +3.91 [-10.79, +18.60] |
| tight_passive | 2026093002 | -28.71 [-58.13, +0.71] | -3.52 [-15.47, +8.44] |
| tight_passive | 2026093003 | -60.94 [-100.69, -21.18] | -5.86 [-21.08, +9.36] |
| loose_passive | 2026093001 | -56.45 [-117.22, +4.33] | +7.03 [-41.60, +55.66] |
| loose_passive | 2026093002 | -10.74 [-73.47, +51.99] | -30.08 [-88.45, +28.30] |
| loose_passive | 2026093003 | -27.54 [-84.38, +29.30] | -1.17 [-50.69, +48.34] |
| tight_aggressive | 2026093001 | -25.98 [-69.78, +17.83] | +4.69 [-29.98, +39.36] |
| tight_aggressive | 2026093002 | -46.68 [-100.59, +7.24] | -10.94 [-40.74, +18.86] |
| tight_aggressive | 2026093003 | -16.99 [-59.78, +25.79] | -5.08 [-43.23, +33.08] |
| loose_aggressive | 2026093001 | -75.98 [-161.69, +9.74] | -79.30 [-150.08, -8.51] |
| loose_aggressive | 2026093002 | -24.22 [-106.38, +57.94] | +0.39 [-66.09, +66.87] |
| loose_aggressive | 2026093003 | -71.09 [-156.51, +14.32] | -39.84 [-120.15, +40.46] |
| pot_pressure | 2026093001 | +33.01 [-23.96, +89.98] | -23.05 [-53.11, +7.02] |
| pot_pressure | 2026093002 | +18.95 [-36.49, +74.38] | -10.16 [-35.43, +15.12] |
| pot_pressure | 2026093003 | -21.48 [-79.40, +36.43] | -3.91 [-33.87, +26.06] |
| train_pressure | 2026093001 | -16.60 [-75.02, +41.82] | +14.84 [-42.53, +72.21] |
| train_pressure | 2026093002 | +23.24 [-44.97, +91.45] | +14.06 [-39.06, +67.19] |
| train_pressure | 2026093003 | -43.55 [-107.36, +20.25] | -13.28 [-86.68, +60.12] |
| native-pressure | 2026093001 | -85.35 [-216.68, +45.98] | -78.12 [-221.04, +64.79] |
| native-pressure | 2026093002 | -51.95 [-181.85, +77.94] | -124.22 [-275.18, +26.74] |
| native-pressure | 2026093003 | -96.48 [-216.28, +23.31] | -218.75 [-343.86, -93.64] |
| selective-stackoff | 2026093001 | -31.64 [-70.59, +7.31] | -7.03 [-33.92, +19.85] |
| selective-stackoff | 2026093002 | -11.91 [-50.00, +26.17] | -13.28 [-46.10, +19.54] |
| selective-stackoff | 2026093003 | -25.39 [-56.01, +5.23] | -5.86 [-33.44, +21.72] |
| lbr | 2026093001 | -82.81 [-237.92, +72.30] | -33.59 [-186.38, +119.19] |
| lbr | 2026093002 | -47.27 [-193.61, +99.08] | -222.66 [-378.13, -67.19] |
| lbr | 2026093003 | -6.64 [-157.95, +144.67] | -24.22 [-208.31, +159.87] |

## Large-pot tails and lookup exposure

Counts sum the three lineages for description only; they are not independent replicate counts for uncertainty. Each arm has 1,536 hands/panel, or 768 for LBR. A large raise demands at least 800 additional chips (8BB) from the opponent; the opportunity denominator counts target decisions with such a menu action. A jam raise commits the acting player’s remaining stack; full-stack wins/losses are net +2,000/−2,000 chips, not merely winning a showdown.

| Panel | Arm | Full-stack wins / losses | Large actions / opportunities | Jam actions / opportunities | Opponent folds / continues after large actions | Fallback / target decisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uniform | v1 | 109 / 48 | 175 / 735 | 154 / 802 | 83 / 92 | 2 / 3269 |
| uniform | v2 | 89 / 61 | 239 / 574 | 224 / 636 | 123 / 116 | 180 / 2903 |
| passive | v1 | 118 / 43 | 261 / 691 | 165 / 695 | 0 / 261 | 1 / 5598 |
| passive | v2 | 146 / 94 | 346 / 705 | 249 / 722 | 0 / 346 | 473 / 5292 |
| minraise-cap2 | v1 | 327 / 164 | 511 / 2341 | 464 / 2495 | 0 / 511 | 26 / 5385 |
| minraise-cap2 | v2 | 337 / 263 | 660 / 1528 | 571 / 1729 | 0 / 660 | 590 / 4411 |
| pressure-cap2 | v1 | 85 / 84 | 288 / 1429 | 217 / 1510 | 93 / 195 | 3 / 4401 |
| pressure-cap2 | v2 | 75 / 153 | 414 / 998 | 352 / 1104 | 160 / 254 | 295 / 3685 |
| tight_passive | v1 | 2 / 6 | 6 / 25 | 6 / 25 | 0 / 6 | 7 / 1527 |
| tight_passive | v2 | 2 / 12 | 14 / 31 | 11 / 32 | 0 / 14 | 71 / 1462 |
| loose_passive | v1 | 26 / 26 | 80 / 245 | 57 / 249 | 3 / 77 | 22 / 3361 |
| loose_passive | v2 | 28 / 40 | 106 / 246 | 80 / 251 | 4 / 102 | 252 / 3143 |
| tight_aggressive | v1 | 9 / 14 | 8 / 51 | 10 / 56 | 2 / 6 | 0 / 1443 |
| tight_aggressive | v2 | 7 / 25 | 16 / 52 | 19 / 62 | 4 / 12 | 35 / 1429 |
| loose_aggressive | v1 | 75 / 54 | 83 / 376 | 79 / 428 | 11 / 72 | 1 / 2680 |
| loose_aggressive | v2 | 65 / 85 | 124 / 283 | 131 / 335 | 16 / 108 | 187 / 2454 |
| pot_pressure | v1 | 29 / 65 | 109 / 332 | 74 / 362 | 57 / 52 | 376 / 1645 |
| pot_pressure | v2 | 26 / 61 | 106 / 325 | 64 / 353 | 53 / 53 | 382 / 1614 |
| train_pressure | v1 | 29 / 43 | 89 / 306 | 56 / 338 | 25 / 64 | 142 / 2315 |
| train_pressure | v2 | 30 / 42 | 84 / 214 | 77 / 256 | 21 / 63 | 213 / 2083 |
| native-pressure | v1 | 242 / 135 | 526 / 2046 | 531 / 2266 | 196 / 330 | 36 / 5222 |
| native-pressure | v2 | 168 / 163 | 585 / 1436 | 601 / 1688 | 259 / 326 | 445 / 4320 |
| selective-stackoff | v1 | 3 / 12 | 26 / 77 | 21 / 77 | 3 / 23 | 0 / 2181 |
| selective-stackoff | v2 | 1 / 9 | 25 / 62 | 17 / 64 | 3 / 22 | 99 / 1940 |
| lbr | v1 | 55 / 65 | 140 / 530 | 124 / 601 | 54 / 86 | 5 / 1743 |
| lbr | v2 | 67 / 89 | 164 / 420 | 148 / 456 | 54 / 110 | 128 / 1669 |

Concrete diagnostic leads: min-raise full-stack losses increase **164→263**, while wins increase 327→337; native-pressure wins decrease **242→168** and losses increase 135→163. Under LBR, wins increase 55→67 but losses increase 65→89. These counts fit the negative overall estimates without proving which card feature or individual wager caused them. Selective-stackoff losses actually fall 12→9 while wins fall 3→1 and overall profit declines: a stack-loss count alone would miss that result.

### First-large-raise whole-hand partition

This table assigns a hand’s entire net result once, using only its first large target raise. `N / total BB` is the hand count and sum of whole-hand chip returns divided by 100; it is neither BB/100 nor bet EV. The no-large group, any no-response group and these groups partition every hand in the JSON. All observed large raises have a recorded rival response.

| Panel | Arm | First large raise folded: N / total BB | First large raise continued: N / total BB | No large raise: N / total BB |
| --- | --- | ---: | ---: | ---: |
| uniform | v1 | 83 / +588.00 | 91 / +585.00 | 1362 / +488.50 |
| uniform | v2 | 121 / +855.00 | 114 / +121.00 | 1301 / +172.00 |
| passive | v1 | 0 / +0.00 | 247 / +1725.00 | 1289 / -46.00 |
| passive | v2 | 0 / +0.00 | 327 / +1156.00 | 1209 / +463.00 |
| minraise-cap2 | v1 | 0 / +0.00 | 509 / +2975.00 | 1027 / -1138.00 |
| minraise-cap2 | v2 | 0 / +0.00 | 646 / +1039.00 | 890 / -1526.00 |
| pressure-cap2 | v1 | 93 / +512.00 | 195 / +101.00 | 1248 / -444.50 |
| pressure-cap2 | v2 | 160 / +866.00 | 247 / -1744.00 | 1129 / -306.00 |
| tight_passive | v1 | 0 / +0.00 | 6 / -40.00 | 1530 / +837.00 |
| tight_passive | v2 | 0 / +0.00 | 14 / -188.00 | 1522 / +643.50 |
| loose_passive | v1 | 3 / +19.00 | 74 / +19.00 | 1459 / +35.00 |
| loose_passive | v2 | 3 / +26.00 | 101 / -387.00 | 1432 / +129.50 |
| tight_aggressive | v1 | 2 / +15.00 | 6 / +40.00 | 1528 / +616.50 |
| tight_aggressive | v2 | 4 / +33.00 | 12 / -118.00 | 1520 / +498.00 |
| loose_aggressive | v1 | 11 / +89.00 | 71 / +365.00 | 1454 / +103.00 |
| loose_aggressive | v2 | 16 / +140.00 | 107 / -211.00 | 1413 / -114.50 |
| pot_pressure | v1 | 57 / +277.00 | 52 / -476.00 | 1427 / +314.00 |
| pot_pressure | v2 | 53 / +270.00 | 53 / -480.00 | 1430 / +308.00 |
| train_pressure | v1 | 25 / +152.00 | 64 / -330.00 | 1447 / +206.00 |
| train_pressure | v2 | 21 / +124.00 | 63 / -368.00 | 1452 / +217.50 |
| native-pressure | v1 | 196 / +1539.00 | 330 / +1833.00 | 1010 / -651.50 |
| native-pressure | v2 | 259 / +1863.00 | 326 / -450.00 | 951 / -369.00 |
| selective-stackoff | v1 | 3 / +14.00 | 23 / -183.00 | 1510 / +724.50 |
| selective-stackoff | v2 | 3 / +19.00 | 22 / -263.00 | 1511 / +556.00 |
| lbr | v1 | 54 / +382.00 | 85 / +248.00 | 629 / -1190.00 |
| lbr | v2 | 54 / +287.00 | 110 / -492.00 | 604 / -889.00 |

### Street visit density and fallback

The following aggregates reached target decisions across the separately reported panels; they describe exposure, not strength. Detailed histograms per panel/seed remain in the independent JSON. `0` visits corresponds to the unchanged uniform missing-key fallback. Stored visits are traverser updates, not observations of every actor.

| Arm | Street | Decisions | 0 visits | 1–9 visits | 10–99 visits | 100+ visits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1 | preflop | 18494 | 196 (1.1%) | 36 (0.2%) | 624 (3.4%) | 17638 (95.4%) |
| v1 | flop | 10692 | 140 (1.3%) | 50 (0.5%) | 557 (5.2%) | 9945 (93.0%) |
| v1 | turn | 6803 | 145 (2.1%) | 248 (3.6%) | 1197 (17.6%) | 5213 (76.6%) |
| v1 | river | 4781 | 140 (2.9%) | 220 (4.6%) | 1072 (22.4%) | 3349 (70.0%) |
| v2 | preflop | 18026 | 195 (1.1%) | 41 (0.2%) | 395 (2.2%) | 17395 (96.5%) |
| v2 | flop | 9707 | 962 (9.9%) | 2366 (24.4%) | 3602 (37.1%) | 2777 (28.6%) |
| v2 | turn | 5404 | 1197 (22.2%) | 1880 (34.8%) | 1844 (34.1%) | 483 (8.9%) |
| v2 | river | 3268 | 996 (30.5%) | 1342 (41.1%) | 816 (25.0%) | 114 (3.5%) |

The density loss is particularly large on flop/turn/river while preflop remains dense. Fallback is retained, not translated or masked. This supports a sampling-density diagnostic lead; it does not separate dilution, visited-state changes, fallback action mixtures and intrinsic representation quality. #144’s separately frozen card×history experiment is needed for its own interaction contrast; these 13-panel results are not a common four-cell comparison.

### Post-hoc LBR review: separate v1’s residual weakness from v2’s dilution

I independently reproduced Claude’s final-review counts from the committed raw
hands, with no model loads or new games. The [post-hoc output](hu20-card-v2-artifacts/lbr-posthoc.json)
pins all three raw-file hashes and the analysis script. This section was added
after outcomes were opened; it is not a prospective endpoint.

**LBR-only reached target decisions:** the pooled histogram above also includes
other opponents, so it should not be read as the LBR histogram.

| Street | v1 decisions | v1 ≥100 visits | v2 decisions | v2 ≥100 visits | v2 <10 visits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preflop | 1,007 | 99.4% | 933 | 99.9% | 0.0% |
| Flop | 513 | 91.6% | 531 | 17.9% | 43.3% |
| Turn | 168 | 54.2% | 164 | 4.3% | 71.3% |
| River | 55 | 43.6% | 41 | 0.0% | 82.9% |

**Whole-hand net returns grouped by the last recorded betting action:** an
all-in on an earlier street may still run out the board to the river. These
labels describe where betting stopped, not the final board street or the
location of an erroneous decision. The arms can enter different groups.

| Last betting street | v1 hands / total net BB | v2 hands / total net BB |
| --- | ---: | ---: |
| Preflop | 383 / −111 | 392 / −295 |
| Flop | 251 / −471 | 248 / −588 |
| Turn | 90 / −2 | 93 / −186 |
| River | 44 / +24 | 35 / −25 |
| Total | 768 / −560 | 768 / −1,094 |

The totals reproduce −72.92/−142.45 BB/100. V1’s preflop/flop-stop groups
sum to −582 BB while its later-stop groups sum to +22 BB. Its 15 hands
with any target turn/river key below 10 visits net +125 BB; 63 hands with
minimum late visits 10–99 net −58 BB; 56 with all late target visits ≥100
net −45 BB; 634 with no target late decision net −582 BB. These are disjoint
hand partitions. Pooling the latter two categories gives the review’s
690 hands / −627 BB, but hides the much larger no-late-decision category.

This **raises the priority of v1’s preflop/flop strategy and commitment
patterns**, including potential card-aliasing at well-visited keys. Sparse
turn/river keys do not describe the main observed v1 loss partition here.
It does not establish that the mistake occurred on the flop, or that the
flop abstraction is its unique cause. A hand can end after earlier strategic
choices; visited keys can still be unconverged, and later continuation
values affect earlier regret updates. These 128 paired blocks and bounded
responder remain a small, post-hoc diagnostic.

V2’s extra −184 BB in preflop-stop hands is compatible with degraded future
values feeding into its earlier strategy; that mechanism is untested. The
preflop 169-class representation is unchanged, but full information-key IDs
are schema-specific and learned strategies differ. Likewise, the 43 flop
buckets in #139 are **observed types in its uniform-card sample**, not an
exhaustive schema-wide bucket limit.

A replication on #136’s larger, completed B100M/B500M LBR records would be a
useful next measurement, with Doctor Research’s artifact ownership respected.
Retained v2 quarter-checkpoint curves could test whether the negative gap
recovers with work; shrinking/flat gaps alone would not uniquely distinguish
dilution from representation quality. Its ~300-second final evaluation timing
was measured on Linux 64GB pods, so any M1 follow-up needs a memory/runtime
preflight and a separately frozen budget. No additional model evaluation,
training, M4 work or new abstraction was started for this review.

Reproduce this raw-only partition with a new output file:

```sh
python -m scripts.analyze_hu20_card_v2_lbr \
  --root docs/reports/hu20-card-v2-artifacts/production \
  --out results/card-v2-lbr-posthoc-reproduction.json
```

### Post-hoc flop fold/sizing screen

I checked the review’s new lead on the same saved records, without models or
new hands. The [fold-screen output](hu20-card-v2-artifacts/fold-posthoc.json)
pins the input and script hashes, includes all 13 panels, seed and visit splits,
and separates first bets, raise responses and stack-capped calls.

For each reached target decision facing a wager, the screening ratio is
`call_amount / current_pot`; the strategy measure is its recorded fold
probability, not an inferred probability from the selected action. For an
**uncapped first bet** of `b` into `P0`, this ratio equals `b/(P0+b)`, the
idealized zero-equity-bluff break-even fold frequency (one minus MDF).
For raises or capped calls it is only a sizing screen, not a general MDF
threshold. Even for first bets, averaging selected reached holdings does not
recover the defender’s full range or establish exploitability. All means below
weight decisions, including repeated decisions within a hand; they are not
independent samples or paired causal estimates.

| LBR flop subset | Decisions | Mean fold probability | Actual fold fraction | Mean call/pot screen |
| --- | ---: | ---: | ---: | ---: |
| v1, all | 273 | 50.1% | 50.5% | 37.2% |
| v1, uncapped first bets | 227 | 51.4% | 52.0% | 36.9% |
| v1, ≥1,000 visits | 124 | 63.4% | 63.7% | 33.8% |
| v2, all | 308 | 29.1% | 33.1% | 29.9% |
| v2, uncapped first bets | 215 | 28.4% | 30.7% | 31.4% |

V1’s all-decision gap is **+12.9 percentage points**, and the ≥1,000-visit
subset’s gap is **+29.6 points**. Its three lineage means are 53.5%, 48.0%
and 48.9%, versus screens of 36.5%, 38.8% and 36.0% (90/99/84 decisions).
This supports investigating folding at well-visited flop keys; a visit count
does not prove convergence. V2’s mean fold probability is 29.1%, versus a
29.9% screen, but its realized fold fraction is **33.1%**, not within one
percentage point of that mean.

LBR bets on 227/291 (78.0%) v1 flop first-bet opportunities, versus 215/254
(84.6%) for v2. **113/227 v1 bets are exactly pot-sized**; I do not describe
that as a majority. Counts use the executed amount, not a preset label.

Whole-hand returns split by ending clarify, but do not locate, the losses:

| Flop last betting action | v1 hands / net BB | v2 hands / net BB |
| --- | ---: | ---: |
| Target folds | 138 / −466 | 102 / −509 |
| Rival folds | 41 / +235 | 45 / +181 |
| Showdown (including earlier all-ins) | 72 / −240 | 101 / −260 |

The largest negative v1 flop-ending group is target folds, **but showdown
losses are material too**. V2’s turn-ending target folds net −248 BB in 31
hands. These remain whole-hand results; they are not the EV of folding,
calling or any individual bet.

The same all-decision screen on non-LBR panels does not show a uniform gap:

| Opponent | v1 decisions | v1 fold / screen | v2 decisions | v2 fold / screen |
| --- | ---: | ---: | ---: | ---: |
| loose_aggressive | 160 | 39.4% / 48.6% | 127 | 34.1% / 48.0% |
| loose_passive | 6 | 51.1% / 33.3% | 6 | 74.2% / 33.3% |
| minraise-cap2 | 1028 | 19.3% / 14.6% | 899 | 17.3% / 14.6% |
| native-pressure | 1105 | 18.2% / 14.8% | 963 | 18.6% / 15.5% |
| passive | 0 | — | 0 | — |
| pot_pressure | 78 | 28.3% / 56.7% | 77 | 27.4% / 56.3% |
| pressure-cap2 | 783 | 23.8% / 16.5% | 659 | 20.6% / 16.8% |
| selective-stackoff | 23 | 35.5% / 27.0% | 21 | 42.5% / 28.7% |
| tight_aggressive | 17 | 50.0% / 45.8% | 26 | 32.1% / 44.4% |
| tight_passive | 2 | 0.0% / 33.3% | 3 | 0.0% / 33.3% |
| train_pressure | 82 | 30.2% / 31.5% | 81 | 36.3% / 31.9% |
| uniform | 452 | 35.8% / 34.7% | 397 | 33.5% / 34.5% |

Uniform is close to its screen; pot-pressure and loose-aggressive are below
it, whereas min-raise/native-pressure are somewhat above. Several style
samples have fewer than 30 decisions and are particularly unstable. Different
opponents select different holdings, public histories and wager sizes; this
does not isolate a pot-bet-key defect or justify interpreting the gap as a
range-wide exploit estimate. In particular, LBR’s result cannot be generalized
to every opponent that bets pot.

My concrete lead is therefore **selected flop defence at well-visited v1
keys**, with preflop range quality, card aliasing, action/history encoding and
continuation values still possible contributors. #136’s larger completed
B100M/B500M records are the appropriate next replication, coordinated with
Doctor Research; no #136 records or M4 jobs were touched here. A future
flop-only representation experiment would require its own frozen design and
budget. Nothing here authorizes training or changes #144’s frozen descriptor.

Reproduce all screening and ending arithmetic with a fresh output path:

```sh
python -m scripts.analyze_hu20_fold_screen \
  --root docs/reports/hu20-card-v2-artifacts/production \
  --out results/card-v2-fold-posthoc-reproduction.json
```

## Training work, growth and resources

Linux production/evaluation source is **`aae1036a5cf2f3372a75a1f686832a303dddd84d`**. The later independent report reader/report commits do not alter that runtime. Frozen descriptor SHA-256 is `190d530ce66d65a031334d95600ce0f005d81324a7353170dcebf1d64a6ccd92`, schema `hu20-native-reopening-ordered-history-card-v2`. All workers use Python 3.11.15, Rust 1.90.0, NumPy 1.26.4, SciPy 1.17.1 and native-engine revision `5db20e3d5d6862b32a7402035c1340b622d3b005`; native binary and host provenance are pinned in each attempt/result.

| Seed | Actual nodes (overshoot) | Iterations | v1 keys / visits per key | v2 keys / visits per key | Training nodes/sec | Train seconds | With checkpoints/export seconds | Eval seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026093001 | 100,000,179 (179) | 290,133 | 1,496,914 / 13.486 | 10,556,005 / 1.947 | 16,632 | 6012.58 | 6551.97 | 307.13 |
| 2026093002 | 100,000,275 (275) | 288,986 | 1,494,799 / 13.511 | 10,515,605 / 1.955 | 18,763 | 5329.67 | 5795.54 | 279.15 |
| 2026093003 | 100,000,317 (317) | 288,149 | 1,499,679 / 13.445 | 10,517,259 / 1.955 | 17,770 | 5627.50 | 6124.79 | 301.25 |

Equal nodes is equal **traversal-work accounting**, not equal CPU seconds or equal convergence. Mature v1 baselines are existing pinned artifacts, not retrained concurrently for a host-throughput race. Training seconds exclude measured save/export/hash interruptions. Segment throughput and checkpoint timing are retained for capacity planning.

### Quarter-point growth

Each street cell is `keys / mean stored traverser visits per key`. Segment throughput covers the preceding quarter, excluding saves/hashes. Complete iterations cause small overshoots at each checkpoint.

| Seed | Actual nodes | All keys | Preflop | Flop | Turn | River | Segment nodes/sec | Save / hash seconds | Compressed checkpoint bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026093001 | 25,000,171 | 3,480,388 | 15,977 / 24.67 | 450,596 / 2.08 | 1,123,478 / 1.37 | 1,890,337 / 1.18 | 16,693 | 39.30 / 1.53 | 149,865,514 |
| 2026093001 | 50,000,007 | 6,180,769 | 17,119 / 43.25 | 722,181 / 2.59 | 1,978,291 / 1.57 | 3,463,178 / 1.30 | 16,587 | 70.58 / 2.70 | 266,240,032 |
| 2026093001 | 75,000,011 | 8,499,413 | 17,836 / 60.71 | 933,211 / 3.00 | 2,705,717 / 1.74 | 4,842,649 / 1.41 | 16,655 | 97.99 / 3.72 | 366,627,267 |
| 2026093001 | 100,000,179 | 10,556,005 | 18,183 / 77.10 | 1,101,352 / 3.36 | 3,345,006 / 1.88 | 6,091,464 / 1.50 | 16,592 | 121.67 / 4.61 | 456,194,800 |
| 2026093002 | 25,000,323 | 3,485,822 | 15,966 / 24.73 | 450,710 / 2.07 | 1,122,655 / 1.37 | 1,896,491 / 1.18 | 18,905 | 34.36 / 1.37 | 150,405,709 |
| 2026093002 | 50,000,104 | 6,194,442 | 17,023 / 43.29 | 719,521 / 2.60 | 1,983,741 / 1.57 | 3,474,157 / 1.30 | 18,946 | 61.96 / 2.43 | 267,334,085 |
| 2026093002 | 75,000,234 | 8,490,913 | 17,306 / 61.50 | 916,660 / 3.03 | 2,703,671 / 1.75 | 4,853,276 / 1.41 | 18,637 | 86.77 / 3.33 | 367,239,828 |
| 2026093002 | 100,000,275 | 10,515,605 | 17,490 / 78.90 | 1,077,101 / 3.40 | 3,330,238 / 1.90 | 6,090,776 / 1.51 | 18,569 | 107.85 / 4.12 | 455,744,927 |
| 2026093003 | 25,000,024 | 3,473,452 | 15,576 / 24.98 | 448,422 / 2.10 | 1,121,925 / 1.37 | 1,887,529 / 1.18 | 18,260 | 38.58 / 1.50 | 149,823,111 |
| 2026093003 | 50,000,487 | 6,157,545 | 16,613 / 43.40 | 710,435 / 2.62 | 1,971,936 / 1.59 | 3,458,561 / 1.31 | 17,149 | 61.57 / 2.44 | 265,922,661 |
| 2026093003 | 75,000,070 | 8,455,334 | 16,969 / 61.60 | 913,387 / 3.04 | 2,691,729 / 1.75 | 4,833,249 / 1.42 | 18,049 | 86.50 / 3.39 | 365,703,646 |
| 2026093003 | 100,000,317 | 10,517,259 | 17,510 / 78.46 | 1,082,716 / 3.40 | 3,334,028 / 1.90 | 6,083,005 / 1.51 | 17,662 | 116.79 / 4.40 | 455,502,557 |

### Memory and storage

Owned-process RSS is sampled once/second. The separate trainer `ru_maxrss` can retain shorter peaks; it is not perfectly synchronized with the window samples. Save/export windows come from explicit timestamps. “Training/hash” excludes those windows but includes streaming hashes; evaluation includes model/checkpoint loading, pilot and inference, not just per-action inference. Small host swap changes do not identify which process swapped.

| Seed | Training/hash sampled peak GiB | Save/export sampled peak GiB | Trainer high-water GiB | Evaluation sampled peak GiB | Host swap growth MiB | Lowest free disk GiB | Final export seconds | Export compressed bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026093001 | 6.902 | 10.768 | 11.145 | 10.717 | 0.246 | 36.31 | 137.41 | 277,804,601 |
| 2026093002 | 6.875 | 11.118 | 11.107 | 10.686 | 0.973 | 36.31 | 118.42 | 277,093,554 |
| 2026093003 | 6.896 | 10.538 | 11.109 | 10.733 | 0.984 | 36.31 | 131.11 | 277,066,138 |

No resource guard fired. Actual cgroup limit was 64,000,000,000 bytes per pod; owned RSS guard was the lesser of 45GiB and 80% of the limit, swap-growth ceiling 0.5GiB, free-disk floor 8GiB, entry ceiling 20M. Final checkpoint files retain real reach/iteration-weighted accumulators for later diagnostics; only the current exports enter this comparison.

### Projected versus actual

| Measure | Pretraining proposal | Measured result |
| --- | --- | --- |
| Keys at 100M | 14.11M power / 16.89M linear; review suggested possible 6.5–7M | 10.52–10.56M; lower than original projections, higher than the review calibration |
| Save/export memory | 23.66GiB extrapolated peak before headroom | 10.54–11.12GiB sampled owned save/export RSS; trainer high-water 11.11–11.15GiB |
| Training runtime | About 1.85h/seed, with slowdown/setup/retrieval reserve | 1.48–1.67h pure traversal; 1.61–1.82h including checkpoints/export |
| Rental ceiling | $8.55 nominal five-hour allocation, $10 hard total | $3.203879 conservative total including failed setup; billing not settled |
| Capacity | 20M entries, 64GB shape | 10.52–10.56M entries; every guard remains inactive |

Local 1M→2M growth was not a mature forecast. The realized 2M→100M v2 key-growth exponent is about 0.87, versus the early 0.944 estimate. The 64GB shape supplied safe headroom but no assumed single-worker speedup. These actuals are not authorization for another rental or extrapolation to 500M.

### Rental ledger and teardown

Each corrected pod was CPU5-memory (`cpu5m`), 8vCPU/64GB, one worker, no GPU, 40GB ephemeral disk. The returned hourly quote was $0.52; the conservative ledger uses $0.57/pod-hour to reserve disk charges. Elapsed hours below are estimated billable lifetime from request to successful DELETE, **not settled billed hours**. Separate storage/transfer charges were not provided by the returned API metadata and remain unverified. No retained billable volume is configured.

| Seed | Exact pod ID | Requested UTC | DELETE acknowledged UTC | Elapsed hours | Conservative upper USD |
| --- | --- | --- | --- | ---: | ---: |
| 2026093001 | `fg4o4yqg7rr8a9` | 2026-10-01T15:02:51+00:00 | 2026-10-01T17:00:53+00:00 | 1.9674 | 1.121394 |
| 2026093002 | `seqetu9zw59clr` | 2026-10-01T15:02:52+00:00 | 2026-10-01T16:47:57+00:00 | 1.7514 | 0.998272 |
| 2026093003 | `yj9yhawimwcn2t` | 2026-10-01T15:02:53+00:00 | 2026-10-01T16:53:42+00:00 | 1.8469 | 1.052710 |

The initial uv catalogue failure occurred before any tests, models or training and contributes $0.031504 conservative upper spend. It remains in the [setup-failure record](hu20-card-v2-artifacts/setup-attempt-1/manifest.json). The corrected attempt contributes $3.172375. Both use the original absolute cutoff; no clock reset or extra training budget occurred. [Lease/pod ledger](hu20-card-v2-artifacts/production/pods.json), [operator completion](hu20-card-v2-artifacts/production/operator-finished.json), [API absence check](hu20-card-v2-artifacts/production/api-absence.json) and the existing M4 coordination note record teardown. Controller/caffeinate/watchdog processes have exited.

## Artifact identity, availability and reproduction

| Seed | Current-export compressed SHA-256 | Compressed bytes | Final checkpoint SHA-256 |
| --- | --- | ---: | --- |
| 2026093001 | `6dd75ccffce785a3165e8d893231c46ea19e1c48c30d753275ed715680a11be1` | 277,804,601 | `0c933433a8246866452cc1a74989d1f33cfc5829262f0bf977955251af9fc26d` |
| 2026093002 | `d28da8f86347f4bfa843bbd5fe5402621e87d911d018a738140402e4bf94b7e7` | 277,093,554 | `f71e2fe25fa7fa2ab189ce667a240bc6770eee33730579dad3d7ed9aa83a5a60` |
| 2026093003 | `bb8df2bdbc8bf12e231b7107f31049316c898ddc83130e81f5ec747bab9035b0` | 277,066,138 | `26e8a164423a54dfe2b643df40a269ca1c7e92f1687ab64f12850d0dda0a8cc8` |

All compressed/uncompressed sizes and hashes, 25/50/75/100M checkpoint hashes, baseline identities and engine origins are retained in each [production lineage directory](hu20-card-v2-artifacts/production/). These are inference **current** exports with v2 schema, not replacements for the v0.4 B100M v1 model. Full resumable checkpoints, current exports, original iteration logs and transported archives are retained outside Git under `results/card-v2-rental-20261001-repair/{seed}/results/work/training` in the separate experiment worktree. They are locally retained research artifacts, **not public model-download assets**; I can supply exact hash-identified files for a separately authorized follow-up. The PR includes compact metadata, raw generated hands and compressed resource logs, not large binaries or credentials.

The [public manifest](hu20-card-v2-artifacts/production/public-manifest.json) pins the copied bytes. Each original work manifest also indexes omitted checkpoints/logs, so an indexed original file is not a promise that it is in Git. Resource JSONL copies are compressed without changing their uncompressed content. Model/export bytes are never recompressed. Raw hands are generated simulator records, not private human sessions. `observation.card_bucket` retains the legacy v1 descriptor for comparison; the actual `target_abstraction`, `target_key` and visit count identify the real arm. Recompute the v2 descriptor from its own cards/current board and the frozen implementation.

### Validation scope

- 29 focused checks pass on M1 and separately on every Linux pod: descriptor collisions/refinement/suit invariance/information isolation, unchanged menu/history, explicit artifact schema rejection, current-export guard, generated native replay, recovery/cutoffs/archive checks and independent metric arithmetic.
- Every pod passed full-payload final/current/next-iteration comparison against the 100k M1 reference and its fresh-process midpoint resume before production. Linux direct/resumed files are byte-identical; M1/Linux uncompressed payloads are identical. Current-export gzip OS headers differ across hosts, recorded separately. This validates the bounded reference, not a universal cross-host floating-point claim.
- Every loaded policy/checkpoint pair was hash-verified and all current probabilities checked against its saved regrets. All 38,400 final hands passed native action/event/payoff replay on Linux. The independent streaming M1 reader rejects duplicate/missing/unpaired coordinates and exactly reproduces every per-seed panel/tail summary.
- Full CI at `0998146` passed before final evidence; final evidence changes only documentation/artifacts. The actual frozen trainer/evaluator remains `aae1036`. No runtime change after the real comparisons.

For model-free reproduction, use the pinned Python/native dependency environment and the [protocol commands](../hu20-card-v2-protocol.md#reproduction-commands). Recompute published raw arithmetic without loading a model:

```sh
python - <<'PY'
import json
from pathlib import Path
from scripts.summarize_hu20_cards_v2 import summarize_records
root = Path('docs/reports/hu20-card-v2-artifacts/production')
saved = json.loads((root / 'independent-summary.json').read_text())
for seed, expected in saved['per_seed'].items():
    actual = summarize_records(root / seed / 'evaluation/hands.jsonl.gz')
    assert actual['panels'] == expected['panels']
    assert actual['raw_sha256'] == expected['raw_sha256']
print('All published paired/tail/visit arithmetic matches')
PY
```

Replay any generated raw row with `replay_row(row)` from `scripts/play_robustness.py`; no model is required. Re-running training/evaluation needs its retained hash-pinned binaries/environment, the frozen run plan and a new independently approved compute allocation. Do not silently reuse this exhausted rental approval.

## Decision and unresolved questions

I keep v1 as the playable default and retain v2 as a negative fixed-budget experiment. For v2, the immediate leads are sparse postflop coverage, fallback exposure and the sharp full-stack-loss increase against min-raise, alongside unchanged remaining kicker collisions. For v1’s residual LBR weakness, the separately verified post-hoc early betting-stop groups at well-visited keys raise preflop/flop strategy and commitment as leads; they do not locate individual mistakes. It remains unresolved whether a more economical representation, history compression or additional convergence could recover the useful distinctions. Those are different experiments, not changes or extra training folded into this result. #144 can test its separately frozen history interaction; this task makes no causal claim from that unfinished campaign.

---

## Pre-training record (retained chronology)

The sections below record the prospective gates, original projections and progress checkpoints. Statements that execution/results were pending describe those earlier checkpoints; the completed results above supersede that status.
I refine only the postflop private-card representation over merged #139/#142.
The [prospective protocol](../hu20-card-v2-protocol.md) and
[resource plans](../../configs/diagnostics/hu20-card-v2-preflight-v2.json)
precede any strength outcomes. Draft #143 is separate from active #136.

## Model-free collision and growth gate

All retained #139 files pass their byte/hash checks. No model is loaded and
no equity is re-estimated. On the same 3,072 uniform-card holdings, all five
highlighted concrete same-board failures separate. V2 retains v1 as a prefix:
old buckets can split, but cannot merge. Suit and card-order invariance pass.

| Street | Sample holdings | v1 buckets/keys | v2 buckets/keys | Same-board different-made-value pairs separated / old pairs |
| --- | ---: | ---: | ---: | ---: |
| flop | 1024 | 43 | 508 | 2056/2246 |
| turn | 1024 | 69 | 609 | 1219/1351 |
| river | 1024 | 35 | 519 | 1654/1783 |

**4,929/5,380 (91.62%)** sampled pairs with different made-hand values separate;
**451 residual pairs remain**. These are pair counts clustered within boards,
not independent examples. The largest residual retained river collision has
uniform-equity spread .1818 (8 versus T as third kicker in the same band).
Splitting measured collisions is a representation check, not proof of strength
or a causal explanation of historical losses.

Key counts use one identical check-through public/menu template per street,
varying only private cards/current board; therefore they match descriptor
counts in this audit. This is not reached-state occupancy or a mature-table
projection. [All row descriptors/keys](hu20-card-v2-artifacts/collision-audit/descriptors.jsonl),
[full audit and residual collisions](hu20-card-v2-artifacts/collision-audit/summary.json)
retain the source hash and every distinction. Audit: **0.56 seconds / 56.7 MiB**.

## Sequential M1 resource prefixes

No playing outcomes. Both use the retained seed 2026093001 and original recipe:
K1, uncapped native menu, unchanged history, 3M safety entry limit, 250k
per-iteration node limit and 300-second iteration watchdog. Each path starts
from zero and stops at the first complete iteration crossing each node target.

| Representation | Complete nodes | Iterations | Entries | Mean visits/key | Training nodes/sec | Training seconds | Whole prefix seconds | Peak with save/export |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 2,000,143 | 5,549 | 193,295 | 2.155 | 14,945 | 133.83 | 138.60 | 291.20 MiB |
| v2 | 2,000,268 | 5,070 | 351,592 | 1.147 | 15,055 | 132.86 | 140.78 | 504.34 MiB |

| Representation | Preflop keys | Flop keys | Turn keys | River keys |
| --- | ---: | ---: | ---: | ---: |
| v1 | 9,317 | 15,152 | 59,466 | 109,360 |
| v2 | 9,312 | 57,062 | 112,420 | 172,798 |

V2 has **1.819×** as many entries at this prefix and mean visits/key falls
2.155→1.147. Actual nodes/iterations differ slightly because new postflop
strategies change later sampled paths; both use the same completed-node rule,
not the same iteration count. The coarse preflop partition is unchanged.
Equal nodes is equal traversal-work accounting, not equal convergence or
identical CPU seconds. Training coverage/fallback counters and all four
milestones are in [v1](hu20-card-v2-artifacts/preflight-v1/summary.json) and
[v2](hu20-card-v2-artifacts/preflight-v2/summary.json); full iteration logs are compressed without changing their contents. The retained
per-prefix manifests pin their original bytes; [transport hashes](hu20-card-v2-artifacts/iteration-log-transport.json)
pin both the original and compressed logs.
M1 macOS arm64, Python 3.11.15, pinned native engine; both workers exited.

## Resource proposal before approval — no paid run at that checkpoint

The early 1M→2M v2 entry-growth exponent is **0.944**. Extrapolating that
local curve gives **14.11M keys** at 100M; holding the last absolute growth
rate gives **16.89M**. Neither is a mature-table forecast: saturation
can reduce growth, different seeds/strategies can increase it. The retained v1
3M safety limit is therefore a material risk; I do not silently raise it or
coarsen v2. Scaling observed save/export peak by the linear entry estimate gives
about **23.66 GiB**,
before headroom.

The [exact resource proposal](../../configs/diagnostics/hu20-card-v2-resource-proposal.json)
requests a **20M safety entry limit**, **three CPU5-memory 8-vCPU/64-GB pods**,
one worker/pod, **5-hour absolute rental cutoff**, **$10 total cap**. The live
catalog compute rate is $0.52/h/pod; total admitted rate must be ≤$0.57/h/pod
including disk. Three full five-hour allocations at that ceiling cost $8.55,
leaving reserve. Extra vCPUs buy the memory shape; no single-worker speedup is
assumed. Prefix timing projects 1.85 training hours/seed, with 1.5× slowdown
and 75 minutes setup/recovery/save/retrieval reserve below the cutoff.

This changes only an inactive abort threshold, not regret math, sampling,
actions, history or the 100M stopping rule. Approval is needed because the
original recipe's safety capacity and the resource/cost allocation differ.
Stop on cap/RSS/swap/disk/time failure; preserve partials. No silent coarsening,
extra nodes, paid follow-on or model promotion. Linux parity/recovery and the
final paired panel budget were pending at this proposal checkpoint; the later
approved freeze below records their resolution.

## Validation and remaining work

24 focused descriptor/existing HU20 tests pass: the five concrete collisions,
card order/suit permutation, hidden-world observation isolation, own-card/
draw features, unchanged menu/history/seeds, artifact schema rejection,
current-export guard and next-iteration checkpoint recovery. The new schema
is isolated; production defaults and #142's river player stay unchanged.
CI is tracked in draft #143. No policy-strength results have been opened.

I notified Doctor Research and appended to the existing M4 coordination note.
That was a brief network-only coordination action; no M4 compute, allocation,
campaign-file changes or extra transfers occurred. I use already cached M1
baseline artifacts, separate RunPod ownership and cost accounting. #136 stays
untouched.

## Approved execution freeze (October1, before rentals)

I approved three CPU5 8vCPU/64GB pods, one worker each, $10 total /five hours
maximum per pod and a20M entry safety ceiling. The descriptor and scientific
100M/seed budget remain frozen. The [run plan](../../configs/diagnostics/hu20-card-v2-run.json)
and [protocol](../hu20-card-v2-protocol.md#approved-rental-and-evaluation-freeze)
pin the recovery, resource, checkpoint and paired-evaluation rules. The
additional generated-fixture campaign tests exercise native replay, v1/v2
schema isolation, exact tail/paired arithmetic, outcome-blind timing and
archive verification. No paid rental or strength result has run at this
checkpoint; the final evidence will report actual admission, failures and cost.

### Independent pretraining review

Claude reviewed descriptor/schema/audit at12dd2ff and found no correctness
problem. Its useful capacity calibration is that v1's own early exponent
overpredicted its actual100M key count by about2×; equivalent saturation is
not guaranteed for v2. I retain the conservative approved shape and will
compare projections to actual25/50/100M growth, late throughput, separately
identified serialization memory windows, disk/swap and per-pod cost ledger.
The final decision records also permit per-street visit bands0/<10/<100,
so denser card information can be read alongside sparser visits. The unchanged
small LBR budget is exploratory, not a high-power primary test. A null/negative
v2 result cannot establish that card resolution is irrelevant. #144's separate
four-cell interaction requires its own prospective schedule; I will not pool
this two-cell experiment into it without that common evaluation.

## First rental attempt: setup failure retained

At14:54:55–59UTC three approved CPU5-memory pods were created. uv0.8.22
could not resolve the frozen Python3.11.15 Linux download; all three exited
before tests, recovery, model loading or training. Their archives were
transport-hash checked and retained; no work manifest exists because no
worker ran. Teardown was API-confirmed at14:56:06UTC. Conservative
compute+disk upper spend is$0.0315037; settled billing remains pending.
[Raw setup logs and lease ledger](hu20-card-v2-artifacts/setup-attempt-1/manifest.json)
retain the original source6fbffeb, exact pod IDs/names and failure.

The corrected pinned uv0.12.21 catalogue includes Python3.11.15 for Linux
x86_64; the versioned public mirror is checked locally before another
rental. Descriptor and100M scientific budget stay frozen. A manual repair
uses the original19:54:53UTC rental cutoff and includes failed-attempt spend
within the same$10 cap, rather than resetting the clock.

## Production25M resource checkpoint — before playing outcomes

All three corrected Linux pods passed29focused checks and exact M1/Linux
plus fresh-process recovery gates before production. Sourceaae1036; the
new independent report reader is separate from that frozen trainer runtime.
No playing outcomes are opened here. Full CI also passed at699f332.

| Seed | Completed nodes | Keys | Training nodes/sec | Checkpoint save seconds | Compressed bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026093001 | 25,000,171 | 3,480,388 | 16,693 | 39.30 | 149,865,514 |
| 2026093002 | 25,000,323 | 3,485,822 | 18,905 | 34.36 | 150,405,709 |
| 2026093003 | 25,000,024 | 3,473,452 | 18,260 | 38.58 | 149,823,111 |

[Quarter-point metadata and complete checkpoint hashes](hu20-card-v2-artifacts/quarter-25m/)
retain per-street key/visit/coverage counts. The20M safety ceiling remains
inactive; the old3M ceiling would already have stopped each lineage. This
is capacity evidence, not a strength result. Final100M exports, comparisons
and final resource/cost audit remain pending.
