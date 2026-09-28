# HU20 capped versus native-reopening: completed M4 comparison

## Result and decision

**The uncapped candidate passes both prespecified aggregate tests.** Against the unchanged native-pressure rule, its target profit changes from **−254.75 to +102.20 BB/100**, a paired **+356.94 [97.5% interval +337.01, +376.87]**. The gain is positive for every seed and both roles.

Against the common original-cap2 LBR, B−A is **+34.39 [−5.55, +74.33] BB/100**. Its lower bound exceeds the frozen **−10-BB/100** material-regression margin, so aggregate non-inferiority passes. The interval includes zero: improved LBR resistance is **not** established. B still loses **−95.88 BB/100 [−131.07, −60.69]** to this limited responder. Per-seed and big-blind LBR intervals remain much wider than the pooled safeguard.

This is a successful repair of the measured native-pressure weakness under the fixed attacker class. It does not establish broad poker competence, exact exploitability, professional strength, or multiplayer improvement. Capped-minraise and passive-control regressions remain. Draft PR #115 remains open; no model is promoted.

## Frozen experiment and completed work

Three paired fresh seeds × two arms, equal 20M completed-node budgets, K1 external sampling, unchanged card/history features and min/pot/conditional-jam sizes, final-current C, no free fold. A keeps cap2; B uses `raise_cap=None`, subject to native reopening. The game/menu/artifact identity changes explicitly; existing checkpoint behavior remains intact. Pair designation does not imply identical traversal streams after the trees diverge.

All six runs finished from zero, recording 24 fixed 2M/5M/10M/20M checkpoint/export pairs. Actual work: **120,001,431 nodes**, **1,431 whole-iteration overshoot nodes**, **367,998 completed outer iterations**, and no discarded main iteration or failed training attempt. All six separate-seed 1M-node preflights passed; their deliberate cancellation/recovery work is retained separately.

The frozen evaluation completed **177 panels and 1,148,928 hands**: 1,105,920 cheap stress/control hands, 6,144 LBR hands and 36,864 secondary-panel hands. Native replay additionally verified all **240 outcome-suppressed preflight timing hands**. The candidate human smoke completed and replayed **20** additional hands. There was no failed or omitted evaluation hand.

## Primary realized returns

All amounts below are target profit. Attacker profit is its exact negative in this no-rake zero-sum game. BB/hand is BB/100 divided by 100; 20BB buy-ins/100 is BB/100 divided by 20. Native B earns +1.022 BB/hand in this panel; LBR B loses −0.959 BB/hand. Native improvement is +17.85 20BB buy-ins/100, not a generic full-game win rate.

| Fixed attacker | A target BB/100 | B target BB/100 | B−A BB/100 | 97.5% paired interval |
| --- | ---: | ---: | ---: | --- |
| Pressure-native | -254.75 | +102.20 | +356.94 | [+337.01, +376.87] |
| LBR-original-cap2 | -130.27 | -95.88 | +34.39 | [-5.55, +74.33] |

Native-pressure absolute 97.5% intervals: A **[−269.74, −239.75]**, B **[+87.62, +116.77]**. LBR absolute intervals: A **[−164.34, −96.21]**, B **[−131.07, −60.69]**. The two claims use prespecified two-sided 97.5% intervals. Each independent deal block averages both seat rotations and the three seed-paired contrasts before uncertainty is computed; policies sharing a deal are not independent samples.

| Seed | Native-pressure B−A | 97.5% interval | LBR B−A | 97.5% interval |
| --- | ---: | --- | ---: | --- |
| 2026093001 | +365.22 | [+336.65, +393.80] | +54.49 | [-14.37, +123.35] |
| 2026093002 | +349.37 | [+320.35, +378.38] | +2.93 | [-64.30, +70.16] |
| 2026093003 | +356.24 | [+326.93, +385.54] | +45.75 | [-17.31, +108.82] |

Seed intervals are retained diagnostics; the non-inferiority claim is the prespecified aggregate, not a guarantee for every trained seed. All three B policies have negative realized LBR profit, and each per-policy 95% interval excludes zero. The realized return comes from played hands against the fixed target, not the maximum of the responder's noisy internal estimates.

### Role-specific primary results

Units: BB/100. Role-specific intervals are descriptive alongside the role-balanced primary.

| Role / attacker | A target | B target | B−A | 97.5% paired interval |
| --- | ---: | ---: | ---: | --- |
| big_blind / LBR-original-cap2 | -167.45 | -143.16 | +24.28 | [-33.50, +82.06] |
| big_blind / Pressure-native | -281.75 | +72.54 | +354.30 | [+320.75, +387.84] |
| button_small_blind / LBR-original-cap2 | -93.10 | -48.60 | +44.50 | [-8.36, +97.36] |
| button_small_blind / Pressure-native | -227.74 | +131.85 | +359.59 | [+329.76, +389.41] |

The big-blind LBR contrast does not establish the −10 margin separately. Its absolute B loss is substantial. No role-specific robustness certificate is claimed.

## Controls, historical references and secondary panel

The original-cap2 attackers use the same old attacker menu for both targets. B keeps its own expanded target menu. Native minraise pressure remains a separate contract. No attacker was selected by its observed profit.

| Control | A target | B target | B−A | Exploratory 95% interval |
| --- | ---: | ---: | ---: | --- |
| Pressure-original-cap2 | +31.32 | +28.80 | -2.51 | [-14.25, +9.22] |
| Minraise-original-cap2 | +92.60 | +60.48 | -32.12 | [-45.57, -18.68] |
| Minraise-native | -240.62 | +64.02 | +304.64 | [+285.83, +323.45] |
| Passive | +90.84 | +77.94 | -12.90 | [-21.56, -4.25] |

The negative capped-minraise and passive contrasts are retained rather than hidden behind the primary win. B still has positive absolute returns against those controls; that does not remove the relative regressions. These contrasts and checkpoint curves use exploratory 95% intervals without confirmatory multiple-comparison claims.

The original six-opponent secondary schedule uses 512 blocks per final policy/opponent:

| Control | A target | B target | B−A | Exploratory 95% interval |
| --- | ---: | ---: | ---: | --- |
| loose_passive | +7.03 | +7.24 | +0.21 | [-17.30, +17.72] |
| loose_aggressive | -22.97 | -9.16 | +13.80 | [-9.48, +37.09] |
| tight_passive | +40.23 | +39.91 | -0.33 | [-6.38, +5.73] |
| tight_aggressive | +46.11 | +61.44 | +15.33 | [+3.17, +27.49] |
| pot_pressure | +5.24 | +3.76 | -1.48 | [-17.64, +14.68] |
| hu20_uniform | +139.32 | +161.26 | +21.94 | [-9.47, +53.35] |

B's mean loose-aggressive return remains negative; its point estimate and uncertainty are preserved in the artifact summary. The three original #112 final checkpoints were evaluated unchanged on all five cheap contracts. Their per-policy results and hashes are retained; they are historical references, not substitutes for the fresh A controls. No new untrained policy or trained-minus-uniform contrast was added to the frozen A/B design.

### Fixed checkpoint curve

| Complete-node milestone | Native-pressure B−A | Exploratory 95% interval |
| --- | ---: | --- |
| 2M | +308.61 | [+290.26, +326.96] |
| 5M | +339.84 | [+322.20, +357.48] |
| 10M | +370.68 | [+353.46, +387.91] |
| 20M | +356.94 | [+339.52, +374.37] |

The final checkpoint is reported despite the larger 10M paired point estimate. This is not proof of a 10M optimum or a demonstrated 10M→20M regression; the curve reuses paired deals and is exploratory. Intermediate LBR checkpoints were not part of the frozen schedule, so no LBR learning curve is inferred.

## Action boundaries, fallbacks and density

Across final native-pressure trajectories, A has **26,084 target decisions after original off-menu history, all fallback**. B has **21,097** such decisions, with **19,961 trained and 1,136 fallback** (94.62% hits). Native-pressure target hit rate rises from **66.35% to 98.32%**. The paths and action choices differ, so this is exposure telemetry, not an isolated causal estimate for fallback reduction.

Native pressure made 14,721 A-facing and 14,956 B-facing actions outside original cap2. Every B-facing pressure action was on B's own menu. B itself made 13,446 target actions outside original cap2, all on its own menu. During LBR play, all responder actions remain on original cap2; B made 287 target actions outside that old menu, and all were on its own menu. Actual B-action likelihoods still enter the Bayesian updates.

| Suite / arm | Street | Trained decisions | Fallback decisions | Hit rate |
| --- | --- | ---: | ---: | ---: |
| Pressure-native / A | preflop | 33,141 | 8,024 | 80.51% |
| Pressure-native / A | flop | 12,012 | 10,414 | 53.56% |
| Pressure-native / A | turn | 4,542 | 5,298 | 46.16% |
| Pressure-native / A | river | 1,896 | 2,429 | 43.84% |
| Pressure-native / B | preflop | 42,478 | 256 | 99.40% |
| Pressure-native / B | flop | 18,055 | 250 | 98.63% |
| Pressure-native / B | turn | 7,067 | 463 | 93.85% |
| Pressure-native / B | river | 2,124 | 222 | 90.54% |
| LBR-original-cap2 / A | preflop | 3,585 | 0 | 100.00% |
| LBR-original-cap2 / A | flop | 2,620 | 4 | 99.85% |
| LBR-original-cap2 / A | turn | 947 | 17 | 98.24% |
| LBR-original-cap2 / A | river | 344 | 8 | 97.73% |
| LBR-original-cap2 / B | preflop | 3,715 | 0 | 100.00% |
| LBR-original-cap2 / B | flop | 2,408 | 12 | 99.50% |
| LBR-original-cap2 / B | turn | 760 | 25 | 96.82% |
| LBR-original-cap2 / B | river | 229 | 15 | 93.85% |

Per-policy folds, checks, calls and raises by street, both action memberships, prior-history flags and decision-weighted visits are retained in the [compact summary](hu20-native-reopening-m4-artifacts/summary.json) and [suite telemetry](hu20-native-reopening-m4-artifacts/diagnostics.json). Raw visit histograms and every hand remain on M4. Exposure does not assign an entire terminal loss to one street or prove a sizing error from the realized hidden deal.

The same **4,248 frozen independent public observations** were replayed for both schemas. See [all-six training measurements](hu20-native-reopening-training.md): B's flop coverage is 99.57–99.74% versus A's 75.54–75.80%; turn coverage is 81.52–88.64% versus 35.18–35.28%; river coverage is 37.82–54.98% versus 31.25–31.36%. The set deliberately includes repeated raises and is not a whole-game distribution. B has more entries but lower mean visit counts on this same set.

Own LBR trajectories are different: river hit rates are **97.73% A / 93.85% B**, with mean visits **71.7 / 41.2**. Own native-pressure river hit rates are **43.84% / 90.54%**. Independent coverage and self-selected trajectory coverage are distinct diagnostics; no intersection of incompatible key hashes was used.

### Cost of expanded training

The ≥2-raise columns include decisions responding to the second raise in both arms. The >2 columns explicitly count decisions beyond the old boundary. Revisit counts are repeated updates, not distinct keys.

| Run | New / revisited key visits at ≥2 raises | Decision visits at >2 raises | Regret-update visits at >2 raises | Outer p99 / max seconds |
| --- | ---: | ---: | ---: | ---: |
| A-2026093001 | 106,460 / 1,228,137 | 0 | 0 | 0.042 / 0.173 |
| A-2026093002 | 106,574 / 1,228,842 | 0 | 0 | 0.042 / 0.187 |
| A-2026093003 | 106,661 / 1,222,150 | 0 | 0 | 0.041 / 0.192 |
| B-2026093001 | 385,005 / 1,402,348 | 2,347,515 | 924,366 | 0.057 / 0.385 |
| B-2026093002 | 385,756 / 1,412,552 | 2,361,285 | 929,845 | 0.057 / 0.369 |
| B-2026093003 | 382,386 / 1,409,996 | 2,370,016 | 934,038 | 0.057 / 0.360 |

A creates about 271k entries; B about 723k–730k. B spends roughly 2.35M–2.37M decision visits per seed after more than two raises, including 0.92M–0.93M regret-update visits. Fewer completed outer iterations and broader, less dense representation accompany the pressure gain; the experiment jointly changes training exposure and available target actions through one cap setting. It does not identify fewer misses as the sole cause.

## LBR completion and resources

All **14,057 LBR decisions completed**, with zero soft-budget overruns and zero logged zero-likelihood events. Maximum solver decision time was **0.602 seconds** under the unchanged five-soft-second/four-sample contract. The approximation remains one-step/checkdown with sampled future chance; a small return would not certify low exploitability. HU LBR says nothing about TP worst-case quality.

| Suite / arm | Target decisions | Target mean / max ms | LBR mean / max seconds |
| --- | ---: | ---: | ---: |
| Pressure-native / A | 77,756 | 0.035 / 0.680 | — |
| Pressure-native / B | 70,915 | 0.033 / 0.927 | — |
| LBR-original-cap2 / A | 7,525 | 0.039 / 0.704 | 0.252 / 0.539 |
| LBR-original-cap2 / B | 7,164 | 0.038 / 0.436 | 0.248 / 0.602 |

The action timers cover the instrumented lookup/sampling/key path or responder `choose_action`; initial menu preparation and native hand transitions are outside those timers. Whole-phase wall measurements include that work.

The measured campaign finished at **2026-09-28T22:36:08.217220+00:00** (00:36 Madrid), **3.97 hours** after preflight began, before the unchanged 06:37 Madrid deadline. Six training runs cost 20.7–21.9 minutes each; evaluation cost 71.1 minutes; native audit 12.3 minutes; the independent arithmetic pass 19.2 seconds. Peak process RSS was **2.16 GiB**, minimum free disk **41.67 GiB**, and swap stayed at 761.38 MiB. All RSS/disk/swap/time guards pass. Single-process observed training throughput is roughly 15–16k completed nodes/second including recorded overhead; no parallel-scaling claim is made.

Report postprocessing and publication also finish under the original deadline. Nothing was run to consume spare hours.

## Audit, lineage and retention

- All **216** final inventory files match their recorded bytes and SHA-256; no unlisted root file or failed campaign attempt was found. There are **16** completed supervisor attempts (six preflight, six training, evaluation, audit, independent statistics and human smoke). All **177** evaluation panel attempts completed.
- The native audit checked actor order, legality, chip conservation, exact terminal event digests, target-policy RNG replay, trained/fallback flags, own/original action memberships and visit counts for every hand. It matched every checkpoint's current-regret probabilities to its exported profile across all fixed milestones.
- Pairing and two rotations were checked, with independent target/responder action streams. Fresh evaluation and independent diagnostic deals have zero overlap with main/preflight training and 84,120 prior opened deal seeds across 255 hand files. All 4,248 independent observations replay exactly.
- Original preflight checksums, supervisor record and resource log are preserved. Deliberate two-node cancellations and byte-identical next-iteration recovery are retained, without publishing partial regrets or counting that validation work as main training.
- A second arithmetic implementation reproduces both primary means and intervals to 1e−8. Its raw-hand SHA-256 matches the final inventory. The compiled native engine hash matches the preflight identity; M4 source remains clean at `a87e9f8805d211e2b21dbb710339ada9083eedf6`.
- Old HU20 and both TP20 demo artifact hashes match their published model cards. Their interfaces and historical reports remain intact. The new 20-hand human smoke has **72 trained / 1 fallback bot decisions**, and all histories replay. The transcript exposes bot cards only through legitimate settlement disclosures; it is a usability check, not a strength comparison.

The [manifest](hu20-native-reopening-m4-artifacts/manifest.json), [post-stop verification](hu20-native-reopening-m4-artifacts/verification.json), [independent statistics](hu20-native-reopening-m4-artifacts/independent-summary.json), [frozen plan](hu20-native-reopening-m4-artifacts/frozen-plan.json), [model identities](hu20-native-reopening-m4-artifacts/models.json), compact results and 12 fixed-order repeated-raise replays are committed. Those replay cases are exploratory and were selected by fixed order/history, not profit. The original full report and raw histograms remain on M4; the committed summary identifies its exact source hash and transformation.

Raw artifact root on `ssh m4`:
`/Users/dberweger/Local/hu20-native-reopening-ab/results/hu20-native-reopening-m4-20260928`.
The original directory totals 1.72 GiB. Retrieve the entire directory and final inventory, or only named files:

```sh
scp -r m4:/Users/dberweger/Local/hu20-native-reopening-ab/results/hu20-native-reopening-m4-20260928 ./
scp m4:/Users/dberweger/Local/hu20-native-reopening-ab/results/hu20-native-reopening-m4-20260928-manifest.json ./
scp m4:/Users/dberweger/Local/hu20-native-reopening-campaign-20260928.log ./
```

Verification, diagnostics and preserved-demo postprocessing records are siblings ending `-verification.json`, `-diagnostics.json` and `-preserved-demos.json`; their methods and copies are committed. Source validation remains **856 full-suite tests**, **22 focused tests**, and passing CI on `3803459`; this final publication changes reports/model-card documentation only.

## Play and one next recommendation

The [candidate model card](../hu20-native-reopening-model-card.md) provides hash-pinned play/replay commands for **B seed 2026093001**, selected by fixed order, not profit. Missing learned entries remain explicitly counted uniform fallback on B's own menu.

**One proposed next experiment:** a separately authorized, bounded **HU20 same-recipe work-scaling confirmation of uncapped B**, preserving these saved 20M A/B policies as baselines and using fresh confirmation deals. Retain native pressure, original-cap2 LBR and both regressing controls; measure growth before choosing the work budget. This result supports retaining uncapped B as an experimental candidate, while the residual LBR losses and control regressions set quality objectives for the next comparison. It does not justify card changes by exclusion or a multiplayer jump. No new campaign, paid host, merge or default promotion is launched.
