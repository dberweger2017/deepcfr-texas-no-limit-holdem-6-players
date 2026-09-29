# Completion of the frozen HU20 scaling diagnostics

The [M4-only recovery](hu20-scaling-m4-recovery.md) reached all three 100M-node checkpoints and completed the two prespecified primary comparisons, but its original ten-hour window left 99 diagnostic panels pending. At the owner's request, a **separate, bounded M4-only evaluation** ran those exact pending panels without training or changing any saved policy, opponent rule, deal root, block count, or scientific source. The old attempt and deadline remain sealed. The diagnostic runner was frozen against source `874ba641fc6110a2d0998af986608633abef8898`; [freeze.json](hu20-scaling-diagnostics-artifacts/freeze.json) records all 15 policy hashes and the parent plan and manifest hashes.

**Outcome:** all 99 diagnostic panels and 534,528 hands completed. Combined with the unchanged 73,728 primary hands, #116 now has **111 panels and 608,256 natively replayed hands**, with no pending panel or audit failure. The primary 97.5% results are byte-for-byte unchanged: 100M minus each lineage's 20M policy is **+25.94 [7.43, 44.45] BB/100** against original-cap2 bounded LBR and **+19.39 [2.86, 35.93]** against native pressure. Both prespecified aggregate gates pass. The 100M target still loses **−73.14 BB/100** to that bounded LBR; no robustness or player-strength promotion follows.

## Controls and checkpoint curve

All entries below are **target BB/100** against fixed opponents. The same three saved lineages and paired deal/rotation blocks are used at 20M, 40M, 80M, and 100M. The contrast averages the three seed differences inside each block; its two-sided 95% interval is **exploratory and unadjusted** for the many comparisons. The 40M/80M results are playing diagnostics against the listed rules, **not intermediate LBR measurements**.

| Fixed rule | B20M | B40M | B80M | B100M | B100M − own B20M, 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original-cap2 minraise/check-call | +49.73 | +81.40 | +78.66 | +97.09 | **+47.36 [34.47, 60.24]** |
| Native-legal minraise/check-call | +48.19 | +72.70 | +75.12 | +92.25 | **+44.06 [30.51, 57.62]** |
| Passive check-call | +89.87 | +94.07 | +107.97 | +103.49 | **+13.63 [4.43, 22.82]** |
| Original-cap2 pressure | +30.16 | +33.10 | +34.46 | +39.15 | +8.99 [−1.59, 19.58] |
| Native-legal pressure | +106.70 | +113.35 | +131.70 | +126.09 | **+19.39 [4.94, 33.85]** |

The capped A20M references on these same schedules earned +78.04 BB/100 against original-cap2 minraise, +93.02 against passive, and −252.14 against native pressure. B100M is above those A20M control point estimates, although this report does not claim a new cross-arm interval. The losses against the two original controls reported for B20M in #115 have recovered by B100M **relative to B's own 20M policies**. The curve is uneven: passive and native-pressure point estimates peak at 80M, while minraise performance rises further at 100M. It does not support extrapolating another fivefold gain.

The minraise improvement appears in all three lineages: original-cap2 changes are +45.38, +45.31, and +51.37 BB/100; native-legal changes are +43.52, +35.40, and +53.27. Passive changes are +5.98, +43.02, and **−8.11**. Against the original-cap2 pressure rule, one lineage changes by −3.06. These exploratory seed effects prevent a claim that every saved policy improved against every rule. Per-policy position results and intervals are retained in [combined-results.json](hu20-scaling-diagnostics-artifacts/combined-results.json).

## Broader fixed opponents

The six secondary rules had only **256 paired rotation blocks per policy**. Their 95% intervals are exploratory and often wide.

| Opponent | B20M BB/100 | B100M BB/100 | B100M − own B20M, 95% interval |
| --- | ---: | ---: | ---: |
| HU20 uniform | +151.99 | +124.41 | −27.57 [−67.49, 12.35] |
| Loose aggressive | +3.48 | +32.10 | +28.61 [−1.32, 58.55] |
| Loose passive | +33.33 | +20.64 | −12.70 [−35.97, 10.58] |
| Pot pressure | +15.62 | **+3.29** | −12.34 [−30.59, 5.91] |
| Tight aggressive | +8.24 | +33.14 | +24.90 [9.56, 40.24] |
| Tight passive | +32.16 | +38.70 | +6.54 [−4.08, 17.17] |

Pot pressure is a particularly weak **absolute** result: seed 1's B100M estimate is −10.94 BB/100, while the three-seed aggregate is only +3.29. The 256-block panel cannot establish that this policy is worse than B20M. It does show why the primary gains alone should not be presented as broad playing improvement.

## Coverage, accounting, and limits

The native audit replayed every diagnostic hand and checked legal actions, chip settlement, target RNG, concrete menu/key/visit membership, and event digests; it recorded **zero failures**. It did not recompute each internal LBR action-value estimate. An independent parser of all 608,256 raw hands reproduced **135 estimates**, including both original 97.5% primary contrasts and 22 available exploratory curves, to 1e−8 BB/100. Every block has both positions. The [independent result](hu20-scaling-diagnostics-artifacts/independent.json) records hashes of all 111 raw panels.

The sealed [manifest](hu20-scaling-diagnostics-artifacts/final-manifest.json) covers **123 files**, all rehashed successfully after the runner and logs stopped. The additional [deal check](hu20-scaling-diagnostics-artifacts/additional-verification.json) found zero overlap with 100,016 earlier opened deal seeds, the independent observations, or preflight deals. **4,096 native-pressure deal seeds intentionally overlap the primary panels**: this was the frozen shared schedule for comparisons, not a fresh independent sample. The remaining diagnostic coordinates are distinct. The parent checkpoint, export, training-work, and recovery lineage checks remain in the sealed recovery report; this attempt loaded their verified saved artifacts and performed no new training.

In the diagnostic panels, the B100M policies used fallback on 353 of 86,892 target decisions against original-cap2 minraise, 67 of 89,791 against passive, and 491 of 83,465 against native minraise. Against pot pressure, they used fallback on **454 of 1,757** target decisions; 513 decisions followed a history outside the target's restricted sizing menu. Those events overlap but are not identical. A losing hand cannot be attributed to one fallback or street from its final payoff. Full street breakdowns are in [telemetry.json](hu20-scaling-diagnostics-artifacts/telemetry.json); in particular, the target's trained-key decisions still dominate the bounded LBR play documented in the recovery report.

The separate diagnostic run lasted about **24 minutes**, including evaluation, replay, combination and sealing. Peak sampled owned-job RSS was **2.97 GiB**, minimum free disk **38.94 GiB**, and measured swap growth **zero**. All 287 resource samples were on AC. These are [sampled resources](hu20-scaling-diagnostics-artifacts/telemetry.json), not a claim about an unobserved instantaneous peak. The 10.5-GiB memory and 8-GiB disk guards were respected. The evaluation and full attempt finished before their unchanged separate cutoffs. No paid host or M1-heavy work was used.

Large raw hands, audits and retained logs remain on M4 at:

`/Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-diagnostics-20260929`

```sh
rsync -a m4:/Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-diagnostics-20260929/ ./hu20-scaling-diagnostics/
scp m4:/Users/dberweger/Local/hu20-training-scaling-pr116/results/hu20-scaling-diagnostics-20260929-final-manifest.json ./
```

The #112/#113/#115 demos and the fixed-first-seed B100M human-play smoke/replay remain as documented in the recovery report. Bot cards were hidden until legitimate disclosure. This diagnostic continuation did not replace or promote a playable model.

## One next recommendation

**Diagnose a bounded set of actual decision values against the saved LBR and pot-pressure policies before specifying another training or abstraction change.** More unchanged training improved the prespecified aggregate attacks and recovered the minraise/passive controls, but B100M still loses substantially to an attacker whose hands mostly reach trained keys; the small pot-pressure panel is weak and includes many off-menu histories. A focused, information-safe conditional audit can separate those mechanisms. The present data do not justify assuming that fallback changes alone, a 500M continuation, or paid hardware will fix them. No further campaign is launched by this report.
