# v0.5.3: heads-up 100 BB, larger bet menu

A new heads-up 100 BB model, trained on a larger betting menu (min, half pot, pot, 1.5× pot, 2× pot and all-in) with finer postflop card buckets. **It beats v0.5.2 head to head by +37.10 [30.16, 44.04] BB/100**, with translation on for both models, over 131,072 duplicate blocks. All three training seeds beat v0.5.2 on their own: +47.04, +40.95 and +45.42 BB/100 (#296).

- **Model:** PR272's frozen linear-CFR opponent-sampled average, seed 2026100601, chosen before evaluation without seed selection. 5.0 billion nodes, 97.3 million stored entries, K50 postflop card buckets. Translation for off-menu bet sizes stays on, with a new adapter for this menu (512 states, 128 events).
- **Scripted opponents**, BB/100 (#296):

  | Opponent | BB/100 [95%] |
  |---|---|
  | check/call | +268.70 [252.02, 285.37] |
  | random | +106.91 [71.01, 142.81] |
  | tight-aggressive | +37.29 [27.87, 46.71] |
  | loose-aggressive | +34.87 [15.09, 54.65] |
  | pot-size pressure | +27.24 [15.27, 39.21] |
  | matched-pot pressure | +20.46 [7.61, 33.30] |
  | half-pot pressure | +16.18 [2.60, 29.75] |

  Pot-size pressure, which beat v0.5.2, is now beaten. The margins against half-pot and matched-pot pressure are the smallest, and against matched-pot pressure this model scores less than v0.5.2 on the same deals: −26.57 [−40.79, −12.34] BB/100.
- **Translation:** turning it on changes nothing against any of the seven scripted opponents (exactly 0.00 BB/100 on the same deals), so it does no harm. Those opponents almost never bet off this menu, so a benefit against human bet sizes is not measured yet.
- **Limits:** scripted opponents and our own models only; no external benchmark. No human, multiplayer or exploitability claim. See the model card for every number and limit.

**Running it.** v0.5.3 runs with `--hu100-release`, which also loads v0.5.0–v0.5.2 bundles. On an M1 MacBook Pro it took about ten minutes to load and peaked at 1.7 GB of RAM. The HU20 releases are unchanged: v0.4.2 stays the default on the HU20 table.

**Assets:** the model in three ordered parts (each at most 1 GiB; the whole model is 3.2 GB), the flop, turn and river card-bucket tables, the model card, these notes, install instructions, `release-manifest.json`, `SHA256SUMS` and a standard-library verifier, `verify_hu100_bundle.py`. The verifier checks every part, the whole model's fixed hash, the tables and the publication binding to this release's source commit.

[Science report](https://github.com/dberweger2017/CFR-NLHE-6max/blob/PACKAGE_SOURCE_COMMIT/docs/reports/hu100-v053-translation.md) · [package readiness](https://github.com/dberweger2017/CFR-NLHE-6max/pull/298) · [install](INSTALL.md)
