# HU200 frozen 20B research seeds

This **pre-release is not Latest and is not v0.5.5**. It publishes the three
frozen 20B-node seeds that are the ingredients of v0.5.5's sequence-form blend:
2026101302, 2026101421 and 2026101422. Seed 2026101302 was the original v0.5.5
candidate. The collection includes each seed's average export **and raw
checkpoint**, in ordered parts no larger than 1 GiB, plus the pinned K50 tables,
original blend configuration and a standalone stdlib verifier. No private Drive
access is required to rebuild the blend from these GitHub assets.

The seeds **did not qualify under their declared rule**. Protocol 1 required
every seed's 10,000-hand raw 95% lower bound against Slumbot to exceed −50 BB/100.
[#304](https://github.com/dberweger2017/CFR-NLHE-6max/pull/304) reported:

| Seed | Raw BB/100 [95% interval] | Original seed gate |
|---|---:|---|
| 2026101302 | −23.08 [−56.53, +10.38] | Failed by 6.5 BB/100 |
| 2026101421 | −10.66 [−40.53, +19.21] | Passed |
| 2026101422 | −7.87 [−39.39, +23.65] | Passed |

The three-seed recipe failed. [#316's later AIVAT analysis](https://github.com/dberweger2017/CFR-NLHE-6max/pull/316)
attributes the failure mostly to card luck: seed 1302 had about −9 BB/100 of it.
That does not change the predeclared raw-statistic verdict or qualify these seeds
retrospectively. The separately published blend qualified under a new, prospective
protocol v2 at −19.67 [−38.57, −0.76] BB/100 over 30,000 fresh hands (#318); it still
loses to Slumbot by a small detectable margin.

#306's internal blend-minus-seed results are +3.61 /+0.37 /+1.85 BB/100, all within
noise. Its blend loses to scripted 1.5× pot pressure at −29.39 [−55.24, −3.53].
These research ingredients do not establish HU100, multiway or professional strength.

The blend was built and tested on a base 16 GB MacBook Pro M4. #306's full
construction/evaluation pipeline peaked at 12,748,324,864 bytes family RSS.
The [install guide](INSTALL.md) describes the self-contained reconstruction;
the [model card](MODEL_CARD.md) pins all six original payloads. The table app
supports the qualified v0.5.5 blend; this seed collection is verification-only.
