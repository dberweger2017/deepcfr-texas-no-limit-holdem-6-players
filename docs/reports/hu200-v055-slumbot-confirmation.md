# v0.5.5 Slumbot confirmation: two seeds pass, seed 2026101302 fails

**v0.5.5 does not qualify with this recipe.** Under the owner-approved [protocol](../hu200-v055-acceptance-protocol.md), each of three seeds plays its first 10,000 completed hands against Slumbot, and each 95% lower bound must be above −50 BB/100. Seeds 2026101422 and 2026101421 pass; the frozen 20B candidate, seed 2026101302, misses by 6.5 BB/100. The protocol allows no rerun, extension or seed substitution, so this result stands. What comes next is the owner's decision.

| Seed | Model | BB/100 [95%], raw | Pass | All-in adjusted | Abandoned |
|---|---|---|---|---|---:|
| 2026101422 | [#278](https://github.com/dberweger2017/CFR-NLHE-6max/pull/278) confirmation seed | −7.87 [−39.39, +23.65] | **yes** | −7.93 [−38.10, +22.24] | 108 |
| 2026101421 | #278 confirmation seed | −10.66 [−40.53, +19.21] | **yes** | −9.04 [−36.96, +18.88] | 60 |
| 2026101302 | #265 frozen 20B, the candidate | −23.08 [**−56.53**, +10.38] | **no** | −13.79 [−45.32, +17.73] | 33 |

Each row is exactly 10,000 completed hands, with translation 512/128 as declared. 1 BB/100 is one big blind (100 chips) per 100 hands.

## Runs and validity

All three first parts stopped early on the morning of October 15, during Slumbot service problems (timeouts and protocol errors); the service was then down from about 12:25 to 18:01 Madrid. As the protocol prescribes, each seed continued in a second run with its next policy seed, and the two parts are pooled in play order. Only each seed's first 10,000 hands count.

| Seed | Part 1 (policy seed …01) | Part 2 (policy seed …02) |
|---|---|---|
| 2026101302 | 271 hands, stopped by a protocol error at 08:48 Madrid | 9,729 hands, completed 22:35 |
| 2026101421 | 431 hands, stopped by a protocol error at 09:22 | 9,569 hands, completed 22:36 |
| 2026101422 | 1,439 hands, ended at 11:20 without a summary | 8,561 hands, completed 22:13 |

Part 2 ran at commit `98658f7`, which adds #290's protocol-error detail to the client; the playing code is unchanged.

- **Journals:** all six hash chains verify, both on the server and again independently on the M1.
- **Session totals:** for every run that wrote a summary, Slumbot's session total equals our verified net, with zero unverified chips. Seed 2026101422's part 1 ended without a summary, so its Slumbot session total was not recorded; its journal still verifies and its hands settle from the public actions and cards.
- **Abandoned hands:** 33, 60 and 108, all below the limit of 200. Every abandonment was an ambiguous timeout. Seed 2026101422 also had 2 hands rejected by the service (limit 50).

## Why seed 2026101302 failed: our interpretation

This section is interpretation after the fact. It does not change the verdict, and none of it was declared in advance.

**1. Most of the gap is all-in luck.** All-in adjustment replaces the result of each hand where a player is all-in before the last card with its expected value, which removes the luck of the remaining cards. It moves seed 2026101302 from −23.1 to −13.8 BB/100, a 9.3 BB/100 swing, against 0.1 and 1.6 for the other seeds. That seed also had more all-in hands (44, against 33 and 30) and lost 12.9 BB/100 at showdown, where the other two won 3.7 and 3.0. With the luck removed, the three seeds sit close together, between −8 and −14.

**2. The same model did better in earlier play.** Seed 2026101302's model, frozen 20B with translation, played 20,000 exploratory hands on October 14 ([#271](https://github.com/dberweger2017/CFR-NLHE-6max/pull/271) and its follow-up runs): about −5 ± 22 BB/100. The difference from this run's −23.1 is about 0.9 standard errors, an ordinary swing. Over all 30,000 hands the model averages about −11 BB/100. The protocol keeps exploratory hands out of the decision, and this report keeps it that way.

**3. The rule was harder to pass than its wording suggests.** At 10,000 hands the interval is about ±31.6 BB/100, so a seed passes only when its measured mean is above about −18.4. Each seed is a separate chance to fail:

| True strength (BB/100) | One seed passes | All three pass |
|---:|---:|---:|
| +5 | 93% | 80% |
| 0 | 87% | 66% |
| −5 | 80% | 51% |
| −10 | 70% | 34% |
| −14 | 61% | 22% |
| −20 | 46% | 10% |

The protocol's expected-behavior note gave the chance for one seed (75–90% at break-even); it did not give the chance that all three pass. At our best estimate of the recipe's strength, around −10 to −14 BB/100, the three-seed rule passes only about a quarter to a third of the time. In practice it asks whether the model is close to break-even against Slumbot, rather than whether it stays above −50.

**4. The recipe as a whole is well above −50.** Pooling the three seeds equally gives −13.9 [−32.1, +4.4] BB/100 over 30,000 hands. That is not a pass, because pooling was not declared, but every estimate points the same way: the recipe loses a little to Slumbot and is not close to −50.

**By seat, within noise:** the big blind loses more than the button in seeds 2026101422 and 2026101302 (−22.4 against +6.6, and −41.8 against −4.4 BB/100), but not in 2026101421 (−9.5 against −11.8). Most of seed 2026101302's shortfall against the other two is in the big blind.

## What this means for the next attempt

- **The recipe:** a new candidate needs its own fresh confirmation. The open candidates are the three-seed blend ([#297](https://github.com/dberweger2017/CFR-NLHE-6max/pull/297)) and the 301M-entry tables ([#301](https://github.com/dberweger2017/CFR-NLHE-6max/pull/301)).
- **The test design:** any new protocol should state, before play, the chance that all seeds pass at a few plausible true strengths, as in the table above. Options include more hands per seed, or a pooled three-seed test with the per-seed results reported alongside. Choosing the design is the owner's decision.

## Evidence

The six journals, five summaries, three pooled analyses, the part-2 launchers and the exploratory runs' summaries are archived on Google Drive; see [RESULTS_INDEX](../../RESULTS_INDEX.md). This folder holds the small receipts: the three [pooled 10,000-hand analyses](hu200-v055-slumbot-confirmation/) from `scripts.analyze_slumbot_run … --hands 10000` and the five run summaries.

Reproduce a seed's row from the restored archive:

```sh
python -m scripts.analyze_slumbot_run PART1_DIR PART2_DIR --hands 10000 --out pooled.json
```

No release, tag, default or model change.
