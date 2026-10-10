# v0.5.1 model card: heads-up 100 BB

The second heads-up 100 BB release: the same recipe as v0.5.0, trained twice as long. **It beats v0.5.0 head to head by +29.51 [26.82, 32.20] BB/100.** Like v0.5.0, it is evaluated with internal checks and scripted opponents only. No external benchmark exists yet, so there's no claim about strength against established bots or people.

## What the model is

- **Training:** linear CFR with opponent-sampled averaging, trained natively to **2,000,000,460 nodes** (iteration 1,812,907) from seed **2026100601**. The table holds **54,626,283** information sets. It's the same training run as v0.5.0, continued from 1B to 2B nodes.
- **Abstraction:** unchanged from v0.5.0: the v1 card descriptor and ordered, size-bucketed betting history, with a menu of min-raise, pot and conditional all-in (`hu100-native-reopening-ordered-history-card-v1`).
- **Policy:** the normalized stored average. Information sets with no stored average play uniformly over the menu.
- **Translation, always on:** when a real bet size leaves the trained menu's support, the bot maps the public history to the nearest one the menu can produce (`hu100-public-menu-translation-v1`, at most 512 states and 128 events). Real wagers are never changed, and translation uses public information only.
- **Not included:** search, opponent modelling, other stack depths.

The model is #223's exported average, unchanged: it isn't re-extracted, and it isn't the best of several seeds. #226 regenerated it from the seed byte for byte, and its full audit covered all 54,626,283 entries.

| File | Bytes | SHA256 |
|---|---:|---|
| `O2B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz` | 1,603,422,956 | `e6f79ccac39a651352424e05382681f4d1994bb570b025f692556d170f87e9ae` |

Source checkpoint SHA256 `e84039c7a934a966a2c01f237748a23951124a8b665edc2585ef4809e5681676`; training source `38c83f2eab09b5247a8c16344ce6d0a3d00c43af`; [model index](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/reports/hu100-3b-ladder-artifacts/model-input-index.json).

## Table and information

- **Table:** two seats, 10,000 chips each, reset every hand; blinds 50/100; no ante or rake; uncapped no-limit betting; alternating button.
- **What the bot sees:** only its own seat's legal observation. It never sees the other hole cards, the deck or seeds.
- **Modes:** human play (restricted or free sizing) and a self-play spectator view.
- **Journals:** private journals keep what independent replay needs. Responses to the human never include hidden bot cards.
- **Other depths:** HU20, HU200 and mixed-depth tables are rejected.

## Evidence

All results are BB/100 on fresh paired deals with seats swapped.

**Head to head against v0.5.0** ([#223](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/reports/hu100-3b-ladder.md)): **+29.51 [26.82, 32.20]**, over 524,288 duplicate blocks. v0.5.0's model is exactly this run's 1B checkpoint.

**Independent seeds** ([#226](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/226)): the same step from 1B to 2B nodes, repeated from two other seeds, gains almost exactly the same. Each uses 524,288 blocks with a three-contrast Bonferroni 98.33% interval.

| Seed | 2B vs 1B, BB/100 |
|---|---|
| 2026100601 (this release; #223, 95%) | +29.51 [26.82, 32.20] |
| 2026100901 | +28.14 [24.87, 31.41] |
| 2026100902 | +27.90 [24.65, 31.15] |

Head-to-head matches between the 2B checkpoints of different seeds show no significant strength difference; both intervals cross zero.

**Against the scripted opponents, translation on** (#223, 4,096 blocks per opponent), alongside v0.5.0 on the same deals:

| Opponent | v0.5.1 [95%] | v0.5.0 [95%] | Difference [95%] |
|---|---|---|---|
| check_call | +132.12 [110.96, 153.29] | +115.50 [94.68, 136.31] | +16.63 [−0.19, 33.44] |
| random | +106.90 [33.81, 179.98] | +125.48 [52.92, 198.04] | −18.59 [−50.30, 13.13] |
| tight_aggressive | +54.87 [35.47, 74.27] | +46.50 [25.98, 67.02] | +8.37 [−8.66, 25.40] |
| loose_aggressive | +42.18 [7.36, 77.00] | +40.02 [3.61, 76.43] | +2.16 [−31.28, 35.60] |
| pot_pressure | **−37.09 [−69.66, −4.52]** | −45.09 [−77.55, −12.64] | +8.00 [−20.21, 36.22] |

**No scripted difference is significant either way.** The scripted bots no longer separate these models, while head-to-head play clearly does.

**Pot-size pressure still beats both models in this sample.** v0.5.0's own published estimate came from #215's different sample, at −9.33 [−39.10, 20.44]. On these deals, v0.5.0 loses −45.09, so the opponent is harder here than #215's figure suggests.

## Limits

- **No external benchmark.** No suitable public 100 BB opponent has been found; v0.5.5 moves to 200 BB to play Slumbot.
- **Pot-size pressure:** this release loses to the pot-pressure opponent in #223's sample.
- **What the intervals cover:** they're over deals and action streams for fixed policies, not over training seeds. The seed results above address that separately.
- **Coverage:** the table is larger than v0.5.0's, but many information sets are still sparsely visited.
- **More training still pays.** In #226, a 4B checkpoint from the same run beats this one head to head, so a stronger release is expected.
