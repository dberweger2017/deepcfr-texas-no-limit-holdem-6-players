# v0.5.2 model card: heads-up 100 BB

The third heads-up 100 BB release: the same recipe and training run as v0.5.1, continued from 2B to 4B nodes. **It beats v0.5.1 head to head by +17.90 [14.98, 20.83] BB/100**, and two independent seeds repeat the gain. Like the earlier v0.5.x releases, it is evaluated with internal checks and scripted opponents only. No external benchmark exists yet, so there's no claim about strength against established bots or people.

## What the model is

- **Training:** linear CFR with opponent-sampled averaging, trained natively to **4,000,000,174 nodes** (iteration 3,716,073) from seed **2026100601**. The table holds **70,466,972** information sets. It's the same training run as v0.5.0 and v0.5.1: their models are this run's 1B and 2B checkpoints.
- **Abstraction:** unchanged: the v1 card descriptor and ordered, size-bucketed betting history, with a menu of min-raise, pot and conditional all-in (`hu100-native-reopening-ordered-history-card-v1`).
- **Policy:** the normalized stored average. Information sets with no stored average play uniformly over the menu.
- **Translation, always on:** when a real bet size leaves the trained menu's support, the bot maps the public history to the nearest one the menu can produce (`hu100-public-menu-translation-v1`, at most 512 states and 128 events). Real wagers are never changed, and translation uses public information only.
- **Not included:** search, opponent modelling, other stack depths.

The model is #226's exported average, unchanged: it isn't re-extracted, and it isn't the best of several seeds. #226 fully audited all 70,466,972 entries, and #236 checked its bytes again before its scripted evaluation.

| File | Bytes | SHA256 |
|---|---:|---|
| `O4B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz` | 2,121,415,396 | `0ed7f9512996bb3e9535a7258a5d0e4bc8c00f22734d511c642c0e5b95cbd3c0` |

Source checkpoint SHA256 `65fd61d6999d7a06f36fc1c718aea20e7c23856aaa491df7d517e312c1c33368`; training source `d3120c2c195b5eca136b7f0d9969e32ba5159fff`; [model index](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/reports/hu100-4b-seed-ladder-artifacts/model-index.json).

## Table and information

- **Table:** two seats, 10,000 chips each, reset every hand; blinds 50/100; no ante or rake; uncapped no-limit betting; alternating button.
- **What the bot sees:** only its own seat's legal observation. It never sees the other hole cards, the deck or seeds.
- **Modes:** human play (restricted or free sizing) and a self-play spectator view.
- **Journals:** private journals keep what independent replay needs. Responses to the human never include hidden bot cards.
- **Other depths:** HU20, HU200 and mixed-depth tables are rejected.

## Evidence

All results are BB/100 on fresh paired deals with seats swapped.

**Head to head against v0.5.1** ([#226](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/reports/hu100-4b-seed-ladder.md)): **+17.90 [14.98, 20.83]** over 524,288 duplicate blocks, with a three-contrast Bonferroni 98.33% interval. v0.5.1's model is exactly this run's 2B checkpoint.

**Independent seeds** ([#236](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/reports/hu100-4b-seed-replication.md)): two other seeds, trained from scratch past 2B, gain almost exactly the same over their own 2B checkpoints. Their intervals are two-contrast Bonferroni 97.5%. Both stopped at the campaign's memory-derived entry cap just short of 4B, at 3.885B and 3.967B nodes.

| Seed | Gain over own 2B, BB/100 | Blocks |
|---|---|---:|
| 2026100601 (this release; #226) | +17.90 [14.98, 20.83] | 524,288 |
| 2026100901 (3.885B) | +18.67 [15.94, 21.40] | 524,288 |
| 2026100902 (3.967B) | +19.42 [15.58, 23.25] | 262,144 |

The three seeds' final models are close to each other head to head: this release against the other two plays +0.38 [−1.92, 2.68] and +0.43 [−1.86, 2.71] (95%). So the gain doesn't come from a lucky seed. Crossing zero doesn't prove equal strength.

**Against the scripted opponents, translation on** (#236, 4,096 blocks per opponent), alongside v0.5.1 on the same deals:

| Opponent | v0.5.2 [95%] | v0.5.1 [95%] | Difference [95%] |
|---|---|---|---|
| check_call | +140.41 [119.47, 161.34] | +134.38 [112.21, 156.56] | +6.02 [−11.62, 23.67] |
| random | +138.42 [66.43, 210.41] | +136.19 [64.69, 207.69] | +2.23 [−20.91, 25.36] |
| loose_aggressive | +82.15 [49.55, 114.74] | +47.07 [13.14, 81.00] | **+35.08 [7.77, 62.38]** |
| tight_aggressive | +49.49 [32.41, 66.57] | +49.64 [32.40, 66.88] | −0.15 [−13.71, 13.40] |
| pot_pressure | **−49.44 [−82.28, −16.61]** | −45.59 [−78.41, −12.76] | −3.86 [−30.06, 22.35] |

The release safeguard was predeclared: no opponent's difference may have a 95% upper bound below −20 BB/100. **No opponent comes close to that.** The loose-aggressive gain is significant on its own interval, but it wasn't a predeclared claim. The pot-pressure and random intervals still reach below −20, so a loss of that size against them isn't ruled out.

**Pot-size pressure still beats both models.** Scripted results vary between samples: on #223's deals, v0.5.1 lost −37.09 to it, against −45.59 here.

## Limits

- **No external benchmark.** No suitable public 100 BB opponent has been found; v0.5.5 moves to 200 BB to play Slumbot.
- **Pot-size pressure:** this release loses to the pot-pressure opponent, as v0.5.0 and v0.5.1 do.
- **What the intervals cover:** they're over deals and action streams for fixed policies, not over training seeds. The seed results above address that separately.
- **Coverage:** visits per information set grow with training, but many river and turn sets are still sparsely visited.
- **Size:** the model is 26 MB under GitHub's 2 GiB asset limit, so a larger HU100 model will need another way to be distributed.
