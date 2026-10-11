# v0.5.0 model card: heads-up 100 BB

The first release for heads-up no-limit hold'em at 100 big blinds. It's a usable local opponent, evaluated with internal checks and scripted opponents only. **No external benchmark exists yet**, so this release makes no claim about strength against established bots or people.

## What the model is

- **Training:** linear CFR with opponent-sampled averaging, the production recipe of v0.4.1 and v0.4.2, trained natively to **1,000,002,065 nodes** (iteration 885,307) from seed **2026100601**. The table holds **41,010,014** information sets.
- **Abstraction:** the v1 card descriptor and ordered, size-bucketed betting history, with a menu of min-raise, pot and conditional all-in (`hu100-native-reopening-ordered-history-card-v1`).
- **Policy:** the normalized stored average. Information sets with no stored average play uniformly over the menu.
- **Translation, always on:** when a real bet size leaves the trained menu's support, the bot maps the public history to the nearest one the menu can produce (`hu100-public-menu-translation-v1`, at most 512 states and 128 events). Real wagers are never changed, and translation uses public information only.
- **Not included:** search, opponent modelling, other stack depths.

The model is #207's exported average, unchanged: it isn't re-extracted, and it isn't the best of several seeds.

| File | Bytes | SHA256 |
|---|---:|---|
| `O1B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz` | 1,173,264,021 | `47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9` |

Source checkpoint SHA256 `cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec`; training source `bd0e7a417064f736091dc2b667954b50becb4b69`; [model index](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/reports/native-hu100-growth-1b-artifacts/model-index.json).

## Table and information

- **Table:** two seats, 10,000 chips each, reset every hand; blinds 50/100; no ante or rake; uncapped no-limit betting; alternating button.
- **What the bot sees:** only its own seat's legal observation. It never sees the other hole cards, the deck or seeds.
- **Modes:** human play (restricted or free sizing) and a self-play spectator view.
- **Journals:** private journals keep what independent replay needs. Responses to the human never include hidden bot cards.
- **Other depths:** HU20, HU200 and mixed-depth tables are rejected.

## Evidence

All results are BB/100 against the scripted opponents, on fresh paired deals with seats swapped.

**This model, translation on** ([#215](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/reports/hu100-independent-stages.md), 4,096 blocks per opponent):

| Opponent | BB/100 [95%] |
|---|---|
| check_call | +100.51 [81.49, 119.54] |
| random | +76.89 [4.82, 148.96] |
| loose_aggressive | +57.10 [21.67, 92.54] |
| tight_aggressive | +36.30 [18.61, 53.99] |
| pot_pressure | **−9.33 [−39.10, 20.44]** |

Translation changes only the pot_pressure result: −99.48 with it off.

**Independent seeds.** Two other seeds, trained the same way, beat the same four opponents too. Their pot_pressure results are also negative, with intervals crossing zero.

**The recipe's formal qualification failed.** #215's growth test from 39.4M to 1B nodes passed eight of nine adjusted contrasts. The ninth, seed 2026100902 against tight_aggressive, is +32.54 [−5.11, 70.20] and inconclusive. That decision stands; it isn't re-read from a descriptive interval.

**Head to head** ([#223](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/reports/hu100-3b-ladder.md), same seed):
- this 1B model beats its 500M checkpoint by **+38.93 [35.92, 41.95]**;
- the 2B checkpoint beats this model by **+29.51 [26.82, 32.20]**.

More training still pays, so stronger HU100 releases are expected.

## Limits

- **No external benchmark.** No suitable public 100 BB opponent has been found; v0.5.5 moves to 200 BB to play Slumbot.
- **Pot-size pressure:** profit against the pot-pressure opponent is unproven.
- **What the intervals cover:** they're over deals and action streams for fixed policies, not over training seeds.
- **Scripted opponents measure little at this level.** They no longer separate 1B from 2B (#223), even though head-to-head play does.
- **Coverage is still sparse,** at about 4.8 traverser visits per key.
