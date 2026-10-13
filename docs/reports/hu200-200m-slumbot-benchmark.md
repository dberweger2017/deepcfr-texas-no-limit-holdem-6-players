# First Slumbot benchmark: the 200M HU200 model

**−103 ± 110 BB/100 over 1,796 hands** (95%; −112 ± 99 with all-in equity adjustment). This is a pipeline benchmark of [#248](hu200-200m-model.md)'s small HU200 model: v1 cards, 200M training nodes, trained in minutes. It is far below any release budget and shows that the whole chain works against Slumbot's live API: model, client, settlement and analysis. It is not a v0.5.5 result.

## The run

- **Model:** #248's average, seed 2026101301, 200,001,087 nodes, 31.6M entries, SHA256 `b7c15eb642f2edec82df04983b28a8f80b8c3db310f3d6ea53e21aab6d120402`. Policy sampling seed 2026101303, translation off.
- **Session:** the owner's Slumbot account, October 13, 02:17–04:11 Madrid, with `scripts.play_slumbot` from [#249](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/249) and 0.5 s between requests.
- **Hands:** 1,796 verified; 14 abandoned after requests without a complete answer, which cost −1,200 chips by Slumbot's session total. The run then stopped on an HTTP error from the service after 1,811 hands of the planned 3,000. The status was not recorded; the summary now records it.
- **Pace:** 5,813 requests at a median latency of 0.67 s (90th percentile 1.85 s, maximum 21.9 s), 3.8 s per hand.
- **Checks:** every hand's settlement was recomputed from the public actions and both revealed hands and equals Slumbot's reported winnings, and the journal's hash chain verifies.

## Result

| | BB/100 [95%] | Hands |
|---|---|---:|
| **Raw** | **−103 ± 110** | 1,796 |
| All-in adjusted | −112 ± 99 | 1,796 |
| Big blind, all-in adjusted | −114 ± 139 | 898 |
| Small blind/button, all-in adjusted | −110 ± 140 | 898 |

Hands dealt fresh by the service are unpaired, so these are ordinary intervals over hands. **All-in adjustment** counts the client's exact equity share of the pot whenever both players were all-in before the river, over every possible runout. It removes runout luck without changing the expected result. Only 11 hands qualified (5 preflop, 3 flop, 3 turn), so it narrowed the interval by about 10%.

**Slumbot's `baseline_winnings` is not used.** It follows the client's own actions, so winnings minus baseline cancels much of the client's play along with the luck. A random-action bot, 300 anonymous hands: raw −365 ± 411, baseline mean −490 ± 452, and winnings minus baseline **+125 ± 192**, which no random player could truly achieve. On this run the difference was exactly 0 at the final hand by coincidence; its running total had ranged from −44,600 to +5,900 chips.

## Where the chips went

| How hands ended | Hands | Chips |
|---|---:|---:|
| We folded preflop | 680 | −62,450 |
| Slumbot folded preflop | 258 | +52,900 |
| Showdown | 232 | **−248,400** |
| Slumbot folded, flop to river | 383 | +302,450 |
| We folded, flop to river | 237 | −149,250 |
| All-in before the river | 6 | −80,000 |
| **Total** | **1,796** | **−184,750** |

Our actions by street, out of 4,001 decisions:

| Street | Fold | Check | Call | Raise | All-in |
|---|---:|---:|---:|---:|---:|
| Preflop | 680 | | 572 | 689 | 2 |
| Flop | 117 | 283 | 44 | 573 | 3 |
| Turn | 63 | 196 | 39 | 311 | 6 |
| River | 58 | 134 | 33 | 192 | 6 |

Facing a Slumbot bet or raise, by its size relative to the pot before it (blind-only preflop spots excluded):

| Slumbot's size | Fold | Call | Raise or all-in |
|---|---:|---:|---:|
| ≤½ pot | 468 | 317 | 401 |
| ½–¾ pot | 19 | 11 | 12 |
| About pot | 57 | 94 | 111 |
| 1.2–2× pot | 12 | 8 | 18 |
| >2× pot | 6 | 4 | 9 |
| All-in | 7 | 10 | 0 |

**Stored policy coverage:** a decision without a stored policy plays uniformly over the menu. That happened on 15 of 1,943 preflop decisions (0.8%), 18 of 1,020 flop (1.8%), 37 of 615 turn (6.0%) and 34 of 423 river (8.0%).

## What it shows

- **Showdowns are the main loss:** −248,400 chips over 232 hands, about 10.7 BB each. The model raises a lot on every street, and reaches showdown with weaker hands than Slumbot's.
- **Preflop folding is frequent:** 680 of 1,943 preflop decisions (35%). Of these, 349 were our first action as small blind, folding 39% of our small blinds outright. Another 296 were big-blind folds to Slumbot's min-raise to 200, where the call gets 3:1, and 35 were folds to re-raises. After the flop, 172 folds came against bets of at most half the pot. Folding this often to cheap prices gives up good pot odds.
- **Slumbot folds a lot to our pressure:** +355,350 chips from its folds. The aggression wins small pots and loses big ones.
- **Unseen situations are rare** in this sample, up to 8% on the river. They don't explain the loss; the stored strategy does. That fits a model with only 200M training nodes.

Precision needs many more hands or stronger variance reduction: raw ±50 BB/100 takes about 16,000 hands. The next useful benchmark is a stronger model, such as [#250](hu200-k50-first.md)'s K50 HU200, on the same footing.

## Reproduce

```
python -m scripts.analyze_slumbot_run RUN_DIR --model AVERAGE.gz --model-sha256 SHA
```

`src/arena/external/analysis.py` verifies the journal, computes exact all-in equities in parallel, and writes `analysis.json`. Journals and analyses are archived as indexed in RESULTS_INDEX.md.
