# Slumbot benchmark: the K50 752M HU200 model

**−81 ± 56 BB/100 over 5,000 hands** (95%; −74 ± 51 with all-in equity adjustment). #250's K50 HU200 average, 752,593,175 training nodes, loses to Slumbot by more than the noise. Against #248's 200M model on the same footing ([#253](hu200-200m-slumbot-benchmark.md), −103 ± 110 over 1,796 hands), the difference is +22 ± 123: not detectable at these sample sizes. Neither is a v0.5.5 result.

## The run

- **Model:** #250's terminal average, seed 2026101302, 752,593,175 nodes, 85,373,907 entries, SHA256 `60bd37c979e018f11f136370f468cc41b8e1435550531707e89d45f28709ec7c`, with #163's pinned K50 tables (`084e243d…`, `c0bc6a8a…`, `70c1cb5e…`). Translation off; it does not support K50.
- **Account and machine:** the owner's account `dberwegerch`, October 13, on the M1, with `scripts.play_slumbot` and no spacing between requests. One request at a time.
- **Three parts, 5,000 verified hands in all.** Each restart kept its journal and took a fresh policy seed:

| Part | Policy seed | Hands | BB/100 [95%] | Ended |
|---|---:|---:|---:|---|
| 1 (09:44–09:59) | 2026101305 | 186 | +126 ± 441 | 6 requests without an answer, over the limit of 5 then |
| 2 (10:25–10:44) | 2026101306 | 228 | +9 ± 177 | Slumbot refused a legal call with HTTP 400 |
| 3 (10:57–12:42) | 2026101307 | 4,586 | −94 ± 58 | Completed, nothing abandoned |

  Parts 1 and 2 led to #257's changes: a request without an answer, or one Slumbot refuses, now abandons only its hand, and a refusal starts a fresh session. Part 3 ran 15,976 requests at a median latency of 0.36 s (90th percentile 0.47 s), 1.4 s per hand.
- **Checks:** every settlement was recomputed from the public actions and both revealed hands and equals Slumbot's reported winnings; each journal's hash chain verifies. In every part, Slumbot's own session total equals our verified net, so the 13 abandoned hands cost nothing.
- **Not included:** 81 hands this model played on `dberwegerch` from the Linux server by mistake (−191 ± 278), and the dedicated account `deepcfr-k50-752m`'s run, which is still going. They are reported separately.

## Result

| | BB/100 [95%] | Hands |
|---|---|---:|
| **Raw** | **−81 ± 56** | 5,000 |
| All-in adjusted | −74 ± 51 | 5,000 |
| Big blind, all-in adjusted | −61 ± 76 | 2,499 |
| Small blind/button, all-in adjusted | −88 ± 67 | 2,501 |

The all-in adjustment counts the client's exact equity share whenever both players were all-in before the river: 20 hands (4 preflop, 6 flop, 10 turn). The other 14 all-ins were on the river and keep their result.

## Where the chips went, against the 200M model

Each ending's chips per 100 hands, so the two runs compare directly:

| How hands ended | K50 752M hands | K50 752M BB/100 | 200M BB/100 |
|---|---:|---:|---:|
| We folded preflop | 1,503 | −28.4 | −34.8 |
| Slumbot folded preflop | 677 | +21.8 | +29.5 |
| Showdown | 898 | **−49.8** | **−138.3** |
| Slumbot folded, flop to river | 1,041 | +138.4 | +168.4 |
| We folded, flop to river | 865 | −106.9 | −83.1 |
| All-in before the river | 16 | −56.0 | −44.5 |
| **Total** | **5,000** | **−80.8** | **−102.9** |

Our actions by street, out of 12,437 decisions:

| Street | Fold | Check | Call | Raise | All-in |
|---|---:|---:|---:|---:|---:|
| Preflop | 1,504 | | 1,958 | 1,947 | |
| Flop | 386 | 1,210 | 266 | 1,523 | 2 |
| Turn | 273 | 739 | 184 | 908 | 14 |
| River | 206 | 530 | 150 | 623 | 14 |

Facing a Slumbot bet or raise, by its size relative to the pot before it (blind-only preflop spots excluded; the big blind facing a min-raise counts as ≤½ pot):

| Slumbot's size | Fold | Call | Raise or all-in |
|---|---:|---:|---:|
| ≤½ pot | 1,394 | 1,376 | 893 |
| ½–¾ pot | 63 | 23 | 24 |
| About pot | 250 | 353 | 209 |
| 1.2–2× pot | 26 | 20 | 33 |
| >2× pot | 13 | 15 | 33 |
| All-in | 19 | 19 | 0 |

**Preflop folds:** 604 of our 2,504 small-blind first actions were folds (24%; the 200M model folded 39%). As big blind we folded to Slumbot's min-raise to 200 in 751 of 2,264 spots (33%), at 3:1. Another 149 folds came against re-raises.

**Stored policy coverage:** a decision without a stored policy plays uniformly over the menu. That happened on 48 of 5,409 preflop decisions (0.9%), 78 of 3,387 flop (2.3%), 97 of 2,118 turn (4.6%) and 97 of 1,523 river (6.4%), similar to the 200M model's 0.8–8.0%.

## What it shows

- **Showdowns cost much less:** −50 BB/100 against −138. The K50 model reaches showdown more often (18% of hands against 13%) but loses 2.8 BB per showdown against 10.7.
- **Postflop folds cost more, and Slumbot folds pay less:** −107 against −83, and +138 against +168. It gives up more pots after the flop, and its pressure works less often. The net gain over the 200M model is small and within the noise.
- **Slumbot rarely overbets:** 140 of the 4,763 bets and raises we faced were above 1.2× the pot. Unlike #251's pot-pressure opponent, Slumbot's pressure comes in small and pot-sized bets, so the overbet menu (#256) is not obviously what this matchup needs.
- **Unseen situations are rare:** at most 6.4% of river decisions. The loss comes from the stored strategy itself, as it did for the 200M model.

Precision: at this run's spread, ±25 BB/100 takes about 25,000 hands. The dedicated-account run of this model on the Linux server (up to 200,000 hands) will provide that. The 200M model is being played to 5,000 hands on its own account, `deepcfr-v1-200m`, for an equal-precision comparison.

## Reproduce

```
python -m scripts.analyze_slumbot_run PART1 PART2 PART3 --out ANALYSIS.json \
    --model AVERAGE.gz --model-sha256 SHA --card-buckets TABLES --policy-cache CACHE
```

The analysis now pools several runs of one model, verifying each journal on its own and keeping each part's result. Journals and the analysis are archived as indexed in RESULTS_INDEX.md.
