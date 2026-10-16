# AIVAT on real Slumbot hands

**Status: in progress.** All three of #304's 10,000-hand confirmation runs are analyzed, with the pooled 30,000 hands. The v0.5.5 blend's exploratory run follows in this same PR.

[#314](https://github.com/dberweger2017/CFR-NLHE-6max/pull/314), contributed by @mstampfli, added AIVAT luck removal to `scripts.analyze_slumbot_run`. Its author had no Slumbot journals. This report runs the merged code on our real hands.

Nothing here changes a verdict. [#304](hu200-v055-slumbot-confirmation.md)'s rule was raw BB/100, declared in advance. Using AIVAT in any acceptance rule needs a new protocol declared before the hands it judges.

## What was run

- **Input:** #304's runs, each seed's first 10,000 completed hands with both parts pooled in play order, from the [archive](../../RESULTS_INDEX.md) `hu200-v055-slumbot-confirmation-20261015.zip`.
- **Code:** #314's commit `515e0595`, identical to its merge.
- **Cards only:** `--hands 10000 --aivat`, on cygnus-x1 with 4 workers (seeds 2026101302 and 2026101421). All 11 tests in `tests/test_aivat.py` pass there. Seed 2026101422's run there was stopped by the heat guard at VRM 88 °C and redone on our Slumbot server with identical code.
- **Cards + own actions:** the same, plus `--model`, its SHA256, the K50 tables, the run's own read-only policy cache and translation 512/128, on our Slumbot server.
- **Prerequisite, checked on real data:** Slumbot reveals its hole cards on every hand, folds included (9,729 of 9,729 hands in one run). So every hand gets card corrections, with no selection.
- **Cost:** about 1.5 CPU seconds per hand, about an hour per seed with 4 workers.

## Results

BB/100 with 95% intervals over 10,000 hands. "Worth" is raw variance divided by the estimate's variance: how many raw hands one hand is worth.

**Seed 2026101302** (the run that failed #304's rule):

| Estimate | BB/100 [95%] | Worth |
|---|---|---:|
| Raw | −23.07 [−56.53, +10.38] | 1.00 |
| All-in adjusted (existing) | −13.79 [−45.32, +17.73] | |
| AIVAT, cards | −13.95 [−40.89, +12.99] | 1.54 |
| AIVAT, cards, weighted | −11.78 [−38.15, +14.59] | 1.61 |
| AIVAT, cards + own actions | −23.15 [−50.03, +3.73] | 1.55 |
| AIVAT, cards + own actions, weighted | −15.49 [−40.82, +9.85] | 1.74 |
| Card luck removed | −9.13 [−23.39, +5.14] | |
| Own-action luck removed | +9.20 [−6.79, +25.20] | |

**Seed 2026101421:**

| Estimate | BB/100 [95%] | Worth |
|---|---|---:|
| Raw | −10.66 [−40.53, +19.21] | 1.00 |
| All-in adjusted (existing) | −9.04 [−36.96, +18.88] | |
| AIVAT, cards | −11.15 [−34.88, +12.59] | 1.58 |
| AIVAT, cards, weighted | −10.72 [−33.54, +12.10] | 1.71 |
| AIVAT, cards + own actions | −12.40 [−37.38, +12.58] | 1.43 |
| AIVAT, cards + own actions, weighted | −9.74 [−31.75, +12.27] | 1.84 |
| Card luck removed | +0.49 [−12.57, +13.54] | |
| Own-action luck removed | +1.25 [−14.69, +17.20] | |

**Seed 2026101422:**

| Estimate | BB/100 [95%] | Worth |
|---|---|---:|
| Raw | −7.87 [−39.39, +23.65] | 1.00 |
| All-in adjusted (existing) | −7.93 [−38.10, +22.24] | |
| AIVAT, cards | −10.53 [−36.66, +15.59] | 1.46 |
| AIVAT, cards, weighted | −5.79 [−32.11, +20.53] | 1.43 |
| AIVAT, cards + own actions | −7.14 [−35.64, +21.36] | 1.22 |
| AIVAT, cards + own actions, weighted | −3.89 [−30.10, +22.33] | 1.45 |
| Card luck removed | +2.66 [−10.59, +15.91] | |
| Own-action luck removed | −3.39 [−18.55, +11.76] | |

**All three seeds pooled (30,000 hands).** The seeds have equal size, so each pooled mean is the average of the three, and its interval combines the three intervals.

| Estimate | BB/100 [95%] |
|---|---|
| Raw | −13.87 [−32.14, +4.40] |
| All-in adjusted (existing) | −10.26 [−27.52, +7.01] |
| AIVAT, cards | −11.88 [−26.68, +2.93] |
| AIVAT, cards, weighted | −9.43 [−23.99, +5.14] |
| AIVAT, cards + own actions | −14.23 [−29.72, +1.26] |
| AIVAT, cards + own actions, weighted | −9.70 [−23.90, +4.49] |
| **Card luck removed** | **−1.99 [−9.81, +5.82]**, z = −0.50 |
| **Own-action luck removed** | **+2.35 [−6.71, +11.42]**, z = +0.51 |

**The zero-mean check passes:** over 30,000 real hands, both removed luck terms are well within noise of zero, as the method requires. On our data the corrections show no sign of bias.

## What it shows so far

1. **AIVAT works on real data, and consistently.** On every seed one hand is worth 1.2–1.8 raw hands; the weighted cards + own actions estimate is worth 1.74, 1.84 and 1.45. Pooled, its interval is ±14.2 against raw's ±18.3. Each estimate differs from raw only by the luck it measured, and that luck averages zero across the three seeds. 
2. **The luck behind #304's failure.** Seed 2026101302 had about 9 BB/100 of bad card luck, and its own sampled actions ran about 9 BB/100 lucky. That partly offset it, so its raw result was mostly card luck. Seeds 2026101421 and 2026101422 had luck within a few BB/100 of zero. With luck removed, the three seeds sit between −4 and −15 BB/100 (weighted cards + own actions).
3. **The value estimate is the limit, and points to the next step.** On all three seeds the cross-fitted weights scale the deal's correction by about 4.5–5.7× and the flop's by about 2–2.6×. The check-down value (pot × equity) prices early luck at the small pot of that moment, while early cards steer much larger later pots. A value estimate that anticipates later betting, such as the model's own action values, should remove considerably more than the 1.5–1.8× seen here; published AIVAT results are far larger. #314's structure keeps that a contained change.
4. **By seat:** the big blind trails the button on all three seeds after luck removal (weighted cards + own actions: −39.3 versus +8.4, −17.1 versus −2.4, and −10.7 versus +2.9 BB/100). Each gap is within its own noise, but the direction is consistent, which makes big-blind play worth a closer look.

## Pending in this PR

- The v0.5.5 blend's 10,000-hand exploratory run (`deepcfr-v055-bx`, model SHA256 `ce3ecc2c…e6485`), both modes.

The receipts in [`slumbot-aivat-real-runs/`](slumbot-aivat-real-runs/) are the analyzer's full JSON outputs, including per-seat results and the cross-fitted weights.
