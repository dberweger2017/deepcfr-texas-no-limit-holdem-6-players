# AIVAT on real Slumbot hands

**Status: in progress.** Two of #304's three 10,000-hand confirmation runs are analyzed. Seed 2026101422, the pooled 30,000 hands and the v0.5.5 blend's exploratory run follow in this same PR.

[#314](https://github.com/dberweger2017/CFR-NLHE-6max/pull/314), contributed by @mstampfli, added AIVAT luck removal to `scripts.analyze_slumbot_run`. Its author had no Slumbot journals. This report runs the merged code on our real hands.

Nothing here changes a verdict. [#304](hu200-v055-slumbot-confirmation.md)'s rule was raw BB/100, declared in advance. Using AIVAT in any acceptance rule needs a new protocol declared before the hands it judges.

## What was run

- **Input:** #304's runs, each seed's first 10,000 completed hands with both parts pooled in play order, from the [archive](../../RESULTS_INDEX.md) `hu200-v055-slumbot-confirmation-20261015.zip`.
- **Code:** #314's commit `515e0595`, identical to its merge.
- **Cards only:** `--hands 10000 --aivat`, on cygnus-x1 with 4 workers. All 11 tests in `tests/test_aivat.py` pass there.
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

**Seed 2026101422:** pending.

## What it shows so far

1. **AIVAT works on real data, and consistently.** On both seeds one hand is worth 1.4–1.8 raw hands, and each estimate differs from raw only by the luck it measured. The weighted cards + own actions estimate is the strongest on both seeds (1.74 and 1.84), turning ±30–33 BB/100 into about ±22–25 at 10,000 hands.
2. **The luck behind #304's failure.** Seed 2026101302 had about 9 BB/100 of bad card luck, and its own sampled actions ran about 9 BB/100 lucky. That partly offset it, so its raw result was mostly card luck. Seed 2026101421's luck was close to zero on both counts, and there raw and AIVAT agree. Neither luck term is individually significant; the pooled check below tests whether they average zero.
3. **The value estimate is the limit, and points to the next step.** The cross-fitted weights scale the deal's correction by about 5× and the flop's by about 2.5×. The check-down value (pot × equity) prices early luck at the small pot of that moment, while early cards steer much larger later pots. A value estimate that anticipates later betting, such as the model's own action values, should remove considerably more than the 1.5–1.8× seen here; published AIVAT results are far larger. #314's structure keeps that a contained change.
4. **By seat:** the big-blind deficit #304 noted stays after luck removal for seed 2026101302 (weighted −39.3 ± 34.4 versus +8.4 ± 37.2 on the button). For seed 2026101421 the gap is smaller and within noise.

## Pending in this PR

- Seed 2026101422, both modes.
- The pooled 30,000 hands, with the removed card and own-action luck tested for zero mean across all hands.
- The v0.5.5 blend's 10,000-hand exploratory run (`deepcfr-v055-bx`, model SHA256 `ce3ecc2c…e6485`), both modes.

The receipts in [`slumbot-aivat-real-runs/`](slumbot-aivat-real-runs/) are the analyzer's full JSON outputs, including per-seat results and the cross-fitted weights.
