# AIVAT for Slumbot runs

`python -m scripts.analyze_slumbot_run RUN_DIR --aivat` adds an `aivat` block to the analysis: the same
hands with their luck removed, so the interval is narrower for the same number of hands. Add
`--model` (with the run's own model, hash and options) to also remove the luck of our own
sampled actions. Exact preflop equity costs about a second of CPU per hand; `--workers` shares it.

## What it removes

AIVAT ([Burch et al., 2018](https://arxiv.org/abs/1612.06915)) subtracts, wherever an outcome is
drawn from known odds, how far that outcome moved a value estimate from its average over every
outcome that could have been drawn. Each correction averages exactly zero, so the expected result
is unchanged for any value estimate; a better estimate only removes more noise.

- **Cards.** The deal, flop, turn and river are uniform over the unseen cards. Slumbot reveals
  its hand at the end of every hand, so every completed hand qualifies. The correction is
  `pot * (equity after - equity before)`, with exact equity over every remaining board. Equity
  before a card is exactly the average of equity after it; before the deal it is exactly 1/2.
- **Our own actions** (with `--model`). The client samples each action from its menu
  probabilities. Journals record only the chosen action, so the analysis looks every decision up
  again in the exact model; translation and fallback lookups are deterministic. The correction is
  the chosen action's value minus the probability-weighted average over the menu.
- **Not Slumbot's actions:** its strategy is unknown.

The value estimate is the check-down value `pot * equity - own contribution`, with a raise valued
as if called. A hand all-in before the river ends at the existing all-in adjustment, minus the
corrections before the all-in.

## Plain and weighted

The check-down value prices early luck at the small pot of the moment, though early cards steer
much larger later pots. The analysis therefore reports two estimates:

- `bb_per_100`: every correction at weight 1, the textbook estimate.
- `weighted_bb_per_100`: each street's card and action corrections scaled by least-squares
  weights. Hands alternate between two halves, and each half uses weights fitted only on the other
  half. Those hands are independent of it, so the weights act as fixed numbers and every scaled
  correction still averages zero. `weights_by_half` lists them.

`plain_variance_vs_raw` and `weighted_variance_vs_raw` give each estimate's variance relative to
raw, and `*_hands_worth` their inverse: how many raw hands one hand is worth. The command line
summary prints the latter. Intervals are nominal 95% intervals over hands; the weighted one ignores the small dependence
that shared weights introduce. `card_luck_bb_per_100` and `action_luck_bb_per_100` show what was
removed; both average zero in expectation.

## Evidence

- The vectorized evaluator equals `src.game.showdown.hand_value` on 20,000 random seven-card hands
  and edge cases; equities equal `allin_equity`'s exact counts on later streets and two preflop
  matchups.
- Zero mean is tested exactly: equity before each turn and river card equals the average after it,
  preflop equities are antisymmetric, and action corrections average exactly zero under the policy.
- Planted faults (wheel straights, kickers, dead cards, pot sizes, raise sizes, fold values,
  separator handling, policy weights, cross-fit leakage, street placement) each fail
  `tests/test_aivat.py`.
- A model plays fixture hands end to end against a scripted Slumbot transport, and the analysis
  reproduces every hand's corrections from the model.
