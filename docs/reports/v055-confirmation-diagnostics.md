# v0.5.5 confirmation: card luck and losing public-action paths

**Research in progress:** public-action tables have run; the profile cross-check, full-sample AIVAT and archive acceptance are pending. This PR is not complete.

[Protocol version 2](../hu200-v055-acceptance-protocol-v2.md) uses **raw BB/100**. The [#318 confirmation](hu200-v055-blend-confirmation.md) remains **−19.67 [−38.57, −0.76] BB/100** and its qualification verdict stands. Everything here is informational and descriptive. There is no new play, training, model loading or own-action correction.

## Inputs and method

The analysis uses the three 10,000-hand confirmation accounts, [#304’s three 10,000-hand seeds](hu200-v055-slumbot-confirmation.md), and the blend’s 10,000 exploratory hands from [#316](slumbot-aivat-real-runs.md): **70,000 distinct completed hands**. Exploratory hands never enter the confirmation pool.

All three indexed ZIPs match their SHA256s; every member matches its manifest; all eleven journal hash chains verify. The #316 index names the exploratory journals directly rather than an indexed journal ZIP. Its 8,300-hand verified prefix and 1,700-hand continuation were retrieved from that recorded source; the prefix matches its preservation receipt. Their ordered hand IDs, positions and winnings will also be checked against #316’s archived per-hand corrections. Both journals will be preserved in this PR’s archive.

The existing `scripts.profile_slumbot_runs` tool profiles completed terminal records. The [public-action helper](v055-confirmation-diagnostics/public_tables.py) splits those events by size, position, preflop pot type and betting role. It reports final net chips, including blinds. The complete [readable tables](v055-confirmation-diagnostics/public-tables.md) and [CSV receipts](v055-confirmation-diagnostics/tables/) cover every account and comparison run.

A response opportunity is a bet/raise followed by an answer on the same flop. A re-raise creates another opportunity. Response counts therefore need not equal hand counts. Terminal flop folds and the BB first-response partitions each count a hand once; the two partitions overlap and must not be added together.

Sizes follow the profile tool: **raise beyond a call / pot after calling**, with all-ins separate. A half-pot bucket is (0.4, 0.6], the next is (0.6, 0.85], approximately pot is (0.85, 1.2], and large is (1.2, 2.5]. The inherited `<0.4` label includes exactly 0.4. The analyzer JSON’s `responses_by_bet_size` pools streets and uses a different convention for raises; it cannot by itself answer the flop question. Its `actions_by_street` and raw totals are cross-checked separately.

## Card-only AIVAT

Pending the completed 30,000-hand run. Each account uses #316’s [per-hand wrapper](slumbot-aivat-real-runs/dump_aivat.py) and the existing analyzer. The pool concatenates accounts A, B, C and fits its own cross-fitted weights, rather than averaging account fits. Earlier runs use only the four saved card columns from #316; their own-action columns are excluded.

Plain and weighted intervals, worth, removed luck and its zero-mean check, both halves’ street weights, seat results and six charts will be added here. Intervals use the analyzer’s nominal 95% convention over hands; weighted intervals omit the small dependence introduced by shared fitted weights. No acceptance test is recalculated.

## What accounts for the flop-fold loss

The terminal public actions reproduce #318 exactly: **our 3,775 flop folds lose 1,161,000 chips (−38.70 BB/100)**, while **Slumbot’s 2,795 flop folds win 773,250 (+25.775 BB/100)**, net **−12.925 BB/100 over all 30,000 hands**.

These are fold counts, not matched-strategy comparisons. We and Slumbot use different bet sizes and reach different ranges and pot types. Folding more often does not by itself establish an error.

The three largest recurring groups of our flop folds are:

| Rank | Concrete path | Confirmation folds / opportunities | Loss over all hands | Earlier-run losses |
|---|---|---:|---:|---:|
| 1 | BB calls a 2 BB button open, checks, then folds to a half-pot bet | 2,333 / 4,018 | −15.55 BB/100 | −13.76 to −16.24 |
| 2 | Button faces a half-pot c-bet in a 3-bet pot | 188 / 335 | −4.60 BB/100 | −2.91 to −5.25 |
| 3 | Button limps, calls the BB’s raise to 3 BB, then folds to a half-pot c-bet | 371 / 698 | −3.71 BB/100 | −3.48 to −3.87 |

Each path appears at similar frequency in **every one of the seven individual runs**. Their combined gross loss is −23.86 BB/100; other terminal flop-fold paths net +10.94, leaving −12.925. The second group’s two main exact paths are opening to 2 BB and calling a 6 BB 3-bet (116 folds, −2.32), and opening to 3 BB and calling a 9 BB 3-bet (70 folds, −2.10). The remaining two folds are in other 3-bet-or-larger pots.

The [flop response tables](v055-confirmation-diagnostics/public-tables.md) give fold/call/raise shares for both players, every observed size and both positions. The [fine table](v055-confirmation-diagnostics/tables/flop-responses.csv) further separates single-raised, limped and 3-bet-or-larger pots, continuation bets, non-aggressor bets and raises. Slumbot’s folds against our smaller and pot-sized bets offset much of the loss against its half-pot bets. Those different mixes prevent a direct “excess folds at equal ranges” interpretation.

## Big blind: opening sizes and response outcomes

Slumbot **always opens to 2 BB when it enters from the button** in all seven runs: no limp and no other open size. In the confirmation it opens 13,535 of 15,000 buttons (90.23%) and folds the other 1,465.

| Our first BB response | Hands | Share facing a 2 BB open | Final net chips | Chips per selected hand | Adds BB/100 over BB hands |
|---|---:|---:|---:|---:|---:|
| Fold | 2,417 | 17.86% | −241,700 | −100.00 | −16.11 |
| Call | 7,991 | 59.04% | −501,300 | −62.73 | −33.42 |
| Raise | 3,127 | 23.10% | +177,850 | +56.88 | +11.86 |
| Slumbot folds; no response | 1,465 | — | +73,250 | +50.00 | +4.88 |
| **All BB hands** | **15,000** | | **−491,900** | **−32.79** | **−32.79** |

The call group loses in every run, contributing −14.41 to −43.94 BB/100 over BB hands. Its confirmation conditional mean is **−62.73 ± 23.56 chips per call hand**. That is less loss than folding’s posted 100 chips, but selected ranges differ: it is not a counterfactual comparison of actions. Raise outcomes vary in sign (seed 1302 loses); these tables do not show that raising more would help.

**Preflop endings almost balance:** our preflop folds contribute −20.33 BB/100 over BB hands, and Slumbot’s preflop folds +20.46, net **+0.12**. Hands continuing past preflop contribute **−32.92**: our later folds −125.32, Slumbot’s later folds +96.86, showdowns −4.45. This allocates final outcomes by where a hand ends, including its preflop investment; it is not a causal assignment of losses to streets.

Among the 7,991 initial calls, our flop folds contribute −34.45 BB/100 over BB hands, turn folds −20.74 and river folds −26.99. Slumbot’s folds and showdowns offset those gross losses. The largest BB call/check-fold path is the same path ranked first above, so those numbers overlap.

## Ranked targets for future diagnosis

1. **BB call of the 2 BB open → check facing the half-pot c-bet.** This is the largest specific recurring terminal loss and sits inside the largest losing BB response group. Examine its range and continuation support before proposing search or training changes.
2. **Button calls a 3-bet → faces a half-pot flop c-bet.** It is less frequent but loses 6–9 BB on most folds. Both principal preflop sizes recur across seeds and blend accounts.
3. **Button limp-call of a 3 BB isolation raise → faces a half-pot flop c-bet.** This is a separate, stable path with a narrow and repeatable monetary contribution.

These are descriptions of where chips leave the observed samples. No alternative action has been evaluated, no fix is claimed, and rare one-run paths are not promoted into findings. The full [consistency table](v055-confirmation-diagnostics/public-tables.md#consistency-of-the-largest-specific-flop-fold-paths) retains all runs, including variation in BB raises and showdowns.

## Reproduction and evidence

Pending final AIVAT receipts, commands and accepted archive. The initial launcher failure occurred before saving corrections; its traceback and the successful smoke check are retained with the research evidence. The PR records plumbing failures as they occur. The owner decides merge.
