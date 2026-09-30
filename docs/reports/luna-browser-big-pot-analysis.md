# Luna primary: where the result came from

**Independent, post-hoc analysis.** This was written by a reviewing agent after the primary session ended and its results were published. It is not part of the frozen [protocol](../luna-browser-benchmark-protocol.md): the decomposition, the thresholds and the questions were all chosen after looking at the data. It uses only the committed public-view files [`primary/hands.csv`](luna-browser/primary/hands.csv) and [`primary/decisions.csv`](luna-browser/primary/decisions.csv), with no private journal, bot cards or RNG. The [primary report](luna-browser-benchmark.md) remains the authoritative record of the session. Reproduce with `python -m scripts.analyse_luna_big_pots`.

## Summary

- **The headline is statistically a tie.** Luna's +26.75 BB/100 over 400 hands has a per-hand 95% interval of about **[−17, +71] BB/100**. The session was also stopped at 400 of its planned 500 hands after the running score was visible.
- **All of Luna's profit comes from a small set of large pots.** Luna won **14 full stacks and lost none**. In the 15 hands where Luna called or raised against a bet of at least 8 BB, it won 14 and lost 0, for **+276 BB**. In the other **367 hands, Luna lost −87 BB (−23.7 BB/100)**.
- **B100M's measured weakness here is specific:** it keeps committing its stack against a player who only continues with strong hands. Outside those pots, it outplayed Luna on this sample.

The first point says the result is noise-level overall. The second and third say where the losses concentrate, and that pattern is much less likely to be noise.

## 1. The headline, with uncertainty

| Measure | Value |
| --- | ---: |
| Hands | 400 |
| Luna result | +107 BB, **+26.75 BB/100** |
| Per-hand standard deviation | 4.49 BB |
| 95% interval (per hand) | **[−17.3, +70.8] BB/100** |
| 95% interval (consecutive-hand pairs, one per button assignment) | [−16.6, +70.1] BB/100 |
| Without Luna's 10 largest wins | **−23.25 BB/100** |

Hands are treated as independent because each has a fresh deal and 20 BB stacks reset. Pairing consecutive hands as one block changes the interval very little. Neither interval accounts for the early stop.

## 2. Decomposition by large-bet exposure

A hand counts as **large-bet** if Luna ever faced a call of at least the threshold. Luna's action at its first such decision classifies it as **folded** or **continued** (call or raise). The main threshold is **8 BB**, where the bet is 40% of a 20 BB stack or more, which is usually a jam or near-jam. Results at 5 BB and 12 BB are shown for robustness.

| Threshold | Luna folded | Luna continued | All other hands |
| --- | --- | --- | --- |
| 5 BB | 32 hands, −124 BB | 16 hands, **+287 BB (15 won, 0 lost)** | 352 hands, −56 BB (−15.9 BB/100) |
| **8 BB** | **18 hands, −82 BB** | **15 hands, +276 BB (14 won, 0 lost)** | **367 hands, −87 BB (−23.7 BB/100)** |
| 12 BB | 11 hands, −52 BB | 7 hands, +136 BB (7 won, 0 lost) | 382 hands, +23 BB (+6.0 BB/100) |

(Continued hands that are neither won nor lost were ties.)

**Every threshold gives the same shape.** When Luna put more chips in against a big bet, it never lost. At 12 BB the "other hands" row turns slightly positive, because some big-bet hands then move into it. Either way, the large positive total always comes from the continued row.

**How unusual is 14 of 15?** If Luna's calls against big bets won at a realistic rate, getting 14 or more wins out of 15 would be rare:

| Assumed per-call win rate | Chance of ≥14 wins out of 15 |
| ---: | ---: |
| 60% | 0.5% |
| 70% | 3.5% |
| 80% | 16.7% |

These rates are illustrative, not estimates. A player calling only with its strongest hands can win often, but only if the bettor's range contains too few hands that beat it. The pattern suggests B100M's large bets and calls in these spots were weak relative to the hands that continued.

## 3. What Luna did

- **Very tight.** It folded in 214 of 400 hands (53.5%), losing an average of 1.80 BB each time.
- **Folded often when facing postflop bets.** 57% on the flop, 44% on the turn and 55% on the river.
- **Committed its stack only with strong made hands.** The table shows Luna's cards and the board at its last recorded decision in each hand it won a full stack. The board may have changed after that decision.

| Hand | Luna position | Luna cards | Board at last Luna decision |
| ---: | --- | --- | --- |
| 39 | Button/SB | 5h 5c | Kc 5d 7c 2s |
| 41 | Button/SB | Ac Kc | Ks Jh Ah 3h |
| 46 | BB | Qs 6s | Qd 8d 9c 6h Qh |
| 89 | Button/SB | 8s 7s | 8h 9s 7h Jd Qc |
| 134 | BB | Jc Th | Td Tc 7d 4d 3h |
| 137 | Button/SB | Kd Tc | 8c Kh Kc 9d |
| 186 | BB | 6c Th | 8h Qh 9c 7h 6h |
| 208 | BB | 7h 6d | 8d 5c 4d Jc 5s |
| 216 | BB | As Ac | preflop |
| 224 | BB | 2s Ad | Kd 4d 8d 8s 5d |
| 237 | Button/SB | Ad 8s | Qh Kd Jd Td Tc |
| 277 | Button/SB | Qh Qc | 7s Qs Ts Jd 9c |
| 307 | Button/SB | 2s Qs | 2c Qd Qc 2d |
| 382 | BB | 8h 7h | 8s 7c 2c 5h |

These are sets, trips, straights, a nut flush, full houses, pocket aces and two-pair hands. In several of them, Luna first made a small bet or min-raise, and the next decision faced a much larger bet. That is consistent with B100M raising big over small bets from a strong hand, but the public files don't record B100M's actions directly.

The 14 stack wins are spread across the whole session (hands 39–382). They don't cluster in one stretch, so this doesn't look like a single run of good cards.

## 4. B100M's side

Taking the 8 BB threshold from B100M's perspective:

- **When Luna folded to a big bet,** B100M won 82 BB over 18 hands, about **4.6 BB per fold**.
- **When Luna continued,** B100M lost 276 BB over 15 hands, about **18.4 BB per call**.
- **Net in these 33 hands:** about **−194 BB** for B100M.

For big bets like these to break even, the opponent must fold about 18.4 / (18.4 + 4.6) ≈ **80%** of the time. Luna folded **55%** (18 of 33). A range that bets big this often only works against a player who folds much more.

The [primary report](luna-browser-benchmark.md) records **894 trained lookups and zero fallback** for B100M. This loss therefore comes from the learned strategy itself, not from the bot drifting outside its tree.

## 5. Limits

- **Post-hoc and exploratory.** The thresholds and the decomposition were chosen after seeing the results. Nothing here has been adjusted for multiple comparisons, and it isn't confirmation.
- **No B100M cards.** The public files don't include B100M's cards, even when they were shown at showdown, so they can't show *what* the bot bet or called with. The "bluffing too much" versus "value-betting too thin" question is open.
- **One opponent, one seed, one session.** 400 hands against a single LLM player and the first-seed B100M policy only.
- **Restricted mode.** Both players used the trained min/pot/jam menu. Free sizing might change the pattern.
- **The bet threshold is approximate.** It's read from the "Call X BB" button label, which is the amount owed, not the bet's size relative to the pot.

## 6. Suggested follow-ups

1. **Use legitimately revealed showdown cards.** For the 15 continued hands, pull B100M's cards from the private journal wherever they were shown at showdown. That would show whether its big bets and calls were bluffs or overplayed medium hands. It stays within the benchmark's information rules and changes no result.
2. **Pass this pattern to the posterior audit** as a candidate mechanism: large bets and all-ins, especially after a small raise from the opponent. Check whether B100M's high-visit mistakes concentrate on those decisions, and whether the conditional-jam part of the menu or coarse card buckets are involved.
3. **Add a fixed trap-style opponent** to the adversarial panel. It folds often, min-bets or min-raises strong hands, and continues against big bets only with strong made hands. It's cheap and fixed, and it targets exactly this pattern.
4. **Report large-pot decompositions** alongside headline BB/100 in future human and LLM benchmarks. At 20 BB, a few stack-offs can dominate a 400-hand result.
