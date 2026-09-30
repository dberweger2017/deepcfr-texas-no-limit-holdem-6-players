# Luna primary: where the result came from

**Independent, post-hoc analysis.** I examined the published primary session after its results were known. This decomposition, its thresholds and its questions were not part of the frozen [protocol](../luna-browser-benchmark-protocol.md). I used only committed [hand results](luna-browser/primary/hands.csv), [human decisions](luna-browser/primary/decisions.csv) and [public event history](luna-browser/primary/public-history.json). No private journal, unrevealed cards or RNG is needed. The [primary report](luna-browser-benchmark.md) remains authoritative. Reproduce the descriptive figures and disclosed-card table with `python -m scripts.analyse_luna_big_pots`.

## Summary

- Luna's realized result was **+107 BB / +26.75 BB per 100** over 400 completed hands. This does not establish a strength advantage or a statistical tie. The original target was 500; I stopped at 400 after progress/scores were visible, and technical interruptions are retained.
- Large-call hands drove the profit. At the 8 BB amount-owed threshold, Luna continued in 15 hands: **14 wins, one tie, no losses, +276 BB**. It folded in another 18 hands for −82 BB. The other 367 hands contributed −87 BB.
- Luna won **12 exact full stacks of 20 BB**, plus **two 19 BB near-full stacks**, and lost no full stack. Removing the ten largest winning hands leaves **−93 BB over 390 hands / −23.85 BB per 100**. This is a post-hoc sensitivity calculation, not an alternative estimate of the original result.
- The concentration is a useful candidate mechanism for further diagnosis of B100M's large wagers against strong continuations. It does not establish that the pattern is unlikely to be chance, that B100M generally outplays Luna elsewhere, or that a particular betting range is profitable.

## 1. Descriptive result and uncertainty limits

| Measure | Value |
| --- | ---: |
| Completed hands | 400 |
| Luna net | +107 BB / +26.75 BB per 100 |
| Per-hand observed standard deviation | 4.49 BB |
| Exact +20 BB hands / exact −20 BB hands | 12 / 0 |
| Additional +19 BB hands | 2: hands 237 and 277 |
| After removing the ten largest wins | −93 BB / 390 hands / −23.85 BB per 100 |

I report no confidence interval or significance test. Fresh deals and resetting stacks do not alone establish independent, identically distributed outcomes: Luna retained one context and could change its actions across hands. Pairing consecutive positions does not resolve that dependence or the stop after results were visible. A simple mean ± 1.96 standard errors calculation would require assumptions that were not validated here; an interval containing zero would still not demonstrate equivalence.

The script's earlier top-ten calculation subtracted those winnings but divided by all 400 hands, effectively zeroing the removed outcomes. The corrected calculation divides by the 390 hands actually retained. Neither procedure is evidence that the omitted wins should be discarded from the benchmark.

## 2. Decomposition by large-call exposure

A hand has **large-call exposure** if Luna owed at least the threshold at any recorded decision. Its first such decision classifies it as folded or continued (call or raise). The threshold is the exact displayed amount owed, read from `Call X BB`; it is not raise-to, pot fraction, total invested, or a general definition of a jam. The 5, 8 and 12 BB thresholds were selected after looking at outcomes and are overlapping decompositions of one session, not independent replications.

| Amount owed threshold | Luna folded | Luna continued | Other hands |
| --- | --- | --- | --- |
| 5 BB | 32 hands, −124 BB | 16 hands, +287 BB; 15 wins, 1 tie, 0 losses | 352 hands, −56 BB / −15.91 BB per 100 |
| **8 BB** | **18 hands, −82 BB** | **15 hands, +276 BB; 14 wins, 1 tie, 0 losses** | **367 hands, −87 BB / −23.71 BB per 100** |
| 12 BB | 11 hands, −52 BB | 7 hands, +136 BB; 7 wins, 0 ties, 0 losses | 382 hands, +23 BB / +6.02 BB per 100 |

Each row partitions all 400 hands and sums to +107 BB. The continued category is strongly positive in this realized sample. The other category changes sign at 12 BB as the membership changes. These selected outcomes cannot establish optimality, general relative strength or a causal effect of the bet threshold.

I removed the illustrative binomial tail table. Arbitrarily assuming a 60–80% constant independent win probability for selected strong-hand continuations does not supply a validated null model or a significance test for this post-hoc pattern.

## 3. Observable decisions and legitimate disclosures

Luna folded in 214 of 400 hands (53.5%), losing an average of 1.80 BB in those hands. Among decisions with a displayed call, its fold rates were 60/105 (57.1%) on the flop, 34/77 (44.2%) on the turn and 26/47 (55.3%) on the river. These are descriptive counts, not evidence that folding was excessive.

The table covers **all 15 continued hands at 8 BB**, including the +18 BB hand and the tie. Human cards and board are from the first qualifying decision. Bot cards are **later, legitimately shown cards** from public history; they were not available to Luna at the earlier decision. “Not shown” remains unknown: it must not be filled from private replay.

| Hand | First exposure street | Luna cards | Board then | Later shown bot cards | Luna net BB |
| ---: | --- | --- | --- | --- | ---: |
| 39 | Turn | 5h 5c | Kc 5d 7c 2s | 4s Ac | +20 |
| 41 | Flop | Ac Kc | Ks Jh Ah | Qc Qh | +20 |
| 46 | River | Qs 6s | Qd 8d 9c 6h Qh | 3c 5c | +20 |
| 89 | Flop | 8s 7s | 8h 9s 7h | Not shown | +20 |
| 134 | Turn | Jc Th | Td Tc 7d 4d | Not shown | +20 |
| 136 | River | 7d 8s | 7h 7c 4d 8c Ks | Kc 8h | +18 |
| 137 | Turn | Kd Tc | 8c Kh Kc 9d | Qs Js | +20 |
| 186 | River | 6c Th | 8h Qh 9c 7h 6h | Td 6d | +20 |
| 208 | River | 7h 6d | 8d 5c 4d Jc 5s | Kd 3d | +20 |
| 216 | Preflop | As Ac | None yet | Not shown | +20 |
| 224 | River | 2s Ad | Kd 4d 8d 8s 5d | Tc 7d | +20 |
| 237 | Turn | Ad 8s | Qh Kd Jd Td | Jh 7s | +19 |
| 277 | Turn | Qh Qc | 7s Qs Ts Jd | Jc 6s | +19 |
| 320 | River | 9c 3s | 6s 7s 8c 6d Tc | Kd 9h | 0 |
| 382 | Flop | 8h 7h | 8s 7c 2c | Not shown | +20 |

These continuations show strong made hands or pocket aces; hand 320 plays a straight shared with the opponent. That supports a strong-hand-selection hypothesis for these particular calls. It does not show that every stack commitment or every unsampled decision was good. In particular, winning a selected hand is not an independent estimate of its decision EV.

Public history already records both players' actions and legitimate bot disclosures in **11 of these 15 hands**. For example, in hand 46 Luna bet to 1 BB on the river, B100M raised to 17 BB, and Luna called the remaining 16 BB. B100M later showed 3c 5c against Luna's full house. That is a concrete public trace worth investigating; it does not quantify the frequency or intended role of that action across the policy's entire range.

## 4. B100M's realized side, not betting EV

At 8 BB, B100M gained +82 BB in the 18 folded hands and lost −276 BB in the 15 continued hands: **−194 BB across this selected 33-hand subset**. These are whole-hand net payoffs, including earlier investments and subsequent actions.

The previous roughly 80% break-even fold calculation used those whole-hand averages as if they were the gain and risk of one bet. It cannot establish a betting-EV threshold: that requires the pot and incremental wager at each decision, continuation equity and subsequent play. I removed the claim that the range “only works” against opponents folding more often. No counterfactual EV or optimal fold frequency is estimated here.

The primary audit records 894 trained lookups and zero fallback. Uniform missing-key fallback therefore did not contribute to the recorded run. This identifies the policy path used, not the underlying cause of the large-pot losses.

## 5. Limits and follow-up

- Post-hoc categories, one opponent, one saved policy seed, one interrupted and shortened session. No multiple-comparison adjustment, valid confidence claim or confirmation is supplied.
- Restricted native HU20 with 20 BB reset each hand. This does not establish free-sizing, three-player or six-player capabilities.
- Legitimate disclosures are incomplete and selected by the showdown rules. Mucked or folded cards remain unavailable; shown hands do not represent the whole betting range.
- Outcomes spread across the session can still arise from favorable deals. Their dispersion alone does not rule out variance.
- No new model run, training, private-journal extraction or compute campaign was performed for this analysis.

The next LLM benchmark recommendation remains the primary report's single proposed time-budgeted replication with stable browser recovery and a preregistered completion target. Before any new run, this public decomposition can inform a **prospective** large-call diagnostic; it is not permission to launch one or revise the model from these selected outcomes.

## Validation

`python -m scripts.analyse_luna_big_pots` reproduces the threshold partitions, exact versus near-full-stack counts, corrected top-ten denominator, fold rates and disclosed-card table. Regression tests check the recorded 400-hand partition/conservation, the 390-hand denominator, exact 20 BB classification, first-threshold decision, empty groups and reliance on public disclosures only. `python -m pytest tests/play_ui -q` passes **39 tests**. No existing benchmark result or runtime is changed.
