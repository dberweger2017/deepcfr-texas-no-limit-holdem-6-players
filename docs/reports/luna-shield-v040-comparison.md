# Luna against v0.4.0 and Shield

This compares the 400 completed hands from [#127's v0.4.0 B100M match](luna-browser-benchmark.md) with [#174's Shield match](luna-shield-chrome.md). It is a **post-hoc, descriptive comparison of two unpaired sessions**. Positive numbers below mean Luna wins. The script reads only committed public events, exports and accepted-action records; it does not recover hidden opponent cards or private reasoning.

## Results and position split

| Measure | v0.4.0 B100M, September 30 | 0.4.0-shield, October 6 | Shield match minus old match |
|---|---:|---:|---:|
| Completed hands | 400 | 400 | 0 |
| Luna net BB | +107 | −82 | **−189** |
| Luna BB/100 | +26.75 | −20.50 | **−47.25** |
| Button / small blind, 200 hands each | +54.5 BB (+27.25 BB/100) | −120.5 BB (−60.25 BB/100) | **−175 BB** |
| Big blind, 200 hands each | +52.5 BB (+26.25 BB/100) | +38.5 BB (+19.25 BB/100) | −14 BB |
| Wins / losses / ties | 164 / 231 / 5 | 180 / 217 / 3 | +16 wins, −14 losses, −2 ties |
| Average terminal pot | 7.0925 BB | 8.075 BB | +0.9825 BB |
| Exact +20 BB / −20 BB hands | 12 / 0 | 12 / 3 | Same full-stack wins; three additional full-stack losses |

**175 of the 189 BB swing (92.6%) comes from the button/small blind.** Luna remained profitable from the big blind in both recorded samples. Its higher number of winning hands against Shield did not produce a profit: hand size matters, not just the win count. These are realized sample differences, not a measured 47.25 BB/100 increase in Shield's strength.

## The earlier large-pot profit still appears, but is smaller

The [earlier large-pot analysis](luna-browser-big-pot-analysis.md) found that strong continuations against large amounts owed supplied much of Luna's profit. Applying the same categories to Shield gives the following disjoint partition. A hand is classified by Luna's **first decision owing at least 8 BB**; a call or raise is continued, a fold is folded, and hands without such a decision are other. Figures are final whole-hand payoffs, including earlier investment and subsequent play.

| First exposure to ≥8 BB owed | v0.4.0 B100M | Shield | Net payoff difference |
|---|---|---|---:|
| Luna continued | 15 hands; 14 wins, 0 losses, 1 tie; **+276 BB** | 14 hands; 11 wins, 2 losses, 1 tie; **+160 BB** | −116 BB |
| Luna folded | 18 hands; **−82 BB** | 25 hands; **−125 BB** | −43 BB |
| Other hands | 367 hands; **−87 BB** | 361 hands; **−117 BB** | −30 BB |
| Total | 400 hands; **+107 BB** | 400 hands; **−82 BB** | **−189 BB** |

Luna still earned substantial profit in the large-call continuation subset. It also encountered more qualifying folds and less favorable continuation outcomes. These three differences sum to the overall swing, but **do not establish which bets or calls caused it**: the memberships, cards and earlier action paths differ across sessions. Thresholds of 5 and 12 BB are retained in the [raw comparison](luna-shield-chrome/comparison.json); the thresholds were inherited from an exploratory historical analysis, not prospectively frozen for this run.

Both sessions' ten largest wins sum to +200 BB. Removing them leaves −93 BB over 390 hands against B100M and −282 BB over 390 against Shield. That sensitivity calculation describes concentration; it does not justify discarding legitimate wins or replacing the actual benchmark scores.

## Luna's button play changed

| Observable button behavior | v0.4.0 B100M | Shield |
|---|---:|---:|
| First action: raise | 122/200 (61.0%) | 163/200 (81.5%) |
| First action: limp | 53/200 (26.5%) | 32/200 (16.0%) |
| First action: fold | 25/200 (12.5%) | 5/200 (2.5%) |
| Hands with an observed response to the bot's three-bet after Luna opened | 32 | 52 |
| First response: fold / call / re-raise | 20 / 11 / 1 | 27 / 22 / 3 |
| Continued rather than folded | 12/32 (37.5%) | 25/52 (48.1%) |
| Final payoff in three-bet-called hands | 11 hands, **+59 BB** (5 wins / 6 losses) | 22 hands, **−51 BB** (5 wins / 17 losses) |

The accepted records support Luna's retrospective about difficulties against three-bets: it opened more often, called more such re-raises and had worse realized results in those called hands. They do **not** prove that any individual continuation was an expected-value error or that opening less would have improved its score. The extra raises could reflect the new context, different deals, opponent responses or adaptation; this comparison cannot separate them. The three-bet and large-call subsets overlap and must not be added together as independent loss explanations.

## What is comparable, and what changed

Both runs used runtime-confirmed `gpt-6-luna` / `high`, restricted heads-up Hold'em, 20 BB reset stacks, no rake/ante, alternating positions and one continuing player context within each match. All 400 hands replay in each run, and both policies recorded zero missing-key fallback lookups.

The earlier opponent was shipped B100M seed 2026093001's current policy; Shield was CFR+ seed 2026100601's 1B-node traverser-reach average. Training recipe, budget and extraction therefore all changed. Deals were not paired. The old player was a fresh child in the internal browser; the new player was launched independently in native Chrome with a different conversation and handoff. Retained observations permit adaptation within either run, but do not make the two player strategies fixed or identical.

The earlier 500-hand plan was stopped at 400 for time after progress/scores were visible; its status remains ABORTED. Shield's target was 400 before play and completed exactly. The old run logged 979 attempts for 978 accepted decisions, with one attempt/accepted mismatch, one retry and missing acknowledgment metadata retained; its audit detected no prohibited capability. Shield logged none of the planned attempt/observation records, used tools outside the prescribed boundary and did not receive the frozen player text. Its 1,012 accepted decisions are verified, but browser intent, retries and decision latency are unverified. Thus this is not an otherwise identical repetition of the prior benchmark.

**Interpretation:** Shield had the better realized result against these two Luna sessions, chiefly through the button/small-blind swing. That is useful matchup evidence and suggests reviewing three-bet continuations and large-pot trajectories. It neither proves general superiority to v0.4.0 nor demonstrates that Luna learned or exploited a stable weakness. In the separate [fresh paired direct campaign](hu20-cfr-plus.md), the exact Shield seed used here lost to shipped R1 at **−9.64 [−13.07, −6.21] BB/100**. That interval belongs to the independent direct campaign, not this browser comparison. The opposite results answer different matchup questions and do not overturn the declined release decision.

## Reproduction

Run `python -m scripts.compare_luna_matches` and compare its output with [comparison.json](luna-shield-chrome/comparison.json). The reconstruction checks every accepted action against both CSVs and every older amount owed against rendered call labels, then reproduces the published historical 8 BB partition and full-stack counts. All partitions conserve 400 hands and the final score; input SHA256s are included. No new play, training, private-journal extraction, confidence interval for Luna or counterfactual action-value estimate is included.

After integrating current main, all **57 focused play/audit/average/release tests pass**. The comparison reproduces exactly from the retained inputs; diff and relative-link checks pass.
