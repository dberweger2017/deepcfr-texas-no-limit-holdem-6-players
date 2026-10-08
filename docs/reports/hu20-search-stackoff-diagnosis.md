# HU20 turn search against selective stackoff

October 8, 2026 · [PR #208](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/208) · M1 only · recorded-hand diagnosis, no new play.

## Finding

The proposed **over-calling large opposing bets or jams** mechanism is contradicted by these hands. Selective-stackoff v1 never makes those bets here: its 800-chip `LARGE_CALL` constant describes **the amount it will call**, not the amount it bets. It raises the minimum, with probability 35%, when strong and facing a small or free wager. Search called where base folded once, costing 0.07 BB/100; that price was small.

There is a reproducible mismatch in **search's belief about callers and their next response**. In the clearest losing hand, after the opponent called a 400-chip turn bet, search's river range was only 17–20% strong; the frozen rules imply 100% strong. Search then always bet large and expected roughly 52–55% folds, whereas that opponent always calls. This is a real model mismatch, but it does not establish that correcting it would recover the entire observed regression. Smaller bets and checks also contribute substantially, and some large negative contrasts are search wins compared with bigger base wins.

The performance regression is **plausible, not established across the exploratory family**. The original −10.35 BB/100 result reproduces exactly. Its 13-panel Bonferroni interval includes zero. All three frozen lineages have negative contrasts, but they share deals and are not independent confirmations. Recommend a fresh, predeclared replication before choosing a policy change. Search adoption remains unresolved.

## Inputs and method

The fresh isolated checkout started at main `9da625f`. Search, tree and frozen-opponent source are unchanged from #166's launched revision `8632ccf`. All three 500M-node average policies are the exact frozen exports, seeds 2026093001–3003. The engine is pinned to `5db20e3d5d6862b32a7402035c1340b622d3b005`.

The restored [#166 closeout](hu20-fixed-work-arena/attempt-2-closeout.md) ZIP is 5,645,638,363 bytes, SHA256 `cac0d2a766a8d0276820662112b68bb12ffe3688b8597faa0f63d43a829b3a0e`. The whole ZIP was verified before extraction. Every selected member was size/hash checked against `RESEARCH_MEMBER_HASHES.json`, whose SHA256 is `83830d63ea88f56272da4b8fe402dd9e9f56a3bb245e120eca212b3a50365dcc`. Only 42 compressed arena traces and the plan were initially extracted. Later retrieval verified the nested solver archives, nine selected original requests and the three inference exports. Working files stayed in ignored, non-synced `results/stackoff-diagnosis/` until archival.

All **3,072 selective-stackoff hands /9,552 decisions** replay through the pinned native engine with identical legal transitions, final stacks and event digests. There are 1,536 paired hand coordinates: 256 deal blocks × three lineages × two position rotations. Within each independent deal block, average the six search-minus-base outcomes; use those **256 joint blocks**, not 768 lineage blocks or 1,536 hands, for uncertainty. Units throughout are BB/100, 100 chips per BB. Results are conditional on these three fixed lineages, not a population of training seeds.

For attribution, assign the **entire paired hand's outcome difference** to its first differing action. Identical paths must settle identically. Public contexts at the first difference and all preceding decisions must match. Each table below is a disjoint partition; different tables overlap and must not be added together. These are realized whole-hand contributions, **not the causal EV of one decision**. Later runouts and both subsequent policies contribute.

## Where the loss comes from

Only 158 paired hands differ; the remaining 1,378 contribute zero. Their net difference is −159 BB, or −10.35 BB/100.

| First-difference dimension | Hands | Contribution, BB/100 |
|---|---:|---:|
| Turn | 108 | −8.14 |
| River | 50 | −2.21 |
| **Total** | **158** | **−10.35** |

| Search action at first difference | Hands | Contribution, BB/100 |
|---|---:|---:|
| Bet/raise | 100 | −7.62 |
| Check | 53 | −2.54 |
| Fold | 4 | −0.13 |
| Call | 1 | −0.07 |

Calling large turn/river bets that base folds contributes **zero**, because there are no such opportunities. It is misleading to describe all 100 differing raises as additional aggression: 28 are smaller than base's raise.

| Base → search change | Hands | Contribution, BB/100 |
|---|---:|---:|
| Check → raise | 45 | −2.28 |
| Call → raise | 3 | −1.17 |
| Fold → raise | 1 | −0.33 |
| Larger raise | 23 | −0.26 |
| Smaller raise | 28 | −3.58 |
| Raise → check | 53 | −2.54 |
| Raise → fold | 1 | −0.39 |
| Call → fold | 3 | +0.26 |
| Fold → call | 1 | −0.07 |

Smaller raises plus checks account for −6.12 BB/100, about 59% of the net loss. Thus the observed deficit is heterogeneous; a blanket aggression veto has no demonstrated recovery value.

| Price faced at first difference | Hands | Contribution, BB/100 |
|---|---:|---:|
| Free | 143 | −8.72 |
| 1–200 chips | 15 | −1.63 |
| Above 200 chips | 0 | 0.00 |

| Pot at first difference | Hands | Contribution, BB/100 |
|---|---:|---:|
| Below 800 chips | 149 | −8.46 |
| 800–1,999 chips | 9 | −1.89 |
| 2,000+ chips | 0 | 0.00 |

The 100 differing search raises comprise 62 minimum raises (−4.10), 36 pot raises (−2.86) and two jams (−0.65 BB/100). Ten leave the opponent at least 800 chips to call; those hands contribute −1.95 BB/100. The other raises contribute −5.66. Search raises after an earlier opposing raise on the same street in ten hands, contributing −1.43.

| Immediate opposing response to the first differing search raise | Hands | Contribution, BB/100 |
|---|---:|---:|
| Strong calls | 20 | −6.90 |
| Strong raises | 4 | −0.13 |
| Medium calls | 20 | −3.06 |
| Medium folds | 18 | +2.15 |
| Weak folds | 34 | +0.33 |
| Board-only folds | 4 | 0.00 |
| First difference is not a raise | 58 | −2.73 |

Hands where a strong opponent immediately continues contribute −7.03 BB/100, about 68% of the deficit. This supports inspecting value continuations and traps, but includes smaller search raises and later decisions. “Strong” is the opponent's coarse rule tier: generally two pair or better, including some strength supplied by the board; exact river board-only hands are excluded. It is not a nuts range or an equity measure.

## Same public spots facing bets

There are **35 shared turn/river decision opportunities facing a positive price**, all 1–200 chips. Include the first differing decision, then stop the common prefix. Never pool post-divergence spots as if they matched. All three lineages and both rotations are included.

| Frequency at these same 35 spots | Base call /fold /raise | Search call /fold /raise |
|---|---|---|
| Recorded actions | 16 /10 /9 | 11 /12 /12 |
| Recorded actions, percent | 45.71 /28.57 /25.71 | 31.43 /34.29 /34.29 |
| Mean recorded policy probabilities, percent | 45.66 /31.39 /22.95 | 37.93 /38.69 /23.38 |

Expected calls decrease and folds increase. The increase in realized raises mostly reflects sampling: mean raise probability changes by only 0.43 percentage points. For prices of 201–799 or 800+, **the denominator is zero and frequencies are undefined**. This panel cannot answer how search reacts to large opposing bets; the LBR/native-pressure panels contain such bets, but have different opponents. All-path, unmatched counts are retained separately in the archive.

## What search believed

Select the most negative first-divergence hand per lineage, then the next two losing contrasts with distinct public contexts. Inspect each first difference and its largest subsequent late-street wager: five hands, eight snapshots, nine necessary distribution queries/native solves including one prerequisite. This is a deterministic, intentionally selected diagnostic sample, not a random EV sample.

For every solve, reconstruct and exactly match the original scientific request before running the external native tool. Keep 50 iterations, six threads, uncompressed profiles and the original menus; lower only native memory admission from the configured 5 GiB to 3 GiB to respect the family guard. All nine menus match; **maximum absolute error against the recorded action probabilities is 0.0**. The binary SHA256 is `a172854c88e9ef17e61b30b8dce19b5570d765ed34e2ff533d8e4395b82776b1`, with the frozen harness source hash checked before use.

Search ranges below condition its root weights on known hero cards and the solved likelihood of observed opposing actions. The diagnostic frozen-rule posterior starts uniformly over card-compatible holdings, uses every prior public opposing action and integrates the 35% trap draw. It never conditions on the opponent's recorded hidden cards or realized random draws. Hidden cards are reported only to explain actual outcomes. Solved response probabilities come from the already completed profiles; frozen responses apply the rules to that posterior. **Neither alternative ranges nor rule knowledge are supplied to a solver or policy.** Full holding weights, root ranges, categories and request provenance are archived; [compact examples](hu20-search-stackoff-diagnosis/offline-examples.json) retain the summaries.

| Selected spot | Search strong mass | Frozen-rule strong mass | Search's predicted folds to actual wager | Frozen-rule folds |
|---|---:|---:|---:|---:|
| 3001 /block 14, river | 17.45% | 100% | 54.67% | 0% |
| 3003 /block 14, river | 19.88% | 100% | 51.73% | 0% |
| 3002 /block 208, river | 88.43% | 100% | 54.61% | 0% |
| 3002 /block 176, turn | 5.27% | 8.88% | 70.94% | 91.12% |
| 3003 /block 127, river | 36.68% | 100% | 51.85% | 0% |

Seed suffixes abbreviate 2026093001–3003. The two block-14 rows are the same deal, not independent examples.

**Block 14:** hero J♣T♣, board J♠2♥6♦5♦, opponent actually 5♣5♠. Search bets 400 on the turn; that call, being outside the rule's small-wager region, rules out medium hands. After river 9♥ and an opposing check, search still has 80–83% medium mass. It puts 100% probability on a pot bet of 1,200 or a 1,400-chip jam; base at that *same search-path public spot* checks 91–95%. Actual search bets 1,200 and gets called by trips. Search loses 18 BB versus base's 3 BB loss: −15 BB for each of lineages 3001 and 3003. This isolates a credible offensive error after a revealing call. Base's river distribution is an offline same-state query; it did not reach that state in the base arm.

**Block 208, lineage 3002:** hero A♣5♥, board 9♠K♥Q♦J♠9♥, opponent actually T♥J♥, a straight. Facing a 100-chip river minimum bet with pot 900, base calls; search raises to 1,100. Search assigns 88.43% strong mass, yet its solved opponent folds 54.61% to this raise. The rules imply 100% strong and 100% calling. Search puts 21.14% on pot raise/jam, 77.42% on fold and 1.44% on call; base calls with 74.74%. The whole-hand contrast is −10 BB. **Narrowing the root range alone may not fix an incorrect future response model.**

**Counterexamples to a single over-aggression explanation:** block 176 /3002 has hero A♠T♣ on K♠A♣J♦2♦. Search bets 600, opponent J♠T♠ folds; search wins 3 BB, while base's smaller bet leads to a 20 BB win. Its −17 BB contrast is the largest per-hand deficit, not an over-call or a losing bluff. Block 127 /3003 first differs by search checking where base bets 100; hero A♣3♣ later improves on the river. Search wins 9 BB versus base's 20 BB. These realized differences include future-card variance and foregone value; they are not proof that either first decision has negative EV.

### Likelihood floor

The arena's **actual floor was 0**, not the class default 0.01. Changing a default would not explain this result. An analytical 0→0.01 comparison at each round root, with recorded turn strategy factors fixed, changes normalized opponent weights by **0.27–2.55% total variation**; newly admitted root support carries at most **0.381%** mass. This is only a root-belief sensitivity. No alternate-floor solve or gameplay was run, and existing profiles cannot evaluate newly admitted holdings at later nodes. Increasing a floor generally restores low-likelihood holdings; it is not evidence for narrowing this caller range. Floor-only tuning is not recommended from these data.

## Is the regression real?

| Contrast, search minus base | Estimate | 95% interval, BB/100 |
|---|---:|---|
| Joint selective-stackoff, original unadjusted | −10.35 | [−18.89, −1.82] |
| Joint selective-stackoff, 13-panel Bonferroni | −10.35 | **[−22.99, +2.29]** |
| Lineage 3001, descriptive unadjusted | −7.81 | [−19.09, +3.46] |
| Lineage 3002, descriptive unadjusted | −9.57 | [−22.06, +2.92] |
| Lineage 3003, descriptive unadjusted | −13.67 | [−24.18, −3.17] |

Joint SE 4.334, t(255)=−2.388, two-sided p=0.01765; Bonferroni-adjusted p=0.2294. The family interval uses t(255) at 1−0.05/(2×13), preserving the original block estimator. It is a conservative adjustment to the specified 13-panel exploratory family, not a claim to cover every possible post-hoc diagnostic.

Only 57 joint blocks have nonzero contrasts: 35 negative and 22 positive. The worst five account for −7.10 BB/100, about 69% of the net loss. This concentration and the winning-hand counterexamples make chance contribution credible. Do not delete these blocks or select a favorable subset. The repeatable model mismatch and consistently negative lineage estimates make “likely noise” too strong; **plausible but unconfirmed regression** is the supported conclusion.

## Conditional fix design

First replicate unchanged search. If a candidate is then authorized, make **public observation update both opponent holdings and predicted continuation behavior**. Track public call sizes, pot fractions and aggression, plus legitimately shown cards from prior hands. Shrink an opponent model toward the blueprint when observations are sparse; maintain uncertainty over both the range and response frequencies. A large call should constrain next-street continuation beliefs, rather than silently reverting to blueprint/GTO behavior.

For offensive departures from base, require a positive estimated advantage across credible models before betting more or reopening a pot; otherwise use base's distribution. This is a design proposal, not implemented policy or a justified threshold. Data/model fitting must be separate from this diagnosis: **no hidden cards, v1 rule labels, exact frozen-rule posteriors, opponent identity or hard-coded 800-chip exception from these hands may feed into it**. Diagnostic examples are evaluation evidence only. Defensive search decisions can remain available.

This localization aims to preserve gains against wide aggressive opponents; it cannot promise to preserve them. The recorded native-pressure gain is +25.82 [14.48,37.17], with first-difference calls contributing +13.48 BB/100. LBR gains +39.35 [30.43,48.26], with first-difference target raises contributing +18.93. Thus an indiscriminate raise veto risks substantial LBR benefit. These are exposure partitions, not simulated fallback-policy effects. LBR can itself differ before the target's first action because it queries hypothetical future target policy; those rival-first hands are explicitly partitioned, not forced into a target decision. A later candidate needs fresh selective-stackoff, LBR and native-pressure safeguards, as well as the still-missing direct search-versus-no-search match.

## Fresh confirmation proposal and quote — not run

Propose **unchanged search versus base**, fixed 1,024 fresh joint selective-stackoff blocks, all three frozen lineages and both position rotations: 12,288 total hands. Use a new root `202610080166`, reserved only on owner approval. Primary estimator averages the six paired outcomes within each block. Predeclare one two-sided 95% t interval; confirm the regression only if its upper bound is below zero and the estimate is at most −5 BB/100. A lower bound above −5 supports that practical noninferiority margin; all other outcomes are inconclusive. No sample extension or outcome-based reruns. This fresh single primary is separate from the exploratory 13-panel family.

Reuse identical preflop/flop paired state prefixes and RNG state for this non-querying frozen opponent. Resolve only reachable turn/river decision states where search is eligible to depart from base, cache identical scientific requests and branch the paired paths as needed. Still settle **every whole hand**, including zero differences; do not cherry-pick the recorded losers or only decisions that happened to differ in #166. No hidden-card or frozen-rule posterior can enter search.

**Machine: M1. Quote: 8–12 elapsed hours, $0 paid compute, hard 12-hour total cap.** Sequential nice-15 workers; 4 GiB family RSS, normal memory pressure, reserve admission, ≤0.5 GiB new swap, ≥10 GiB free disk, and AC guard unless the owner explicitly retains the power waiver. Native allocation ≤3 GiB; scientific work stays 50 iterations/six threads/uncompressed/original menus/floor 0. Stop and preserve the entire partial attempt on guard failure, illegal action, missing/compressed/incomplete profile, numerical failure or information-boundary violation; never silently substitute base or drop a hand to meet the cap.

The original 256-block panel used 490 fresh native solves (279 turn, 211 river). Scaling suggests roughly 1,960 solves. The nine selected M1 reproductions total 102 seconds of native work; turn solves take 12.06–24.38 seconds and river solves 0.22–0.50 seconds. At four times the old panel, this implies roughly 4–8 hours of native work, plus range compilation, replay and owner-load headroom. The selected spots are not an unbiased timing sample; admission/load can stop a fixed-N confirmation short. **This is a quote, not authorization or scheduled work.** No new arena, search sweep, training or default change was run.

## Reproduce and verify

Use the pinned Python environment/dependencies from `docs/development.md` and #166's external exact-turn harness. Commands run from a fresh checkout of this PR on `dberweger-m1`, with the original ZIP at its indexed local path and the frozen external binary at `~/Local/hu20-fixed50-search-tool/harness/target/release/hu20-exact-flop-tool`:

```sh
python -m scripts.run_hu20_search_stackoff_diagnosis restore
python -m scripts.run_hu20_search_stackoff_diagnosis analyze
python -m scripts.run_hu20_search_stackoff_diagnosis retrieve
python -m scripts.run_hu20_search_stackoff_diagnosis resolve
python -m scripts.run_hu20_search_stackoff_diagnosis tests
python -m scripts.diagnose_hu20_search_stackoff compact --inputs results/stackoff-diagnosis --out docs/reports/hu20-search-stackoff-diagnosis
```

Each stage has a bounded cap, sequential execution, an exclusive phase lock, nice-15 priority and 4 GiB family/pressure/swap/disk guards. The wrapper refuses non-M1 or non-ignored/synced working roots. `--root`, `--archive` and `--binary` allow practical local paths. `--owner-power-waiver` records the owner's explicit waiver used here; default runs require AC. Restoration refuses an existing input directory. Resolve can reuse only completed local profiles whose member hashes still match.

The successful local native phase peaked at **2.60 GiB whole-family RSS**, below the 4 GiB ceiling. The M4 and other processes were untouched. Initial Drive hydration timed out and then failed a whole-ZIP hash gate; after hydration completed, independent SHA256 verification and guarded restoration matched the pinned hash. A mistakenly retained AC check stopped model preparation before any solve; continuation used the owner's waiver. Two analysis-only failures were preserved: an overly strict rival-first assumption for LBR and an unsupported-row floor calculation. The final analysis explicitly handles LBR's policy queries and restricts floor sensitivity to the round root. No arena hand was rerun. All original failures, profiles and guard logs accompany the derived archive.

Ten focused tests pass, covering joint-block accounting, disjoint contribution conservation, duplicate/incomplete pairs, matching public contexts and settlements, hydration seek/retry integrity, scientific request normalization, integrated frozen-rule probabilities and the large-call threshold. A fixture-construction failure during test development is retained with the other logs; it changed no analysis. [One independent review](hu20-search-stackoff-diagnosis/independent-review.json) independently reproduced statistics, shared-prefix counts, all eight diagnostic posteriors/response models and all nine scientific request/member checks, with no open findings. Final-head CI is recorded in the PR. [RESULTS_INDEX](../../RESULTS_INDEX.md) records the accepted derived archive, member hashes, original input/model dependencies, retrieval commands and local cleanup receipt.
