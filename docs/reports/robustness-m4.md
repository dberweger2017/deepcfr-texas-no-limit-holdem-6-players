# Saved HU20/TP20 robustness: completed M4 diagnostic

## Finding

Draft [PR #114](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/114) completed **947,200 evaluation hands**, plus 168 resource-preflight hands. Every hand replayed through native settlement with matching actions, public-event digests and chip ledgers. All 139 evaluation panels and four final audit/probe/demo phases completed; no failed, discarded or partial hand is omitted. Saved models and the #112/#113 human-play interfaces are preserved. No training, rental, merge or model promotion occurred.

The models improve on uniform, but have repeatable weaknesses:

- **Native pressure:** final HU loses **264.01 BB/100**, and final TP loses **279.86 BB/100** against two pressure opponents. Allowing raises beyond the training cap changes both the available actions and reached histories; this is a deployment limitation.
- **Inside the training menu:** final HU beats the reactive pressure rule, but loses **111.96 BB/100** against the information-safe local response. TP loses against homogeneous pressure and minraise lineups even within its menu.
- **Training curve:** the local-response comparison improves from 2M to 20M, but does not establish further improvement from 10M to 20M. This neither proves convergence nor identifies a card-abstraction ceiling.

**One recommended next intervention:** a separately authorized HU20 action-cap A/B that preserves the existing recipe and baseline while making the existing raise sizes available whenever the native engine permits them. Details and acceptance measurements are below. It is not launched by this PR.

## Scope, provenance and freeze

The [protocol](../20bb-robustness.md) and [frozen configuration](../../configs/blueprint/robustness-m4.json) specify all 15 saved inference hashes: three HU seeds at 2M/5M/10M/20M, three final TP seeds, and same-game uniform controls. Extraction remains final-current C. Preflight ran at `83c7e87`; confirmation ran at **`4883849a485ef49c9b2da6b97779f234f6dd8671`**; the finalizer ran at **`641bbd36875a03a4b5bfea7d5d19c33ee91bf783`**. Configuration digest: `f8d57b5289e5fb5a438be1f18aaa00ee6cd67e0f79aabb5b06ce3b67ae7c82f3`.

Claude's original temporary script was absent on both Macs. Pressure is a **precise reconstruction**, not a byte-identical reproduction: minraise whenever allowed; otherwise check when free; otherwise call with any pair, any ace or both ranks at least ten preflop, or pair-or-better postflop (including a board pair); otherwise fold. Minraise/check-call and passive check-call are separately committed controls. The requested `blueprint-learning-analysis-v3.md` was unavailable and requested; it was not independently reviewed. The new tests evaluate the actual saved campaign models.

| Phase | Frozen work | Completed hands |
| --- | --- | ---: |
| HU cheap stress | 13 policies × 3 rules × 2 contracts × 4,096 two-position blocks | 638,976 |
| TP cheap stress | 4 policies × 6 lineups × 2 contracts × 2,048 three-position blocks | 294,912 |
| HU local response | All 13 policies × 512 two-position blocks; four future samples per positive-mass holding, five soft seconds | 13,312 |
| Resource preflight/development timing | Separate deals; two/eight samples; outcomes suppressed | 168 |

Counts, 5M/10M inclusion and the common local-response budget were frozen before confirmation outcomes. Cheap stress finished first. Evaluation was sequential with one heavy process. Cards and positions are paired across targets; actual policy action streams and response sampling are separate. TP mixed rival order is balanced across blocks, with independent rival streams. Fresh roots have zero overlap with 29,184 prior HU and 29,824 prior TP deal seeds inventoried from 253 old hand files.

All intervals here are **exploratory 95% Student-t intervals over independent rotation-block means**, with seed outcomes or paired contrasts averaged within each shared block. Repeated policies on a deal are not independent observations. There is no multiplicity-adjusted promotion claim. Full per-seed, role, checkpoint, lineup and telemetry results remain in [results.json](robustness-m4-artifacts/report/results.json); [the complete table](robustness-m4-artifacts/report/report.md) includes every aggregate arm.

## Cheap attacks: absolute losses and uniform contrasts

Positive profit is from the target's perspective. Both action contracts use the same rule priority: **menu** selects only `choices(..., free_fold=False)` with the two-raise cap; **native** permits minraises beyond that cap. The saved target's lookup and fallback are unchanged.

Final HU, averaged across three seeds:

| Rule / contract | Target BB/hand | Target BB/100 [95% interval] | Trained − uniform BB/100 [95% interval] |
| --- | ---: | --- | --- |
| Pressure / menu | +0.3513 | +35.13 [24.52, 45.75] | +167.41 [151.83, 183.00] |
| Pressure / native | −2.6401 | −264.01 [−277.08, −250.95] | +47.77 [31.35, 64.20] |
| Minraise / menu | +0.7182 | +71.82 [58.24, 85.40] | +348.16 [325.68, 370.64] |
| Minraise / native | −2.5225 | −252.25 [−266.76, −237.73] | +61.29 [43.96, 78.62] |
| Passive / either | +1.0140 | +101.40 [92.40, 110.39] | +102.87 [91.32, 114.43] |

The paired native-minus-menu effect is **−299.15 [−313.09, −285.20] BB/100** for pressure and **−324.07 [−339.60, −308.54]** for minraise. Passive produces identical results under both contracts, as expected.

Final TP, averaged across three seeds:

| Lineup | Menu target BB/100 [95% interval] | Native target BB/100 [95% interval] |
| --- | --- | --- |
| Pressure / pressure | −25.80 [−40.00, −11.61] | −279.86 [−291.61, −268.11] |
| Minraise / minraise | −79.93 [−102.91, −56.95] | −304.65 [−316.40, −292.90] |
| Passive / passive | +116.31 [96.16, 136.47] | +116.31 [96.16, 136.47] |
| Pressure / minraise | −46.23 [−65.57, −26.90] | −285.22 [−296.27, −274.16] |
| Pressure / passive | +23.08 [1.23, 44.93] | −227.20 [−249.32, −205.09] |
| Minraise / passive | −26.28 [−50.92, −1.64] | −226.13 [−248.90, −203.36] |

TP pressure/pressure is **−0.2580 BB/hand** inside the menu and **−2.7986 BB/hand** under native stress. Its paired native-minus-menu effect is **−254.06 [−272.17, −235.95] BB/100**. Trained-minus-uniform is positive in both contracts: **+98.41 [78.03, 118.78]** and **+57.97 [43.16, 72.77]** respectively. Positive relative improvement does not remove the absolute losses. Individual-seed intervals also show native pressure losses for HU/TP and menu pressure losses for TP.

BB/hand is BB/100 divided by 100; 20BB buy-ins/100 is divided by **20**. HU native pressure's −264.01 BB/100 is approximately −13.20 buy-ins/100. TP results do not measure three-player exploitability.

## Validated HU local response

The committed instrument follows the one-step/checkdown construction of [Lisy and Bowling](https://arxiv.org/html/1612.07547v2), with documented approximations. It tracks the full compatible opponent range, conditions on actual target probabilities at the **pre-action** observation, includes declared uniform fallback, and removes newly public collisions. Target probabilities may be queried for hypothetical holdings; actual rival cards, future deck and live target RNG are never read or consumed.

The responder compares only the training menu. After an immediate raise, its heuristic models the target's fold probability and treats nonfold replies as calls followed by checkdown. Future boards are sampled conditional on each positive-mass holding and shared across candidate actions; river settlement is exact. **The played target continues its saved policy**, including raising. Hidden-state values are aggregated before maximizing; reported profits are actual chip outcomes, not noisy internal maxima.

Completed action-comparison batches publish together. A soft deadline is checked between batches, with at least one attempted batch; the external resource guard remains authoritative. The confirmation recorded **31,645/31,645 completed response decisions, zero soft overruns and zero zero-likelihood events**, with maximum 0.569 seconds. This instrument was not mostly timed out. Four future samples remain an approximation; outcome-suppressed two/eight-sample development timings do not establish that this budget is strategically optimal.

Production-code tests cover nonuniform Bayesian updates, explicit zero likelihood, card removal, hidden-card/deck invariance, unmodified live opponent RNG, legal actions, native accounting/runouts, an easily exploitable target, hidden-information maximization, an exact tiny-game best-response upper bound and a low-exploitability royal-flush reference. The initial full suite passed 842 tests; the final reporting revision's CI passed **845 tests**, with 11 focused instrument/report checks. These fixtures validate the instrument's mechanics; they are not an exact full-Hold'em best-response certificate.

| Saved work | Target BB/hand | Target BB/100 [95% interval] |
| --- | ---: | --- |
| 2M | −1.8162 | −181.62 [−214.25, −149.00] |
| 5M | −1.6823 | −168.23 [−200.48, −135.98] |
| 10M | −1.1292 | −112.92 [−141.44, −84.41] |
| 20M | −1.1196 | −111.96 [−143.48, −80.45] |
| Uniform | −2.8682 | −286.82 [−334.00, −239.63] |

Final trained-minus-uniform is **+174.85 [120.27, 229.44] BB/100**. The paired 20M-minus-2M target effect is **+69.66 [28.85, 110.47]**; 20M-minus-10M is **+0.96 [−33.03, 34.95]**, an exploratory contrast, not evidence of an abstraction ceiling. Final target seed results are −96.04 [−139.77, −52.32], −102.98 [−149.94, −56.02] and −136.87 [−180.87, −92.86]. The seed curves are not all monotonic.

| Target role | Target BB/100 [95% interval] | Opposite attacker role | Attacker BB/100 [95% interval] |
| --- | --- | --- | --- |
| Button / small blind | −129.00 [−193.79, −64.22] | Big blind | +129.00 [64.22, 193.79] |
| Big blind | −94.92 [−163.80, −26.05] | Button / small blind | +94.92 [26.05, 163.80] |
| Balanced | −111.96 [−143.48, −80.45] | Balanced | +111.96 [80.45, 143.48] |

In this zero-sum declared game, the legal attacker's **true expected payoff** lower-bounds best-response payoff. These finite-sample intervals describe realized returns; raw profit is not exact profile exploitability. A small/negative local-response estimate would not certify robustness. The [independent aggregation reader](robustness-m4-artifacts/independent/check_lbr.py) recomputed these checkpoint/role contrasts from the raw archive and agrees with the production report; its [summary](robustness-m4-artifacts/independent/lbr-summary.json) records the archive SHA-256.

## Exposure, fallback and conditional decisions

Counts below aggregate final seeds for pressure and legal local response (HU), or pressure/pressure (TP). Each cell is **trained / fallback**, separated by the preceding public-history contract. They measure decisions, not profit assigned to a street.

| Game / contract / preceding history | Preflop | Flop | Turn | River |
| --- | ---: | ---: | ---: | ---: |
| HU menu / menu-history | 33,320 / 0 | 16,564 / 21 | 10,207 / 135 | 5,934 / 123 |
| HU legal local response / menu-history | 3,581 / 0 | 2,551 / 7 | 914 / 18 | 336 / 9 |
| HU native / menu-history | 33,320 / 0 | 13,180 / 15 | 5,565 / 58 | 2,059 / 47 |
| HU native / off-menu-history | 0 / 7,432 | 0 / 10,376 | 0 / 5,420 | 0 / 2,250 |
| TP menu / menu-history | 22,516 / 70 | 8,845 / 430 | 2,931 / 1,116 | 753 / 768 |
| TP native / menu-history | 19,765 / 14 | 0 / 0 | 0 / 0 | 0 / 0 |
| TP native / off-menu-history | 0 / 17,331 | 0 / 0 | 0 / 0 | 0 / 0 |

Every flagged off-menu-history lookup in **these pressure panels** missed. This was measured rather than assumed: history bucketing can merge other off-menu events into trained keys. TP native pressure ended target decision-making preflop; its terminal losses cannot be assigned to a single preflop action merely from payoff. TP menu's thin trained river coverage is a separate on-menu limitation. Target action latency summaries are retained per panel; the largest recorded panel p95 estimate is 0.097 ms.

The [12 replayable flagged histories](robustness-m4-artifacts/report/exploratory-replay-cases.json) are the first eligible cases in the declared seed/block order, without profit selection. The separate [12 conditional river probes](robustness-m4-artifacts/river-probes.json) integrate a declared artificial uniform range of 24 compatible rival holdings against the fixed reactive rule and saved target continuation. All target lookups were trained. Eight initial-action mixture gaps are positive, four zero; maximum 11.25 BB. Native enumeration visited 2,112 nodes.

Those probes demonstrate conditional action-value losses **for that artificial range**, not for the true historical posterior. They do not attribute losses to card buckets, certify full-game equilibrium, use the realized opponent holding as hindsight evidence, or establish that any particular AA/AK sizing is inherently wrong. No same-bucket counterfactual pair was tested.

## Resources and integrity

| Measurement | Observed | Limit / verification |
| --- | ---: | --- |
| Preflight | 40.93 seconds | Began original absolute deadline |
| Main evaluation | 9,292.17 seconds | Completed all frozen counts |
| Through final inventory | 2.80 hours from preflight | Ten-hour absolute limit, never reset |
| Peak RSS including preflight | 1.187 GiB | 10.5 GiB |
| Peak finalizer RSS | 0.453 GiB | 10.5 GiB |
| Minimum free disk | 44.10 GiB | 8 GiB |
| System swap used | 761.38 MiB, unchanged | At most 0.5 GiB growth |
| Native replay | 947,200 main + 168 preflight | Every retained hand |
| Frozen model hashes | All 15 matched | Original inference lineages preserved |
| Global file inventory | 33 original files | Sealed after evaluation/audit logs stopped |

The [environment record](robustness-m4-artifacts/environment.json) was captured retrospectively before sealing: M4 arm64, macOS 26.6, Python 3.11.14, `pokers` 0.2.0 at native commit `5db20e3d5d6862b32a7402035c1340b622d3b005`; binary SHA-256 and package versions are retained. Every retrieved original artifact matched the global inventory byte count/hash. The independent reader/summary are additional local derived artifacts, not original files in that sealed M4 inventory. No evaluation retry or outcome-selected variant occurred.

## ONE next intervention: coherent HU action-cap A/B

Prioritize the large measured native-minus-menu gap while preserving the on-menu local-response test as a quality requirement. Propose a **versioned HU20 menu without the artificial two-raise cap**, retaining the current min/pot/conditional-jam sizes, card descriptor, ordered history, K1 updates and current extraction. Native reopening/stack bounds still govern legality. Preserve the three saved models and existing game version as the baseline; do not translate arbitrary histories into invented trained entries or change fallback ad hoc.

Before owner authorization for training, run an outcome-free expanded-tree resource preflight: measure branching, useful completed work, entry growth, RSS, throughput and interruption/recovery. Removing the cap can make the table sparser and training more costly. A proposed subsequent comparison would use three paired fresh seeds and equal declared node work for baseline versus expanded-menu training, with fresh paired menu/native pressure and the same fixed-budget HU local-response measurements. The existing 20M-per-seed budget is a starting proposal subject to that preflight, not an authorized campaign.

The objective is smaller native-stress losses without sacrificing on-menu quality; favorable reactive-rule profit alone is insufficient. This report does not justify changing card buckets by exclusion, guarantee further same-recipe scaling, or request paid hardware. Three-player on-menu weaknesses remain real and its worst-case quality remains unmeasured. Review this draft before approving the next experiment.

## Retrieval and run/replay

All large artifacts remain on `ssh m4`, in:

- `/Users/dberweger/Local/robustness-pr114/results/robustness-m4-20260928`
- `/Users/dberweger/Local/robustness-pr114/results/robustness-m4-20260928-audit`

The [global manifest](robustness-m4-manifest.json) supplies exact remote paths, sizes and SHA-256 values, including the 256,996,153-byte confirmation archive. Compact scientific outputs are committed here; the four audit logs and full hand archive remain remote. Supervisor logs are also retained at `/Users/dberweger/Local/robustness-campaign-20260928.log` and `/Users/dberweger/Local/robustness-finalizer-20260928.log` (outside the 33-file inventory).

```sh
mkdir -p results/robustness-retrieved
scp m4:/Users/dberweger/Local/robustness-pr114/results/robustness-m4-20260928/confirmation/hands.jsonl.gz results/robustness-retrieved/
shasum -a 256 results/robustness-retrieved/hands.jsonl.gz
# Expected: 1cc074de06bd30362f19367fa09d06ed27a42755ca42a69c2a12e7fdace609aa
python docs/reports/robustness-m4-artifacts/independent/check_lbr.py --hands results/robustness-retrieved/hands.jsonl.gz
```

The diagnostic command completed and natively replayed a pinned final seed-1 model against native pressure. Its one-hand outcome is not a strength estimate. Run another history on M4 without overwriting the retained demo:

```sh
ssh m4
cd /Users/dberweger/Local/robustness-pr114
PY=/Users/dberweger/Local/deepcfr-texas-no-limit-holdem-6-players/.venv/bin/python
$PY -m scripts.play_robustness --plan configs/blueprint/robustness-m4.json \
  --policy 2p-2026092801-20M --rule pressure --contract native \
  --history results/robustness-new-demo.json
$PY -m scripts.play_robustness --replay results/robustness-new-demo.json
```

`minraise` and `passive` select the other controls; use `--rule lbr --contract menu` for the declared local response. Existing [HU human play](../hu20-model-card.md) and [human-plus-two-TP-bots play](../tp20-model-card.md) remain intact, with original verified hashes and bot cards hidden until legitimate disclosure.
