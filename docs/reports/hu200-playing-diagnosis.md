# HU200 improved against loose-aggressive; important weaknesses remain

**100M improves over 20M against loose-aggressive, but still loses to it and
pot-pressure.** The other four paired gains are inconclusive after the declared
five-comparison adjustment. Recommend **action/history-support work next and
defer the Slumbot pilot**. Higher visits plausibly help the covered aggressive
branches, but cannot populate histories excluded by the unchanged training menu.
No new training, recipe change, live match or release occurred.

[PR217](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/217)
starts from current main `4c4e04c`; evaluation source is immutable
`57d9705de423e70370ba59243d9f398b62104207`. It uses only #216's audited
**20,002,716 /100,000,034-node** opponent-sampled averages, training seed
**2026100905**, at the original HU200 game/schema/menu/card abstraction,
uniform missing/zero-mass fallback and translation off. Hashes, bytes,
checkpoint provenance, iteration and identities match its
[model index](hu200-feasibility-artifacts/model-index.json). Existing local
originals were read directly; models, checkpoints and native runtime were neither
copied nor subjected to another full training-accumulator audit.

## Frozen comparison and precision

The [prospective protocol](../hu200-playing-diagnosis.md) preceded outcomes.
Five primary 100M-minus-20M comparisons use **2,048 fresh paired blocks/opponent**,
identical deals and rotated candidate seats: **4,096 hands/checkpoint/opponent,
40,960 hands total**. Both checkpoints share separately labeled private action
streams. Tables reset each hand. Deal root 2026100906 is disjoint from timing
root 2026100907 and all earlier smoke/evaluation schedules. Timing outcomes were
retained without producing or inspecting winrate summaries. The
[frozen plan](hu200-playing-diagnosis-artifacts/frozen-plan.json) was sealed at
154.64 seconds, before final hands or winnings inspection. The full target fit;
no reduction, optional stopping, restart or outcome-dependent selection occurred.

All values below are **BB/100**. Absolute intervals are descriptive 95% Student t
intervals over seat-averaged blocks. Gain intervals are 99% per opponent, giving
simultaneous coverage of at least 95% under Bonferroni's five-comparison family.
Rotations are not independent samples. These are deal/action-stream intervals
for one training lineage, not uncertainty across training seeds or general poker
strength. The gain half-widths range from 41.6 to 201.5 BB/100; four inconclusive
contrasts must not be read as equality or a plateau.

| Opponent | 20M [descriptive 95%] | 100M [descriptive 95%] | Paired gain [simultaneous 95% family] |
|---|---:|---:|---:|
| random | +45.9 [−154.0,245.7] | +132.1 [−64.4,328.5] | +86.2 [−56.0,228.4] |
| check_call | +92.1 [61.2,123.0] | +70.0 [43.6,96.3] | −22.1 [−63.7,19.5] |
| tight_aggressive | +5.1 [−67.6,77.7] | −23.0 [−94.6,48.6] | −28.1 [−127.1,71.0] |
| loose_aggressive | −476.0 [−627.6,−324.4] | −197.4 [−325.8,−69.0] | **+278.6 [77.1,480.0]** |
| pot_pressure | −404.8 [−516.9,−292.7] | −271.3 [−375.9,−166.7] | +133.5 [−5.4,272.4] |

[Exact primary/absolute results and disjoint hand categories](hu200-playing-diagnosis-artifacts/summary.json).
No opponent pool score or release gate was invented after seeing these results.

## Coverage, support and visits

The existing #201 finite legal-history/menu witness search is compatible with
HU200's ordered key. Present keys are stored training witnesses; missing keys on
all-menu paths are supported but absent. Other missing histories receive an
exhaustive compatible-token-tree search, capped at 5,000 states/case and 120s/model.
Alternate witnesses are legally replayed and their **full HU200 key** checked.
Only exhaustion declares unsupported; neither absence nor an off-menu amount
alone does. Final searches used **387 /369 public signatures** and **0.303 /0.281s**
for 20M/100M; **zero unresolved** cases. In 20M pot-pressure, four off-menu
missing decisions have supported alternate paths. They would be misclassified
by a simple missing-key/off-menu rule.

| Opponent | Missing 20M →100M | Supported absent 20M →100M | Unsupported 20M →100M |
|---|---:|---:|---:|
| random | 1,053 →1,026 | 12 →5 | 1,041 →1,021 |
| check_call | 59 →7 | 59 →7 | 0 →0 |
| tight_aggressive | 11 →0 | 11 →0 | 0 →0 |
| loose_aggressive | 92 →10 | 92 →10 | 0 →0 |
| pot_pressure | 976 →934 | 7 →0 | 969 →934 |

Counts are **decisions**, not unique keys/histories. Changed trajectories change
denominators. More unchanged-menu training cannot reach those proven unsupported
abstract histories, although it can alter how often play encounters them.

Positive-mass coverage and low visits differ. The complete
[40-row opponent/street/checkpoint table](hu200-playing-diagnosis-artifacts/coverage-visits.csv)
retains missing, zero mass and **0 /1 /2–9 /10–99 /100+** traverser-visit bands.
Here each cell is **20M →100M**; low-visit percentages use stored-key decisions,
including zero mass, and exclude missing keys.

| Opponent /street | Positive mass (%) | Stored visits <10 (%) |
|---|---:|---:|
| random /preflop | 82.6 →83.2 | 11.5 →0.5 |
| random /flop | 82.0 →82.6 | 9.6 →2.0 |
| random /turn | 81.2 →80.4 | 35.7 →9.3 |
| random /river | 85.0 →87.2 | 60.4 →20.3 |
| check_call /preflop | 100.0 →100.0 | 11.2 →0.0 |
| check_call /flop | 99.7 →100.0 | 7.6 →0.9 |
| check_call /turn | 98.9 →99.8 | 33.5 →6.8 |
| check_call /river | 99.3 →99.9 | 54.4 →16.3 |
| tight /preflop | 100.0 →100.0 | 5.7 →1.1 |
| tight /flop | 98.9 →100.0 | 10.6 →1.6 |
| tight /turn | 97.8 →99.7 | 26.0 →11.0 |
| tight /river | 97.5 →100.0 | 37.8 →13.4 |
| loose /preflop | 99.8 →100.0 | 16.3 →2.2 |
| loose /flop | 99.3 →100.0 | 11.9 →3.2 |
| loose /turn | 95.9 →99.4 | 43.6 →17.3 |
| loose /river | 94.0 →98.7 | 65.7 →29.1 |
| pot /preflop | 88.9 →89.5 | 6.6 →0.3 |
| pot /flop | 67.3 →68.6 | 8.3 →0.8 |
| pot /turn | 50.8 →53.0 | 20.0 →4.7 |
| pot /river | 36.0 →34.5 | 42.1 →8.9 |

At 100M loose-aggressive's river, **64 decisions at stored keys have zero traverser
visits**, all with positive average mass; 19 have one and 165 have 2–9. Four
other stored river decisions have zero mass despite positive traverser visits. Positive average
mass can coexist with zero traverser visits. Supported absent states, stored
zero visits and zero mass remain separate in the
[decision strata](hu200-playing-diagnosis-artifacts/decision-strata.csv).
Those strata include every win/tie/loss and decision-weighted final hand payoffs;
repeated decisions repeat the outcome. They are not street losses or action values.

## Covered losses, sizes and stack-offs

At 100M, loose-aggressive's **2,959 all-positive hands** contribute **−216.66
BB/100** of its −197.40 total. Ten supported-missing hands contribute +17.48,
seven zero-mass/no-missing hands −11.89, and no-decision hands +13.67.
Tight's 2,245 all-positive hands contribute −45.61, offset by +22.58 from
no-decision hands; its overall interval remains inconclusive.
Pot-pressure instead has **543 ever-unsupported hands** contributing −310.11,
1,893 all-positive hands +18.58 and no-decision hands +20.26, totaling −271.26.
These categories are disjoint and use all 4,096 hands as the denominator.
**These are conditional associations, not causal attribution or estimates of
what support work, more training or a different action would earn.**

Actual action frequencies below are decision percentages, in **fold/check/call/raise**
order. Raise-to is the mean total street wager in BB, conditional on raising;
ratio is the raise increment over the prior highest wager divided by pot plus
call. All per-street counts, expected action probabilities and ratio bands remain
in [coverage/behavior](hu200-playing-diagnosis-artifacts/coverage-behavior.json);
[bet-size summaries](hu200-playing-diagnosis-artifacts/bet-sizes.json) retain street
counts, means, medians and ranges, including stack-off score/commitment proxies.

| Opponent | Frequencies 20M →100M | Mean raise-to BB 20M →100M | Mean ratio 20M →100M |
|---|---|---:|---:|
| random | 23.6/13.8/21.0/41.6 →25.1/14.8/21.7/38.4 | 4.17 →4.28 | .68 →.64 |
| check_call | 4.1/34.2/3.5/58.2 →5.2/38.6/3.6/52.6 | 3.55 →3.04 | .62 →.59 |
| tight | 20.9/14.2/17.7/47.3 →25.1/14.8/17.4/42.6 | 9.62 →9.95 | .73 →.70 |
| loose | 15.3/17.3/15.0/52.4 →19.0/19.0/15.2/46.8 | 19.28 →16.48 | .68 →.64 |
| pot | 19.6/11.3/19.9/49.2 →23.3/12.5/19.7/44.6 | 15.00 →14.28 | .69 →.67 |

A large pot reaches >=64BB before settlement, including the final action's paid
chips. Stack-off means voluntarily paying the entire remaining stack; winning
settlement/refunds do not erase that event. These definitions prevent terminal
runout and winner-accounting omissions.

| Opponent | Large-pot losses 20M →100M | Stack-offs /losses 20M →100M | Net contribution of all stack-off hands, BB/100, 20M →100M |
|---|---:|---:|---:|
| random | 790 →756 | 514/243 →510/232 | +25.88 →+102.91 |
| check_call | 33 →26 | 0/0 →0/0 | 0 →0 |
| tight | 50 →47 | 62/31 →59/33 | −27.20 →−48.05 |
| loose | 336 →209 | 307/171 →230/119 | −289.77 →−114.65 |
| pot | 202 →159 | 130/88 →111/67 | −265.70 →−153.12 |

At 100M loose's stack-offs split **153 raises /77 calls**, by preflop 84,
flop 39, turn 47, river 60. Tight has 42 raises/17 calls; pot 64/47. Check-call
never raises, and the candidate's conditional-jam menu never voluntarily exhausts
its stack in that panel. [All tail counts and loss contributions](hu200-playing-diagnosis-artifacts/tail-losses.json).
Large-pot and stack-off subsets overlap and must not be added. Reduced aggression,
smaller bets and fewer stack-offs accompany the loose gain, but this experiment
does not establish that any one of them caused it.

## Representatives and reproducibility

The prospectively declared lowest coordinate SHA256 within each checkpoint,
opponent, exposure/sign and large-pot-loss/stack-off-loss category selects
**74 unique hands**, including wins. Multiple category tags are merged for the
same coordinate. [Representative index](hu200-playing-diagnosis-artifacts/representative-index.json)
retains every coordinate, rank, tag and raw-record hash; full cards/actions/events,
menus/probabilities, private seeds and support proofs are in the science ZIP.
No largest loss or visually interesting hand replaced this selection.

Some 100M examples, all selected by that rule:

- Loose, block **181 /seat 1**, all-positive loss: 53o folds preflop, −0.5BB;
  its key has 496 visits and 88.63% fold probability. The corresponding win
  category, block **1330 /seat 0**, raises K4s to 2BB and wins 1BB.
- Loose, block **835 /seat 1**, stack-off loss: T7s raises the river to 80BB on
  Tc As Kh 6d 9s, then calls a jam and loses 200BB. The final covered key has
  three visits and a 50/50 fold/call average. This is a sparse covered branch,
  not a measured action-value error.
- Pot, block **1623 /seat 1**, unsupported loss: 87o reaches unsupported turn
  and river histories; uniform fallback calls, raises the river to 48BB and
  folds to a further raise, losing 56BB.
- Pot, block **1985 /seat 0**, unsupported win: 82o makes trips, follows
  unsupported turn branches and wins 200BB after a stack-off. Support gaps do
  not imply losing every exposed hand.

Restore the ZIP into a fresh ignored root as indexed in RESULTS_INDEX. Restore
or locate the two #216 averages with their exact hashes. To reproduce a selected
hand, use the evaluation source and admitted environment, load its average once,
and call `play_hand(model, 2026100906, opponent, block, seat, {}, [0.])` from
`scripts.evaluate_hu200_diagnosis`; compare the returned record's SHA256 with the
index. Support proof search counts can depend on the cache: compare the exact
actions/events/settlements/probabilities and record hash after using the original
full schedule/cache when exact proof counters are required. For a complete
reproduction, run each frozen-plan worker at the indexed source, then the pinned
native `parity` command and reporter under a separately admitted budget. This
instruction does not authorize another run.

## Verification, cost and storage

All **40,960 hands /155,954 actions /71,482 candidate decisions** pass legal
observation/action checks, 40,000-chip conservation, full event/settlement replay,
and candidate/rival policy reproduction. The #216 SHA-pinned native runtime
independently checks every hand's actor/street/pot/legal bounds/menu/key/settlement:
20M **79,663 decisions**, 100M **76,291**, **zero mismatched hands**.
Independent raw-record/hash/count readback reconstructs all five paired estimates
and adjusted intervals, ten absolute means, and coverage/visit/support/action
counts. [Recount receipt](hu200-playing-diagnosis-artifacts/independent-recount.json).
28 focused checks pass, with regressions rejecting changed primary and coverage
reports. No game/rules/information-boundary/training-math source was changed.

The M1 exclusive lock and process inventory passed; M4 and other PR roots were
untouched. The clock begins at admission and includes hash/header/source snapshot,
calibration, final play/replay, primary readback/report, archive/readback and
closeout: **361.63s /6.03min**, without failures, retries or guard latches.
The measured admission quote was **1,126.75s /18.78min** of remaining work,
including a 600s closeout reserve and 2x planning margin; it was a conservative
quote, not an observed runtime. Fixed calibration loads **35.75 /113.45s**
were kept separate from scalable play/replay and native startup costs. Final
loads **35.84 /114.95s**, all-panel play/replay **20.72 /20.09s**. Fixed source
snapshot **0.545s**, local ZIP/readback **1.624s**. #211's fixed-cost lesson is
therefore addressed without scaling loads by the hand count.

Across 669 resource samples: peak family **2.066GiB**, kernel command peak
**2.027GiB** (different scopes), maximum within-operation gap **0.607s**, minimum
free memory **64% /5.015GiB psutil available**, minimum free disk **27.695GB**,
zero swap growth, normal pressure and AC throughout. Limits remain 3/4GiB family
soft/hard, 15.5GiB disk floor and original-baseline +512MiB swap/3GB total.
[Resources](hu200-playing-diagnosis-artifacts/resources.json),
[costs](hu200-playing-diagnosis-artifacts/costs.json),
[closeout](hu200-playing-diagnosis-artifacts/closeout.json).
Compact post-closeout table rendering/recount took 2.08s then 3.96s (the latter
includes added verification); it creates no new hands or inference. This
administrative report work is distinct from the guarded primary science clock.

The locally verified **31,933,966-byte ZIP /63 members**, SHA256
`d109861dda08a410c7bc356709c15f836cd3c8a4d7d55bb64153e3959ea7b8ce`, is in
[PR217's Research-Cloud folder](https://drive.google.com/drive/folders/1yBymLKissQIjA3QVSmvyW4nzafO_Slqo).
Embedded `ARCHIVE-MANIFEST.json` SHA256
`238709f951943ebc08d2888e671ad7c2a2cd405ccc02f4f8d0b0611014bfa800`;
every member's size/SHA256 passed local readback. It retains timing/final raw
hands, native fixtures, support/behavior/primary/representative outputs, resources,
logs, plan and source. Mutable archive-monitor logs and final closeout/Drive
metadata remain compact Git receipts. #216's model/runtime archive is referenced,
not duplicated. [Archive receipt](hu200-playing-diagnosis-artifacts/archive-receipt.json).
**Upload-pending owner/later-agent handoff:** one-time Drive metadata identifies
the folder and ZIP, but no native upload-acceptance check or remote byte download
was performed. No upload wait, cloud acceptance claim, deletion or forced offload.
All originals remain. [Drive handoff](hu200-playing-diagnosis-artifacts/drive-handoff.json).

Source and final evidence review were performed by the primary agent; these are
**not independent reviewer approvals**. Resolved source findings, final diff and
validation are recorded in [review](hu200-playing-diagnosis-artifacts/final-review.json).
The PR remains unmerged and ready for owner review.

## Concrete hypotheses for later confirmation

1. **Support:** explicit HU200 action/history support or a separately reviewed
   translation experiment can reduce pot-pressure fallback exposure. Confirm on
   fresh paired deals with the two policies fixed, legal witness parity and a
   declared gain family. The observed −310.11 exposure contribution is not a
   forecast of the fix's gain.
2. **Covered learning:** higher visits may further improve loose-aggressive's
   covered turn/river stack-off branches. The 20M→100M gain supports testing this,
   while 29.1% of stored loose river decisions remain below ten visits. A new
   bounded training experiment needs its own memory/save/load/replay quote and
   fresh confirmation; this task authorizes none.
3. **Residual representation/averaging:** if covered losses persist at adequate
   visits, test lifetime-average carryover versus later-window averages, card
   abstraction and omitted exact prices/stacks in controlled comparisons. This
   sample cannot separate those mechanisms, and does not show a convergence
   plateau, encoding failure or inevitable over-aggression.

Do not use this result to qualify HU100 or declare Slumbot strength/readiness.
A Slumbot pilot also needs its separately reviewed service lifecycle and owner
permission; no live endpoint was contacted here.
