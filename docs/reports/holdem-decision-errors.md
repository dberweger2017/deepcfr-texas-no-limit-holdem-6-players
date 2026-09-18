# What the saved poker decisions reveal

September 18, 2026. This is a post-hoc analysis of the completed
[M4 full-game run](holdem-local-fullgame.md) and
[Runpod multistreet benchmark](holdem-multistreet-campaign.md).
No new training, policy inference, rental or sealed-test comparison was performed.
The interpretation and source review were performed by the primary agent;
Luna prepared the multistreet extraction code only.

**The full-game models frequently commit 100 BB preflop. The supervised models
make a smaller number of expensive, sometimes elementary postflop mistakes.**
These findings give us concrete behaviors to investigate. They do not identify
one common cause or justify promoting a model.

## Full game: preflop commitment dominates the observed losses

Each final scripted-pool evaluation contains 6,144 candidate hands, arranged in
1,024 deal blocks with six seat rotations. These are saved validation hands,
not fresh release confirmation. We identify the candidate through the recorded
lineup and count all-ins using its remaining stack after blinds and earlier bets.
A first-action shove means its first action was a preflop all-in raise.

| At iteration 256 | Seed 2026091802 | Seed 2026091803 | Uniform control |
| --- | ---: | ---: | ---: |
| First-action shoves | 2,288 (37.24%) | 2,152 (35.03%) | 680 (11.07%) |
| First-action shoves facing at most 1 BB to call | 1,928 (31.38%) | 1,863 (30.32%) | 544 (8.85%) |
| Hands with any preflop all-in | 2,527 (41.13%) | 2,272 (36.98%) | 1,194 (19.43%) |
| Total result, BB/100 | −1,056.11 | −869.63 | −1,262.13 |
| Contribution from preflop-all-in hands, BB/100 | −733.37 | −693.86 | −379.26 |
| Contribution from later-all-in hands, BB/100 | −183.56 | −90.08 | −370.39 |
| Contribution from hands without an all-in, BB/100 | −139.18 | −85.69 | −512.48 |

Percentages use all candidate hands as the denominator, including walks. The
three contribution rows partition each policy's recorded return and sum to its
total. The control is identical across both seeds and is shown once.

Preflop-all-in hands account for **69.44% and 79.79% of the models' recorded net
losses**. This is accounting, not an estimate of recoverable losses. Policies
select different hands into these groups; differences between groups or arms
are not causal effects of going all-in. Facing at most 1 BB also includes
limped pots and should not be described as always opening an untouched pot.

The trained models improve overall on uniform play while shoving first much
more often. Their smaller losses in the other commitment groups do not make
the resulting strategy competent.

### Actual actions

These examples come from the final saved hands. Cards are available only when
shown in the event record. The extractor takes the first three shown, unpaired
shove examples without an ace, king or queen, in file order; it does not select
on whether the hand won. This showdown-selected subset cannot estimate the
population distribution of dealt hands or shove holdings.

| Seed / block / rotation | Hero cards | Recorded decision | Realized result |
| --- | --- | --- | ---: |
| 1802 / 1 / 3 | 2♦4♥ | Raised to 100 BB with 7 BB left to call | +300.5 BB |
| 1802 / 4 / 0 | 4♦3♥ | Raised to 100 BB with 2.5 BB left to call | −100 BB |
| 1803 / 1 / 5 | 6♦5♣ | Raised to 100 BB into a 1.5 BB pot, with 1 BB to call | −100 BB |
| 1803 / 4 / 5 | 8♦5♥ | Raised to 100 BB into a 2.5 BB pot, with 0.5 BB to call | −100 BB |

The first example wins a very large pot. That does not establish that its shove
was good, just as the losing examples alone do not establish their action EV.
The important observation is the recorded commitment behavior, supported by
the full-hand frequencies above. We need counterfactual values to assess its
cost under specified opponents.

### Has this behavior improved with training?

| Iteration | Seed 1802 first-action shove rate | Seed 1802 BB/100 | Seed 1803 first-action shove rate | Seed 1803 BB/100 |
| --- | ---: | ---: | ---: | ---: |
| 64 | 42.25% | −1,118.38 | 31.95% | −843.74 |
| 128 | 40.95% | −1,110.02 | 33.72% | −880.23 |
| 192 | 38.62% | −1,042.71 | 37.26% | −826.99 |
| 256 | 37.24% | −1,056.11 | 35.03% | −869.63 |

One seed gradually reduces first-action shoves; the other does not follow that
trajectory. Neither produces a convincing progression toward competent play.
These checkpoints reuse validation deals and are correlated. We do not select
an earlier checkpoint after seeing this table or report it as an independent win.

## Restricted benchmark: expensive errors, concentrated on few boards

This benchmark starts with **2 BB stacks**, fixed private-card ranges and
continuation policies. Its values are not recommendations for a normal 100 BB
game. The analysis uses the already-selected durations and saved train, tuning
and validation decisions: 8,640 records across three architectures and three
seeds. No additional sealed results are inspected or generated.

For each decision we reconstruct
`sum(p[action] * (max(reference Q) - reference Q[action]))`.
Action labels come from the hash-verified production cache: open contexts have
fold/check/raise; facing contexts have fold/call. The second array element must
not be assumed to mean call everywhere.

### Specific costly decisions

| Context | Visible cards and situation | Saved decision | Cost versus best reference action |
| --- | --- | --- | ---: |
| river-917 | 6♣6♦ on 6♠7♣Q♦9♣K♦, facing a bet | Baseline folds 100%, 100%, 97.2%; explicit features fold 100% in all seeds | 2.81 BB for a pure fold |
| river-935 | K♣K♥ on the same board, facing a bet | Card branch seed 941 folds; its other seeds call | 9.44 BB for folding |
| flop-912 | 6♣6♦ on 6♠7♣Q♦, with checking available | Card branch seed 941 folds; its other seeds bet | 4.29 BB for folding |
| river-833 | A♣9♣ on 8♣8♠7♣8♦Q♠, facing a bet | All nine models call | 1.00 BB for calling |

The first two holdings make sets. The final example has three eights on the
board; hero's hole cards do not make a set. This distinction is strategically
meaningful, even though both belong to the coarse made-hand category “trips.”
The learned card branch fixes river-917 in all three seeds but introduces a
large error on river-935 in one seed. Its better average hides this fragility.

For river-917 the reference values are fold −1 and call +1.80556 BB. For
river-935 they are fold −1 and call +8.44444 BB. For river-833 they are fold −1
and call −2 BB. These three action differences have **zero observed standard
error across the saved worlds**. That is conditional evidence within the
specified benchmark, not proof about unsampled worlds or arbitrary ranges.

On flop-912, fold/check/bet values are −1, +1.59843 and +3.29206 BB. The gap
between the best and second-best action is 1.69363 BB, with paired-world SE
0.10917 BB. Folding is considerably worse than either alternative here.
Best-action selection and example selection are post-hoc; these SEs are
explanatory diagnostics, not simultaneous confidence guarantees.

These cases cannot reasonably be summarized as only tiny near-tie action
changes. In particular, unstable flop/turn labels cannot explain the shared
river-833 error or the observed large, stable river gaps.

### Concentration and training coverage

Across the three seeds, the five validation contexts with the largest
seed-averaged costs account for 44.16% of baseline cost, 48.48% of card-branch
cost and 42.96% of explicit-feature cost. There are 192 validation contexts,
but only eight independent board families. Family 38, whose full board is
6♠7♣Q♦9♣K♦, accounts for 27.35%, 44.66% and 44.52% respectively.

Folds contribute 40.89% of baseline cost, 50.31% of card-branch cost and 71.42%
of explicit-feature cost. These are shares of reference decision cost, not
fold frequencies. Calls, checks and raises account for the remainder.

The relevant coarse training coverage is thin:

- River/facing trips on an unpaired board with two cards of one suit:
  15 training contexts from **five families**. This includes the broad category
  containing river-917 and river-935.
- Flop/open trips on an unpaired, three-suit board: 11 training contexts from
  **four families**, the category containing flop-912.
- River/facing trips on a paired board with maximum suit count two: only
  **one training context from one family**, the category containing river-833.

These categories do not establish equivalent ranges, hand strength or histories.
They show that “seen trips before” is a weak coverage claim. A board's trips
and a pocket pair making a set can require very different decisions.

## Interpretation and next decision

The full-game failure includes a gross preflop behavior, despite most replay
already being preflop. The next useful question is whether those records teach
the value of committing 100 BB, rather than merely how many records exist.
The supervised benchmark separately shows poor and seed-sensitive transfer
on concrete postflop decisions. Their causal connection remains untested.

**Next task: design and implement a bounded 100 BB preflop action-value and
card-sensitivity probe before another broad self-play run.** Start with
first-to-act roots, a declared set of weak/medium/strong holdings, the existing
legal action menu, and frozen continuation policies. This avoids silently
redealing uniform opponents after observing informative bets.

Compare saved-model action probabilities and predicted regrets with paired
reference action values for folding, ordinary raises and shoving. Report
uncertainty in action differences, reference decision cost and sampling cost.
Use a small local calibration to choose a feasible reference budget before
committing to a larger run. If adding facing-action roots later, define the
history-conditioned ranges explicitly. Retain all-in as a legal action.

Use the postflop examples above as diagnostic regression cases, with fresh
board families for any eventual confirmation. They motivate broader coverage
and explicit tests of how hole cards contribute to made hands; they do not
justify training directly on these validation cases and claiming improved
held-out generalization.

This analysis does not choose a new optimizer, promote the card branch, ban
shoves, or establish that Deep CFR is unsuitable. It chooses the next concrete
learning signal to inspect. Production defaults remain unchanged.

## Reproduction and evidence

The extraction scripts are `scripts/analyse_saved_play.py` and
`scripts/analyze_multistreet_decision_errors.py`. The compact machine-readable
[analysis evidence](holdem-decision-errors.json) contains input hashes, M4
summaries, selected benchmark examples and coverage/family summaries. Full
per-decision outputs remain in the ignored `results/decision-errors/` directory.
Raw archives and checkpoints remain at the locations inventoried in the two
campaign reports; they are not stored in Git.

From the repository root:

```sh
python -m scripts.analyze_multistreet_decision_errors
```

For M4, pass all eight `outcomes-{64,128,192,256}.json` files with repeated
`--input` arguments to `python -m scripts.analyse_saved_play`, then use
`--out results/decision-errors/fullgame.json`. Their exact paths and SHA-256
hashes are in the evidence file. The output reconstructs both arms and all
commitment groups; it does not load a checkpoint or play additional hands.

Review checks cover candidate-seat identity, blind accounting, walks,
all-in calls versus raises, return reconciliation, action-label joins, paired
world uncertainty, probability normalization, saved regret matching, source
and cache identity, complete split rosters, and reconstruction of every saved
decision cost. Eleven focused tests pass. Full repository/CI results are
recorded on the accompanying PR.
