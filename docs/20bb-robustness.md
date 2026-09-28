# Saved 20BB robustness diagnostic

This dependent draft preserves #112/#113 and changes no trainer, extraction,
keys or action menu. No model is promoted. The temporary Claude script was absent
on both Macs. The committed pressure reconstruction qualifies preflop with any
pair, any ace, or both ranks at least ten; postflop it requires pair or better,
including a pair on the board. Min-raise takes precedence, then free check,
then qualified call, otherwise fold. Minraise control always raises if possible,
otherwise checks/calls. Passive always checks/calls. Each has menu-restricted
(two raises per street, no free fold) and native-legal versions. Native minraises
continue beyond the training cap. Target inference and uniform fallback stay fixed.

## Local response contract

Based on [Lisy and Bowling, arXiv:1612.07547v2](https://arxiv.org/html/1612.07547v2).
The responder tracks all 1,225 holdings compatible with its own initial cards.
Target actions condition the range using exact saved probabilities at the
**pre-action** public observation; board arrivals remove collisions. Fallback
probabilities are part of the target. An action with zero total likelihood logs
an explicit event and preserves the compatible prior. There is no smoothing
of positive likelihoods or reading the actual rival cards/deck/RNG.

Each decision compares restricted-menu actions after integrating the hidden
range. Raise values incorporate the target's immediate fold probability and
assume all nonfold replies call, then both players check down. The actual target
continues its saved strategy, including raises. Future board samples are uniform
conditional on each holding and shared across all candidate actions. River ranks
are exact. Equal initial HU stacks, two players and no rake imply showdown
contributions match after uncalled refunds: win/loss pays +/- matched chips,
tie pays zero. Tests compare this ledger to independently replayed native deals.
No six-player/100BB reference assumptions are imported.

Full comparison batches publish together; soft deadlines are checked between
batches. At least one batch is attempted and may exceed the soft budget. Requested
and completed batches, wall time, over-budget decisions and zero-likelihood
paths are retained. The external RSS/deadline guard terminates the process if
needed and preserves partial rows. This deterministic timeout rule prevents
menu-order selection bias; it is not a guarantee of a hard live decision bound.

## Freeze and inference

The initial config pins all saved inference hashes. Outcome-free preflight uses
separate validation deals and suppresses realized chip outcomes. Timing of two
and eight chance samples guides one common work budget and 64/128/256/512 block
count, frozen in a second commit before confirmation. Cheap suggested counts
are retained only if the measured cost fits. 5M/10M LBR inclusion is decided
before opening confirmation returns. Checkpoint comparisons are exploratory.

Deal roots 2026131001 (stress), 2026132001 (LBR), 2026133001 (resource preflight),
2026134001 (higher-work calibration) are new. Each rule/lineup has a separate
stress stream; contracts and policies share physical cards and rotations.
Mixed TP opponent order reverses on odd blocks. Actual policy RNGs use separate
logical-player streams; LBR chance uses the opponent stream and never target RNG.
All models run sequentially on M4 under 10.5 GiB RSS, 8 GiB minimum disk,
0.5 GiB swap-growth cap and one ten-hour deadline beginning with preflight.
Thirty minutes are reserved for report/audit. No failed attempt is rerun or omitted.

Reported returns are realized target chip movement: BB/hand = chips/100,
BB/100 = chips, 20BB buy-ins/100 = BB/100 divided by 20. Intervals cluster on
whole rotation blocks, averaging training-seed contrasts within each shared
block. Role-specific results remain visible. Attacker profits negate target
profits in HU only. True legal-attacker payoff lower-bounds best-response payoff
in this restricted game; a noisy or negative estimate does not certify robustness.
TP stress does not measure multiplayer exploitability. No profile-exploitability
claim is made, and whole-hand outcomes are not attributed to one street.
