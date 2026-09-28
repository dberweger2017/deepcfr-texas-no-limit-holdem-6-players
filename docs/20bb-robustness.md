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

## Frozen resource decision

The M4 preflight completed 168 outcome-suppressed hands in 40.93 seconds,
peak RSS 1.19 GiB with unchanged swap. Two/eight samples took respectively
2.66/7.91 seconds for eight early-target hands, 2.85/9.40 seconds for eight
final-target hands and 4.59/10.60 seconds for eight uniform hands. These are
small timing samples, not guarantees across the full tree.

Freeze **four chance samples, five soft seconds, 512 two-position LBR blocks**
for **all thirteen HU arms**, including 5M/10M before returns are opened.
The eight-sample development calibration is higher work than this candidate;
it is outcome-suppressed and is not used to select an attacker on confirmation.
Retain the proposed 4,096 HU and 2,048 TP stress blocks. Cheap stress runs first
across every model; then LBR reloads each HU artifact sequentially. The estimated
LBR cost is about three hours, leaving substantial headroom under the original
absolute deadline. A slower actual campaign is retained as incomplete; no count,
seed or deadline is changed after looking at outcomes.

## Conditional probes and replay

The first six reached river target decisions per pressure/passive rule from the
first seed's final policy, menu contract and first 64 frozen blocks are retained
as exploratory probes. The 24-holding uniform compatible opponent range is
explicitly declared; it is **not** inferred from actual hidden cards or presented
as the true action-conditioned belief. Native settlement enumerates each possible
holding and the target's saved continuation probabilities against the fixed rule,
then aggregates before action comparison. This adapts #108's independent native
oracle method to two equal 20BB stacks and the no-free-fold menu; it does not
import the general 100BB river solver or change its game. Neutral probes and
failures remain in the record. Any gap is conditional on that artificial range.

`python -m scripts.play_robustness --plan configs/blueprint/robustness-m4.json
--policy 2p-2026092801-20M --rule pressure --contract native --history NEW.json`
runs a pinned saved model and saves concrete actions. Use
`python -m scripts.play_robustness --replay NEW.json` to verify its native replay.
Large outputs remain on M4 under `Local/robustness-pr114/results/robustness-m4-20260928`.

The requested analysis-v3 document was not found in the repository, attachments
or M4; its path/attachment was requested. No claim is made to have independently
reviewed that absent exploratory seed experiment. This evaluation uses the three
saved campaign seeds and a precisely labeled reconstructed rule.
