# B100M posterior-conditioned decision values: frozen protocol

This is a new diagnostic after sealed draft PR #117. It does not change the
three saved B100M current policies, the bounded original-cap2 LBR, the game,
training, or release criteria. No selected decision's conditional value may be
opened before this protocol, selector, likelihood estimator, and negative
controls are committed and tested. The heavy work is confined to the M4.

## Source and outcome-blind selection

Use the three B100M raw-hand archives from #117's **fresh LBR curve** (root
`202610010300`, 512 paired blocks per policy). This root is disjoint from the
#117 primary decision audit and #116 confirmation. The curve's *aggregate*
returns have been opened, so this is not a sealed new playing schedule; the
selector reads no return, final stack, hidden attacker card, future deck, or
LBR value. Reusing retained hands preserves the roughly two-hour M4 compute
allowance for the much more expensive reverse-LBR likelihood. Report this
source distinction explicitly.

First exclude every #117 primary selected coordinate and every #117
card-collision root. A coordinate is `(seed, source file, block, rotation,
action index)`. Reconstruct each remaining target decision's actual button
position from the native public hand start, and keep only recorded trained
decisions whose checkpoint visit count is available. The 24 cells are exactly
three seeds × four streets × button/small-blind versus big-blind positions.
An outcome-free eligibility pass counts cells at thresholds 100, 300, and
1,000 visits. Freeze the **largest** threshold with all 24 cells occupied; if
none covers all 24, use 100 and report each empty cell. In every occupied cell,
choose the lowest SHA-256 rank of
`hu20-b100-posterior-v1|seed|file|block|rotation|action_index`.
An empty cell has no primary estimate. Its highest-visit decision may be
retained as a clearly labeled secondary example, selected before outcomes.
Never substitute a lower threshold silently. Record source/model/checkpoint
hashes, all eligibility counts, excluded coordinates, selected coordinates,
and the final selection digest.

## Reverse-LBR holding posterior

For a target/B100M decision, start uniform over all two-card LBR holdings
compatible with the target's own cards and public board. Process the visible
history in order. Board cards remove impossible holdings exactly. For every
observed LBR action, each still-compatible hypothetical LBR holding receives
the likelihood of that action under the **unchanged** bounded LBR algorithm,
evaluated from the LBR-visible public prefix and hypothetical LBR cards.
Earlier B100M actions remain in those prefixes because they affect the LBR's
own belief; they are not direct evidence about its hidden cards from the
target's perspective. Do not use `LocalBestResponse.update()` as the target's
posterior: it infers the opposite player's holding. Do not use the actual LBR
cards, original hand RNG, future deck, or recorded LBR internal values.

Marginalize LBR internal chance randomness using a common frozen number of
independent algorithm runs per hypothetical holding and observed LBR action.
Fresh seeds derive only from a new root, selected public coordinate, public
event index, hypothetical holding, and sample index. Choose the common count
from 1/2/4/8 through an **outcome-free** timing preflight. Keep the exact
action equality test, including raise-to amount. Do not substitute a softmax
or heuristic. Record every action likelihood and the posterior's compatible
and positive-mass counts, effective sample size, entropy, maximum weight,
normalization, zero-evidence handling, and total-variation distance from the
compatible uniform prior. A zero-evidence observation preserves the preceding
compatible posterior and flags the case; such a case cannot support a strong
training conclusion. LBR timer-limited likelihood samples remain visible.

## Feasibility gate before values

On hash-ranked, outcome-blind public prefixes, time the unchanged LBR on a
small fixed hash-ranked subset of compatible hypothetical holdings, one fresh
internal seed each. Discard the returned actions and any likelihood numbers;
only timing, RSS, swap, disk, and event counts enter the freeze. Extrapolate to
all compatible holdings, observed LBR actions, and all occupied selected
decisions with at least 1.25× headroom. Reserve time for 96 paired value
worlds per decision, suit controls, the river reference, and reporting.
Choose the largest common likelihood count and world count from respectively
1/2/4/8 and 96/192/384 that fit the **initial two-hour heavy M4 allowance**.
Require at least four likelihood samples and 96 worlds for the primary claim;
if that cannot fit, stop before opening conditional values and publish the
measured cost and smallest defensible alternative. No outcome-dependent budget
change is permitted. Record one frozen root and choices before running values.

## Paired uniform and posterior action values

For each same selected decision, evaluate the exact saved target action menu
and probabilities under both the uniform compatible range and the reverse-LBR
posterior. Draw independent fresh worlds with as much paired randomness as
the distinct holding distributions permit; each world's candidate actions
share its cards and future policy/LBR streams. Preserve all raw world rows,
failed worlds and soft-limited LBR batches. The first half of each condition's
worlds chooses that condition's alternative action. The disjoint second half
estimates that frozen action's paired gain over the saved policy mixture and
its 95% world-clustered interval. Descriptive all-world action means cannot
affect the held-out gap. Record the uniform/posterior chosen actions, gaps,
intervals, their paired difference, menu, probabilities, key, visits, regrets,
entropy, street, actual position, seed, and posterior diagnostics. No losing
realized hand is treated as proof of a bad decision. Neither conditional model
is exact full-game exploitability.

## Independent controls and stopping

Prespecify at least one hash-ranked decision per street for global suit
permutation. Permute *the complete player-visible state* and couple each
sampled LBR holding, future board, and policy/LBR random stream through the
same permutation. Check matching keys, menus, distributions, posterior
weights under the permutation, and paired action returns. A material
unexplained discrepancy is an evaluator failure: stop before making a
training recommendation. Retain the failed case.

Prespecify up to three hash-ranked river decisions for an independent exact
reference. Enumerate compatible opponent holdings and posterior mass; where
the remaining action tree is tractable, enumerate its continuation rather
than rerunning the Monte Carlo evaluator with a huge sample. If only a
smaller terminal/checkdown subproblem is exact, freeze and label its boundary
before results. A material reference disagreement also blocks inference.

Run one heavy M4 process at a time under the established 10.5-GiB owned RSS,
0.5-GiB swap-growth and 8-GiB free-disk guards. M1 performs only light edits,
Git, compact transfer and status. Retain all attempts and partials, native
replay and independent arithmetic where applicable, and a final hash
inventory. Do not start training, rent compute, merge, promote, or schedule a
follow-on automatically. The final report must state whether conditioning
changed gaps, whether high-visit gaps persist, whether all controls pass,
whether river references agree, which mechanism is actually supported, one
next experiment (or the smallest missing measurement), and M4 versus paid
compute. The two-hour ceiling is not a workload target.
