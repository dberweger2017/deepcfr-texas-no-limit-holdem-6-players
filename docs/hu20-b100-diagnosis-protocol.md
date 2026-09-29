# B100M HU20 decision diagnosis: frozen protocol

This protocol is fixed before reading any selected decision's terminal result or
conditional action value. It uses the three B100M current-policy lineages and
the retained, fully replayed #116 original-cap2 LBR hands. It does not change
training, the opponent's action contract, or the formal release criteria.

## Primary decision selection

Read only `policy`, `block`, `rotation`, and each action's `index`, `street`,
`logical_player`, `seat`, `kind`, and `target_trained` from the six B100M LBR
raw-hand files. The selector must not read `target_chips`, final stacks, LBR
estimated values, hidden cards, future deck, or terminal outcome. Include
recorded decisions even if a hand later failed; retain all failures.

For each of three training seeds, four streets, and two target seats, select
one target decision facing a wager and one with a free check, when available.
Facing/free is reconstructed from the pre-action public legal menu, never the
eventual action or payoff. If one context is empty, take the two lowest-ranked
decisions from the other. Rank by SHA-256 of
`hu20-b100-audit-v1|seed|block|rotation|action_index`; ties use the four
integer coordinates. Select at most 48 base decisions. Then select up to four
additional *fallback* target decisions per seed, excluding base choices,
ranked by the same hash. Do not replace any empty cell. Report all cell counts,
shortfalls, duplicates, and the final immutable selection list and digest.
This creates at most 60 primary decisions without selecting for realized loss.

The selection's context, training status, model identity, and public view are
reconstructed by native replay from the recorded actions and deal seed. The
replay is checked against the recorded public action trace and event digest.
Only the target's `Observation` crosses into value evaluation; the evaluator
never receives the actual opponent holding or remaining deck.

## Primary conditional values

At each selected target observation, compare every action in the **actual
uncapped saved-policy menu**. The declared conditional model draws the
opponent's two cards uniformly from holdings compatible with the target's
cards and visible board, and draws undealt cards uniformly without replacement.
It deliberately does **not** condition that range on earlier LBR actions.
This is a reproducible local policy-quality probe, not an estimate of the
original LBR hand's posterior or full-game exploitability. Report this limit
beside every aggregate diagnosis.

Use 96 independent world samples per decision, paired across all candidate
actions. Reconstruct each public prefix on a sampled world, apply a candidate
action, then complete the hand with the unchanged B100M target policy and the
unchanged four-future-sample, five-soft-second original-cap2 LBR. All future
target-action, LBR, world, and chance RNG streams derive from a new audit root,
the selected coordinates, world index, and distinct purpose tags. The actual
hand's seed and target action RNG are not used in value estimation. Publish
every attempted world, failure and timeout; never drop a candidate because it
was slow or unfavorable. The same world compares all actions and an unfinished
comparison batch has no action-value estimate.

Record the public history, target cards' abstraction and key, legal/menu
actions, saved policy probabilities and actual selected action, trained/fallback
status, checkpoint visits and regrets, paired action returns in BB, per-action
means and 95% world-clustered intervals, best estimated action, sampled
policy value, and policy-value gap. To avoid selecting and estimating a gap
on the same noisy worlds, worlds 0–47 select the best estimated action and
worlds 48–95 estimate its paired gain over the saved policy, with a 95%
world-clustered interval. Full-96 action means remain descriptive. This
analysis refinement was committed before any new conditional values or
confirmation returns were inspected. Treat overlapping or wide intervals as
uncertain. The exact realized hand outcome is not an action-value label.
Cross-check payoff accounting on a deterministic small river reference and
verify independence from hidden opponent cards/future deck in tests.

The primary analysis relates positive estimated gaps and their uncertainty to
street, seed, position, action context, visit-count bands (0, 1-2, 3-9, 10-99,
100+), policy entropy, and available regret summaries. Visit correlation is
descriptive. A card/history/action-abstraction mechanism requires a concrete
shared-key or missing-action counterexample with a conditional value contrast;
otherwise label it unproven. Any highest-gap examples chosen after this
selection are explicitly exploratory.

If the primary audit finishes with time for a targeted collision check, take
the lowest-hash *trained* primary decision on each street. Enumerate other
compatible concrete hero holdings with the identical information key and
choose the lowest and highest exact hand-rank tuple on the visible board
(lexicographic tie-break; on preflop choose lexicographic endpoints). Evaluate
each alternative under the same 96-world conditional model. This comparison
is prespecified but diagnostic, not a claim that those hands occur at a given
frequency. It may show a concrete lost card distinction; absence of a
contrast in four roots does not validate the abstraction.

## Additional fixed comparisons

For the missing LBR curve, evaluate saved 20M/40M/80M/100M current policies
for all three seeds on **one new paired two-seat schedule**. Use the unchanged
original-cap2 LBR, four future samples, five-soft-second limit, existing
Bayesian update and realized-return code. An eight-block per-policy resource
preflight writes no returns and selects the largest feasible common count
among 64/128/256/512 blocks. The selection uses observed seconds/block and
remaining time only, reserving at least 90 minutes for audits/reporting; it
never uses apparent profit. Freeze count/root before confirmation outcomes.
All policies receive the same deals and rotations, with separate actual-game
and attacker internal streams. Report absolute and own-20M paired BB/100,
seed/seat effects, block intervals, failures and soft-limited comparisons.
This is bounded LBR, not exact exploitability.

For opponent-action translation, use the same B100M policies. A is the exact
existing lookup/fallback. B maps each opponent raise outside the saved policy's
uncapped min/pot/conditional-jam menu to the closest legal abstract raise by
absolute log ratio of **paid chips to pre-action pot**; ties prefer min, then
pot, then jam. Replace only that event's raise-size/all-in **lookup label**.
The real engine action, public events, pot, stacks, legal target menu, and
settlement remain exact. Recompute lookup on the actual target observation.
If there is no compatible abstract raise or translated key/name match, use A.
This deterministic control is potentially exploitable at boundaries.

Freeze paired schedules for pot-pressure (256 blocks per seed and variant),
one-third-pot and two-thirds-pot pressure (128 each), native minraise and passive
controls (128 each), with both positions. No outcome-driven rule selection.
The retained `pot_pressure` style already wagers approximately 1.5 times the
post-call pot; a separate 1.5-pot arm would duplicate it. This correction was
made before any new outcome was opened.
For each, report absolute and paired BB/100, 95% block intervals, lookup hit,
fallback, translation frequency and size pairs, invalid/missing histories,
latency and every failed hand. Improved hits alone do not establish value.
Do not implement or name a pseudo-harmonic method without a validated primary
reference.

## Execution and stopping

All model loading, replay, poker evaluation, tests, large audits and native
builds run on the M4 only. The M1 handles source edits, Git and small status
reads. Before first heavy M4 work, record UTC start and hard deadline exactly
ten hours later. Use one heavy process, 10.5-GiB owned RSS, 0.5-GiB swap-growth
and 8-GiB free-disk guards. Preserve all attempts, hashes and partial files.
The order is selector/information-safety tests, primary audit, LBR curve,
translation comparison, native replay and independent arithmetic, report.
Time is a ceiling; stop when the declared work is complete. A phase that cannot
finish before the deadline remains explicitly incomplete. No retraining,
paid host, model promotion, merge, or automatic next campaign is authorized.
