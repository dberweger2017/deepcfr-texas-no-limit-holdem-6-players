# Six-player river learning diagnostic

## Question and scope

Can the existing width-32 betting network recover reliable regret targets, and
how much information does the production sampler lose? This is a bounded local
diagnostic, not self-play training, full-game exploitability or a model promotion.
The [plan](../configs/holdem/river-reference.json) is committed before measurements.

Use the real engine, public observation encoder, candidate menu and net-payoff
convention. Six players start with 2 BB, call preflop and check to the river,
leaving 1 BB each. This deliberately short-stack river has a finite small tree;
it is not the 100 BB starting-game benchmark. Include an unopened river (hero
seat 1) and facing seat 1's all-in (hero seat 2). No game-rule changes or
truncated payoffs. All six players remain live at each root.

Three declared boards each have four hero holdings and both public situations:
24 contexts. For each board and situation, generate two equally likely compatible
opponent assignments with a fixed seed, excluding the board and the union of
the four hero holdings. Persist the actual assignments. They define a small
explicit joint range, not independent uniform poker ranges or a posterior from
unrestricted self-play. The forced prefix defines the diagnostic starting game.
Within each context, both worlds give hero exactly the same observation.

Enumerate every legal candidate continuation for two frozen public policies:
uniform, and probabilities proportional to candidate index plus one. These are
fixed diagnostic policies, not learned opponents. At each hero information set,
aggregate action values over worlds with opponent-reach weights. Oracle baseline
queries receive only hero's observation. They cannot distinguish worlds sharing
that observation. Exact enumeration uses privileged simulator states only to
construct reference targets, never to select a playing action.

## Matched comparisons

For every context/profile and all seeds 421/431/433, collect 32 replicates using
the production outcome sampler, exploration 0.5, and four arms: single-action
zero/oracle baseline and first-own-decision expansion zero/oracle baseline.
Pair world selection and action seeds across arms; require identical executed
paths between baseline variants. Root estimates include all sampled continuation
noise. Persist every root value/regret vector and node count. Report empirical
bias, centered variance, reference MSE and variance times mean node count. Oracle
construction and sampling wall time are separate: an exact lookup is a diagnostic
of available variance reduction, not a free deployable learned critic. It is not
a universal lower bound for all possible baselines/samplers.

Combine the two frozen profiles with weights 1 and 2. Direct tables store the
weighted exact root regrets or the corresponding weighted sample means. Fit
fresh networks to those same exact or sampled means, using the existing joint
regret/value loss. Repeated frozen profiles define this finite regression target;
there is no claim to reproduce an online CFR training history. The sampled
fitting arm uses first-decision expansion with zero baseline, with each seed's
own 32 samples per profile. This isolates a finite noisy target table from neural
compression of that table. It is not a replay-reservoir or optimizer sweep.

Use width 32, Adam 0.001, clipping 1, full batches and 1,024 steps; exact/noisy
fits share initialization per seed. Boards 1–2 train (16 contexts); board 3 is
held out completely (8 contexts), including all its hidden worlds. Retain initial
and final errors, regret-matching policy TV against the exact table and fixed-
continuation decision cost. Evaluate regret error against the weighted exact
regrets, not one profile's instantaneous targets. Report training and held-out
errors separately. Tabular values on held-out contexts are reference lookups,
not evidence of table generalization. Decision cost uses weighted frozen action
values, and is neither exploitability nor a replacement training objective.

## Controls, limits and decision

Add a check-first/fold-otherwise arena policy. Against the existing style pool,
run 64 fresh six-player 100 BB blocks with all six seat rotations, paired with
the existing fold policy. Assert zero voluntary contributions, zero-sum payouts
and the -25 BB/100 bound over each balanced block. This is an accounting/behavior
control, not a trained-model comparison or professional benchmark.

Predeclare diagnostic thresholds: exact-target fitting recovers the training
mapping if regret RMSE / RMS exact regret <= 0.10 on every seed. Held-out <= 0.50
on every seed is separate limited generalization evidence. Zero target scale
requires absolute RMSE <= 1e-6. These thresholds choose follow-up work, not a
poker-strength gate. If clean fitting fails, investigate representation/fitting
before commissioning a critic. If it passes but noisy fitting fails, prioritize
sampling quality. If both pass, broaden to full-game coverage/scaling rather
than claiming the learner is solved. Mixed results remain mixed.

Prioritize a persistent critic only if the oracle reduces mean root-regret
variance times mean node count by at least 25% on unopened-river contexts under
the production first-decision sampler. Also publish every context, profile and
seed: an aggregate cannot hide worse cells. Facing an all-in is a useful negative
control because the first decision is expanded and hero has no later decision;
changing its baseline should do nothing. A failure here does not rule out
history-aware critics or baselines at opponent/chance nodes.

One CPU thread, no rental, at most 900 seconds for the complete experiment and
20,000 nodes per exact world tree. Stop on invalid action, incompatible hidden
world, nonfinite result, broken accounting, mutation or resource limit; retain
partial results and mark unattempted cells. No outcome-dependent retries or
extensions. Commit compact reports and hashes; keep raw samples and model files
locally. Subsequent online training needs its own declared comparison.

## Execution

```bash
python -m scripts.check_river_learning --plan configs/holdem/river-reference.json --out results/river-reference
```

One diagnostic network fits all training contexts, including both hero positions;
this does not pool production role replay or change production ownership. Public
hand identifiers do not encode the hero holding. The card probes vary hero cards
with the board, betting prefix and opponent assignments held fixed. Per-decision
predictions retain this comparison alongside aggregate errors.
