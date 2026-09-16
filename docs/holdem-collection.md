# Hold'em external-sampling collection

[PR #59](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/59) connects the [decision encoding](holdem-encoding.md)
and [bet candidates](holdem-betting.md) to an all-role self-play collector.
The [training loop](holdem-training.md) now adds role reservoirs and weighted
fitting. The [baseline pipeline](holdem-baseline.md) adds strategy averaging, recovery
and fixed-arena evaluation; useful learning remains unproven.
The subsequent [snapshot readiness report](reports/snapshot-readiness.md) passes
the small-game gate. No rental or training campaign was used for this collector
delivery. [Collection profiling](holdem-collection-performance.md) documents its
current performance checks and remaining branching cost.

## Frozen current policies

`FrozenProfile` accepts two to six `BettingNetwork` models, one per **physical
seat**, and makes independent CPU float32 copies in evaluation mode with gradients
disabled. Changing a source model after freezing cannot change collection. Even
if several seats were supplied the same model object, their frozen copies do not
share parameter storage. Parameters are not pooled or fitted across seats here.

An explicit `None` slot means a uniform distribution over the legal candidates.
This gives the future trainer an explicit bootstrap policy; it is not inferred
from an all-zero regret network. Neural slots use the current regret-matched
policy from the betting interface, including its largest-regret fallback. They
do not use the value head, an averaged strategy, or a fixed benchmark opponent.

The profile fingerprint covers physical-seat ordering, model widths and weights,
uniform slots, and the decision/action/profile versions. Integrity checks at
collection boundaries reject changed weights or training mode. The models remain
private implementation state; this is isolation from normal training updates,
not a sandbox against arbitrary Python code modifying internals. The fingerprint
identifies snapshot contents, not complete executable provenance or a saved
training checkpoint.

Policy queries accept only `BetCandidates`. Their source observation belongs to
the current actor. The physical mapping selects that actor's model; an absent
seat does not shift model ownership. A six-seat profile can therefore serve
four-, five- and six-player lineups with empty or inactive physical seats.

## One traversal

`collect_traversal(hand, profile, traverser, ...)` is privileged simulation code.
The traverser is a compact participant index in this hand. At each decision:

1. Obtain only the actor's immutable observation, then generate its legal menu.
2. Query the actor's frozen current policy using that public candidate input.
3. At a traverser node, visit **every candidate**, including actions whose current
   probability is zero. At another player's node, sample one candidate from that
   player's current policy.
4. Apply the exact action and validate its actual `ActionTaken` event against the
   selected candidate. Every visited edge retains its execution record.
5. At settlement, use the traverser's final stack minus its starting stack.
   Include previously committed chips, side-pot winnings and refunds.
6. Back up opponent-node returns directly. At traverser nodes, form every action's
   value and regret targets through `action_targets`, then return the current
   policy's weighted mean continuation value.

The collector passes chip payoffs to the target helper; the helper converts to BB
once. This is each traverser's own payoff, not a two-player sign flip of another
seat's winnings. Returns are sampled continuation values, not the betting
network's predictions. No branch is given a fabricated payoff because it is
expensive, unlikely, folded or out of chips. Terminal accounting stays in the
rules engine and public settlement interface.

External sampling accounts for opponent reach through sampled paths. The
traverser's own reach is not multiplied into the regret targets, and zero-own-reach
branches are not pruned. All later traverser choices are still enumerated. This
uses the sampling structure of the existing [small-game collector](../src/solver/neural/solver.py)
with the no-limit action interface; it does not establish six-player convergence
or validate the whole future neural training algorithm.

Traversal uses an explicit postorder stack rather than recursive Python calls.
There is no fixed history-depth cutoff. The supplied root may be a diagnostic
river position or even a settled hand; its estimate is conditional on that root.
It is **not** an unbiased full-game sample if the caller selected the root with a
biased procedure. Normal collection uses fresh starts through `collect_phase`.

## All-role collection and random streams

`collect_phase(table, profile, ...)` starts the requested number of fresh hands
for every participant against one fixed profile. Each root samples a full deal
through the seeded engine. All branches of that traversal retain this sampled
deal; policies never see its unrevealed cards or future board. Fixing the deal
provides common chance samples across branches without making policy decisions
perfect-information decisions.

Deal and opponent-action seeds are derived separately from the master seed,
iteration, physical traverser seat and sample number under
`holdem-external-sampling-v1`. Streams use local RNG instances. Public hand IDs
label the schedule without embedding the deal seed; neither hand labels nor
collection metadata are numerical policy features. Reordering independently
scheduled traversals does not change their results. Within one traversal,
changing depth-first branch order would change the sequence of opponent draws.

The phase uses a **simultaneous collection profile**: every role is collected
before any fitting. No actor's policy changes halfway through collection. This
is not the small-game runner's alternating fit-after-each-player schedule. A
future trainer must keep this choice explicit, retain separate role memories,
and record the profile actually used when integrating policy averaging.

The result records the table, iteration, master seed, traversal count and profile
fingerprint. Each traversal includes its owner's root observation, action seed,
BB return, target records, execution records, node count and terminal count.
These are privileged training records spanning counterfactual branches and
multiple owners. Never pass the combined batch to a playing policy. The current
collector does not read, append or share a session's prior-hand memory; learning
from earlier observed hands remains the adaptation task.

Different roles start from independently sampled deals. Their root values are
not outcomes from one shared table hand, need not sum to zero, and must not be
reported as BB/100 or evidence of playing strength.

## Limits and completion

A node budget covers the whole phase and a cooperative time deadline is checked
at each visited node. A limit raises `CollectionLimitExceeded` and returns no
collection object, including when earlier roles have finished. No callbacks
publish partial samples. Retrying with the same inputs and sufficient limits
reproduces the complete result. The implementation retains targets and execution
traces in memory; large-scale compact storage and performance work remain ahead.

Do not salvage only the easy traversals from a failed phase or silently skip
failed seeds: completion-dependent filtering can bias the training distribution.
Profile integrity is checked before and after collection. Engine errors and
invalid actions propagate rather than triggering a substitute action.

This collector does not own resume state. The [trainer](holdem-baseline.md) now
provides role reservoirs, snapshot archives, completed-iteration checkpoints and
separate inference exports.

## Reproducible local check

```bash
python -m scripts.check_holdem_collection --players 6 --policy neural
```

This uses fresh, untrained width-16 networks at 100 BB, collects twice and checks
exact equality. `--policy uniform` exercises the explicit bootstrap profile.
Both modes are implementation checks. The JSON report includes the source and
profile fingerprints, engine/environment identity, configuration, complete-result
digest and per-role counts. The command does not save trained weights. A test
also compares the result digest across fresh Python processes.

The recorded local neural smoke used six traversals: **420 visited nodes,
74 terminal branches and 37 traverser decision targets per pass**. Both passes
completed in about 1.3 seconds inside the check on the development machine.
These counts and timing are smoke evidence, not a capacity study or throughput
prediction for later trained policies.

The focused tests cover frozen-copy independence and physical-seat ownership;
hidden-world and suit invariance; exact execution and owner-specific hole cards;
all roles and changing lineups; net settlement utilities; zero-probability and
zero-own-reach branches; deterministic seeds, reordered independent traversals
and global RNG isolation; whole-phase failure; and fresh-process reproduction.
An independent exhaustive evaluator checks a tiny fixed-deal river tree: summing
all six possible opponent-sampling outcomes recovers the exact branch values,
policy value and regrets without statistical tolerances beyond floating-point
rounding.

**Validation:** all 28 focused tests and all 433 repository tests pass locally.
Ruff and `git diff --check` pass.

## Next task

Collection, fitting, averaged play and recovery are implemented; small-game
readiness is passed. The [performance report](reports/holdem-collection-performance.md)
retains the difficult unequal-stack failure. Compare a bounded-work traversal
design before declaring a meaningful Hold'em learning campaign.
