# Hold'em bet candidates and learning targets

This is the second engineering task in roadmap milestone 4. It connects the
[decision encoder](holdem-encoding.md) to exact legal bets, action-conditioned
regret/value predictions, and branch-payoff supervision. The legacy trainer is
unchanged. This is not a trained policy, a complete self-play solver, or a pass of
the small-game readiness gate.

## A versioned action menu

`bet_candidates(observation)` validates and encodes the acting player's public
observation, then returns an immutable `BetCandidates` with exact `Action` values
and their numerical features. It never receives engine state. The action schema
is **`holdem-bets-v1`**; its menu is a starting abstraction to evaluate, not an
optimized selection of poker bet sizes. Changing the menu or feature semantics
requires a new schema and corresponding experiment provenance.

Every available fold, check and call is retained. When raising is legal, include
both the minimum raise-to and the maximum legal raise-to (the player's exact
all-in under the current rules), plus these sizes:

| Street | Raise increment as a fraction of the pot after calling |
| --- | --- |
| Preflop | 1/2, 1, 2 |
| Flop | 1/3, 1/2, 3/4, 1, 3/2, 2 |
| Turn | 1/2, 3/4, 1, 3/2, 2 |
| River | 1/2, 1, 3/2, 2 |

Before the first preflop raise, also include raise-to amounts of 2, 2.5, 3 and
4 BB. These apply in limped pots too; the pot-relative candidates grow with the
limpers' contributions. A minimum raise is always present independently of the
menu. Opening bets on later streets use the same formula with a zero call.

Let `B` be the actor's current street contribution, `C` the exact call payment,
`P` the pot before acting, and `f` the menu fraction. The proposed raise-to is:

```
B + C + round_half_up(f * (P + C))
```

The pot includes all committed chips. This is a sizing convention, not a
pot-limit rule or a calculation of how much of a side pot the actor can win.
Side-pot eligibility remains part of the decision input.

Rational fractions are rounded using integer arithmetic, then clamped to the
observation's legal bounds and deduplicated by exact raise-to amount. No neural
float is converted back into a wager. With 1/2 blinds, the opening raise menu is
`4, 5, 6, 7, 8, 12, 200` at a six-handed 200-chip table. After a raise to 10, the
next actor's full minimum is 18 and its pot-sized raise-to is 33: call 10 into a
13-chip pot, then raise another 23.

Short all-ins can collapse every raise size into one candidate. If a short raise
does not reopen betting for the actor, there are no raise candidates. A short
all-in call is a call, not a manufactured raise. Legal amounts outside the menu
are still accepted by the game and encoded in observed history; the baseline
cannot choose every integer size. Later search/action-abstraction experiments
must evaluate that limitation explicitly.

Non-raises follow `ActionKind` order, then raises increase by exact target. Each
candidate has 11 input features: four action-kind flags, payment/target/increment
in BB, payment relative to the current pot, increment relative to the pot after
calling, remaining stack in BB, and an all-in flag. Pot denominators use at least
one BB, as in the decision encoder. Features approximate amounts in the network's
floating-point dtype; `actions` preserve the exact amounts.

## Predictions and play

`BettingNetwork(width=128)` combines the decision context with each candidate's
features, then uses separate scalar regret and value heads. Parameters are
shared across candidates, but each size receives its own output and gradient.
There is no single generic raise score or a sizing head copying its own guess.
Batches use a flat list of real candidates and split outputs back by decision;
no padded action can enter the probabilities or loss.

`ActionScores.probabilities()` regret-matches positive regret predictions. When
none is positive, it selects the largest legal prediction, breaking ties by the
stable candidate order, matching the existing small-game baseline. All-zero
predictions therefore select the first candidate; the collector's initialization
and exploration policy must be considered explicitly in the self-play task.
Predicted action values do not directly choose the action. `choose(random)` uses
a caller-owned `random.Random` stream and returns the exact selected `Action`.
This is a current regret-matched policy, not the average policy required for
export or evaluation.

```python
from random import Random
from src.holdem.actions import bet_candidates, record_execution
from src.holdem.betting import BettingNetwork

candidates = bet_candidates(observation)
model = BettingNetwork(width=128)
scores = model([candidates])[0]
action = scores.choose(Random(41))
# The game runner applies action and supplies its actual ActionTaken event.
record = record_execution(candidates, candidates.actions.index(action), event)
```

`record_execution` rejects a different actor, street, action amount or payment.
The returned record binds the original observation and candidate index to the
executed event. It verifies agreement with the event supplied by the runner; it
does not itself execute a hand or authenticate an external event source. The
existing hand runner already records every applied action in its public history.

## Supervision boundary

`action_targets(candidates, policy, branch_values)` accepts a probability for
each candidate and an action-keyed mapping of **the acting player's estimated
net chip payoff for every branch**. Net payoff means final stack minus starting
stack for terminal rollouts, including chips already committed. Missing branches,
extra actions, nonfinite values and invalid distributions are rejected. A
zero-probability action still needs a branch estimate. Dictionary order cannot
reassign a small bet's target to an all-in.

Values are divided by BB once. For each action `a`, targets are:

```
value[a]  = branch_payoff_chips[a] / big_blind
baseline  = sum(policy[a] * value[a] for a in candidates)
regret[a] = value[a] - baseline
```

Counterfactual simulation belongs to the collector, outside the policy input.
This helper does not establish unbiased sampling: the collector must supply the
correct player perspective, chance/opponent sampling and any required estimator
corrections. Predictions or the selected bet amount are not branch-payoff labels.

`betting_loss` sums squared errors over all candidates, separately for regret and
value, then averages over decisions with equal head weights. It validates exact
observation/candidate alignment, including source records excluded from numerical
encoding equality. This is an unweighted fitting primitive; reservoir admission,
iteration weighting, policy freezing and averaging remain collector/trainer work.
The auxiliary value head shares the encoder with the regret head. Its effect on
learning must be measured before any strength or convergence claim.

## Validation and next task

The focused tests execute every candidate on all streets at four/five/six players
and across generated unequal-stack hands. They cover minimum raises, fractional
rounding, pot-sized raises, short calls, short raises, cumulative reopening,
all-ins, deduplication and exact execution records.

Learning checks cover per-action branch alignment and gradients, actual settled
payoffs, zero-probability actions, mixed candidate counts, reordered candidates,
invalid inputs and numerical overflow in regret matching. Hidden-deal, suit and
seat changes preserve outputs on the same inference shape. Mixed batches agree
within floating-point tolerance; bitwise equality across different batch layouts
is not promised.

The next task is a consistent all-role self-play collector: freeze the policy
profile during collection, evaluate each traverser's candidate branch, pass its
net-payoff estimates into this target interface, and record exact executed bets.
Integrate replay weighting, fitting and snapshot averaging separately. Recovery
and fresh small-game readiness remain required before substantial training.
