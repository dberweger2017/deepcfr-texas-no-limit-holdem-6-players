# Hold'em decision encoding

[PR #57](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/57) is the first delivery in roadmap milestone 4: a versioned current-hand
representation and a modest sequence encoder for the replacement learner. It
uses the existing [player observation](observations.md) and [rules profile](rules.md).
It does not change game behavior or route the legacy trainer through a new model.
Bet candidates, regret/value heads, self-play and training recovery follow in
separate tasks. The failed small-game readiness gate remains open.

## Input boundary

`encode_decision(observation)` accepts only an `Observation` belonging to the
acting player at a live decision. It rejects engine state, spectators, nonacting
observers, terminal hands and tables above six physical seats. Two- and three-
player shapes are supported; the primary training targets remain four to six.
The snapshot must agree with its public event replay. No simulator is consulted.

The returned immutable `DecisionInput` contains normalized feature tuples and
`source`, the original immutable public observation. Exact integer payments,
raise-to bounds, public identities and owner-specific previous hands remain in
`source`. Future action generation must use these integer bounds, never reconstruct
legal amounts from rounded neural features.

The numerical schema is **`holdem-decision-v1`**. Equality compares those features
and the schema, excluding `source`. Different public labels or a uniform rescaling
of chip amounts can therefore produce equal model inputs while retaining different
exact records. This is not a lossless serialization format for observations.

This first model consumes the **current hand**. Earlier hand records and player
identities are preserved in `source` for the later opponent-history/adaptation
work; they are not embedded as arbitrary numeric IDs. The baseline therefore
does not yet learn a named opponent's behavior across hands. Hand IDs, chip
currency labels and opponent/checkpoint names are not numerical model inputs.

## Feature layout

The exported field tuples in [encoding.py](../src/holdem/encoding.py) define the
column order. Change the schema version when changing those meanings.

| Component | Shape per decision | Meaning |
| --- | --- | --- |
| Context | 18 | Pot, blind/call/raise bounds, table capacity, participant count, street and legal-action kinds |
| Cards | 4 × 52 | Own hole-card set, flop set, turn and river, in separate reveal groups |
| Seats | 6 × 16 | Physical/occupancy/participation masks, folded/all-in/waiting/sitting-out/busted state, button, stack/bet/contribution and effective-stack features |
| Pots | 6 × 14 | Existence, amount in BB, six eligible-seat flags and six provisional-refund-owner flags |
| Events | N × 83 | Every public event in order, with type, relative actor/button, street, action/legal kinds, amounts and revealed cards |

Rows use the owner's physical seat as zero and proceed clockwise modulo actual
table capacity. Compact hand seats are translated through `seat_numbers`. An
empty physical seat has `exists=1, occupied=0`; a padding row beyond table capacity
is entirely zero. Dealt-out occupants, folded participants and all-in participants
remain distinct. The same model accepts changing lineups without changing weights
or confusing a folded six-handed participant with an empty four-handed seat.

The pot rows keep side-pot eligibility and unmatched-contribution refund owners.
A single total-pot scalar would not preserve this information.

### Chips and effective stacks

Amounts are divided by the hand's big blind. Pot-relative fields divide by
`max(pot, big_blind)`, explicitly defining the zero/small-pot case. Each event uses
the pot **before that event's payment**, not the final pot. For example, opening
to 10 after 1/2 blinds produces payment 5 BB and payment/pot `10/3`.

Seat rows retain starting stack, remaining stack, street bet and total contribution
in BB, plus remaining stack/pot. For a live participant, pairwise effective
remaining stack is `min(owner.stack, player.stack)`; effective total is the minimum
of their `stack + contributed` amounts. Both are in BB and are zero for folded or
nonparticipating occupants. These describe pairwise stack coverage, not expected
winnings; side-pot eligibility remains a separate input.

Features become float32 in the default model. The schema supports the engine's
integer amounts, but a float32 network cannot distinguish every large integer
chip value. Exact observation records and action bounds remain available alongside
those approximations.

### Cards and symmetry

Visible card groups are canonicalized jointly by choosing the lexicographically
smallest representation across all 24 suit relabelings. Hole cards and flop cards
are unordered sets. Turn and river stay separate, so swapping their reveal order
changes the encoding even if the final five-card set is unchanged. No hidden card
or future board participates in canonicalization.

Canonical suit labels may change when another board card becomes visible. That
is a representation of the current observation. To query a policy at an earlier
decision, encode the observation available at that earlier time; do not reuse the
current decision's card mapping or splice its event features into an earlier query.

### Complete current-hand history

There is one token for every `HandStarted`, `BlindPosted`, `Decision`, `ActionTaken`
and `BoardDealt` event, including the owner's own actions and previous actors'
public legal bounds. Action tokens distinguish check/call/fold/raise, actual payment
and raise-to amount. Board tokens retain reveal boundaries. The initial token
marks the button and blind size; starting stacks also remain in the seat rows.

No fixed event limit, tail selection or silent truncation is applied. Terminal
settlement/showdown events are rejected in a live-decision input under the current
rules profile. If rules later allow cards to be exposed during betting, extend
this schema and its tests explicitly.

## Reference model

[DecisionEncoder](../src/holdem/model.py) returns a decision representation, not an
action or trained poker strategy. The default width is 128 with **194,816 parameters**.
It projects current context/cards/seats/pots, runs event features through a
one-layer GRU, then combines both into a width-128 vector. The modest starting
size is an implementation choice, not a claim about sufficient Hold'em capacity.

```python
from src.holdem.encoding import encode_decision
from src.holdem.model import DecisionEncoder

encoded = encode_decision(observation)
model = DecisionEncoder(width=128)
context = model([encoded])  # [batch, width]
legal_actions = encoded.source.legal_actions
```

Batches can mix table sizes and history lengths. Packed sequences exclude padding
and preserve batch order. The recurrent calculation processes the full sequence;
it does not guarantee perfect recall inside a finite learned vector. Its cost
grows with event count. There is no capacity sweep, trained checkpoint or strength
claim in this delivery. The next action-candidate layer will supply separate legal
bet choices and their training targets.

## Validation

The focused checks cover all betting streets at two to six players; physical-seat
rotation and every suit permutation; hidden-world changes at four/five/six players;
card order and board-reveal stages; same-pot/different-history examples; exact
payments and pre-action pot ratios; side pots and effective stacks; distinct seat
states; chip-amount rescaling; and six-to-five-to-four session transitions.

A history longer than 256 events remains intact. Thirty generated unequal-stack
hands encode every decision without mutating saved observations. Mixed-length
batches agree with individual inference within floating-point tolerance, and the
reference model backpropagates finite gradients. Engine-state and malformed/live-
decision boundary checks fail explicitly. See [the tests](../tests/test_holdem_encoding.py).

All **45 focused tests** and **363 repository tests** pass locally. Ruff and
`git diff --check` pass. No paid compute or model-training campaign was run.
