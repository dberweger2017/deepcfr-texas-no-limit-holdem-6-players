# Player observations

The playing interface is `Observation -> Action`. The host owns the Rust engine and its hidden state. Policies receive immutable values with no engine reference, deck, seed, transition method, or unrevealed opponent cards.

## Use the interface

```python
from src.game.hand import Hand, Table
from src.game.play import RandomPolicy, play_hand

table = Table(
    player_ids=("alice", "bob", "carol", "dave"),
    stacks=(10_000, 10_000, 10_000, 10_000),
)
hand = Hand.start(table, hand_id="hand-1", seed=123)
policies = {identity: RandomPolicy(100 + seat)
            for seat, identity in enumerate(table.player_ids)}
finished = play_hand(hand, policies)
alice_result = finished.observe(0)
```

`Hand` and `Table` belong to the host. A policy implements `choose_action(observation)` and returns an `Action`. `play_hand` gives each policy only its own observation and rejects sharing one policy instance across multiple seats. Policy sampling has its own random generator; it does not use the hidden deck's random stream.

For a small headless check:

```bash
python -m scripts.check_game --players 6 --hands 20 --seed 0
```

This checks execution and accounting with random policies. It is not a poker-strength benchmark.

## Amounts and legal actions

All amounts in the new interface are integer chips. `chip_unit` is a decimal string describing their currency value. With the defaults, one chip is 0.01 table units, blinds are 50/100 chips, and a 10,000-chip stack is 100 BB.

`Action(ActionKind.RAISE, raise_to=...)` names the player's total contribution on the current street. It is not the amount added after calling. For 1/2 blinds, opening to 10 is `raise_to=10`, and the following minimum full raise-to is 18.

`observation.legal_actions` contains the available kinds, the actual call payment capped at the player's stack, and minimum/maximum raise-to targets. A short all-in has equal minimum and maximum targets. Checks and calls are distinct. An unavailable type, fractional target, or out-of-range target fails without changing the hand; the new interface never repairs the action.

Observers who are not acting receive no actionable choices. The public history still records the acting player's decision bounds, which follow from visible betting and stacks.

## What the observation contains

- Its owner and compact hand seat, physical seat mapping, occupied table roster, capacity, public player identities, hand identifier, button, blinds, and chip denomination. See [session seat semantics](sessions.md#observations-and-history-ownership).
- The owner's two hole cards, including after folding or mucking; an empty tuple for a seated spectator who was dealt out.
- The board and the street on which each part was dealt.
- Every player's starting stack, remaining stack, street bet, total contribution, folded/all-in status, and actually tabled cards.
- Current main/side-pot amounts and eligible seats. Pot amounts are provisional while betting continues; unmatched contributions are marked with a possible refund owner.
- The complete public event sequence, including the owner's actions, and the owner's retained records from previous hands.

Tuples and frozen dataclasses keep the data independent of the engine. The normal construction path contains only those values, strings, integers, booleans, and enums. A policy cannot call a simulation method through an observation or obtain a back-reference to its hand.

This is an API boundary for the poker program. It is not a sandbox for hostile Python code running in the same process.

## Public events and replay

`HandStarted` records schema version 2 and rules profile `nlhe-cash-auto-muck-v1`. A public hand identifier must be independent of the hidden seed. Player identities must name publicly known occupants, not checkpoint names or hidden opponent types.

| Event | Meaning |
| --- | --- |
| `HandStarted` | Public table configuration and starting stacks |
| `BlindPosted` | Actual payment, including a short blind |
| `Decision` | Actor and exact legal bounds |
| `ActionTaken` | Actor, street, chosen action, and chips paid |
| `BoardDealt` | Cards revealed on one street |
| `CardsShown` | Two cards publicly tabled by a live player |
| `CardsMucked` | A live showdown hand discarded without disclosure |
| `HandFinished` | Final stacks and the contribution-based pots settled |

`replay(events, seat, hole_cards, previous_hands=())` reconstructs an observation using public events and separately supplied private cards belonging to that seat. The host obtains those private cards from the engine; public replay must never infer them by opening another player's private record.

A call that closes all betting can produce several board events and settlement. Those events remain in their reveal order; payouts do not appear in an earlier betting event. Fold wins do not expose the winning hand or deal an unnecessary board. Terminal observations have no actor or legal actions, and their stacks include payouts and refunds.

## Showdown and mucking

The selected cash-game procedure shows the last river aggressor first; if there was no river bet, it starts left of the button. Surviving players then proceed clockwise. The default disclosure behavior tables a hand if it can still tie or beat an already shown hand in any pot it can win, and mucks it otherwise. The first hand contesting a pot therefore tables. A side-pot winner must show even when it loses the main pot.

The disclosure comparison uses a player's own cards, the public board, and hands already tabled. It does not use the strength of later, still-private hands to decide whether to muck. The Rust engine remains responsible for payouts; the small Python evaluator is used only for this show/muck choice.

This follows the ordered-showdown and optional losing-hand mucking described by [PokerStars' showdown rules](https://www.pokerstars.com/help/articles/poker-rules-master/) and [showdown explanation](https://www.pokerstars.com/poker/learn/lesson/poker-showdown/). It is a documented default disclosure choice, not a claim that every room requires automatic mucking. Cards that were folded or mucked stay private in every other player's observation and retained history. The owner still remembers their own cards.

Voluntary displays outside this procedure, requests to see a hand, accidental exposure, and tournament rules requiring earlier all-in tabling are not implemented. They need explicit events and rule choices, rather than access to the hidden engine state.

## Memory and counterfactual branches

`Hand.apply` returns a new hand and leaves its parent and event tuple unchanged. Exploring one action does not append events to another branch or to persistent memory.

`PlayerHistory.append` accepts a completed observation belonging to that player. It rejects a different owner, incomplete hands, and duplicate hand identifiers. The record stores the owner's cards and public events; it does not recursively store earlier private histories. `Hand.observe` rejects prior records belonging to another identity, including when the same seat has a new occupant.

The host commits records only after actual completed hands. `play_hand` accepts per-player histories but does not mutate them. Evaluation and play loops commit completed records outside traversal. CFR terminal payoffs remain learning targets; simulated opponent actions and terminal branches no longer update live opponent history.

The initial retention policy keeps all completed records for a match in memory. The existing UI/CLI starts fresh history when its model lineup is reloaded. There is no disk persistence or eviction policy yet. The [session manager](sessions.md) retains identity-owned histories across lineup changes within a session and uses public identity keys for the old opponent model. It adds physical seat mappings and public records for seated spectators; cross-session transfer and disk persistence remain separate work.

## Existing models and entry points

The old models use a fixed feature layout and additional-raise amounts. `src/game/legacy.py` bridges that layout using a `LegacyView` built only from an observation; its `observation` field retains the complete history for future encoders. The view contains no simulator object or transition method. The adapter does not promise checkpoint compatibility or preserve the old game's bugs.

Training, evaluation, tournaments, CLI play, and GUI AI turns now use tracked hands and the observation dispatcher. Both neural agent classes and the shared random agent reject raw simulator state. The neural encoder also rejects it. CLI and GUI card displays use the human's view, including at showdown.

The current networks still encode only a subset of the available public history. Richer history encoding, variable-seat models, CFR corrections, and opponent-model improvements remain later roadmap tasks. The information interface is complete enough to supply them; this change does not establish strong play.

## Verification

The tests cover:

- Alternate hidden deals with the same visible history before the flop, on the flop, and on the turn, at four-, five-, and six-player tables. Observations, encoded inputs, neural outputs, and seeded actions agree for both neural agent types.
- Distinct betting histories remaining distinguishable even when stacks, board, and pot match.
- Public replay and chip accounting over 300 generated unequal-stack hands.
- Legal raise-to bounds, branch immutability, player-specific prior records, and rejection of raw-state policy calls.
- Showdown order, side-pot disclosure, muck privacy, and no contested payouts to mucked hands; the disclosure evaluator is also compared with 1,000 Rust-engine showdown results.
- Counterfactual traversal leaving live opponent histories unchanged, plus real evaluation hands entering only their owner's retained record.
- Headless execution and offscreen rendering of private/tabled cards.
