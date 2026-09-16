# Supported game

The initial benchmark is unraked, no-ante cash-game no-limit Texas Hold'em: a standard 52-card deck, two private cards, one five-card board, table stakes, and best-five-card showdown values. Start with 100 BB stacks and 0.5/1 blinds. Four-, five-, and six-handed games are the primary targets; the engine also handles heads-up action order.

The maintained [pokers fork](https://github.com/dberweger2017/pokers) implements the single-hand rules. Its [rules contract](https://github.com/dberweger2017/pokers/blob/5db20e3d5d6862b32a7402035c1340b622d3b005/RULES.md) specifies minimum raises, per-player reopening after short all-ins, side pots, odd chips, dealing, and settlement, with links to the source rules. The exact installed engine is pinned in [requirements.txt](../requirements.txt); use the rules file at that commit when reproducing a run.

## Amounts and actions

The engine stores integer chips. The Python interface expresses amounts in table units, with `chip_unit=0.01` by default. Blinds, stacks, and executed wagers must be multiples of that unit. Unequal stacks use `stakes=[...]` in seat order. Fractional chip amounts, negative raises, and nonfinite amounts are rejected.

The current integration uses the engine's **additional raise** convention: match the outstanding wager, then add the action's amount. A raise from 2 to 10 has amount 8; the following minimum raise-to is 18. `state.min_raise` supplies the last full increment. A smaller increase is permitted only as an exact all-in. A legal-action list alone does not establish that an arbitrary raise size is legal.

The current typed action interface uses explicit integer raise-to targets and rejects amounts outside the legal bounds. Historical checkpoint inference preserves its old sizing convention: round to the nearest chip, with half chips rounded upward, then clamp to the legal bounds. The hand interface validates the final action once; it does not retry or silently substitute a different action.

Calls never exceed the caller's remaining stack. At settlement, stacks include winnings and refunds; reward is final stack minus initial stack. Committed chips and pot are then zero. Logging records failed transitions and does not change game rules or fabricate check actions.

## Player information

The engine's `State` is a privileged simulator object, containing all hole cards and the deck. Read-only Python fields prevent accidental editing but do not enforce fair information access.

The [player-observation interface](observations.md) now separates policies from the simulator. It supplies immutable snapshots, complete public betting/reveal events, exact integer raise-to bounds, and owner-specific prior records. Current policies and the historical checkpoint encoder accept public observations, never raw engine state. Counterfactual traversal does not write live opponent history.

The observation layer adds the named `nlhe-cash-auto-muck-v1` disclosure profile: the last river aggressor shows first, otherwise the first live seat left of the button; subsequent hands can be mucked only after they are beaten by tabled cards in every eligible pot. Folded and mucked cards remain hidden from other players. See the interface document for sources, events, retention rules, and unsupported disclosure procedures.

## Table sessions and exclusions

Participants and stacks are fixed for a hand. The [session manager](sessions.md) now handles occupied physical seats, public identities, bankrolls, joins, departures, sit-outs, top-ups, and button/blind movement. Its `nlhe-moving-button-wait-bb-v1` profile uses a forward-moving button and big-blind-only entry for new or returning players. Changes are accepted only between settled hands. Private records follow public identities when seats change. The session contract specifies heads-up transitions, all-away reopening, buy-in bounds, public spectator records, and replay.

Rake, antes, straddles, multiple runouts, tournament payouts, and live-dealer irregularities are outside this initial profile. Any addition needs a named rule choice and its own checks. The corrected engine is a foundation for training; it does not establish playing strength or make historical checkpoints valid benchmarks for this game.
