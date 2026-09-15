# Supported game

The initial benchmark is unraked, no-ante cash-game no-limit Texas Hold'em: a standard 52-card deck, two private cards, one five-card board, table stakes, and best-five-card showdown values. Start with 100 BB stacks and 0.5/1 blinds. Four-, five-, and six-handed games are the primary targets; the engine also handles heads-up action order.

The maintained [pokers fork](https://github.com/dberweger2017/pokers) implements the single-hand rules. Its [rules contract](https://github.com/dberweger2017/pokers/blob/5db20e3d5d6862b32a7402035c1340b622d3b005/RULES.md) specifies minimum raises, per-player reopening after short all-ins, side pots, odd chips, dealing, and settlement, with links to the source rules. The exact installed engine is pinned in [requirements.txt](../requirements.txt); use the rules file at that commit when reproducing a run.

## Amounts and actions

The engine stores integer chips. The Python interface expresses amounts in table units, with `chip_unit=0.01` by default. Blinds, stacks, and executed wagers must be multiples of that unit. Unequal stacks use `stakes=[...]` in seat order. Fractional chip amounts, negative raises, and nonfinite amounts are rejected.

The current integration uses the engine's **additional raise** convention: match the outstanding wager, then add the action's amount. A raise from 2 to 10 has amount 8; the following minimum raise-to is 18. `state.min_raise` supplies the last full increment. A smaller increase is permitted only as an exact all-in. A legal-action list alone does not establish that an arbitrary raise size is legal.

The legacy sizing adapter rounds to the nearest chip, with half chips rounded upward, and clamps to the legal amount bounds. It validates that exact amount once. It no longer retries smaller epsilon-adjusted wagers or silently turns an engine rejection into a different raise. Strict runs fail when the requested action type is unavailable or the engine rejects the mapped amount. The replacement typed action interface will use explicit raise-to targets.

Calls never exceed the caller's remaining stack. At settlement, stacks include winnings and refunds; reward is final stack minus initial stack. Committed chips and pot are then zero. Logging records failed transitions and does not change game rules or fabricate check actions.

## Player information

The engine's `State` is a privileged simulator object, containing all hole cards and the deck. Read-only Python fields prevent accidental editing but do not enforce fair information access.

The next implementation step is the immutable player observation described in the [roadmap](../ROADMAP.md#information-available-to-the-agent). It must contain complete observed betting history, the player's own cards, public cards and stacks, legal action bounds, and legitimate card disclosures. It must exclude unrevealed opponent cards, undealt cards, seeds, and future outcomes. The current state-based agent interface has not yet passed this acceptance gate.

## Table sessions and exclusions

Participants and stacks are fixed for a hand. The engine accepts a different lineup and unequal stacks for the next hand, but this project still needs a session manager for occupied seats, public identities, joins, departures, sit-outs, top-ups, and button/blind movement. Showdown exposure and mucking events also need an explicit player-visible implementation.

Rake, antes, straddles, multiple runouts, tournament payouts, and live-dealer irregularities are outside this initial profile. Any addition needs a named rule choice and its own checks. The corrected engine is a foundation for training; it does not establish playing strength or make historical checkpoints valid benchmarks for this game.
