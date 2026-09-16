# Cash-table sessions

`Session` owns the table between hands: physical seats, public identities, bankrolls, admission at the blinds, and private hand records. The single-hand engine still receives only the players dealt into that hand. The session profile is `nlhe-moving-button-wait-bb-v1`, used with the existing `nlhe-cash-auto-muck-v1` hand rules.

## Run a session

```python
from src.game.play import RandomPolicy, play_session_hand
from src.game.session import Session

session = Session("local-table")
policies = {}
for seat in range(6):
    identity = f"player-{seat}"
    session.join(identity, seat, 10_000)
    policies[identity] = RandomPolicy(100 + seat)

session.start_hand(seed=123, opening_button=0)
play_session_hand(session, policies)

session.leave("player-5")
session.top_up("player-0", 1_000)  # Subject to the configured table maximum.
session.start_hand(seed=456)
play_session_hand(session, policies)
```

The default table has six physical seats, 50/100-chip blinds, a 20–200 BB buy-in range, and denomination `0.01`. Configuration also supports two to ten seats. Seats increase clockwise. Use public identities independent of checkpoint names, opponent types, and deal seeds. A policy belongs to one identity; never reuse its private memory for a different player.

For a runnable check with changing lineups:

```bash
python -m scripts.check_session --hands 30 --seed 0
```

This uses separate deal and policy random streams and reports chip flows and actual participant counts. Random policies frequently bust, so the number dealt in can differ from occupied seats. This is an execution check, not evidence of playing strength. `scripts.check_game` remains an independent-hand check with reset stacks.

## Button and admission rules

The [PokerStars moving-button explanation](https://www.pokerstars.com/help/articles/fwd-moving-button/) describes advancing the button to the next remaining player clockwise and posting both blinds every hand, including heads-up. Departures can cause an incumbent to skip a blind; this profile does not introduce a dead button to compensate.

The [PokerStars Live cash rules](https://www.pokerstarslive.com/poker/cashgamerules/) allow waiting for the big blind as an entry option. We adopt that option and specify the following house procedure; we do not adopt that venue's entire rulebook:

- **Opening a game:** the host supplies an occupied `opening_button`. All initially ready players enter. Choose this button with a separate seating draw or a declared evaluation rotation, never from hidden card information.
- **Continuing a game:** advance the button clockwise from its previous physical position to the next continuing player with chips. Empty seats, sitting-out players, busted players, and waiters do not receive the button.
- **Waiting players:** a join after play begins, return from sitting out, reload after busting, or move to another seat waits for a full big blind. Returning players wait even if they were away for less than an orbit. There is no mid-orbit posting, dead-blind debt payment, straddle, or buying the button in this profile.
- **Admission:** find the next continuing player after the button as the prospective small blind. The first continuing player or funded waiter clockwise after that seat is the big blind. If it is a waiter, admit that one player. Other waiters remain out of the hand. This also allows a heads-up table to become three-handed: the other incumbent posts the small blind and the entrant posts the big blind.
- **Heads-up without an entrant:** the button posts the small blind and acts first preflop; the other player posts the big blind and acts first after the flop.
- **One incumbent:** that player holds the button/small blind, and the next funded waiter clockwise enters in the big blind.
- **No incumbents:** ordinary dealing pauses. Once at least two funded players are ready, the host can explicitly reopen the game with `opening_button`; all ready waiters enter together. The journal marks this as an opening. Histories and bankrolls remain intact. A caller cannot reset the button while continuing players remain.

Waiting players need at least a full big blind. An incumbent with fewer chips may continue and post a short blind. A zero stack becomes `busted` after settlement. The button and blind movement tests include every departure position, every waiting seat, and transitions into and out of heads-up.

## Chips and table changes

Only `join`, `leave`, `top_up`, `move`, `sit_out`, and `return_to_play` change the lineup or external chips. They are accepted between settled hands. There is no queued mid-hand departure or top-up: the host must finish and settle first, including when a departing player has already folded. A failed policy leaves the hand available for diagnosis or continuation and keeps the table locked.

`leave` returns the player's full stack. There are no partial withdrawals or transfers. Buy-ins and top-ups require exact positive integer chips. A reload from zero must satisfy the minimum buy-in; a top-up cannot exceed the table maximum. Winnings may exceed that maximum and stay in play. A player returning after cashing out must bring back at least that cashout, even if it exceeds the normal maximum. This conservative obligation lasts for this session; there is no elapsed-time exception.

The public ledger records cumulative chips bought in and cashed out. At every table snapshot:

```
sum(table bankrolls) = chips in - chips out
```

During a hand, table bankrolls are frozen at their starting values; use the hand observation for remaining stacks and committed chips. Settlement transfers final engine stacks back into the physical seats exactly once. Folded and all-in are hand states; they do not mean absent, sitting out, or busted.

## Observations and history ownership

Observation schema **2** adds `seat_numbers`, `table_seats`, `capacity`, and `session_profile` to the public hand header and its replay. Existing action, player, pot, and button indexes remain compact **hand seats**. `seat_numbers[hand_seat]` gives the physical seat. `table_seats` is the immutable roster at the start of that hand, including waiting and sitting-out occupants. Missing physical seats are empty. Use `observation.players` for live hand amounts.

`session.observe(identity)` gives a participant their own hole cards. A seated waiter or busted player can watch: their hand seat is `-1`, hole cards are empty, and legal actions are empty. They still see public actions, boards, and actual showdown disclosures. They cannot receive another participant's cards through spectator replay. In this profile `sitting_out` means away from the table; those players receive no new observations or records until returning.

Settlement stores records by **identity**, including public spectator records for present players dealt out. Moving seats, leaving, and rejoining with the same identity preserves that owner's legitimate records. Replacing an occupant with another identity supplies no previous occupant's private history. Records retain all completed observed hands in memory for the lifetime of the session. Disk persistence, cross-session identity/history transfer, and retention limits are separate work.

The retired opponent model is no longer a consumer of these records. Future adaptation must use public identities, retain only observed information, and keep counterfactual traversal separate from live history.

## Public journal and replay

Each `SessionEvent` is a frozen public snapshot with a change kind, occupied seats, button, table configuration, opening flag, and cash ledger. Start, action, and settlement entries also carry the public hand-event prefix available at that point. It contains no hole cards except publicly tabled cards, no deal seeds, and no engine references.

`replay_session(events)` validates chip conservation and returns the last public snapshot. Replay any included `hand_events` through the observation replay with separately supplied owner cards, or with empty cards and a seated spectator's identity. These are typed, trusted host records; this is not an untrusted JSON importer or a simulator-resume format. The journal uses snapshots for straightforward inspection; storage compaction belongs to later profiling/persistence work.

## Integration limits and validation

The new headless runner executes session hands with observation-based policies. Existing neural networks still have fixed player-count inputs and compact-seat configuration. This PR does not make one old checkpoint support arbitrary table sizes. Legacy GUI/CLI and fixed-stack evaluation remain on their existing tracked-hand runners; the new evaluation arena should use sessions when bankroll continuity is part of its declared benchmark.

Checks cover deterministic 6→5→4 play; sparse seats; departures around the blinds; waiting and heads-up admission; all-away reopening; buy-in/top-up/cashout accounting; failed mid-hand mutations; one-time settlement; owner/spectator/replacement privacy; identity-based opponent features and outcomes; policy failures; and 600 generated session hands across 20 seeds. CI runs the full regression suite and the session CLI. These checks establish environment behavior, not strategic strength.
