"""Cash-table lifecycle. This host owns chips and histories; policies see observations."""

from dataclasses import dataclass, replace

from src.game.hand import Hand, Table
from src.game.observation import Observation, PublicEvent
from src.game.play import PlayerHistory
from src.game.types import Action, TableSeat

SESSION_PROFILE = "nlhe-moving-button-wait-bb-v1"


@dataclass(frozen=True, slots=True)
class SessionEvent:
    kind: str
    seats: tuple[TableSeat, ...]
    button: int | None
    chips_in: int
    chips_out: int
    hand_events: tuple[PublicEvent, ...] = ()
    capacity: int = 6
    small_blind: int = 50
    big_blind: int = 100
    chip_unit: str = "0.01"
    profile: str = SESSION_PROFILE
    session_id: str = ""
    min_buy_in: int = 2000
    max_buy_in: int = 20000
    opening: bool = False


def replay_session(events: tuple[SessionEvent, ...]) -> SessionEvent:
    """Recover a public table snapshot, without restoring any private hand state."""
    if not events:
        raise ValueError("A session replay needs an event")
    for event in events:
        if sum(p.stack for p in event.seats) != event.chips_in - event.chips_out:
            raise ValueError("Session ledger does not conserve chips")
    return events[-1]


class Session:
    def __init__(
        self,
        session_id: str,
        *,
        capacity: int = 6,
        small_blind: int = 50,
        big_blind: int = 100,
        chip_unit: str = "0.01",
        min_buy_in: int = 2000,
        max_buy_in: int = 20000,
    ):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Provide a public session identity")
        if type(capacity) is not int or not 2 <= capacity <= 10:
            raise ValueError("A table needs 2–10 physical seats")
        config = Table(
            ("a", "b"),
            (min_buy_in, max_buy_in),
            small_blind=small_blind,
            big_blind=big_blind,
            chip_unit=chip_unit,
        )
        if not big_blind <= min_buy_in <= max_buy_in:
            raise ValueError("Buy-in bounds must be ordered and cover the big blind")
        self.session_id = session_id
        self.capacity = capacity
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.chip_unit = config.chip_unit
        self.min_buy_in = min_buy_in
        self.max_buy_in = max_buy_in
        self._seats: dict[int, TableSeat] = {}
        self._histories: dict[str, PlayerHistory] = {}
        self._cashouts: dict[str, int] = {}
        self._hand: Hand | None = None
        self._button: int | None = None
        self._hands = 0
        self._chips_in = 0
        self._chips_out = 0
        self._events: tuple[SessionEvent, ...] = ()
        self._record("opened")

    @property
    def seats(self) -> tuple[TableSeat, ...]:
        return tuple(self._seats[s] for s in sorted(self._seats))

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        return self._events

    @property
    def participants(self) -> tuple[str, ...]:
        return () if self._hand is None else self._hand.table.player_ids

    @property
    def actor(self) -> str | None:
        if self._hand is None or self._hand.finished:
            return None
        return self._hand.table.player_ids[self._hand.actor]

    @property
    def hand_active(self) -> bool:
        return self._hand is not None

    def _between_hands(self):
        if self._hand is not None:
            raise ValueError(
                "Finish and settle the current hand before changing the table"
            )

    def _seat_number(self, seat: int):
        if type(seat) is not int or not 0 <= seat < self.capacity:
            raise ValueError("Unknown physical seat")

    def _find(self, player_id: str) -> TableSeat:
        for player in self._seats.values():
            if player.player_id == player_id:
                return player
        raise ValueError("Player is not seated")

    def _record(
        self,
        kind: str,
        hand_events: tuple[PublicEvent, ...] = (),
        *,
        opening: bool = False,
    ):
        event = SessionEvent(
            kind,
            self.seats,
            self._button,
            self._chips_in,
            self._chips_out,
            hand_events,
            self.capacity,
            self.small_blind,
            self.big_blind,
            self.chip_unit,
            session_id=self.session_id,
            min_buy_in=self.min_buy_in,
            max_buy_in=self.max_buy_in,
            opening=opening,
        )
        replay_session((event,))
        self._events += (event,)

    def join(self, player_id: str, seat: int, chips: int):
        self._between_hands()
        self._seat_number(seat)
        if not isinstance(player_id, str) or not player_id:
            raise ValueError("Player identity must be a nonempty string")
        if seat in self._seats or any(p.player_id == player_id for p in self.seats):
            raise ValueError("Seat or player identity is already occupied")
        minimum = max(self.min_buy_in, self._cashouts.get(player_id, 0))
        # Cashing out cannot be used to shed a won stack on immediate re-entry.
        maximum = max(self.max_buy_in, minimum)
        if type(chips) is not int or not minimum <= chips <= maximum:
            raise ValueError("Buy-in is outside this player's allowed bounds")
        status = "playing" if self._hands == 0 else "waiting"
        self._seats[seat] = TableSeat(seat, player_id, chips, status)
        self._histories.setdefault(player_id, PlayerHistory(player_id))
        self._chips_in += chips
        self._record("joined")

    def leave(self, player_id: str) -> int:
        self._between_hands()
        player = self._find(player_id)
        del self._seats[player.seat]
        self._cashouts[player_id] = player.stack
        self._chips_out += player.stack
        self._record("left")
        return player.stack

    def top_up(self, player_id: str, chips: int):
        self._between_hands()
        player = self._find(player_id)
        if type(chips) is not int or chips <= 0:
            raise ValueError("A top-up must add positive integer chips")
        total = player.stack + chips
        if total > self.max_buy_in or (player.stack == 0 and total < self.min_buy_in):
            raise ValueError("Top-up is outside the table buy-in bounds")
        status = "waiting" if player.status == "busted" else player.status
        self._seats[player.seat] = replace(player, stack=total, status=status)
        self._chips_in += chips
        self._record("topped_up")

    def sit_out(self, player_id: str):
        self._between_hands()
        player = self._find(player_id)
        self._seats[player.seat] = replace(player, status="sitting_out")
        self._record("sat_out")

    def return_to_play(self, player_id: str):
        self._between_hands()
        player = self._find(player_id)
        if player.status != "sitting_out" or player.stack < self.big_blind:
            raise ValueError(
                "A returning player must be sitting out with a full big blind"
            )
        status = "playing" if self._hands == 0 else "waiting"
        self._seats[player.seat] = replace(player, status=status)
        self._record("returned")

    def move(self, player_id: str, seat: int):
        self._between_hands()
        self._seat_number(seat)
        player = self._find(player_id)
        if seat in self._seats:
            raise ValueError("Destination seat is occupied")
        del self._seats[player.seat]
        status = (
            "waiting" if player.status == "playing" and self._hands else player.status
        )
        self._seats[seat] = replace(player, seat=seat, status=status)
        self._record("moved")

    def _next(self, after: int, seats: list[int]) -> int:
        return min(seats, key=lambda seat: (seat - after - 1) % self.capacity)

    def _lineup(self, opening_button: int | None) -> tuple[list[int], int]:
        playing = [p.seat for p in self.seats if p.status == "playing" and p.stack > 0]
        waiting = [
            p.seat
            for p in self.seats
            if p.status == "waiting" and p.stack >= self.big_blind
        ]
        if self._hands == 0:
            if len(playing) < 2 or opening_button not in playing:
                raise ValueError(
                    "Opening hand needs two players and an explicit occupied button"
                )
            self._seat_number(opening_button)
            return playing, opening_button
        if not playing:
            if len(waiting) < 2 or opening_button not in waiting:
                raise ValueError(
                    "A table with no incumbents needs an explicit new opening button"
                )
            self._seat_number(opening_button)
            return waiting, opening_button
        if opening_button is not None:
            raise ValueError("The session advances the button after the opening hand")
        if len(playing) + len(waiting) < 2:
            raise ValueError("Not enough players to deal a hand")
        button = self._next(self._button, playing)
        if len(playing) == 1:
            return sorted(playing + [self._next(button, waiting)]), button
        # A waiter may enter only where they will actually post the next big blind.
        # For two incumbents, admitting a third player also restores the separate SB.
        sb = self._next(button, playing)
        bb = self._next(sb, playing + waiting)
        if bb in waiting:
            playing.append(bb)
        return sorted(playing), button

    def start_hand(self, *, seed: int, opening_button: int | None = None):
        self._between_hands()
        seats, button = self._lineup(opening_button)
        roster = tuple(
            replace(p, status="playing") if p.seat in seats else p for p in self.seats
        )
        players = [p for p in roster if p.seat in seats]
        table = Table(
            tuple(p.player_id for p in players),
            tuple(p.stack for p in players),
            seats.index(button),
            self.small_blind,
            self.big_blind,
            self.chip_unit,
            tuple(seats),
            roster,
            self.capacity,
            SESSION_PROFILE,
        )
        hand = Hand.start(
            table, hand_id=f"{self.session_id}/{self._hands + 1}", seed=seed
        )
        self._hand = hand
        self._button = button
        self._seats = {p.seat: p for p in roster}
        self._record("hand_started", hand.events, opening=opening_button is not None)

    def observe(self, player_id: str) -> Observation:
        if self._hand is None:
            raise ValueError("There is no current hand")
        return self._hand.observe_player(
            player_id, self._histories.get(player_id, PlayerHistory(player_id)).hands
        )

    def apply(self, player_id: str, action: Action):
        if self._hand is None or self.actor != player_id:
            raise ValueError("Only the current actor may act")
        self._hand = self._hand.apply(action)
        self._record("action", self._hand.events)

    def settle(self):
        if self._hand is None or not self._hand.finished:
            raise ValueError("Only a completed hand can be settled")
        hand = self._hand
        histories = dict(self._histories)
        seats = dict(self._seats)
        for player in self.seats:
            if player.status != "sitting_out":
                view = self.observe(player.player_id)
                histories[player.player_id] = histories[player.player_id].append(view)
        result = hand.observe(0)
        for seat, physical in enumerate(hand.table.seat_numbers):
            player = seats[physical]
            stack = result.players[seat].stack
            seats[physical] = replace(
                player, stack=stack, status="playing" if stack else "busted"
            )
        self._histories = histories
        self._seats = seats
        self._hands += 1
        self._hand = None
        self._record("hand_settled", hand.events)
