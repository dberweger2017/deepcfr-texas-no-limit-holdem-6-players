"""Play complete Slumbot hands and summarize them.

A hand is a `new_hand` request, one `act` request per client decision, and the terminal
response. Every decision and the terminal settlement go to the journal; the final public
action must continue exactly what the client saw and sent.
"""

from dataclasses import dataclass
import math
import statistics

from src.arena.external.slumbot import TOKEN, SlumbotAdapter, replay_record, terminal_record
from src.game.types import Action, ActionKind


class CheckCall:
    """Checks when it can, otherwise calls: Slumbot's own sample strategy."""
    scripted = True
    name = 'check-call'

    def choose_action(self, view):
        kinds = view.legal_actions.kinds
        return Action(ActionKind.CHECK if ActionKind.CHECK in kinds else ActionKind.CALL)


def hu200_average(path, sha256: str, card_buckets=None):
    """A pinned HU200 average: v1 cards, or K50 with the pinned table directory."""
    from src.blueprint.abstraction import HU200_EQUITY_SCHEMA, HU200_SCHEMA
    from src.blueprint.average import AveragePolicy
    # Native exports are written in key order; the sorted reader checks that as it loads.
    return AveragePolicy(path, sha256, expected_schema=HU200_SCHEMA if card_buckets is None else HU200_EQUITY_SCHEMA,
                         sorted_rows=True, card_buckets=card_buckets)


@dataclass
class Hand:
    hand_id: str
    decisions: list
    terminal: dict


def play_hand(connection, player, hand_id: str, journal) -> Hand:
    adapter = SlumbotAdapter()
    response = connection.request('new_hand')
    decisions = []
    while 'winnings' not in response:
        view = adapter.observation(response, hand_id)
        record = replay_record(response, hand_id, player.choose_action(view))
        journal.append(record)
        decisions.append(record)
        response = connection.request('act', record['selected'])
    terminal = terminal_record(response, hand_id)
    if decisions:
        last = decisions[-1]
        if not terminal['action'].startswith(last['public']['action'] + last['selected']):
            raise ValueError('The final action does not continue the client decisions')
        if (terminal['client_pos'], terminal['hole_cards']) != (last['public']['client_pos'], list(last['public']['hole_cards'])):
            raise ValueError('The terminal seat or hole cards changed during the hand')
    journal.terminal(terminal)
    return Hand(hand_id, decisions, terminal)


def interval(values):
    """Mean and nominal 95% half-width; hands are independent fresh deals."""
    if len(values) < 2:
        return {'mean': statistics.fmean(values) if values else None, 'half_width_95': None, 'n': len(values)}
    return {'mean': statistics.fmean(values), 'n': len(values),
            'half_width_95': 1.96 * statistics.stdev(values) / math.sqrt(len(values))}


def without_decisions(hand: Hand) -> Hand:
    """The hand with only its decision count: the journal already holds every decision record,
    and keeping them in memory grows by about 20 KB a hand over long runs."""
    return Hand(hand.hand_id, [None] * len(hand.decisions), hand.terminal)


def summarize(hands: list[Hand]) -> dict:
    """BB/100 equals mean chips per hand, since the big blind is 100 chips."""
    terminals = [h.terminal for h in hands]
    by_seat = {seat: [t['winnings'] for t in terminals if t['client_pos'] == seat] for seat in (0, 1)}
    adjusted = [t['winnings'] - t['baseline_winnings'] for t in terminals if t['baseline_winnings'] is not None]
    return {
        'hands': len(hands),
        'decisions': sum(len(h.decisions) for h in hands),
        'net_chips': sum(t['winnings'] for t in terminals),
        'bb_per_100': interval([t['winnings'] for t in terminals]),
        # Slumbot's baseline follows the client's own actions (see docs/reports/slumbot-client.md),
        # so this difference is kept for reference only; it does not measure skill.
        'baseline_difference_bb_per_100': interval(adjusted),
        'bb_per_100_by_seat': {('big_blind' if seat == 0 else 'small_blind_button'): interval(v) for seat, v in by_seat.items()},
        'endings': {kind: sum(1 for t in terminals if ending(t) == kind) for kind in ('client_fold', 'slumbot_fold', 'showdown')},
    }


def ending(terminal) -> str:
    if not terminal['action'].endswith('f'):
        return 'showdown'
    # The folder acted last; replaying the turn order tells who it was.
    actor = 1
    for token in TOKEN.findall(terminal['action'])[:-1]:
        actor = 0 if token == '/' else 1 - actor
    return 'client_fold' if actor == terminal['client_pos'] else 'slumbot_fold'


def session_accounting(hands: list[Hand]) -> dict | None:
    """Slumbot's own running totals across the run, which include abandoned hands.

    Each terminal reports the session total after that hand, so a session's first one minus its
    winnings is where that session started. A new session (after a login or a refused request)
    restarts the hand count. The difference from our verified total is what hands without a
    verified terminal (abandoned ones) cost, as Slumbot counted them.
    """
    terminals = [h.terminal for h in hands]
    if not terminals or any(type(t.get('session_total')) is not int for t in terminals):
        return None
    sessions = [[terminals[0]]]
    for previous, terminal in zip(terminals, terminals[1:]):
        if terminal['session_num_hands'] <= previous['session_num_hands']:
            sessions.append([])
        sessions[-1].append(terminal)
    slumbot_net = sum(s[-1]['session_total'] - (s[0]['session_total'] - s[0]['winnings']) for s in sessions)
    verified = sum(t['winnings'] for t in terminals)
    return {'slumbot_net_chips': slumbot_net, 'verified_net_chips': verified,
            'unverified_chips': slumbot_net - verified, 'sessions': len(sessions),
            'slumbot_hands': sum(s[-1]['session_num_hands'] - s[0]['session_num_hands'] + 1 for s in sessions)}
