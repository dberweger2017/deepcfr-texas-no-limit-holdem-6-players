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
        'baseline_adjusted_bb_per_100': interval(adjusted),
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
    """Slumbot's own running total across the run, which includes abandoned hands.

    Each terminal reports the session total after that hand, so the first one minus its
    winnings is where the run started. The difference from our verified total is what hands
    without a verified terminal (abandoned ones) cost, as Slumbot counted them.
    """
    terminals = [h.terminal for h in hands]
    if not terminals or any(type(t.get('session_total')) is not int for t in terminals):
        return None
    start = terminals[0]['session_total'] - terminals[0]['winnings']
    slumbot_net = terminals[-1]['session_total'] - start
    verified = sum(t['winnings'] for t in terminals)
    return {'slumbot_net_chips': slumbot_net, 'verified_net_chips': verified,
            'unverified_chips': slumbot_net - verified,
            'slumbot_hands': terminals[-1]['session_num_hands'] - terminals[0]['session_num_hands'] + 1}
