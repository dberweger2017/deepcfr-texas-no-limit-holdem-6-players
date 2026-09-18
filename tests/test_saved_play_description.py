import pytest

from scripts.analyse_saved_play import describe_hand, summarize


def hand(hero=1):
    return {
        'status': 'completed', 'mode': 'fixed', 'rotation': hero, 'big_blind': 2,
        'block': 0, 'hand': 0, 'arm': 'candidate', 'candidate_chips': -200,
        'net_chips': [200, -200],
        'events': [
            {'event': 'HandStarted', 'player_ids': ['player-1', 'player-0'], 'stacks': [200, 200]},
            {'event': 'BlindPosted', 'seat': 1, 'amount': 1},
            {'event': 'BlindPosted', 'seat': 0, 'amount': 2},
            {'event': 'Decision', 'seat': 1, 'legal_actions': {'call_amount': 1}},
            {'event': 'ActionTaken', 'seat': 1, 'street': 'preflop', 'paid': 199,
             'action': {'kind': 'raise', 'raise_to': 200}},
        ],
    }


def test_blind_is_counted_when_recognizing_an_all_in():
    result = describe_hand(hand())
    assert result['commitment'] == 'preflop_all_in'
    assert result['first_action_shove_facing_at_most_one_bb']
    assert result['actions'][0]['paid_bb'] == 99.5
    assert result['actions'][0]['pot_before_bb'] == 1.5


def test_all_in_call_is_not_a_shove_and_returns_reconcile():
    row = hand()
    row['events'][-1]['action'] = {'kind': 'call', 'raise_to': None}
    result = summarize([row])['candidate']
    assert result['first_action_shoves'] == 0
    assert result['commitment']['preflop_all_in']['hands'] == 1
    assert sum(v['contribution_bb_per_100_all_hands'] for v in result['commitment'].values()) == result['profit_bb_per_100']


def test_candidate_seat_must_match_the_saved_identity():
    with pytest.raises(ValueError, match='lineup'):
        describe_hand(hand(hero=0))


def test_walk_does_not_invent_a_hero_decision():
    row = hand()
    row['events'] = row['events'][:3]
    row['candidate_chips'] = 2
    row['net_chips'] = [-2, 2]
    result = describe_hand(row)
    assert result['actions'] == []
    assert result['commitment'] == 'never_all_in'
    assert not result['first_action_shove']
