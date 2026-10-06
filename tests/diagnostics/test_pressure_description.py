"""Situations come from actual native amounts, not blind obligations or menu labels."""
from scripts.describe_hu20_cfr_plus_pressure import situations


def test_blind_completion_is_not_facing_a_raise_and_jam_uses_actual_stack():
    def action(logical, kind, amount=None, **observed):
        observation = {'call_amount': 0, 'stack': 1900, 'street_bet': 100,
                       'menu': [{'kind': 'raise', 'raise_to': 200}], **observed}
        return {'logical_player': logical, 'kind': kind, 'raise_to': amount,
                'street': 'preflop', 'observation': observation, 'average_mass_status': 'missing'}
    blind = action(0, 'call', call_amount=50)
    jam = action(1, 'raise', 2000)
    response = action(0, 'call', call_amount=1900)
    responses, groups = situations({'actions': [blind, jam, response]})
    assert len(responses) == 1
    assert responses[0]['facing'] == 'jam'
    assert responses[0]['off_menu_history']
    assert responses[0]['committed_chips'] == 1900
    assert groups['last_response'] == 'preflop/jam/call'
    assert groups['off_menu'] == 'off-menu-seen'


def test_min_raise_in_menu_is_not_off_menu_and_new_street_resets_pressure():
    observations = {'call_amount': 100, 'stack': 1900, 'street_bet': 100,
                    'menu': [{'kind': 'raise', 'raise_to': 200}]}
    opponent = {'logical_player': 1, 'kind': 'raise', 'raise_to': 200, 'street': 'preflop',
                'observation': observations, 'average_mass_status': None}
    target = {**opponent, 'logical_player': 0, 'kind': 'call', 'raise_to': None,
              'average_mass_status': 'positive_mass'}
    next_street = {**target, 'street': 'flop', 'kind': 'check',
                   'observation': {**observations, 'call_amount': 0}}
    responses, groups = situations({'actions': [opponent, target, next_street]})
    assert len(responses) == 1
    assert responses[0]['facing'] == 'raise'
    assert groups['off_menu'] == 'all-menu-sizes'
    assert groups['last_street'] == 'preflop'
