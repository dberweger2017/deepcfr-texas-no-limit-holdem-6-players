"""Reproduce retained human/bot decisions using legal seat observations only."""

import json
from random import Random

from src.arena.runner import public_events
from src.arena.schedule import digest
from src.game.hand import Hand
from src.game.types import Action, ActionKind
from src.play_api.configuration import distribution, policy_table, recorded_table
from src.play_api.service import _tuple, _model_info
from src.play_api.spectator import observation_record


def audit_state(state, policy):
    table = policy_table(policy)
    rows = list(state['history'])
    assert len(rows) == state['handsPlayed'], 'Hand count'
    if state['current']:
        if state['current'].get('completed'):
            assert state['current'] == rows[-1], 'Current/history mismatch'
        else:
            rows.append(state['current'])
    total = 0
    sampling_end = None
    for number, row in enumerate(rows):
        assert row['button'] == number % 2 and recorded_table(row) == table, 'Table/reset/rotation'
        hand = Hand.start(table.table(('human', 'trained'), row['button']),
                          hand_id=row['handId'], seed=row['dealSeed'])
        generator = Random()
        audited = 'botDecisions' in row
        if table.stack == 10000:
            assert audited and row['modelIdentity'] == _model_info(policy), 'Research decision identity'
        if audited:
            generator.setstate(_tuple(row['samplingStart']))
            if sampling_end is not None:
                assert _tuple(row['samplingStart']) == sampling_end, 'Sampling continuity'
        decisions = iter(row.get('botDecisions', ()))
        for index, action_row in enumerate(row['actions']):
            assert hand.actor == action_row['seat'], 'Actor'
            view = hand.observe(hand.actor)
            legal = view.legal_actions
            assert action_row['legal'] == {'kinds': [k.value for k in legal.kinds],
                'call': legal.call_amount, 'minRaiseTo': legal.min_raise_to,
                'maxRaiseTo': legal.max_raise_to}, 'Legal bounds'
            action = Action(ActionKind(action_row['kind']), action_row['raiseTo'])
            legal.validate(action)
            if audited and hand.actor == 1:
                record = next(decisions)
                menu, probabilities, _, telemetry = distribution(policy, view)
                expected = [{'label': item.name, 'kind': item.action.kind.value,
                             'raiseTo': item.action.raise_to, 'probability': p}
                            for item, p in zip(menu, probabilities)]
                assert record['actionIndex'] == index, 'Bot action index'
                assert record['observation'] == observation_record(view), 'Legal observation'
                assert record['menu'] == expected, 'Distribution'
                assert record['telemetry'] == json.loads(json.dumps(telemetry)), 'Translation receipt'
                selected = generator.choices(range(len(menu)), weights=probabilities, k=1)[0]
                assert record['selectedIndex'] == selected and menu[selected].action == action, 'Sample'
            hand = hand.apply(action)
        assert next(decisions, None) is None, 'Extra bot decisions'
        if audited:
            sampling_end = generator.getstate()
        if number < state['handsPlayed']:
            assert row.get('completed') and hand.finished, 'Settlement'
            net = hand.observe(0).players[0].stack - table.stack
            assert sum(p.stack for p in hand.observe(0).players) == 2 * table.stack, 'Conservation'
            assert row['humanChips'] == net, 'Net chips'
            assert row['publicEventsSha256'] == digest(public_events(hand.events)), 'Events'
            total += net
        else:
            assert not hand.finished and not row.get('completed'), 'Unrecorded settlement'
    assert total == state['totalChips'], 'Session total'
    if sampling_end is not None:
        assert _tuple(state['botRng']) == sampling_end, 'Persisted sampling state'
