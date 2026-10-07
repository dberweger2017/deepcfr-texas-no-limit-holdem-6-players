"""Read-only spectator audit, reconstructing decisions without service replay code."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from random import Random

from src.arena.runner import public_events
from src.blueprint.abstraction import information_key
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _tuple(value):
    return tuple(_tuple(item) if isinstance(item, list) else item for item in value)


def audit_states(states: list[dict], policies: dict, identities: dict) -> dict:
    counts = Counter()
    for state in states:
        assert state['protocol'] == 'hu20-spectator-v1', 'Spectator protocol'
        models = state['models']
        assert len(models) == 2, 'Bot identities'
        for model in models:
            assert model == identities[model['version']], 'Session model/manifest identity'
            assert policies[model['version']].spec.sha256 == model['sha256'], 'Loaded model hash'
        rows = list(state['history'])
        assert len(rows) == state['handsPlayed'], 'Missing retained history'
        current = state['current']
        if current is not None:
            if current.get('completed'):
                assert rows and current == rows[-1], 'Completed current/history mismatch'
            else:
                rows.append(current)
        total = [0, 0]
        sampling_end = None
        for number, row in enumerate(rows):
            assert row['number'] == number and row['button'] == number % 2, 'Seat rotation'
            assert row['models'] == models, 'Hand model/manifest identity'
            if sampling_end is not None:
                assert row['samplingStart'] == sampling_end, 'Sampling continuity'
            generators = [Random() for _ in range(2)]
            for generator, start in zip(generators, row['samplingStart']):
                generator.setstate(_tuple(start))
            hand = Hand.start(Table(('bot-a', 'bot-b'), (2000, 2000), button=row['button']),
                              hand_id=row['handId'], seed=row['dealSeed'])
            for index, record in enumerate(row['decisions']):
                assert record['number'] == index and record['seat'] == hand.actor, 'Decision actor/order'
                seat = hand.actor
                view = hand.observe(seat)
                expected_view = asdict(view)
                expected_view['history'] = [{'event': type(event).__name__, **asdict(event)}
                                            for event in view.history]
                assert record['observation'] == _json_value(expected_view), 'Decision observation'
                assert record['model'] == models[seat], 'Decision model/manifest identity'
                policy = policies[models[seat]['version']]
                menu, probabilities, trained = policy.distribution(view)
                expected_menu = [{'label': item.name, 'kind': item.action.kind.value,
                                  'raiseTo': item.action.raise_to, 'probability': probability}
                                 for item, probability in zip(menu, probabilities)]
                assert record['menu'] == expected_menu, 'Action distribution mismatch'
                key = information_key(view, menu, schema=policy.abstraction)
                source = ('missing' if not trained else
                          'zero-mass' if key in getattr(policy, 'zero_mass', ()) else 'trained')
                assert record['informationKey'] == key and record['lookup'] == source, 'Lookup status'
                selected = generators[seat].choices(range(len(menu)), weights=probabilities, k=1)[0]
                assert type(record['selectedIndex']) is int and record['selectedIndex'] == selected, 'Sampled action'
                chosen = record['menu'][selected]
                action = Action(ActionKind(chosen['kind']), chosen['raiseTo'])
                view.legal_actions.validate(action)
                hand = hand.apply(action)
                counts['decisions'] += 1
                counts[source] += 1
                counts[f'bot_{seat}_decisions'] += 1
            sampling_end = _json_value([generator.getstate() for generator in generators])
            if number < state['handsPlayed']:
                assert row.get('completed') and hand.finished, 'Nonterminal settlement'
                net = [player.stack - 2000 for player in hand.observe(0).players]
                assert sum(net) == 0 and net == row['netChips'], 'Settlement mismatch'
                event_sha = hashlib.sha256(json.dumps(public_events(hand.events), sort_keys=True,
                                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                assert event_sha == row['publicEventsSha256'], 'Public event digest'
                total = [old + delta for old, delta in zip(total, net)]
                counts['hands'] += 1
                counts[f'button_{row["button"]}_hands'] += 1
            else:
                assert not hand.finished and not row.get('completed'), 'Unrecorded settlement'
        if sampling_end is not None:
            assert state['botRngs'] == sampling_end, 'Persisted sampling state'
        assert total == state['totalChips'], 'Session settlement total'
    return {'status': 'verified', 'sessions': len(states), 'hands': counts['hands'],
            'decisions': counts['decisions'], 'counts': dict(counts)}
