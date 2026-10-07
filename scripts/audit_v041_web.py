"""Replay retained web journals and compare every bot position with the arena reader.

Read-only evidence audit: no service replay/reporting methods, no private game
state passed to either policy, and no inferred poker-strength claim.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from src.arena.catalog import Checkpoint
from src.arena.runner import public_events
from src.blueprint.abstraction import information_key
from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT
from src.blueprint.average import AveragePolicy
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.policies.v041 import MODEL_SHA256, load_policy
from src.policies.v040 import load_policy as load_b100m, EXPECTED_SHA256 as R_SHA256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_states(database):
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        return [json.loads(row[0]) for row in connection.execute('SELECT state FROM sessions ORDER BY id')]


def audit_states(states, web, arena, expected_sha):
    counts = Counter()
    positions = []
    hands = []
    for state in states:
        assert state['modelSha256'] == expected_sha, 'Session model hash'
        rows = state['history']
        assert state['handsPlayed'] == len(rows), 'Missing history'
        assert state['current'] is None or state['current'].get('completed'), 'Unfinished hand'
        profit = 0
        for number, row in enumerate(rows):
            assert row['completed'] and row['modelSha256'] == expected_sha
            assert row['button'] == number % 2, 'Seat rotation'
            hand = Hand.start(Table(('human', 'trained'), (2000, 2000), button=row['button']),
                              hand_id=row['handId'], seed=row['dealSeed'])
            lookups = []
            for index, record in enumerate(row['actions']):
                assert hand.actor == record['seat'], 'Actor mismatch'
                view = hand.observe(hand.actor)
                legal = view.legal_actions
                assert record['legal'] == {'kinds': [kind.value for kind in legal.kinds],
                                           'call': legal.call_amount, 'minRaiseTo': legal.min_raise_to,
                                           'maxRaiseTo': legal.max_raise_to}, 'Legal bounds'
                action = Action(ActionKind(record['kind']), record['raiseTo'])
                legal.validate(action)
                if hand.actor == 1:
                    actual = web.distribution(view)
                    expected = arena.distribution(view)
                    assert actual == expected, 'Web/arena distribution mismatch'
                    menu, probabilities, trained = expected
                    key = information_key(view, menu, schema=arena.abstraction)
                    source = ('missing' if not trained else
                              'zero_mass' if key in getattr(arena, 'zero_mass', ()) else 'trained')
                    counts[source] += 1
                    lookups.append({'street': view.street.value, 'trained': bool(trained)})
                    positions.append({'handId': row['handId'], 'action': index,
                                      'key': key, 'source': source,
                                      'menu': [item.name for item in menu], 'probabilities': probabilities})
                    # A fallback is expected only where the pinned arena policy
                    # itself has no entry or stored zero mass.
                    if source != 'trained':
                        assert probabilities == (1 / len(menu),) * len(menu), 'Unexpected fallback'
                elif record['kind'] == 'raise' and record['raiseTo'] == 201:
                    counts['exact_201_chip_human_raises'] += 1
                hand = hand.apply(action)
                counts['actions'] += 1
            assert lookups == row['lookup'], 'Lookup telemetry mismatch'
            assert hand.finished, 'Nonterminal settlement'
            net = [player.stack - 2000 for player in hand.observe(0).players]
            assert sum(net) == 0 and net[0] == row['humanChips'], 'Settlement mismatch'
            assert digest(public_events(hand.events)) == row['publicEventsSha256'], 'Event digest'
            profit += net[0]
            counts['hands'] += 1
            counts['human_button' if row['button'] == 0 else 'human_big_blind'] += 1
            counts[row['playMode'] + '_hands'] += 1
            hands.append({'handId': row['handId'], 'sourceVersion': row['sourceVersion'],
                          'humanButton': row['button'] == 0, 'playMode': row['playMode'],
                          'actions': len(row['actions']), 'publicEventsSha256': row['publicEventsSha256']})
        assert profit == state['totalChips'], 'Session settlement total'
    return {'status': 'verified', 'model_sha256': expected_sha, 'sessions': len(states),
            'counts': dict(counts), 'unexpected_fallbacks': 0,
            'positions_sha256': digest(positions), 'positions': positions, 'hands': hands}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--version', required=True, choices=('v0.4.1', 'v0.4.0'))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.version == 'v0.4.1':
        web = load_policy(args.model)
        arena = AveragePolicy(args.model, MODEL_SHA256)
        sha = MODEL_SHA256
    else:
        web = load_b100m(args.model)
        arena = FrozenBlueprint(Checkpoint('R1', str(args.model), R_SHA256, HU20_UNCAPPED_FORMAT), args.model)
        sha = R_SHA256
    result = audit_states(read_states(args.database), web, arena, sha)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('positions', 'hands')}))


if __name__ == '__main__':
    main()
