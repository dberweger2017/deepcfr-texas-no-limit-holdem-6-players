"""One-decision HU20 playback; only immutable seat observations reach policies."""

from __future__ import annotations

from dataclasses import asdict
import json
from math import fsum, isfinite
from pathlib import Path
from random import Random
import secrets

from src.arena.runner import public_events
from src.arena.schedule import digest
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, information_key
from src.blueprint.solver import HU20_UNCAPPED_GAME
from src.game.hand import Hand, Table
from src.game.observation import Observation
from src.play_api.service import PlayError, PlayService, _events, _json, _rng

PROTOCOL = 'hu20-spectator-v1'
PLAYERS = ('bot-a', 'bot-b')


def observation_record(view: Observation) -> dict:
    record = asdict(view)
    record['history'] = [{'event': type(event).__name__, **asdict(event)} for event in view.history]
    return json.loads(_json(record))


def replay_hand(row: dict) -> Hand:
    hand = Hand.start(Table(PLAYERS, (2000, 2000), button=row['button']),
                      hand_id=row['handId'], seed=row['dealSeed'])
    from src.game.types import Action, ActionKind
    for record in row['decisions']:
        if hand.actor != record['seat']:
            raise RuntimeError('Spectator journal actor mismatch')
        selected = record['menu'][record['selectedIndex']]
        hand = hand.apply(Action(ActionKind(selected['kind']), selected['raiseTo']))
    return hand


class SpectatorService(PlayService):
    def __init__(self, db_path: Path, policies: dict, identities: dict, *, source_version='unknown'):
        if not policies or set(policies) != set(identities):
            raise ValueError('Spectator models need matching identities')
        self.policies = policies
        self.identities = json.loads(_json(identities))
        for version, policy in policies.items():
            identity = identities[version]
            if (identity['version'] != version or identity['sha256'] != policy.spec.sha256
                    or identity['game'] != policy.game or identity['schema'] != policy.abstraction
                    or policy.game != HU20_UNCAPPED_GAME or policy.abstraction != HU20_UNCAPPED_SCHEMA
                    or getattr(policy, 'benchmark_only', False)):
                raise ValueError('Spectator model identity differs')
        if len({(p.game, p.abstraction) for p in policies.values()}) != 1:
            raise ValueError('Spectator models must share the HU20 game and schema')
        super().__init__(db_path, next(iter(policies.values())), source_version=source_version)

    def _models_match(self, models):
        return (isinstance(models, list) and len(models) == 2
                and all(isinstance(item, dict) and item == self.identities.get(item.get('version'))
                        for item in models))

    def _ack_matches(self, response):
        return self._models_match(response.get('models'))

    def _load(self, session_id):
        row = self.db.execute('SELECT state FROM sessions WHERE id=?', (session_id,)).fetchone()
        if row is None:
            raise PlayError('Unknown session', 404)
        state = json.loads(row[0])
        if state.get('protocol') != PROTOCOL or not self._models_match(state.get('models')):
            raise PlayError('Session models differ from the loaded policies', 409)
        return state

    def create(self, key, body):
        versions = body.get('modelVersions')
        if (set(body) != {'sessionType', 'modelVersions'} or body['sessionType'] != 'spectator'
                or not isinstance(versions, list) or len(versions) != 2
                or any(not isinstance(v, str) or v not in self.policies for v in versions)):
            raise PlayError('Choose two available pinned spectator models')

        def operation(_):
            state = {'sessionId': secrets.token_urlsafe(18), 'sessionType': 'spectator',
                     'protocol': PROTOCOL, 'models': [self.identities[v] for v in versions],
                     'revision': 0, 'handsPlayed': 0, 'totalChips': [0, 0],
                     'dealRng': Random(secrets.randbits(256)).getstate(),
                     'botRngs': [Random(secrets.randbits(256)).getstate() for _ in range(2)],
                     'current': None, 'history': [], 'sourceVersion': self.source_version}
            return state, self._view(state)

        return self._mutate(key, _json(['spectator-create', body]), None, operation)

    def new_hand(self, session_id, key, body):
        def operation(state):
            self._check_revision(state, body, {'revision'}, hand=False)
            if state['current'] is not None and not state['current'].get('completed'):
                raise PlayError('Finish the current hand first', 409)
            deals = _rng(state['dealRng'])
            state['current'] = {'handId': secrets.token_urlsafe(18),
                                'number': state['handsPlayed'], 'button': state['handsPlayed'] % 2,
                                'dealSeed': deals.randrange(2**63), 'decisions': [],
                                'samplingStart': state['botRngs'], 'models': state['models']}
            state['dealRng'] = deals.getstate()
            state['revision'] += 1
            return state, self._view(state)

        return self._mutate(key, _json(['spectator-hand', session_id, body]), session_id, operation)

    def advance(self, session_id, key, body):
        def operation(state):
            self._check_revision(state, body, {'handId', 'revision'})
            row = state['current']
            hand = replay_hand(row)
            if hand.finished:
                raise PlayError('The hand is finished', 409)
            seat = hand.actor
            # All inference and lookup classification uses this seat's view. The
            # sibling view and private replay data are never arguments to a policy.
            view = hand.observe(seat)
            model = state['models'][seat]
            policy = self.policies[model['version']]
            menu, probabilities, trained = policy.distribution(view)
            if (not menu or len(menu) != len(probabilities)
                    or any(not isfinite(p) or p < 0 for p in probabilities)
                    or abs(fsum(probabilities) - 1) > 1e-8):
                raise ValueError('Invalid spectator action distribution')
            for item in menu:
                view.legal_actions.validate(item.action)
            info_key = information_key(view, menu, schema=policy.abstraction)
            status = ('missing' if not trained else
                      'zero-mass' if info_key in getattr(policy, 'zero_mass', ()) else 'trained')
            generator = _rng(state['botRngs'][seat])
            selected = generator.choices(range(len(menu)), weights=probabilities, k=1)[0]
            row['decisions'].append({
                'number': len(row['decisions']), 'seat': seat, 'model': model,
                'observation': observation_record(view), 'informationKey': info_key, 'lookup': status,
                'menu': [{'label': item.name, 'kind': item.action.kind.value,
                          'raiseTo': item.action.raise_to, 'probability': probability}
                         for item, probability in zip(menu, probabilities)],
                'selectedIndex': selected,
            })
            state['botRngs'][seat] = generator.getstate()
            result = hand.apply(menu[selected].action)
            if result.finished:
                net = [player.stack - 2000 for player in result.observe(0).players]
                if sum(net) != 0:
                    raise RuntimeError('Spectator settlement did not conserve chips')
                row.update(completed=True, netChips=net,
                           publicEventsSha256=digest(public_events(result.events)),
                           sourceVersion=state['sourceVersion'])
                state['history'].append(row)
                state['handsPlayed'] += 1
                state['totalChips'] = [total + change for total, change in zip(state['totalChips'], net)]
            state['revision'] += 1
            return state, self._view(state)

        return self._mutate(key, _json(['spectator-advance', session_id, body]), session_id, operation)

    def _view(self, state):
        response = {key: state[key] for key in ('sessionId', 'sessionType', 'protocol', 'models',
                                              'revision', 'handsPlayed', 'sourceVersion')}
        response.update(phase='ready', hand=None, sessionChips=list(state['totalChips']))
        if state['current'] is None:
            return response
        row = state['current']
        hand = replay_hand(row)
        views = [hand.observe(seat) for seat in range(2)]
        response['phase'] = 'finished' if hand.finished else 'playing'
        response['hand'] = {
            'id': row['handId'], 'number': row['number'], 'button': row['button'],
            'actor': hand.actor, 'street': views[0].street.value, 'board': list(views[0].board),
            'pot': sum(p.amount for p in hand.events[-1].pots) if hand.finished else views[0].pot,
            'perspectives': [observation_record(view) for view in views],
            'events': _events(hand), 'decisions': row['decisions'],
            'result': {'netChips': row['netChips']} if hand.finished else None,
        }
        return response

    def history(self, session_id):
        with self.lock:
            state = self._load(session_id)
            hands = []
            for row in state['history']:
                hand = replay_hand(row)
                hands.append({'handId': row['handId'], 'number': row['number'], 'button': row['button'],
                              'models': row['models'], 'netChips': row['netChips'],
                              'events': _events(hand), 'decisions': row['decisions'],
                              'publicEventsSha256': row['publicEventsSha256']})
            return {'sessionId': session_id, 'models': state['models'], 'hands': hands}

    def verify_replay(self, session_id):
        from src.play_api.spectator_audit import audit_states
        with self.lock:
            result = audit_states([self._load(session_id)], self.policies, self.identities)
            return result['hands']

    def act(self, *_):
        raise PlayError('Spectator sessions accept only bot decisions', 409)

    def end_benchmark(self, *_):
        raise PlayError('This is not a human benchmark', 409)

    def benchmark_result(self, *_):
        raise PlayError('This is not a human benchmark', 409)

    def diagnostics(self, *_):
        raise PlayError('Spectator diagnostics are retained with each decision', 409)
