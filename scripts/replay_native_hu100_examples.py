"""Illustrative archived loss paths and supported off-menu witnesses, not inference."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from scripts.diagnose_native_hu100 import PREFIX, rows, write
from src.arena.runner import public_events
from src.arena.schedule import Plan, build_schedule, canonical
from src.blueprint.abstraction import HU100_SCHEMA, choices, information_key
from src.game.hand import Hand, Table, card_name
from src.game.types import Action, ActionKind


def record(initial, actions, *, supported=False):
    hand = initial; decisions = []; chosen = []
    for action in actions:
        view = hand.observe(hand.actor); legal = view.legal_actions
        menu = choices(view, raise_cap=None, free_fold=False)
        if supported and action not in [c.action for c in menu]:
            raise ValueError('Witness contains an off-menu action')
        decisions.append({'actor': hand.actor, 'street': view.street.value, 'pot': view.pot,
            'kinds': [k.value for k in legal.kinds], 'call': legal.call_amount,
            'min_raise_to': legal.min_raise_to, 'max_raise_to': legal.max_raise_to,
            'menu': [[c.name, c.action.raise_to] for c in menu],
            'key': information_key(view, menu, schema=HU100_SCHEMA)})
        chosen.append([action.kind.value, action.raise_to]); hand = hand.apply(action)
    if not hand.finished: raise ValueError('Example fixture must be complete')
    first = 1 - initial.table.button; holes = initial._state.players_state
    deck = [card_name(holes[s].hand[r]) for r in (0, 1) for s in (first, initial.table.button)]
    deck += [card_name(c) for c in initial._state.deck]
    return {'stack_bb': 100, 'seed': 'archived-example', 'button': initial.table.button,
            'deck': deck, 'decisions': decisions, 'actions': chosen,
            'final_stacks': list(hand.events[-1].stacks)}, hand


def replay_examples(root):
    wanted = {}
    for f in rows(root / 'analysis/features.jsonl.gz'):
        if f['nodes'] != 11042440: continue
        if (f['opponent'] in ('tight_aggressive', 'loose_aggressive')
                and f['lookup'] == 'positive-mass-known-key' and f['net_bb'] < 0 and f['pot_bb'] >= 16):
            wanted.setdefault(f['opponent'], f)
        if f['opponent'] == 'pot_pressure' and f['lookup'] == 'missing-key' and f['net_bb'] < 0:
            wanted.setdefault('pot-pressure-missing', f)
        if f['opponent'] == 'pot_pressure' and f['support'] == 'supported':
            wanted.setdefault('off-menu-supported', f)
    representatives = json.loads((root / 'analysis/representatives.json').read_text())
    fixtures = []; examples = []
    for label, f in wanted.items():
        panel = root / 'inputs' / PREFIX / 'final/11042440' / f['opponent']
        plan = Plan.from_dict(json.loads((panel / 'manifest.json').read_text())['plan'])
        block = next(b for b in build_schedule(plan) if b.index == f['block'])
        archived = next(h for h in rows(panel / 'hands.jsonl') if
                        (h['arm'], h['block'], h['rotation']) == ('candidate', f['block'], f['rotation']))
        trace = [d for d in rows(panel / 'decisions.jsonl.gz') if
                 (d['arm'], d['block'], d['rotation']) == ('candidate', f['block'], f['rotation'])]
        ids = tuple(f'player-{(s - f["rotation"]) % 2}' for s in (0, 1))
        initial = Hand.start(Table(ids, (10000, 10000), block.button),
                             hand_id=trace[0]['hand_id'], seed=block.deal_seeds[0])
        actions = [Action(ActionKind(d['action']['kind']), d['action']['raise_to']) for d in trace]
        fixture, hand = record(initial, actions)
        if canonical(public_events(hand.events)) != canonical(archived['events']):
            raise ValueError('Representative archived replay differs')
        fixtures.append(fixture)
        actor = trace[f['decision']]['seat']
        example = {'label': label, 'feature': f,
                   'target_hole_cards': [card_name(c) for c in initial._state.players_state[actor].hand],
                   'path': [{'seat': d['seat'], 'street': d['street'], 'action': d['action'],
                             'lookup': d.get('lookup'), 'probabilities': d.get('probabilities')} for d in trace],
                   'events_replayed': True}
        if label == 'off-menu-supported':
            rep = next(r for r in representatives if r['feature']['key'] == f['key'] and r['proof'])
            witness = [Action(ActionKind(a['kind']), a['raise_to']) for a in rep['proof']['witness_actions']]
            state = initial
            for action in witness:
                if action not in [c.action for c in choices(state.observe(state.actor), raise_cap=None, free_fold=False)]:
                    raise ValueError('Off-menu witness action')
                state = state.apply(action)
            view = state.observe(state.actor); menu = choices(view, raise_cap=None, free_fold=False)
            if information_key(view, menu, schema=HU100_SCHEMA) != f['key']:
                raise ValueError('Alternate path does not reach the missing abstract key')
            example['witness_actions'] = [asdict(a) for a in witness]
            example['witness_target_pot_bb'] = view.pot / 100
            while not state.finished:
                quiet = next(c for c in choices(state.observe(state.actor), raise_cap=None, free_fold=False)
                             if c.name in ('check', 'call'))
                witness.append(quiet.action); state = state.apply(quiet.action)
            fixture, _ = record(initial, witness, supported=True); fixtures.append(fixture)
        examples.append(example)
    write(root / 'failure-examples.json', examples)
    with (root / 'example-native-fixtures.jsonl').open('x') as out:
        for fixture in fixtures: out.write(json.dumps(fixture) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    replay_examples(p.parse_args().root)
