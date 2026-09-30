"""Enumerate own holdings in fixed public contexts; no hidden-card conditioning."""
import argparse
import gc
import gzip
import json
from collections import defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from scripts.evaluate_hu20 import write_json
from scripts.evaluate_hu20_stackoff import Guard
from src.blueprint.abstraction import information_key
from src.diagnostics.saved_hu20 import load_saved, file_hash
from src.diagnostics.stackoff_tails import public_context, snapshot
from src.game.hand import Hand, Table
from src.blueprint.search import DECK
from src.game.types import Action, ActionKind
from src.arena.schedule import digest


def context_view(context):
    origin = context['origin']
    ids = tuple(f"player-{(s - origin['rotation']) % 2}" for s in range(2))
    hand = Hand.start(Table(ids, (2000, 2000), button=origin['button']),
                      hand_id=f"robustness-{origin['phase']}-2-{origin['block']}", seed=origin['deal_seed'])
    for item in context['prefix']:
        if hand.actor != item['seat']:
            raise ValueError('Context prefix actor differs')
        hand = hand.apply(Action(ActionKind(item['kind']), item['raise_to']))
    view = hand.observe(hand.actor)
    if public_context(view) != context['public_context'] or digest(public_context(view)) != context['id']:
        raise ValueError('Context public replay differs')
    return view


def holdings(view):
    excluded = set(view.board)
    # Shown cards, if any, are public. Unrevealed opponent cards are never read.
    for player in view.players:
        excluded.update(player.shown_cards)
    return combinations([c for c in DECK if c not in excluded], 2)


def queries(source, visits, context):
    view = context_view(context)
    for pair in holdings(view):
        candidate = replace(view, hole_cards=tuple(pair))
        menu, probabilities, trained = source.distribution(candidate)
        key = information_key(candidate, menu, schema=source.abstraction)
        yield {'context_id': context['id'], 'key': key,
               **snapshot(candidate, menu, probabilities, trained, visits.get(key, 0))}



def unique_contexts(document):
    result = {}
    for context in document['contexts']:
        if digest(context['public_context']) != context['id']:
            raise ValueError('Selected context digest differs')
        result.setdefault(context['id'], context)
    return list(result.values())

def inspect(plan, inputs, run):
    document = json.loads((run / 'contexts.json').read_text())
    contexts = unique_contexts(document)
    guard = Guard(plan, run)
    summaries = []
    for spec in plan['models']:
        guard()
        source, visits = load_saved(spec, inputs, guard)
        path = run / (spec['name'] + '.inspection.jsonl.gz')
        if path.exists():
            raise FileExistsError(path)
        groups = defaultdict(list)
        with gzip.open(path, 'wt') as handle:
            for context in contexts:
                for row in queries(source, visits, context):
                    row['model'] = spec['name']
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                    groups[(row['context_id'], row['concrete_category'], str(row['card_bucket']))].append(row)
                guard()
        for (context_id, category, bucket), rows in sorted(groups.items()):
            unique = {r['key']: r for r in rows}
            high = {k: r for k, r in unique.items() if r['trained'] and r['visits'] >= plan['minimum_visits']}
            summaries.append({'model': spec['name'], 'context_id': context_id, 'concrete_category': category,
                              'bucket': bucket, 'compatible_holdings': len(rows), 'unique_keys': len(unique),
                              'trained_keys': sum(r['trained'] for r in unique.values()), 'high_visit_keys': len(high),
                              'high_visit_nodes': [{'key': k, 'visits': r['visits'],
                                   'large_raise_probability': r['large_raise_probability'],
                                   'jam_probability': r['jam_probability'], 'menu': r['menu'],
                                   'menu_size': len(r['menu']), 'example_hole_cards': r['hole_cards']}
                                   for k, r in sorted(high.items())]})
        del source, visits, groups
        gc.collect()
    result = {'contexts_sha256': file_hash(run / 'contexts.json'), 'contexts': document,
              'unique_public_contexts': len(contexts), 'rows': summaries, 'interpretation': 'Shared keys are not independent samples; similar bucket aggression alone does not prove an error.'}
    write_json(run / 'inspection-summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args()
    inspect(json.loads((args.run / 'plan.json').read_text()), args.inputs, args.run)


if __name__ == '__main__':
    main()
