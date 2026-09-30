"""Post-hoc simulator-only comparisons; never inputs to a playing policy."""
from collections import Counter, defaultdict

from src.diagnostics.selective_stackoff import LARGE_CALL
from src.diagnostics.stackoff_tails import CATEGORIES
from src.game.showdown import hand_value

VERSION = 'first-large-raise-made-hands-v1'


def first_large_raise(row):
    """Use the board at the raise, not the runout, and count one event per hand."""
    if row['status'] != 'complete':
        raise ValueError('Made-hand analysis requires complete generated hands')
    actions = row['actions']
    for index, action in enumerate(actions):
        if action['logical_player'] != 0 or action['kind'] != 'raise':
            continue
        observed = action['observation']
        selected = next(c for c in observed['menu'] if c['kind'] == 'raise'
                        and c['raise_to'] == action['raise_to'])
        if selected['rival_call_amount'] < LARGE_CALL:
            continue
        rival_raised = any(a['logical_player'] == 1 and a['kind'] == 'raise'
                          and a['street'] == action['street'] for a in actions[:index])
        response = 'no_response'
        rival = actions[index + 1] if index + 1 < len(actions) else None
        if rival and rival['logical_player'] == 1 and rival['street'] == action['street']:
            response = 'folded' if rival['kind'] == 'fold' else 'continued'
        board = tuple(observed['board'])
        target_cards = tuple(observed['hole_cards'])
        # Recorded cards are joined offline. No playing policy receives this comparison.
        rival_cards = next((tuple(a['observation']['hole_cards']) for a in actions
                            if a['logical_player'] == 1), None)
        if rival_cards is None:
            raise ValueError('Missing recorded rival cards')
        target_value = hand_value(target_cards + board) if board else None
        rival_value = hand_value(rival_cards + board) if board else None
        comparison = ('preflop' if not board else 'ahead' if target_value > rival_value
                      else 'behind' if target_value < rival_value else 'tied')
        return {**{k: row[k] for k in ('policy', 'block', 'rotation', 'button')},
                'index': action['index'], 'street': action['street'],
                'situation': 'after_rival_raise' if rival_raised else 'no_rival_raise',
                'response': response, 'board': list(board), 'target_cards': list(target_cards),
                'rival_cards': list(rival_cards), 'raise_to': action['raise_to'],
                'rival_call_amount': selected['rival_call_amount'], 'comparison': comparison,
                'target_category': CATEGORIES[target_value[0]] if board else 'preflop',
                'rival_category': CATEGORIES[rival_value[0]] if board else 'preflop',
                'target_chips': row['target_chips']}
    return None


def summarize_events(events, plan):
    models = {m['name']: m for m in plan['models']}
    groups = defaultdict(Counter)
    for event in events:
        model = models[event['policy']]
        for key in [('aggregate', model['milestone'], event['situation']),
                    (event['policy'], model['milestone'], event['situation'])]:
            counts = groups[key]
            counts['hands'] += 1
            counts['target_chips'] += event['target_chips']
            counts[event['response']] += 1
            if event['response'] == 'continued':
                counts['continued_' + event['comparison']] += 1
                counts['category_' + event['target_category']] += 1
                counts['street_' + event['street'] + '_continued'] += 1
    return {'analysis_version': VERSION, 'post_hoc': True,
            'warning': 'Made-hand order ignores draws; selected correlated hands and whole-hand returns are not equity or individual-bet EV.',
            'first_large_raise_hands': len(events),
            'groups': [{'model': k[0], 'milestone': k[1], 'situation': k[2], 'counts': dict(v)}
                       for k, v in sorted(groups.items())]}


def markdown(result):
    lines = ['## Made hands at the first large target raise (post-hoc)', '',
             'I split each hand by whether the rival had already raised on that street. '
             'Both cards are joined only in offline simulator analysis; each policy still uses its own observation. '
             'Ahead/behind/tied compares made hands on the board at the raise, not the final board or equity. '
             'Preflop comparisons are separate. Whole-hand returns are not individual-bet EV; draws and semi-bluffs are not valued. '
             'These selected, correlated events against one exploitable opponent do not establish the cause of Luna’s results.', '',
             '| Work | Situation | Hands | Folds / continued / no response | Ahead / behind / tied / preflop when continued | One pair / postflop continuations | Whole-hand target BB |',
             '| ---: | --- | ---: | --- | --- | --- | ---: |']
    for row in result['groups']:
        if row['model'] != 'aggregate':
            continue
        c = row['counts']
        response = ' / '.join(str(c.get(k, 0)) for k in ('folded', 'continued', 'no_response'))
        comparison = ' / '.join(str(c.get('continued_' + k, 0)) for k in ('ahead', 'behind', 'tied', 'preflop'))
        postflop = c.get('continued', 0) - c.get('continued_preflop', 0)
        lines.append(f"| {row['milestone']/1e6:g}M | {row['situation']} | {c['hands']} | {response} | {comparison} | "
                     f"{c.get('category_pair',0)}/{postflop} | {c['target_chips']/100:+.2f} |")
    lines += ['', 'Individual-lineage counts, street denominators and all made-hand categories are in '
              '`large-raise-made-hands.json`; exact selected events are in `large-raise-made-hands.rows.jsonl`. '
              'The v1 opponent and original campaign schedule remain frozen.', '']
    return '\n'.join(lines)
