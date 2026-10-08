"""Read-only diagnosis of PR166's frozen paired selective-stackoff hands.

Hidden simulator cards may be inspected here; this module is never a policy input.
Restoration verifies the whole ZIP before reading selected member payloads.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import zipfile
from time import sleep

ZIP_SHA256 = 'cac0d2a766a8d0276820662112b68bb12ffe3688b8597faa0f63d43a829b3a0e'
ZIP_BYTES = 5645638363
SEEDS = (2026093001, 2026093002, 2026093003)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        offset=0
        while True:
            for attempt in range(4):
                try:
                    f.seek(offset)
                    chunk=f.read(1024**2)
                    break
                except TimeoutError:
                    # Drive hydration can time out; resume the same unhashed bytes.
                    if attempt==3: raise
                    print(f'Hydration read timeout at byte {offset}; retry {attempt+1}/3',flush=True)
                    sleep(2)
            if not chunk: break
            h.update(chunk)
            offset+=len(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def restore(archive, out):
    before=archive.stat()
    actual=file_hash(archive)
    if before.st_size != ZIP_BYTES or actual != ZIP_SHA256:
        raise ValueError(f'Whole PR166 ZIP differs: bytes={before.st_size}; SHA256={actual}')
    after=archive.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('Archive changed during verification')
    if out.exists():
        raise FileExistsError('Use a fresh restoration directory')
    if 'CloudStorage' in str(out.resolve()) or 'Research-Cloud' in str(out.resolve()):
        raise ValueError('Restore only outside synced storage')
    out.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        raw = z.read('RESEARCH_MEMBER_HASHES.json')
        manifest = json.loads(raw)
        selected = [n for n in manifest if n.startswith('audit-attempt-2/')
                    and n.endswith('.hands.jsonl.gz')]
        selected += ['plan.json']
        receipts = {}
        for name in sorted(selected):
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256(); size = 0
            with z.open(name) as source, target.open('xb') as dest:
                for chunk in iter(lambda: source.read(1024**2), b''):
                    dest.write(chunk); h.update(chunk); size += len(chunk)
            spec = {'bytes': size, 'sha256': h.hexdigest()}
            if spec != manifest[name]:
                raise ValueError('Member hash differs: ' + name)
            receipts[name] = spec
        (out / 'RESEARCH_MEMBER_HASHES.json').write_bytes(raw)
    write(out / 'restoration.json', {'archive': str(archive), 'archive_sha256': ZIP_SHA256,
          'archive_bytes': ZIP_BYTES, 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
          'archive_stat':{'inode':after.st_ino,'mtime_ns':after.st_mtime_ns},
          'selected_members': receipts, 'whole_zip_verified': True})


def token(action):
    return action['seat'], action['kind'], action['raise_to']


def first_difference(base, search, *, allow_opponent=False):
    """Only an identical prefix defines a paired common decision opportunity."""
    for b, s in zip(base['actions'], search['actions'], strict=False):
        if token(b) != token(s):
            if not allow_opponent and (b['logical_player'] != 0 or s['logical_player'] != 0):
                raise ValueError('Opponent differs before target: RNG/history pairing broken')
            if b['observation']['public_context'] != s['observation']['public_context']:
                raise ValueError('First divergence has different public state')
            return b, s
    if len(base['actions']) != len(search['actions']):
        raise ValueError('Unequal action lengths without divergent action')
    if base['target_chips'] != search['target_chips']:
        raise ValueError('Identical actions with different settlement')
    return None


def interval(values, panels=1):
    from scipy.stats import t
    n = len(values); center = mean(values)
    se = stdev(values) / n**.5
    half = float(t.ppf(1 - .05 / (2 * panels), n - 1)) * se
    return {'n': n, 'mean': center, 'se': se, 'interval': [center-half, center+half],
            't': center / se if se else None,
            'two_sided_p': float(2*t.sf(abs(center/se), n-1)) if se else None}


def bin_price(amount):
    return '0' if amount == 0 else '1-200' if amount <= 200 else '201-799' if amount < 800 else '800+'


def bin_pot(amount):
    return '<800' if amount < 800 else '800-1999' if amount < 2000 else '2000+'


def summarize(rows, blocks=256, *, allow_opponent=False):
    pairs = defaultdict(dict)
    for row in rows:
        key = row['seed'], row['block'], row['rotation']
        if row['arm'] in pairs[key]:
            raise ValueError('Duplicate hand coordinate')
        pairs[key][row['arm']] = row
    expected = {(seed, block, rotation) for seed in SEEDS for block in range(blocks) for rotation in (0,1)}
    if set(pairs) != expected or any(set(v) != {'base', 'search'} for v in pairs.values()):
        raise ValueError('Incomplete frozen selective-stackoff coverage')
    partitions = {k: defaultdict(lambda: [0, 0]) for k in
                  ('street','action','price','pot','mechanism','change','response','choice','raise_price')}
    deltas = {}; contrasts = []; common = Counter(); common_prob = Counter(); all_spots = Counter()
    opponent = Counter()
    for (seed, block, rotation), arms in sorted(pairs.items()):
        b, s = arms['base'], arms['search']
        if b['deal_seed'] != s['deal_seed'] or b['button'] != s['button']:
            raise ValueError('Different paired deals')
        diff = first_difference(b, s,allow_opponent=allow_opponent)
        delta = (s['target_chips'] - b['target_chips']) / 100
        deltas[seed, block, rotation] = delta
        # The full common prefix includes its first differing target decision.
        for ba, sa in zip(b['actions'], s['actions'], strict=False):
            if ba['logical_player'] == 0 and ba['street'] in ('turn', 'river'):
                obs = ba['observation']; price = bin_price(obs['call_amount'])
                if obs['call_amount']:
                    for arm, a in (('base', ba), ('search', sa)):
                        common[arm, price, a['kind']] += 1
                        for choice, p in zip(a['observation']['menu'], a['observation']['probabilities'], strict=True):
                            common_prob[arm, price, choice['kind']] += p
                        common[arm, price, 'opportunities'] += 1
            if token(ba) != token(sa): break
        for arm, row in arms.items():
            for a in row['actions']:
                if a['street'] in ('turn','river'):
                    if a['logical_player'] == 0 and a['observation']['call_amount']:
                        all_spots[arm, bin_price(a['observation']['call_amount']), a['kind']] += 1
                    elif a['logical_player'] == 1 and a['kind'] == 'raise':
                        opponent[arm, a['street'], a['observation']['selection_tier']] += 1
        if diff:
            ba, sa = diff; obs = sa['observation']
            if sa['logical_player']==1:
                # LBR queries the target's counterfactual future strategy. Its own
                # action can change before a live target action changes.
                labels=dict.fromkeys(partitions,'rival-first')
                labels['street']=sa['street'];labels['action']='rival-'+sa['kind']
                for dimension,label in labels.items():
                    partitions[dimension][label][0]+=1;partitions[dimension][label][1]+=delta
                contrasts.append({'seed':seed,'block':block,'rotation':rotation,'delta_bb':delta,
                                  'labels':labels,'logical_player':1})
                continue
            menu_choice = next(c for c in obs['menu'] if (c['kind'],c['raise_to']) == (sa['kind'],sa['raise_to']))
            prior_raise = any(a['logical_player'] == 1 and a['kind'] == 'raise' and a['street'] == sa['street']
                              for a in s['actions'][:sa['index']])
            mechanism = ('call-large-base-fold' if sa['kind']=='call' and ba['kind']=='fold' and obs['call_amount']>=800
                         else 'raise-after-rival-raise' if sa['kind']=='raise' and prior_raise
                         else 'raise-no-rival-raise' if sa['kind']=='raise'
                         else 'call-small-base-fold' if sa['kind']=='call' and ba['kind']=='fold'
                         else 'other')
            change=(f'{ba["kind"]}-to-{sa["kind"]}' if ba['kind']!=sa['kind'] else
                    'larger-raise' if sa['raise_to']>ba['raise_to'] else 'smaller-raise')
            response='not-a-raise'
            if sa['kind']=='raise':
                subsequent=s['actions'][sa['index']+1:sa['index']+2]
                if not subsequent or subsequent[0]['logical_player']!=1: raise ValueError('Raise has no immediate rival response')
                rival=subsequent[0]
                response=rival['observation']['selection_tier']+'-'+rival['kind']
            choice='jam' if menu_choice.get('jam') else menu_choice.get('name',sa['kind'])
            labels = {'street':sa['street'], 'action':sa['kind'], 'price':bin_price(obs['call_amount']),
                      'pot':bin_pot(obs['pot']), 'mechanism':mechanism,'change':change,'response':response,
                      'choice':choice,'raise_price':bin_price(menu_choice.get('rival_call_amount',0)) if sa['kind']=='raise' else 'not-a-raise'}
            contrasts.append({'seed':seed,'block':block,'rotation':rotation,'delta_bb':delta,
                              'base_chips':b['target_chips'],'search_chips':s['target_chips'],
                              'base_action':token(ba),'search_action':token(sa),'decision_index':sa['index'],
                              'observation':obs,'labels':labels, 'selected_menu_choice':menu_choice})
        else: labels = dict.fromkeys(partitions, 'identical')
        for dimension, label in labels.items():
            partitions[dimension][label][0] += 1; partitions[dimension][label][1] += delta
    block_series = [mean(deltas[seed,block,rotation] for seed in SEEDS for rotation in (0,1))*100
                    for block in range(blocks)]
    lineage = {str(seed):interval([mean(deltas[seed,block,r] for r in (0,1))*100 for block in range(blocks)])
               for seed in SEEDS}
    partition_rows = {dimension:[{'label':label,'hands':n,'delta_bb':chips,'contribution_bb_per_100':chips/len(expected)*100}
                                for label,(n,chips) in sorted(table.items())] for dimension,table in partitions.items()}
    if any(abs(sum(r['delta_bb'] for r in table)-sum(deltas.values()))>1e-8 for table in partition_rows.values()):
        raise ValueError('Disjoint contributions do not reconcile')
    def counters(table): return [{'key':list(k),'value':v} for k,v in sorted(table.items())]
    ordinary=interval(block_series)
    result = {'hands_per_arm':len(expected),'paired_hand_coordinates':len(pairs),'aggregate':ordinary,
              'bonferroni_13':interval(block_series,13),'lineages':lineage,'partitions':partition_rows,
              'common_prefix_actual':counters(common),'common_prefix_expected':counters(common_prob),
              'unmatched_actual':counters(all_spots),'opponent_raises':counters(opponent),
              'changed_hands':len(contrasts),
              'joint_blocks':[{'block':i,'contrast_bb_per_100':v,'contribution_bb_per_100':v/blocks}
                              for i,v in enumerate(block_series)],
              'bonferroni_adjusted_p':min(1,13*ordinary['two_sided_p']) if ordinary['two_sided_p'] is not None else None}
    return result, sorted(contrasts, key=lambda r:(r['delta_bb'],r['seed'],r['block'],r['rotation']))


def analyze(inputs, out):
    rows = []
    receipt = json.loads((inputs/'restoration.json').read_text())
    for name, spec in receipt['selected_members'].items():
        path = inputs/name
        if path.stat().st_size != spec['bytes'] or file_hash(path) != spec['sha256']:
            raise ValueError('Restored input changed')
        if name.endswith('.hands.jsonl.gz'):
            with gzip.open(path,'rt') as f:
                for line in f:
                    row = json.loads(line)
                    if row['panel']=='selective-stackoff': rows.append(row)
    # Replay recorded actions only; this never chooses an action or generates new play.
    from src.game.hand import Hand, Table
    from src.game.types import Action, ActionKind
    from src.diagnostics.stackoff_tails import public_context
    from src.arena.runner import public_events
    from src.arena.schedule import digest
    decisions=0
    for row in rows:
        hand=Hand.start(Table(('seat0','seat1'),(2000,2000),button=row['button']),
                        hand_id=row['hand_id'],seed=row['deal_seed'])
        for a in row['actions']:
            view=hand.observe(hand.actor)
            if (hand.actor!=a['seat'] or public_context(view)!=a['observation']['public_context']
                    or list(view.hole_cards)!=a['observation']['hole_cards']):
                raise ValueError('Recorded snapshot does not replay')
            hand=hand.apply(Action(ActionKind(a['kind']),a['raise_to'])); decisions+=1
        chips=[p.stack-2000 for p in hand.observe(0).players]
        if (not hand.finished or chips!=row['net_chips_by_seat'] or sum(chips)
                or chips[row['rotation']]!=row['target_chips']
                or digest(public_events(hand.events))!=row['public_events_sha256']):
            raise ValueError('Recorded settlement/events do not replay')
    summary, contrasts = summarize(rows)
    summary['native_replay']={'hands':len(rows),'decisions':decisions,'verified':True}
    write(out/'summary.json',summary); write(out/'contrasts.json',contrasts)
    write(out/'selective-hands.json',rows)
    print(json.dumps(summary['aggregate']))


def gain_scope(inputs, out):
    """Describe the same first-divergence strata in the two frozen gain panels.

    This is exposure arithmetic, not counterfactual fallback-policy performance.
    Slim records keep exact public-context digests and discard bulky telemetry.
    """
    receipt=json.loads((inputs/'restoration.json').read_text())
    for panel in ('lbr','native-pressure'):
        rows=[]
        for name,spec in receipt['selected_members'].items():
            if not name.endswith('.hands.jsonl.gz'): continue
            path=inputs/name
            if path.stat().st_size!=spec['bytes'] or file_hash(path)!=spec['sha256']:
                raise ValueError('Restored gain-panel input changed')
            with gzip.open(path,'rt') as f:
                for line in f:
                    row=json.loads(line)
                    if row['panel']!=panel: continue
                    slim={k:row[k] for k in ('seed','block','rotation','arm','deal_seed','button','target_chips')}
                    slim['actions']=[]
                    for a in row['actions']:
                        item={k:a[k] for k in ('seat','logical_player','kind','raise_to','street','index')}
                        obs=a['observation']; item['observation']={k:obs[k] for k in
                            ('public_context_id','call_amount','pot','menu','probabilities','selection_tier')}
                        item['observation']['public_context']=obs['public_context_id']
                        slim['actions'].append(item)
                    rows.append(slim)
        summary,_=summarize(rows,2048,allow_opponent=panel=='lbr')
        write(out/(panel+'-scope.json'),summary)
        print(panel,json.dumps(summary['aggregate']),flush=True)


def compact_results(root: Path, out: Path) -> None:
    """Keep report summaries in Git; complete blocks/holdings stay in the archive."""
    summary=json.loads((root/'analysis/summary.json').read_text())
    blocks=summary.pop('joint_blocks')
    nonzero=[b for b in blocks if b['contrast_bb_per_100']]
    summary['joint_block_concentration']={
        'nonzero':len(nonzero),
        'negative':sum(b['contrast_bb_per_100']<0 for b in blocks),
        'positive':sum(b['contrast_bb_per_100']>0 for b in blocks),
        'worst_five':sorted(blocks,key=lambda b:b['contrast_bb_per_100'])[:5]}
    write(out/'summary.json',summary)
    offline=json.loads((root/'offline/resolution-summary.json').read_text())
    offline.pop('records')
    for example in offline['examples']:
        for label in ('search','frozen_rules','search_at_round_root','floor_001_at_round_root_fixed_turn'):
            example[label].pop('top_holdings')
    write(out/'offline-examples.json',offline)
    gains={}
    for panel in ('lbr','native-pressure'):
        value=json.loads((root/'analysis'/(panel+'-scope.json')).read_text())
        gains[panel]={k:value[k] for k in ('aggregate','lineages','partitions')}
    write(out/'gain-exposure.json',gains)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('restore','analyze','gain-scope','compact')); p.add_argument('--archive',type=Path)
    p.add_argument('--inputs',type=Path); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='restore': restore(a.archive,a.out)
    elif a.command=='analyze': analyze(a.inputs,a.out)
    elif a.command=='gain-scope': gain_scope(a.inputs,a.out)
    else: compact_results(a.inputs,a.out)


if __name__=='__main__': main()
