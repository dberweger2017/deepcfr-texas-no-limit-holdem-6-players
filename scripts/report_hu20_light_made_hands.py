"""M1-only post-hoc trends from closed, hash-verified #136 records."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import resource
import subprocess
import time

from src.arena.schedule import digest, stream_seed
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.stackoff_made_hands import first_large_raise, summarize_events
from src.diagnostics.stackoff_tails import hand_tails


PANEL = 'selective_stackoff'
SITUATIONS = ('after_rival_raise', 'no_rival_raise')


def verified_inputs(root):
    manifest = json.loads((root/'transfer-manifest.json').read_text())
    remote = Path(manifest['remote_plan']).parents[2]
    plan_path = root/Path(manifest['remote_plan']).relative_to(remote)
    if file_hash(plan_path) != manifest['plan_file_sha256']:
        raise ValueError('Campaign plan bytes differ')
    plan = json.loads(plan_path.read_text())
    if digest(plan) != manifest['canonical_plan_sha256']:
        raise ValueError('Canonical campaign plan differs')
    for entry in manifest['inputs']:
        folder = root/Path(entry['remote_dir']).relative_to(remote)
        for name, field in (('hands.jsonl.gz', 'hands_sha256'), ('result.json', 'result_sha256')):
            if file_hash(folder/name) != entry[field]:
                raise ValueError('Transferred file hash differs')
        if (folder/'hands.jsonl.gz').stat().st_size != entry['hands_bytes']:
            raise ValueError('Transferred hand-file size differs')
        result = json.loads((folder/'result.json').read_text())
        if result['status'] != 'complete' or result['stage'] != 'light' or any(result['spec'].get(k) != v for k,v in entry['model'].items()):
            raise ValueError('Closed task identity/status differs')
    return manifest, plan, remote


def analyze(root):
    manifest, plan, remote = verified_inputs(root)
    panels = {p['name']:p for p in plan['light_panels']}
    seeds = {p['seed'] for p in plan['parents']}
    expected_tasks = {(s,m) for s in seeds for m in plan['light_totals']}
    observed_tasks, per_seed, events, models = set(), [], [], []
    coupling = {}
    for entry in manifest['inputs']:
        folder = root/Path(entry['remote_dir']).relative_to(remote)
        # The final transfer entry is a hash-pinned identity subset; retain the
        # complete spec from its independently hash-verified closed result.
        spec = json.loads((folder/'result.json').read_text())['spec']
        coordinate = spec['seed'], spec['milestone']
        if coordinate not in expected_tasks or coordinate in observed_tasks:
            raise ValueError('Unexpected/duplicate light task')
        observed_tasks.add(coordinate)
        if spec['players'] != 2 or spec['format'] != 'holdem-hu20-native-reopening-blueprint-v1':
            raise ValueError('Unsupported model identity')
        models.append(spec)
        folder = root/Path(entry['remote_dir']).relative_to(remote)
        seen, counts, chips, selected = set(), Counter(), 0, []
        with gzip.open(folder/'hands.jsonl.gz', 'rt') as handle:
            for line in handle:
                row = json.loads(line)
                key = row['panel'], row['block'], row['rotation']
                if key in seen or key[0] not in panels or key[2] not in (0,1) or not 0<=key[1]<panels[key[0]]['blocks']:
                    raise ValueError('Unexpected/duplicate hand coordinate')
                seen.add(key)
                if (row['policy'] != spec['name'] or row['seed'] != spec['seed'] or
                    row['milestone'] != spec['milestone'] or row['campaign_stage'] != 'light' or
                    row['status'] != 'complete' or not row.get('native_replay_verified')):
                    raise ValueError('Hand model/status/replay evidence differs')
                panel = panels[key[0]]
                if (row['root_seed'] != panel['root'] or row['deal_seed'] != stream_seed(panel['root'],'test','deal',2,key[1]) or
                    row['button'] != key[1]%2 or sum(row['net_chips_by_seat']) != 0 or
                    row['target_chips'] != row['net_chips_by_seat'][key[2]]):
                    raise ValueError('Deal or chip accounting differs')
                if coupling.setdefault(key, (row['deal_seed'],row['button'])) != (row['deal_seed'],row['button']):
                    raise ValueError('Shared checkpoint deals differ')
                if key[0] != PANEL:
                    continue
                tails = hand_tails(row)
                if tails != row['tails']:
                    raise ValueError('Tail arithmetic differs')
                counts.update(tails['counts']);chips += row['target_chips']
                event = first_large_raise(row)
                if event is not None:
                    event.update(seed=spec['seed'],milestone=spec['milestone'],task=entry['task'])
                    selected.append(event);events.append(event)
        expected = {(name,b,r) for name,p in panels.items() for b in range(p['blocks']) for r in (0,1)}
        if seen != expected or len(seen) != json.loads((folder/'result.json').read_text())['hands']:
            raise ValueError('Closed task has incomplete paired hands')
        situations = []
        derived = summarize_events(selected, {'models':[spec]})['groups']
        for situation in SITUATIONS:
            c = next((r['counts'] for r in derived if r['model']=='aggregate' and r['situation']==situation),{})
            subset = [e for e in selected if e['situation']==situation]
            c.update(selected_full_stack_wins=sum(e['target_chips']==2000 for e in subset),
                     selected_full_stack_losses=sum(e['target_chips']==-2000 for e in subset))
            situations.append({'situation':situation,'counts':c})
        per_seed.append({'seed':spec['seed'],'milestone':spec['milestone'],'policy':spec['name'],
                         'counts':dict(counts),'whole_hand_chips':chips,'situations':situations})
    aggregates = []
    for milestone in plan['light_totals']:
        rows = [r for r in per_seed if r['milestone']==milestone]
        if {r['seed'] for r in rows} != seeds:
            continue
        counts = Counter()
        for row in rows:counts.update(row['counts'])
        situations = []
        for situation in SITUATIONS:
            c = Counter()
            for row in rows:c.update(next(s['counts'] for s in row['situations'] if s['situation']==situation))
            situations.append({'situation':situation,'counts':dict(c)})
        aggregates.append({'milestone':milestone,'counts':dict(counts),
                           'whole_hand_chips':sum(r['whole_hand_chips'] for r in rows),'situations':situations})
    return {'version':'hu20-light-made-hand-trends-v1','post_hoc':True,
            'scientific_source':manifest['scientific_source'],'canonical_plan_sha256':digest(plan),
            'transfer_manifest_sha256':file_hash(root/'transfer-manifest.json'),
            'completed_tasks':len(observed_tasks),'pending_tasks':[list(k) for k in sorted(expected_tasks-observed_tasks)],
            'selective_hands':sum(r['counts']['hands'] for r in per_seed),
            'selected_events':len(events),'per_seed':per_seed,'three_lineage':aggregates,
            'warning':'Current made hands are not equity; whole-hand profit is not individual-bet EV. Selected shared-deal counts are descriptive.'},events


def markdown(result):
    lines=['# HU20 light-panel made-hand trends','',
           f"{result['completed_tasks']} closed tasks; {result['selective_hands']:,} selective hands; {result['selected_events']} first-large-raise events. Pending tasks: {len(result['pending_tasks'])}.", '',
           '**Post-hoc/descriptive.** The fixed #136 opponent, hands, source and model identities remain unchanged. '
           'Both cards are joined offline only. Made-hand order uses the current board, not draws or future cards; it is not equity or bet EV. '
           'No new deals, model loads, or M4 analysis. Shared deals/lineages and selected events are correlated.', '',
           '## All selective hands: three-lineage totals','',
           '| Nodes | Hands | Whole-hand BB | +20BB / hands | −20BB / hands | First large raise hands |',
           '| ---: | ---: | ---: | --- | --- | ---: |']
    for row in result['three_lineage']:
        c=row['counts'];selected=sum(s['counts'].get('hands',0) for s in row['situations'])
        lines.append(f"| {row['milestone']/1e6:g}M | {c['hands']} | {row['whole_hand_chips']/100:+.2f} | {c['full_stack_wins']}/{c['hands']} | {c['full_stack_losses']}/{c['hands']} | {selected} |")
    lines += ['', '## First large raises: three-lineage counts','',
              '| Nodes | Situation | Hands | Continued | Ahead / behind / tied / preflop | One pair / postflop continuations | Whole-hand BB | Selected +20 / −20BB |',
              '| ---: | --- | ---: | ---: | --- | --- | ---: | --- |']
    for row in result['three_lineage']:
        for item in row['situations']:
            c=item['counts'];continued=c.get('continued',0);postflop=continued-c.get('continued_preflop',0)
            comparison=' / '.join(str(c.get('continued_'+k,0)) for k in ('ahead','behind','tied','preflop'))
            lines.append(f"| {row['milestone']/1e6:g}M | {item['situation']} | {c.get('hands',0)} | {continued} | {comparison} | {c.get('category_pair',0)}/{postflop} | {c.get('target_chips',0)/100:+.2f} | {c['selected_full_stack_wins']} / {c['selected_full_stack_losses']} |")
    lines += ['', '## Lineages: all-hand tails and one-pair re-raises','',
              '| Seed | Nodes | Hands | Whole-hand BB | +20 / −20BB | One pair / postflop continuations after rival raise |',
              '| --- | ---: | ---: | ---: | --- | --- |']
    for row in sorted(result['per_seed'],key=lambda r:(r['seed'],r['milestone'])):
        c=row['counts'];s=next(x['counts'] for x in row['situations'] if x['situation']=='after_rival_raise')
        postflop=s.get('continued',0)-s.get('continued_preflop',0)
        lines.append(f"| {row['seed']} | {row['milestone']/1e6:g}M | {c['hands']} | {row['whole_hand_chips']/100:+.2f} | {c['full_stack_wins']} / {c['full_stack_losses']} | {s.get('category_pair',0)}/{postflop} |")
    lines += ['','All categories, response counts, street denominators and each lineage’s two situations remain in summary.json. '
              'Exact selected cards/boards/actions/payoffs are in events.jsonl. Zero denominators mean no exposure, not safe play.', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result,events=analyze(a.inputs)
    result.update(reporting_source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (a.out/'events.jsonl').write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in events))
    (a.out/'dashboard.md').write_text(markdown(result))
    (a.out/'manifest.json').write_text(json.dumps({f.name:{'bytes':f.stat().st_size,'sha256':file_hash(f)} for f in a.out.iterdir()},indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('completed_tasks','selective_hands','selected_events','seconds','peak_rss_bytes')}))


if __name__=='__main__':main()
