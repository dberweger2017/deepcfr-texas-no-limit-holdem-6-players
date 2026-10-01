"""Read a fixed snapshot of closed light evaluations; never change gameplay."""

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean
import time

from src.arena.report import estimate
from src.arena.schedule import digest, stream_seed
from src.diagnostics.stackoff_tails import hand_tails
from scripts.evaluate_hu20_stackoff import swap_bytes
from scripts.hu20_platform_pilot import peak_rss


def aggregate(series, seeds, milestone, panel, role=None):
    """Missing lineage means pending; never silently average fewer lineages."""
    keys = [(s, milestone, panel) for s in seeds]
    if not all(k in series for k in keys):
        return None
    blocks = set(series[keys[0]])
    if any(set(series[k]) != blocks for k in keys):
        raise ValueError('Different completed block sets')
    return [mean(series[k][b][role] if role else mean(series[k][b].values())
                 for k in keys) for b in sorted(blocks)]


def cell(values):
    result = estimate(values)
    if values and abs(result['bb_per_100'] - sum(values)/len(values)) > 1e-9:
        raise ValueError('Independent sum disagrees with estimate')
    result['bb_per_hand'] = result['bb_per_100']/100 if values else None
    return result


def summary(root, plan, guard=lambda: None):
    panels = {p['name']: p for p in plan['light_panels']}
    seeds = [p['seed'] for p in plan['parents']]
    series, per_seed, inputs, pending = {}, [], [], []
    coupling = {}
    folders = []
    for seed in seeds:
        for milestone in plan['light_totals']:
            folder = root/'evaluation'/f'light-{seed}-{milestone}'
            path = folder/'result.json'
            if not path.exists():
                pending.append(dict(seed=seed, milestone=milestone, status='not_closed'))
                continue
            result = json.loads(path.read_text())
            if result['status'] != 'complete':
                pending.append(dict(seed=seed, milestone=milestone,
                                    status=result['status'], failure=result.get('failure')))
                continue
            folders.append((folder, result, seed, milestone))
    # Snapshot is fixed before opening these rows. Late completions enter a later report.
    for folder, result, seed, milestone in folders:
        guard()
        spec = result['spec']
        if spec['seed'] != seed or spec['milestone'] != milestone:
            raise ValueError('Task/model coordinate mismatch')
        raw = folder/'hands.jsonl.gz'
        hasher = hashlib.sha256()
        with raw.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                hasher.update(chunk)
        inputs.append(dict(task=folder.name, hands_sha256=hasher.hexdigest(),
                           hands_bytes=raw.stat().st_size, model=spec,
                           result_sha256=hashlib.sha256((folder/'result.json').read_bytes()).hexdigest()))
        seen, values = set(), defaultdict(dict)
        counts, partitions = defaultdict(Counter), defaultdict(dict)
        with gzip.open(raw, 'rt') as handle:
            for line in handle:
                guard()
                row = json.loads(line)
                panel, block, rotation = row['panel'], row['block'], row['rotation']
                key = (panel, block, rotation)
                if key in seen or panel not in panels or not 0 <= block < panels[panel]['blocks'] or rotation not in (0, 1):
                    raise ValueError('Duplicate/unexpected light coordinate')
                seen.add(key)
                if row['seed'] != seed or row['milestone'] != milestone or row['policy'] != spec['name']:
                    raise ValueError('Hand/model identity mismatch')
                if row['status'] != 'complete' or not row.get('native_replay_verified'):
                    raise ValueError('Invalid hand or missing native replay')
                chips = row['target_chips']
                if sum(row['net_chips_by_seat']) or chips != row['net_chips_by_seat'][rotation]:
                    raise ValueError('Chip settlement mismatch')
                definition = panels[panel]
                expected_deal = stream_seed(definition['root'], 'test', 'deal', 2, block)
                if row['root_seed'] != definition['root'] or row['deal_seed'] != expected_deal or row['button'] != block % 2:
                    raise ValueError('Frozen deal coordinate mismatch')
                public = (row['deal_seed'], row['button'])
                if coupling.setdefault(key, public) != public:
                    raise ValueError('Unpaired checkpoint/seed deals')
                tails = hand_tails(row)
                if tails != row['tails']:
                    raise ValueError('Recomputed tail arithmetic differs')
                large_calls = sum(a['logical_player']==0 and a['kind']=='call' and
                                  a['observation']['call_amount']>=800 for a in row['actions'])
                allin_calls = sum(a['logical_player']==0 and a['kind']=='call' and
                                  a['observation']['call_amount']==a['observation']['stack'] and
                                  a['observation']['stack']>0 for a in row['actions'])
                if large_calls != row['large_calls'] or allin_calls != row['allin_calls']:
                    raise ValueError('Call telemetry mismatch')
                role = 'button' if rotation == row['button'] else 'big_blind'
                values[panel].setdefault(block, {})[role] = chips
                for group in ((panel, None), (panel, role)):
                    counts[group].update(tails['counts'])
                    counts[group].update(large_calls=large_calls, allin_calls=allin_calls)
                    part = partitions[group].setdefault(tails['first_large_raise_response'], dict(hands=0, target_chips=0))
                    part['hands'] += 1
                    part['target_chips'] += chips
        expected = sum(p['blocks']*2 for p in panels.values())
        if len(seen) != expected or result['hands'] != expected:
            raise ValueError('Closed task has missing hands')
        for panel, definition in panels.items():
            blocks = values[panel]
            if len(blocks) != definition['blocks'] or any(set(v) != {'button','big_blind'} for v in blocks.values()):
                raise ValueError('Incomplete paired block')
            series[(seed, milestone, panel)] = blocks
            overall = [mean(blocks[b].values()) for b in sorted(blocks)]
            roles = {role: cell([blocks[b][role] for b in sorted(blocks)]) for role in ('button','big_blind')}
            per_seed.append(dict(seed=seed, milestone=milestone, panel=panel,
                                 overall=cell(overall), positions=roles, counts=dict(counts[(panel,None)]),
                                 partitions=partitions[(panel,None)],
                                 positional_tails={role:dict(counts=dict(counts[(panel,role)]),
                                                             partitions=partitions[(panel,role)])
                                                   for role in ('button','big_blind')}))
    aggregates, changes, seed_changes = [], [], []
    for panel in panels:
        baseline = aggregate(series, seeds, 100000000, panel)
        for milestone in plan['light_totals']:
            values = aggregate(series, seeds, milestone, panel)
            if values is not None:
                aggregates.append(dict(panel=panel, milestone=milestone, lineages=3,
                                       overall=cell(values), positions={role:cell(aggregate(series,seeds,milestone,panel,role))
                                                                         for role in ('button','big_blind')}))
                if milestone != 100000000 and baseline is not None:
                    changes.append(dict(panel=panel, baseline=100000000, candidate=milestone,
                                        paired_difference=cell([v-b for v,b in zip(values,baseline,strict=True)]),
                                        positions={role:cell([v-b for v,b in zip(aggregate(series,seeds,milestone,panel,role),
                                                                               aggregate(series,seeds,100000000,panel,role),strict=True)])
                                                   for role in ('button','big_blind')}))
            for seed in seeds:
                base = aggregate(series, [seed], 100000000, panel)
                candidate = aggregate(series, [seed], milestone, panel)
                if milestone != 100000000 and base is not None and candidate is not None:
                    seed_changes.append(dict(seed=seed, panel=panel, baseline=100000000, candidate=milestone,
                                             paired_difference=cell([v-b for v,b in zip(candidate,base,strict=True)])))
    return dict(status='preliminary', captured=time.time(), plan_sha256=digest(plan),
                frozen_training_source='17b4c9a08ed0765d0fb8f05240c0409b21e43977',
                inference='Exploratory unadjusted 95% Student-t block intervals, conditional on fixed saved lineages; no final gates.',
                completed_tasks=len(folders), completed_hands=sum(x[1]['hands'] for x in folders),
                pending_tasks=pending, inputs=inputs, three_lineage_aggregate=aggregates,
                checkpoint_changes=changes, per_seed=per_seed, per_seed_checkpoint_changes=seed_changes)


def formatted(value):
    ci = value['ci95']
    return f"{value['bb_per_100']:+.2f}" + (f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else f" ({value['reason']})")


def markdown(result, panels):
    milestones = sorted({r['milestone'] for r in result['three_lineage_aggregate']})
    lookup = {(r['milestone'],r['panel']):r for r in result['three_lineage_aggregate']}
    changes = {(r['candidate'],r['panel']):r for r in result['checkpoint_changes']}
    lines = ['# Preliminary HU20 100M→500M checkpoint curves', '',
             f"Snapshot: {result['completed_tasks']} closed light tasks, **{result['completed_hands']:,} hands**. "
             f"{len(result['pending_tasks'])} light tasks pending at capture.", '',
             '**Exploratory only.** Unadjusted 95% intervals use 256 paired deal blocks per opponent. '
             'Three seed returns/contrasts are averaged inside each deal/rotation block, then both roles are averaged. '
             'Neither seats nor seeds multiply the sample count. All seven opponents and every available checkpoint are retained. '
             'Broad bounded-LBR/native-pressure panels and fresh final confirmation are pending; no strength or promotion gate is inferred.', '',
             '## Absolute three-lineage returns: BB/100 [95% interval]', '',
             '| Nodes | '+' | '.join(panels)+' |', '| ---: | '+' | '.join(['---']*len(panels))+' |']
    for m in milestones:
        lines.append(f'| {m/1e6:g}M | '+' | '.join(formatted(lookup[(m,p)]['overall']) for p in panels)+' |')
    lines += ['', '## Paired change versus each lineage’s own B100M: BB/100 [95% interval]', '',
              '| Nodes | '+' | '.join(panels)+' |', '| ---: | '+' | '.join(['---']*len(panels))+' |']
    for m in milestones:
        if m != 100000000:
            lines.append(f'| {m/1e6:g}M | '+' | '.join(formatted(changes[(m,p)]['paired_difference']) for p in panels)+' |')
    lines += ['', '## Selective-stackoff: individual seeds and positions', '',
              '| Seed | Nodes | Overall BB/100 [95%] | Button | Big blind |', '| --- | ---: | --- | --- | --- |']
    stress = [r for r in result['per_seed'] if r['panel']=='selective_stackoff']
    for r in sorted(stress,key=lambda r:(r['milestone'],r['seed'])):
        lines.append(f"| {r['seed']} | {r['milestone']/1e6:g}M | {formatted(r['overall'])} | {formatted(r['positions']['button'])} | {formatted(r['positions']['big_blind'])} |")
    lines += ['', '## Selective-stackoff tails: counts and denominators', '',
              'Large raises/calls use the retained 800-chip threshold. Full stack is ±20 BB. '
              'These are descriptive correlated-hand counts; whole-hand partitions are not individual-bet EV.', '',
              '| Seed | Nodes | −20BB / hands | +20BB / hands | Large raises / opportunities | Jams / opportunities | Large calls | All-in calls | Fallback / decisions |',
              '| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- |']
    for r in sorted(stress,key=lambda r:(r['milestone'],r['seed'])):
        c=r['counts']
        lines.append(f"| {r['seed']} | {r['milestone']/1e6:g}M | {c['full_stack_losses']}/{c['hands']} | {c['full_stack_wins']}/{c['hands']} | {c['large_actions']}/{c['large_opportunities']} | {c['jam_actions']}/{c['jam_opportunities']} | {c['large_calls']} | {c['allin_calls']} | {c['fallback']}/{c['target_decisions']} |")
    lines += ['', '## Coverage and interpretation', '',
              'The machine-readable summary and CSV retain all opponents, seeds, positions, checkpoint changes, '
              'fallback by street, tails and whole-hand return partitions. An aggregate requires all three seeds; '
              'a checkpoint available for only one or two seeds is shown individually, not substituted for a three-seed curve. '
              'Missing tasks stay pending. Model/checkpoint identities and raw closed-file hashes are in the input manifest. '
              'Each generated hand was natively replayed during evaluation; this reporting pass checks replay evidence, '
              'frozen coordinates, chip accounting, recomputed tails, pairing and independent sums. It does not rerun gameplay.', '',
              'The selective-stackoff opponent was designed after Luna and remains a regression/stress opponent, '
              'not independent confirmation of Luna. Point estimates can fluctuate; these small panels do not replace '
              'the frozen broader or fresh final schedules. Training, saving and evaluation settings remain unchanged.', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    with Path('/tmp/DR_RESEARCH_M4_HEAVY.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        note=Path('/tmp/DR_RESEARCH_M4_COORDINATION.txt')
        with note.open('a') as h:h.write('\nDoctor Research #136 preliminary reporting CLAIM: closed raw light records only; no gameplay.\n')
        try:
            if shutil.disk_usage(a.root).free < 8*2**30:raise OSError('Reporting disk headroom')
            swap_before=swap_bytes();last=0
            def guard():
                nonlocal last
                if time.time()-last < 2:return
                last=time.time()
                if peak_rss() >= 10.5*2**30:raise MemoryError('Reporting RSS')
                if swap_bytes()-swap_before > .5*2**30:raise MemoryError('Reporting swap growth')
                if shutil.disk_usage(a.root).free < 8*2**30:raise OSError('Reporting disk headroom')
            plan=json.loads(a.plan.read_text());result=summary(a.root,plan,guard)
            (a.out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
            (a.out/'curves.md').write_text(markdown(result,[p['name'] for p in plan['light_panels']]))
            with (a.out/'per-seed-role.csv').open('w') as h:
                w=csv.writer(h);w.writerow(['seed','nodes','opponent','role','bb_per_hand','bb_per_100','ci95_lower','ci95_upper','blocks'])
                for r in result['per_seed']:
                    for role,v in {'overall':r['overall'],**r['positions']}.items():
                        w.writerow([r['seed'],r['milestone'],r['panel'],role,v['bb_per_hand'],v['bb_per_100'],*(v['ci95'] or [None,None]),v['blocks']])
            (a.out/'manifest.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                                         for p in a.out.iterdir() if p.is_file()},indent=2)+'\n')
            print(json.dumps(dict(tasks=result['completed_tasks'],hands=result['completed_hands'],pending=len(result['pending_tasks']))))
        finally:
            with note.open('a') as h:h.write('\nDoctor Research #136 preliminary reporting RELEASE: no heavy reporting child.\n')


if __name__=='__main__':main()
