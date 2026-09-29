"""Audit all legal hands and report exploratory paired, block-clustered returns."""
import argparse,gzip,json,subprocess
from collections import Counter,defaultdict
from pathlib import Path
from time import time
import numpy as np
from scipy.stats import t
from scripts.evaluate_hu20 import write_json
from scripts.play_robustness import replay_row
from scripts.tp20_common import seal
from src.arena.schedule import digest,stream_seed
from src.blueprint.windowed import _hash
from scripts.run_tp20_campaign import swap_bytes


def estimate(chips):
    a=np.asarray(chips,dtype=float);n=len(a)
    if n<2:return {'blocks':n,'bb100':float(a.mean()) if n else None,'ci95':None}
    mean=float(a.mean());half=float(t.ppf(.975,n-1)*a.std(ddof=1)/np.sqrt(n))
    return {'blocks':n,'bb_hand':mean/100,'bb100':mean,'buyins20_per100':mean/20,'ci95':[mean-half,mean+half]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.root;plan=json.loads((root/'confirmation/plan.json').read_text());campaign=json.loads((root/'campaign.json').read_text())
    started=time();failures=[];hashes=0
    for folder in (root/'preflight',root/'confirmation'):
        expected=json.loads((folder/'checksums.json').read_text())
        for name,h in expected.items():
            if _hash(folder/name)!=h:raise ValueError(f'Artifact mismatch {folder/name}')
            hashes+=1
    for spec in plan['policies']:
        if spec['name']!='uniform' and _hash(Path(spec['path']))!=spec['sha256']:raise ValueError('Saved model changed')
    for name,h in json.loads((root/'checksums.json').read_text()).items():
        if _hash(root/name)!=h:raise ValueError(f'Campaign inventory mismatch {name}')
    groups=defaultdict(lambda: {'blocks':defaultdict(list),'roles':defaultdict(lambda:defaultdict(list)),'telemetry':Counter(),'lbr':Counter(),'latencies':[]})
    preflight_replays=0
    with gzip.open(root/'preflight/hands.jsonl.gz','rt') as pref:
        for line in pref:
            row=json.loads(line)
            if row['target_chips'] is not None or row['net_chips_by_seat'] is not None:raise ValueError('Preflight exposed realized returns')
            replay_row(row);preflight_replays+=1
    total=0;replayed=0;dealsets=defaultdict(set);offmenu=Counter();problem=[];coupling={}
    with gzip.open(root/'confirmation/hands.jsonl.gz','rt') as rows:
        for line in rows:
            row=json.loads(line);total+=1;n=row['players'];rules=tuple(row['rules']);contract=row['contract'];name=row['policy']
            key=(n,rules,contract,name);g=groups[key]
            expecteddeal=stream_seed(row['root_seed'],'test','deal',n,row['block'])
            if row['deal_seed']!=expecteddeal or row['button']!=row['block']%n:raise ValueError('Schedule mismatch')
            expected_order=list(rules[::-1] if n==3 and row['block']%2 else rules)
            if row['actual_rival_order']!=expected_order:raise ValueError('Mixed opponent order mismatch')
            ckey=(n,rules,row['block'],row['rotation'])
            pair=(row['deal_seed'],row['button'])
            if ckey in coupling and coupling[ckey]!=pair:raise ValueError('Unpaired policy/contract schedules')
            coupling[ckey]=pair;dealsets[n].add(row['deal_seed'])
            replay_row(row);replayed+=1
            if row['status']!='complete':failures.append({k:v for k,v in row.items() if k!='actions'});continue
            chips=row['target_chips'];g['blocks'][row['block']].append(chips)
            role='button' if row['rotation']==row['button'] else 'big_blind' if row['rotation']==(row['button']+1)%n and n==2 else 'nonbutton'
            g['roles'][role][row['block']].append(chips)
            for item in row['decision_telemetry']:g['telemetry'][tuple(item['coordinates'])]+=item['count']
            for item in row['actions']:
                if not item['on_training_menu']:offmenu[(n,rules,contract,name,item['logical_player']==0)]+=1
                if item['logical_player']==0:g['latencies'].append(item['seconds'])
                if 'lbr' in item:
                    d=item['lbr'];g['lbr']['decisions']+=1;g['lbr']['complete']+=d['completed'];g['lbr']['over_soft_budget']+=d['over_soft_budget'];g['lbr']['samples']+=d['samples']
                    g['lbr']['seconds']+=d['seconds'];g['lbr']['max_seconds']=max(g['lbr']['max_seconds'],d['seconds'])
                    g['lbr']['range_holdings_sum']+=d['positive_range_holdings'];g['lbr']['zero_likelihood_cumulative']+=d['zero_likelihood_events']
            if name=='2p-2026092801-20M' and row['block']<64 and len(problem)<12 and any(not i['on_training_menu'] for i in row['actions']):
                problem.append(row)
    prior_schedule_audit=[]
    for n,parent in ((2,Path('/Users/dberweger/Local/hu20-pr112/results/hu20-m4-20260927')),
                     (3,Path('/Users/dberweger/Local/tp20-pr113/results/tp20-m4-20260928'))):
        prior_deals=set();prior_files=0
        if parent.exists():
            for file in parent.rglob('hands.jsonl*'):
                opener=gzip.open if file.suffix=='.gz' else open
                with opener(file,'rt') as rows:
                    for line in rows:
                        row=json.loads(line)
                        if 'deal_seed' in row:prior_deals.add(row['deal_seed'])
                prior_files+=1
            overlap=dealsets[n]&prior_deals
            if overlap:raise ValueError('Confirmation reuses opened HU20/TP20 deals')
            prior_schedule_audit.append({'players':n,'prior_hand_files':prior_files,'prior_deal_seeds':len(prior_deals),'overlap':len(overlap)})
    attempts=json.loads((root/'confirmation/attempts.json').read_text());expected_panels=13*6+4*12+len(plan['lbr_targets'])
    expected_hands=13*6*plan['hu_blocks']*2+4*12*plan['tp_blocks']*3+len(plan['lbr_targets'])*plan['lbr_blocks']*2
    complete=not failures and len(attempts)==expected_panels and all(x.get('status')=='complete' for x in attempts) and total==expected_hands
    results=[];blocks={}
    for key,g in groups.items():
        n,rules,contract,name=key
        if any(len(v)!=n for v in g['blocks'].values()):complete=False
        values={b:sum(v)/n for b,v in g['blocks'].items() if len(v)==n};blocks[key]=values
        result={'players':n,'rules':rules,'contract':contract,'policy':name,'target':estimate(list(values.values())),
            'roles':{r:estimate([sum(xs)/len(xs) for xs in byblock.values()]) for r,byblock in g['roles'].items()},
            'target_decisions':[{'coordinates':k,'count':v} for k,v in sorted(g['telemetry'].items())],
            'off_menu_target_actions':offmenu[(*key,True)],'off_menu_rival_actions':offmenu[(*key,False)],
            'lbr':dict(g['lbr']),'target_latency_p95':float(np.quantile(g['latencies'],.95)) if g['latencies'] else None}
        if n==2:result['attacker']=estimate([-v for v in values.values()])
        results.append(result)
    comparisons=[]
    for n in (2,3):
        for checkpoint in ([2,5,10,20] if n==2 else [20]):
            policies=[s['name'] for s in plan['policies'] if s['players']==n and s.get('nodes')==checkpoint*1000000]
            if not policies:continue
            panels=plan['hu_lineups'] if n==2 else plan['tp_lineups']
            panels=panels+([['lbr']] if n==2 else [])
            for rules in panels:
                for contract in (['menu'] if rules==['lbr'] else ['menu','native']):
                    names=[(n,tuple(rules),contract,name) for name in policies];uniform=(n,tuple(rules),contract,'uniform')
                    if any(k not in blocks for k in names+[uniform]):continue
                    common=set.intersection(*(set(blocks[k]) for k in names+[uniform]))
                    contrasts=[sum(blocks[k][b]-blocks[uniform][b] for k in names)/len(names) for b in sorted(common)]
                    absolutes=[sum(blocks[k][b] for k in names)/len(names) for b in sorted(common)]
                    comparisons.append({'players':n,'checkpoint_mnodes':checkpoint,'rules':rules,'contract':contract,
                        'training_seeds':policies,'trained_minus_uniform':estimate(contrasts),'trained_absolute':estimate(absolutes),
                        'per_seed_contrasts':{k[-1]:estimate([blocks[k][b]-blocks[uniform][b] for b in sorted(common)]) for k in names}})
    contract_effects=[];learning_effects=[]
    for n in (2,3):
        panels=plan['hu_lineups'] if n==2 else plan['tp_lineups']
        for rules in panels:
            names=[s['name'] for s in plan['policies'] if s['players']==n and s.get('nodes')==20000000]
            pairs=[((n,tuple(rules),'native',name),(n,tuple(rules),'menu',name)) for name in names]
            if pairs and all(k in blocks for pair in pairs for k in pair):
                common=set.intersection(*(set(blocks[k]) for pair in pairs for k in pair))
                vals=[sum(blocks[x][b]-blocks[y][b] for x,y in pairs)/len(pairs) for b in sorted(common)]
                contract_effects.append({'players':n,'rules':rules,'final_target_native_minus_menu':estimate(vals)})
    for rules in plan['hu_lineups']+[['lbr']]:
        for contract in (['menu'] if rules==['lbr'] else ['menu','native']):
            pairs=[]
            for seed in (2026092801,2026092802,2026092803):
                pairs.append(((2,tuple(rules),contract,f'2p-{seed}-20M'),(2,tuple(rules),contract,f'2p-{seed}-2M')))
            if pairs and all(k in blocks for pair in pairs for k in pair):
                common=set.intersection(*(set(blocks[k]) for pair in pairs for k in pair))
                vals=[sum(blocks[x][b]-blocks[y][b] for x,y in pairs)/len(pairs) for b in sorted(common)]
                learning_effects.append({'rules':rules,'contract':contract,'target20M_minus2M':estimate(vals),
                    'per_seed':{x[-1]:estimate([blocks[x][b]-blocks[y][b] for b in sorted(common)]) for x,y in pairs}})
    resources=[json.loads(line) for line in (root/'resources.jsonl').read_text().splitlines()]
    peak=max([r['rss_bytes'] for r in resources]+[json.loads((root/'preflight/result.json').read_text()).get('peak_rss_bytes',0),json.loads((root/'confirmation/result.json').read_text()).get('peak_rss_bytes',0)] if (root/'confirmation/result.json').exists() else [r['rss_bytes'] for r in resources])
    minimum_disk=min(r['free_disk_bytes'] for r in resources)
    swap_values=[swap_bytes(r['swap']) for r in resources];swap_values=[v for v in swap_values if v is not None]
    resource_checks={'rss_within_10_5_gib':peak<=10.5*1024**3,'disk_above_8_gib':minimum_disk>=8*1024**3,
        'swap_growth_within_half_gib':not swap_values or max(swap_values)-swap_values[0]<=.5*1024**3,
        'audit_within_absolute_deadline':time()<=campaign.get('deadline',float('inf'))}
    complete=complete and all(resource_checks.values())
    report={'report_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'status':'complete' if complete else 'incomplete','plan_digest':digest(plan),'campaign':campaign,
        'hands':total,'preflight_native_replayed_hands':preflight_replays,'prior_opened_schedule_audit':prior_schedule_audit,'expected_hands':expected_hands,'native_replayed_hands':replayed,'verified_phase_files':hashes,'failures':failures,
        'attempts':attempts,'results':results,'comparisons':comparisons,'contract_effects':contract_effects,'learning_effects':learning_effects,'resources':{'peak_rss_bytes':peak,'checks':resource_checks,
            'minimum_free_disk_bytes':min(r['free_disk_bytes'] for r in resources),'swap_first':resources[0]['swap'],'swap_last':resources[-1]['swap'],
            'elapsed_since_preflight':time()-campaign['started'],'audit_seconds':time()-started},
        'interval_contract':'exploratory 95% Student t over paired rotation-block means, seeds averaged within block; no multiplicity-adjusted primary claim'}
    a.out.mkdir(parents=True,exist_ok=True);write_json(a.out/'results.json',report);write_json(a.out/'exploratory-replay-cases.json',problem)
    text=['# Saved 20BB robustness — M4 report','',f"Status: **{report['status']}**. {total:,} hands; every retained hand replayed through native settlement. All estimates below are exploratory block-clustered 95% intervals. No model promotion.",'',
        '| Game / work | Attack | Contract | Absolute target BB/100 [95% CI] | Trained − uniform BB/100 [95% CI] |','| --- | --- | --- | --- | --- |']
    def fmt(e):
        if e['ci95'] is None:return str(e.get('bb100'))
        return f"{e['bb100']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]"
    for c in comparisons:text.append(f"| {'HU' if c['players']==2 else 'TP'} {c['checkpoint_mnodes']}M | {' / '.join(c['rules'])} | {c['contract']} | {fmt(c['trained_absolute'])} | {fmt(c['trained_minus_uniform'])} |")
    text+=['','BB/hand is BB/100 divided by 100. 20BB buy-ins/100 is BB/100 divided by 20. HU attacker returns negate target returns; role-specific and individual-seed results are retained in results.json. TP worst-case quality is unmeasured.','',
        'The local response integrates the full compatible range and samples four future boards per holding, uses a five-second soft batch deadline, and plays against the unchanged saved target. Internal maxima are never used as reported profit. A negative or imprecise attacker result is not a robustness certificate.','',
        'Trained/fallback and off-menu-history decision counts are separated by street in the JSON. These are exposure measurements; whole-hand profits are not attributed to individual streets. Selected river probes use a declared artificial range and cannot establish full-game exploitability.','',
        f"Peak sampled RSS: {report['resources']['peak_rss_bytes']/1024**3:.3f} GiB; minimum disk: {report['resources']['minimum_free_disk_bytes']/1024**3:.2f} GiB. Resource records and every attempt are retained."]
    (a.out/'report.md').write_text('\n'.join(text)+'\n');seal(a.out)
    print(json.dumps({'status':report['status'],'hands':total,'audit_seconds':time()-started}));return not complete

if __name__=='__main__':raise SystemExit(main())
