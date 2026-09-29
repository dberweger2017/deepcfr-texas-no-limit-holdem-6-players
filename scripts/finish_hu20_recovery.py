"""M4-only final audits, readable report and publication patch, even if incomplete."""

import argparse
from collections import Counter,defaultdict
import difflib
import gc
import gzip
import json
from pathlib import Path
from time import time

from scripts.evaluate_hu20_scaling import tasks
from scripts.report_hu20_scaling import audit,combine
from scripts.report_hu20_reopening import verify_phase
from scripts.finish_hu20_scaling import independent
from scripts.hu20_scaling_common import acquire, inventory, check
from scripts.hu20_scaling_runtime import validate_inputs
from scripts.train_hu20 import write_json,system
from src.arena.schedule import stream_seed
from src.blueprint.solver import _seed
from src.blueprint.windowed import _hash


def patch(plan,report,out,extra=None):
    source=Path(plan['source']);changes={
        'docs/reports/hu20-scaling-m4-recovery.md':report,
        'docs/reports/hu20-scaling-recovery-artifacts/summary.json':(out/'compact-results.json').read_text()}
    if extra:changes['docs/reports/hu20-scaling-recovery-artifacts/final-manifest.json']=extra
    model='docs/hu20-native-reopening-model-card.md'
    changes[model]=(source/model).read_text()+'\n## M4-only scaling recovery (#116)\n\nSee the [M4-only report](reports/hu20-scaling-m4-recovery.md). No default model is promoted. Partial or resource-limited comparisons do not establish robustness; the original 20M model and previous human interfaces remain available.\n'
    road='ROADMAP.md';original=(source/road).read_text()
    entry='- **M4-only #116 recovery reported:** the [new recovery report](docs/reports/hu20-scaling-m4-recovery.md) retains the separately authorized ten-hour attempt, resumed lineages and actual complete/pending comparisons. Its measured gates and remaining limits are recorded there. No merge, promotion, paid host or subsequent campaign is scheduled.\n\n'
    changes[road]=original.replace('## Current position\n\n','## Current position\n\n'+entry,1)
    result=[]
    for name,text in changes.items():
        old=(source/name).read_text() if (source/name).exists() else ''
        result.extend(difflib.unified_diff(old.splitlines(True),text.splitlines(True),
                 fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
    Path(plan['root']).with_name(Path(plan['root']).name+'-report.patch').write_text(''.join(result))


def finish(plan):
    root=Path(plan['root']);out=root/'report';acquire(out)
    state=json.loads((root/'campaign.json').read_text());before=plan['swap_baselines']['m4']
    validate_inputs(plan)
    paths=[];attempts=[]
    for phase in ['primary','diagnostic']:
        evaluation=root/f'evaluation-{phase}';folder=root/f'audit-{phase}'
        if (evaluation/'result.json').exists():
            if not (folder/'results.json').exists():
                phaseplan={**plan,'panel_filter':phase}
                check(plan,out,plan['deadline']-60,before)
                audit(phaseplan,evaluation,folder)
            paths.append(folder/'results.json')
            attempts+=json.loads((evaluation/'attempts.json').read_text())
    anticipated=plan['baseline_models']+plan['reference_models']
    for seed in plan['training_seeds']:
        anticipated += [{'name':f'B-{seed}-{m}','seed':seed,'arm':'B','milestone':m} for m in plan['milestones']]
    expected=out/'expected-models.json';write_json(expected,anticipated)
    if not paths:
        empty=out/'empty-audit.json';write_json(empty,{'host':'m4','status':'incomplete','panels':[],'native_replayed_hands':0});paths=[empty]
    report=combine({**plan,'coordinator_models':str(expected),'panel_filter':'all'},paths,out/'results.json')
    rawpaths=list(root.glob('evaluation-*/*.jsonl.gz'))
    available=[a for a,r in report['primary'].items() if r['status']=='available']
    if available:
        independent_result=independent(rawpaths,plan['training_seeds'],100000000,available)
        for a,r in independent_result['primary'].items():
            ref=report['primary'][a]['long_minus_20M']
            if r['blocks']!=ref['blocks'] or abs(r['bb100']-ref['bb100'])>1e-8 or any(abs(x-y)>1e-8 for x,y in zip(r['interval'],ref['interval'])):
                raise ValueError('Independent block arithmetic disagreement')
        write_json(out/'independent-summary.json',independent_result)
    prior=set();fresh=set();prior_files=0
    for oldroot in plan['prior_roots']:
        for path in Path(oldroot).rglob('*hands.jsonl*'):
            prior_files+=1;opener=gzip.open if path.suffix=='.gz' else open
            with opener(path,'rt') as f:
                for line in f:
                    row=json.loads(line)
                    if 'deal_seed' in row:prior.add(row['deal_seed'])
    for path in rawpaths:
        with gzip.open(path,'rt') as f:
            for line in f:fresh.add(json.loads(line)['deal_seed'])
    if fresh & prior:raise ValueError('Confirmation reused prior opened deal')
    independent_deals=set();observations=0
    with gzip.open(plan['independent_path'],'rt') as f:
        for line in f:
            row=json.loads(line);independent_deals.add(row['seed']);observations+=1
    preflight_deals={stream_seed(plan['recovery_preflight_root'],'test','deal',2,b) for b in range(8)}
    if fresh & (independent_deals|preflight_deals):raise ValueError('Confirmation overlapped fixture/preflight')
    excluded_deals=fresh|independent_deals|preflight_deals
    lineages=[];resources=[];hashchecks=0
    for seed in plan['training_seeds']:
        folder=root/'training'/f'B-{seed}';resultpath=folder/'result.json';parent=plan['parents'][str(seed)]
        if not resultpath.exists():
            lineages.append({'seed':seed,'status':'not started','completed_nodes':parent['completed_nodes']});continue
        r=json.loads(resultpath.read_text());hashchecks+=verify_phase(folder)
        if r['initial_nodes']!=parent['completed_nodes'] or r['initial_iteration']!=parent['iteration'] or r['parent']['checkpoint_sha256']!=parent['checkpoint_sha256']:
            raise ValueError('Resumed lineage counter/hash mismatch')
        cumulative=parent['completed_nodes'];iteration=parent['iteration'];counters=defaultdict(Counter)
        with (folder/'iterations.jsonl').open() as f:
            for line in f:
                x=json.loads(line);cumulative+=x['nodes'];iteration+=1
                if x['lifetime_completed_nodes']!=cumulative or x['iteration']!=iteration:raise ValueError('Complete iteration work mismatch')
                for field in ['new_entries_by_street','revisited_keys_by_street','traverser_visits_by_street']:
                    counters[field].update(x[field])
                for k,v in x['attempted_work'].items():
                    if isinstance(v,dict):counters['attempted_'+k].update(v)
        if cumulative!=r['completed_nodes'] or iteration!=r['completed_iterations']:raise ValueError('Completed work mismatch')
        for i in range(1,iteration+2):
            for seat in (0,1):
                if _seed(seed,i,seat,0,'deal') in excluded_deals:raise ValueError('Training deal overlap')
        for m in r['milestones']:
            if _hash(Path(m['checkpoint_path']))!=m['checkpoint_sha256'] or _hash(Path(m['policy_path']))!=m['policy_sha256']:
                raise ValueError('Milestone checkpoint/export changed')
            if m['recovered'] and (seed!=2026093001 or m['completed_nodes']!=40000075 or m['iteration']!=107359):
                raise ValueError('Recovered 40M state changed')
        lineages.append({**r,'additional_work_by_street_and_raise':{k:dict(v) for k,v in counters.items()}})
    for path in root.glob('*supervisor/resources.jsonl'):
        records=[json.loads(x) for x in path.read_text().splitlines()]
        if not records:continue
        summary={'path':str(path),'samples':len(records),'peak_aggregate_rss_bytes':max(r['aggregate_job_rss_bytes'] for r in records),
                 'min_free_disk_bytes':min(r['free_disk_bytes'] for r in records),'max_swap_growth_bytes':max(r['swap_growth_bytes'] for r in records),
                 'all_samples_ac':all('AC Power' in r['power'] for r in records)}
        if summary['peak_aggregate_rss_bytes']>=10.5*1024**3 or summary['min_free_disk_bytes']<8*1024**3 or summary['max_swap_growth_bytes']>.5*1024**3:
            summary['resource_violation']=True
        resources.append(summary)
    complete_training=all(r.get('status')=='complete' and r['completed_nodes']>=100000000 for r in lineages)
    if not complete_training:report['status']='incomplete'
    verification={'runtime_input_hashes':len(plan['runtime_inputs']),'training_hash_checks':hashchecks,
                  'fresh_deal_count':len(fresh),'prior_files':prior_files,'prior_overlap':0,'independent_fixture_observations':observations,
                  'all_training_complete':complete_training,'lineages':lineages,'resources':resources,'attempts':attempts,'campaign':state,
                  'native_replayed_hands':report['native_replayed_hands'],'finished':time(),'deadline':plan['deadline']}
    write_json(out/'verification.json',verification);write_json(out/'results.json',report)
    compact={k:report[k] for k in ['status','primary','primary_roles','quality_gate','pressure_safeguard','native_replayed_hands','per_policy','exploratory_curves']}
    compact.update(attempt_id=plan['attempt_id'],deadline=plan['deadline'],expected_hands=plan['expected_confirmation_hands'],
                   pending_panels=[{'policy':x['policy'],'attacker':x['attacker'],'pending_blocks':len(x['pending_blocks'])} for x in report['pending_panels']],
                   training=[{k:r.get(k) for k in ['seed','status','completed_nodes','additional_nodes','discarded_nodes','failure','failure_phase']} for r in lineages],resources=resources)
    if complete_training and time()<plan['deadline']-600:
        from scripts.play_hu20_native import play
        from scripts.play_hu20 import replay_history
        models=json.loads(Path(plan['coordinator_models']).read_text());model=next(s for s in models if s['seed']==plan['training_seeds'][0] and s['milestone']==100000000)
        visible=[]
        choose=lambda prompt:next(x.split('.')[0].strip() for x in reversed(visible) if x.startswith('  ') and ('check' in x or 'call' in x))
        smoke=play(Path(model['path']),model['sha256'],out/'candidate-human-smoke.jsonl',seed=plan['demo_root'],max_hands=20,input_fn=choose,output=visible.append)
        smoke['replayed']=replay_history(out/'candidate-human-smoke.jsonl')
        write_json(out/'candidate-human-smoke-result.json',smoke);write_json(out/'candidate-model.json',model)
        (out/'candidate-human-transcript.txt').write_text('\n'.join(visible)+'\n')
        compact['candidate']=model;compact['human_smoke']=smoke
    write_json(out/'compact-results.json',compact)
    lines=['# M4-only uncapped HU20 recovery','',f"Attempt `{plan['attempt_id']}`: **{report['status']}**.",
       f"Native replayed {report['native_replayed_hands']:,} / {plan['expected_confirmation_hands']:,} planned confirmation hands.",
       'The prior failed attempt and its deadline remain unchanged. This separately authorized attempt resumes retained states; M1 performed only transfer/edit/status work. All computation, automatic transitions and guards ran on M4.',
       '',f"Primary quality gate: **{report['quality_gate']}**. Native-pressure safeguard: **{report['pressure_safeguard']}**.",'']
    for a,r in report['primary'].items():
        if r['status']=='available':
            x=r['long_minus_20M'];lines.append(f"- {a}: final minus own 20M **{x['bb100']:.2f} BB/100**, two-sided 97.5% CI **[{x['interval'][0]:.2f}, {x['interval'][1]:.2f}]**. Early absolute {r['early_absolute']['bb100']:.2f}, final absolute {r['long_absolute']['bb100']:.2f} BB/100; {x['blocks']} independent paired blocks, three lineage contrasts averaged inside each block.")
        else:lines.append(f"- {a}: **unavailable**, prespecified complete pairing/count not achieved.")
    lines+=['','These are realized profits against fixed attackers, not exact exploitability or full-game/human competence. Weak or limited LBR does not certify robustness. Per-lineage/role profits, control regressions, exploratory curves, telemetry and every pending panel/block are retained in the machine-readable outputs. No partial target or resource-limited attack is silently omitted.','',
       '## Retention and audit','',f"M4 root: `{root}`. Complete verification: `report/verification.json`; per-hand replay and telemetry: `audit-*/results.json`; pending exact blocks: `report/results.json`. Source/config/runtime path mapping and all attempts are retained. Original #112/#113/#115 playable models remain unchanged.",'',
       '```sh',f'rsync -a m4:{root}/ ./hu20-scaling-m4-recovery/',f'scp m4:{root}-final-manifest.json ./','```','',
       'The native audit replays actions, target RNG, concrete menus/key/visit membership, event digests and chip settlement. It verifies LBR legality/work telemetry; it does not recompute every internal LBR action-value estimate. Available primary arithmetic is independently rebuilt from raw chips. Public observation density and own-training street work are separate diagnostics.','']
    if 'candidate' in compact:
        m=compact['candidate'];lines+=['## Fixed-first-seed experimental human command','','```sh',f"python -m scripts.play_hu20_native --policy {m['path']} --sha256 {m['sha256']} --history results/recovery-human.jsonl",'python -m scripts.play_hu20_native --replay results/recovery-human.jsonl','```','The bounded smoke/replay used the first seed by fixed order, with bot cards hidden until legitimate disclosure. No model promotion follows.','']
    lines+=['## One next recommendation','',
       'Review this fixed-recipe scaling evidence before another experiment. A complete positive primary supports more work on this recipe; an inconclusive or pending comparison does not diagnose an abstraction ceiling. No follow-on campaign is launched.']
    text='\n\n'.join(lines)+'\n';(out/'REPORT.md').write_text(text);patch(plan,text,out)
    return {'status':report['status'],'hands':report['native_replayed_hands']}


def seal(plan):
    root=Path(plan['root'])
    if time()>=plan['deadline']:raise TimeoutError('Recovery reporting deadline')
    # Called by the detached wrapper only after coordinator AND outer watchdog logs close.
    files={}
    for path in sorted(root.rglob('*')):
        if time()>=plan['deadline']:
            raise TimeoutError('Global inventory reached original recovery deadline')
        if path.is_file():files[str(path.relative_to(root))]={'sha256':_hash(path),'bytes':path.stat().st_size}
    manifest={'attempt_id':plan['attempt_id'],'deadline':plan['deadline'],'finished':time(),'files':files}
    text=json.dumps(manifest,indent=2,sort_keys=True)+'\n'
    root.with_name(root.name+'-final-manifest.json').write_text(text)
    out=root/'report'
    if (out/'REPORT.md').exists():patch(plan,(out/'REPORT.md').read_text(),out,text)
    return {'status':'sealed','files':len(files)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--seal',action='store_true');a=p.parse_args()
    plan=json.loads(a.plan.read_text());print(json.dumps(seal(plan) if a.seal else finish(plan)))
