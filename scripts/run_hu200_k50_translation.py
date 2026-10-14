"""Fixed-budget translation experiment; complete verification precedes all readout."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import gzip
import json
from pathlib import Path
import subprocess
import sys
from time import time

from scripts import evaluate_hu100_direct as direct
from scripts.archive_hu100_4b_seed_ladder import seal
from scripts.guard_hu100_k50_v052 import guard
from scripts.reserve_match_roots import reserve
from scripts.run_hu200_frozen_evaluation import estimate, quote, series, gh, revision, read
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.action_translation import TranslationOptions, version_for_schema
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.blueprint.average import AveragePolicy
from src.policies.files import file_hash

SHA = '9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6'
ROOTS = {'pilot_direct':20261014122101, 'pilot_panel':20261014122102,
         'direct':20261014122103, 'panel':20261014122104}
END = datetime(2026,10,15,6,tzinfo=timezone.utc).timestamp()
SCIENCE_END = END-45*60
LADDER = ((131072,16384),(65536,8192),(32768,4096))
RULE = 'Recommend Slumbot only if pot_pressure paired Student-t nominal95 lower>0 and no other panel or direct upper<-20 BB/100'


def put(path, value):
    direct.put(Path(path),value)


def choose_counts(costs, remaining):
    options=[]
    for n,p in LADDER:
        seconds=1.25*(quote(costs['direct'],n)+sum(quote(costs['panel'][op],p) for op in direct.PANEL_OPPONENTS))
        option=dict(direct=n,panel=p,projected_seconds=seconds,fits=seconds<=remaining)
        options.append(option)
        if option['fits']: return option,options
    raise ValueError('Minimum counts cannot fit by the fixed deadline')


def prepare(config,base):
    item=config['model'];path=Path(item['path']);audit=read(item['audit_path']);checked=audit.get('audit',audit)
    owners={str(n):json.loads(gh('pr','view',n,'--json','state,url,headRefOid')) for n in (163,265,268)}
    tables={s:file_hash(Path(item['card_buckets'])/(s+'-k50.bin')) for s in EQUITY_K50_SHA256}
    if (file_hash(path)!=SHA or item['sha256']!=SHA or path.stat().st_size!=audit['files']['average']['bytes']
            or checked['status']!='verified' or checked['average_sha256']!=SHA
            or checked['all_nodes_verified']!=85_373_907 or tables!=EQUITY_K50_SHA256):
        raise ValueError('Pinned model, audit or tables differ')
    with gzip.open(path,'rt') as stream:meta=json.loads(stream.readline())
    h=meta['checkpoint_header']
    if h['abstraction']!=HU200_EQUITY_SCHEMA or h['native_state']['completed_nodes']!=audit['actual_nodes'] or meta['source_checkpoint_sha256']!=checked['checkpoint_sha256']:
        raise ValueError('Average header/parent differs')
    spec=dict(name='frozen-off',path=str(path),sha256=SHA,bytes=path.stat().st_size,
        format=meta['format'],abstraction=HU200_EQUITY_SCHEMA,card_buckets=item['card_buckets'],
        entries=85_373_907,iteration=h['iteration'],source_checkpoint_sha256=meta['source_checkpoint_sha256'],translation=None)
    pair=[dict(spec,name='frozen-on',translation=asdict(TranslationOptions())),spec]
    put(base/'inputs.json',dict(source=revision(),configuration=config,model=spec,
        actual_nodes=audit['actual_nodes'],tables=tables,owners=owners,header=h,
        retrieval=item['retrieval'],same_model_tables_cache_both_arms=True))
    put(base/'specs.json',pair)
    reserve(list(ROOTS.values()),config['pr'],base/'root-reservation.json')
    # Verify and map the owner's existing cache; never rebuild or alter it.
    m=AveragePolicy(path,SHA,expected_schema=HU200_EQUITY_SCHEMA,card_buckets=Path(item['card_buckets']),
        cache=Path(config['cache']),cache_read_only=True,translation=TranslationOptions())
    put(base/'cache-verified.json',dict(description=m.description,cache=config['cache'],read_only=True))


def pair_worker(base,specs,blocks,root,opponent,cache,pilot):
    registry,primary=direct.execute(specs,blocks,root,'translation',base/'play',revision(),
        stack_bb=200,opponent=opponent,cache=cache,cache_read_only=True)
    del registry;gc.collect()
    checked=direct.audit(base/'play',base/'replay.json',cost_only=pilot)
    registry,repeat=direct.execute(specs,blocks,root,'translation',base/'reproduction',revision(),
        reproduce=base/'play',stack_bb=200,opponent=opponent,cache=cache,cache_read_only=True)
    put(base/'costs.json',dict(primary=primary,repeat=repeat,audit_seconds=checked['seconds'],
        blocks=blocks,pilot=pilot,**({'paired_sd_bb_per_100':checked['paired_sd_bb_per_100']} if pilot else {})))


def run_pair(base,label,blocks,root,opponent,config,pilot=False):
    command=[sys.executable,'-m','scripts.run_hu200_k50_translation','pair','--out',str(base/label),
        '--specs',str(base/'specs.json'),'--blocks',str(blocks),'--root',str(root),'--cache',config['cache']]
    if opponent:command+=['--opponent',opponent]
    if pilot:command+=['--pilot']
    subprocess.run(command,check=True,timeout=max(1,SCIENCE_END-time()))
    return read(base/label/'costs.json')


def freeze(config,base,costs):
    selected,ladder=choose_counts(costs,SCIENCE_END-time())
    plan=dict(source=revision(),inputs=read(base/'inputs.json'),counts=selected,projection_ladder=ladder,
        roots=ROOTS,opponents=direct.PANEL_OPPONENTS,cache=config['cache'],cache_read_only=True,
        translation={'version':version_for_schema(HU200_EQUITY_SCHEMA),**asdict(TranslationOptions())},
        pilot=costs,projection_multiplier=1.25,frozen_at=time(),rule=RULE,
        science_end_utc='2026-10-15T05:15:00Z',deadline_utc='2026-10-15T06:00:00Z',
        guards=dict(sample_seconds=5,sustained_seconds=60,rss_gib=13,critical_pressure=4,
            free_percent=8,swap_growth_bytes=4_000_000_000,disk_floor_gib=16),
        pairing='HU200 reset; 50/100 blinds; duplicate rotated seats; shared panel deals and role RNGs; blocks independent',
        pilots='1024 direct blocks and 64 blocks per opponent; cost and paired SD only; disjoint final roots',
        readout='Every final hand independently replays with own tables and engine witness, then exactly reproduces. Only elapsed seconds excluded.')
    put(base/'canonical-plan.json',plan)
    original=json.loads(gh('pr','view',config['pr'],'--json','body'))['body']
    posted=original+'\n\n## Canonical plan before final hands\n\nSHA256 `'+file_hash(base/'canonical-plan.json')+'`.\n\n```json\n'+json.dumps(plan,sort_keys=True,indent=2)+'\n```\n'
    body=base/'canonical-pr-body.md';body.write_text(posted)
    gh('pr','edit',config['pr'],'--body-file',body)
    if json.loads(gh('pr','view',config['pr'],'--json','body'))['body']!=posted:
        raise ValueError('Canonical plan PR readback differs')
    put(base/'plan-posted.json',dict(sha256=file_hash(base/'canonical-plan.json'),readback_verified=True,at=time()))
    return plan


def percentiles(values):
    values=sorted(values)
    if not values:return None
    return dict(n=len(values),mean=sum(values)/len(values),p50=values[len(values)//2],
                p90=values[int(.9*len(values))],p99=values[int(.99*len(values))],max=values[-1])


def telemetry(folder):
    cells=defaultdict(lambda:dict(modes=Counter(),distances=[],lookup=[],decision=[],states=Counter(),bounds=0,all_in_changes=Counter()))
    with gzip.open(folder/'play/decisions.jsonl.gz','rt') as stream:
        for d in map(json.loads,stream):
            if 'translation' not in d:continue
            info=d['translation'];c=cells[d['policy']+'/'+d['street']]
            c['modes'][info['mode']]+=1;c['lookup'].append(info['lookup_seconds']);c['decision'].append(d['decision_seconds'])
            c['states'][info['states']]+=1;c['bounds']+=int(info['bound_reached'])
            if info['mode']=='translated':
                c['distances'].append(info['distance']);c['all_in_changes'][info['all_in_changes']]+=1
    return {k:dict(decisions=sum(c['modes'].values()),modes=dict(c['modes']),
        firing_share=c['modes']['translated']/sum(c['modes'].values()),distance=percentiles(c['distances']),
        lookup_seconds=percentiles(c['lookup']),decision_seconds=percentiles(c['decision']),
        states=dict(c['states']),bounds=c['bounds'],all_in_changes=dict(c['all_in_changes'])) for k,c in cells.items()}


def readout(base,plan):
    if read(base/'all-final-verified.json')['status']!='verified':raise ValueError('All final hands must verify before readout')
    folders=['direct',*['panel-'+op for op in direct.PANEL_OPPONENTS]]
    for folder in folders:
        if read(base/folder/'replay.json')['status']!='verified' or not read(base/folder/'reproduction/complete.json')['reproduced_all_hands_and_decisions']:
            raise ValueError('Global verification barrier incomplete')
        for arm in ('play','reproduction'):
            for name,pin in read(base/folder/arm/'complete.json')['files'].items():
                path=base/folder/arm/name
                if path.stat().st_size!=pin['bytes'] or file_hash(path)!=pin['sha256']:raise ValueError('Final bytes changed')
    result=dict(direct=estimate(series(base/'direct',plan['counts']['direct'])['candidate']),panel={})
    for op in direct.PANEL_OPPONENTS:
        arms=series(base/('panel-'+op),plan['counts']['panel'],panel=True)
        result['panel'][op]=dict(arms={a:estimate(v) for a,v in arms.items()},
            contrast=estimate([a-b for a,b in zip(arms['candidate'],arms['baseline'],strict=True)]))
    flags={op:r['contrast']['ci95'][1]<-20 for op,r in result['panel'].items() if op!='pot_pressure'}
    flags['direct']=result['direct']['ci95'][1]<-20
    result.update(status='verified',rule=RULE,severe_regressions=flags,
        recommend_slumbot=result['panel']['pot_pressure']['contrast']['ci95'][0]>0 and not any(flags.values()),
        all_final_hands_independently_replayed_and_exactly_reproduced=True,
        canonical_plan_sha256=file_hash(base/'canonical-plan.json'),telemetry={f:telemetry(base/f) for f in folders})
    put(base/'summary.json',result)


def science(config,base):
    prepare(config,base)
    costs=dict(direct=run_pair(base,'pilot-direct',1024,ROOTS['pilot_direct'],None,config,True),panel={})
    for op in direct.PANEL_OPPONENTS:
        costs['panel'][op]=run_pair(base,'pilot-panel-'+op,64,ROOTS['pilot_panel'],op,config,True)
    put(base/'pilot-costs.json',costs)
    plan=freeze(config,base,costs)
    if file_hash(base/'canonical-plan.json')!=read(base/'plan-posted.json')['sha256']:raise ValueError('Frozen plan changed')
    run_pair(base,'direct',plan['counts']['direct'],ROOTS['direct'],None,config)
    for op in direct.PANEL_OPPONENTS:
        run_pair(base,'panel-'+op,plan['counts']['panel'],ROOTS['panel'],op,config)
    put(base/'all-final-verified.json',dict(status='verified',source=revision(),at=time()))
    readout(base,plan)


def archive(config,base):
    owner=json.loads(gh('pr','view',config['pr'],'--json','state,url,headRefOid'))
    files=[(p,'research/'+p.relative_to(base).as_posix()) for p in sorted(base.rglob('*'))
        if p.is_file() and not p.is_symlink() and not p.is_relative_to(base/'guards/archive') and p.name!='archive-receipt.json']
    for p in ('scripts/run_hu200_k50_translation.py','docs/hu200-k50-translation-protocol.md'):
        files.append((Path(p),'source/'+p))
    dest=Path.home()/f"Local/Research-Cloud/PR-{config['pr']}-hu200-k50-translation/hu200-k50-translation-M4-20261014.zip"
    put(base/'archive-receipt.json',seal(files,dest,dict(source=revision(),owner=owner,inputs=read(base/'inputs.json'),originals_retained=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('run','science','pair','archive'))
    p.add_argument('--config',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--specs',type=Path);p.add_argument('--blocks',type=int);p.add_argument('--root',type=int)
    p.add_argument('--cache',type=Path);p.add_argument('--opponent',choices=direct.PANEL_OPPONENTS)
    p.add_argument('--pilot',action='store_true');a=p.parse_args();base=a.out.resolve()
    if a.stage=='pair':
        pair_worker(base,read(a.specs),a.blocks,a.root,a.opponent,a.cache,a.pilot);return
    config=read(a.config)
    if a.stage=='science':science(config,base);return
    if a.stage=='archive':archive(config,base);return
    base.mkdir(parents=True,exist_ok=False)
    for stage,deadline in (('science',SCIENCE_END),('archive',END)):
        guard(base,stage,[sys.executable,'-m','scripts.run_hu200_k50_translation',stage,
            '--config',str(a.config.resolve()),'--out',str(base)],sample_seconds=5,deadline=deadline)
    put(base/'detached-complete.json',dict(at=time(),within_deadline=time()<END))


if __name__=='__main__':main()
