"""One-use HU100 ladder preparation, exact gates, cost freeze and final readout."""
import argparse
from dataclasses import asdict
import gc
import gzip
from collections import Counter,defaultdict
import json
from math import ceil, log2
from pathlib import Path
import subprocess
import shutil
import sys
from time import perf_counter

from scripts.evaluate_hu100_direct import make_plan, put
from scripts.native_hu100_model_metadata import audited_average_spec
from scripts.evaluate_native_hu100_baseline import execute, make_plan as scripted_plan
from scripts.audit_native_hu100_baseline import audit
from scripts.run_hu100_independent_stages import LinkedRegistry
from scripts.run_hu100_seed_qualification import PRIOR_ROOTS
from scripts.report_native_hu100_learning_curves import frozen_schedule, interval
from src.arena.schedule import build_schedule, digest
from src.blueprint.action_translation import TranslationOptions
from src.policies.files import file_hash

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/hu100-3b-ladder'
BINARY=ROOT/'native/hu20-trainer/target/release/hu20-trainer'
SEED=2026100601
ENTRY_CAP=(7*1024**3-100_000_000)//110
GATE_ENTRY_CAP=57_658_644
PILOT_ROOT,FINAL_ROOT=2026100922201,2026100922202
SCRIPT_PILOT_ROOT,SCRIPT_FINAL_ROOT=2026100922203,2026100922204
PINNED={500_000_000:'35f4b46e6c490573b2cf3ebe4953c269a7793845f425dd53e296fc64be439788',
    1_000_000_000:'cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec'}


def read(path):return json.loads(path.read_text())


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    index=read(ROOT/'docs/reports/native-hu100-growth-1b-artifacts/model-index.json')
    models={}
    for target in PINNED:
        row=next(m for m in index['models'] if m['spec']['actual_nodes']//1_000_000==target//1_000_000)
        if row['audit']['audit']['checkpoint_sha256']!=PINNED[target]:raise ValueError('Indexed checkpoint gate differs')
        spec=row['spec'];p=Path(spec['path'])
        if p.stat().st_size!=spec['bytes'] or file_hash(p)!=spec['sha256']:raise ValueError('Indexed #207 original differs')
        models[str(target)]=spec
    put(OUT/'indexed-models.json',models)
    put(OUT/'prepare.json',{'base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'binary_sha256':file_hash(BINARY),'gate_cap':GATE_ENTRY_CAP,'terminal_cap':ENTRY_CAP,
        'gate_pins':PINNED,'originals_owning_pr':207,'owning_pr_live_state':'MERGED',
        'retrieval':'Hash-verify existing indexed originals; no resumed archives or original writes'})


def check_checkpoint(target):
    folder=OUT/'training'/str(target)
    checkpoint=folder/'checkpoint.gz'
    actual=file_hash(checkpoint)
    if target in PINNED and actual!=PINNED[target]:
        put(OUT/'exactness-mismatch.json',{'target':target,'expected':PINNED[target],'actual':actual})
        raise ValueError('STOP exact checkpoint gate mismatch')
    telemetry=read_lines(OUT/'training'/f'{target}-telemetry.jsonl')[-1]
    if telemetry['checkpoint_sha256']!=actual:raise ValueError('Telemetry checkpoint differs')
    put(folder/'gate.json',{'status':'passed','actual':actual,'expected':PINNED.get(target),
        'actual_nodes':telemetry['completed_nodes'],'terminal_capacity_stop':telemetry['status']=='incomplete-target'})


def read_lines(path):return [json.loads(x) for x in path.read_text().splitlines()]


def spec(target):
    folder=OUT/'training'/str(target)
    checked=read(folder/'audit.json')
    gate=read(folder/'gate.json')
    model=audited_average_spec(folder/'average.gz',checked,
        checkpoint_sha256=gate['actual'],actual_nodes=gate['actual_nodes'])
    indexed=read(OUT/'indexed-models.json')
    if str(target) in indexed:
        old=indexed[str(target)]
        if model['sha256']!=old['sha256'] or model['bytes']!=old['bytes']:
            raise ValueError('STOP exact average export mismatch')
    put(folder/'spec.json',model)


def pairs():
    models={int(p.parent.name):read(p) for p in (OUT/'training').glob('*/spec.json')}
    if 1_000_000_000 not in models:raise ValueError('1B gate required before direct evaluation')
    terminal=max(models)
    result={'terminal-vs-1b':[models[terminal],models[1_000_000_000]],
        '1b-vs-500m':[models[1_000_000_000],models[500_000_000]]}
    if 2_000_000_000 in models and terminal>2_000_000_000:
        result['2b-vs-1b']=[models[2_000_000_000],models[1_000_000_000]]
    # If terminal is the 2B save, primary already represents that single rung.
    put(OUT/'pairs.json',result)
    for rung,models in result.items():put(OUT/'specs'/f'{rung}.json',models)


def config(model,blocks):
    settings=read(ROOT/'configs/arena/hu100-playing-baseline-v1.json')
    settings.update(model=model,proposed_final_blocks_per_opponent=blocks,
        action_translation=asdict(TranslationOptions()),pilot_root=SCRIPT_PILOT_ROOT,final_root=SCRIPT_FINAL_ROOT)
    return settings


def reached_visits(run,registry,model):
    """Count both exact-key and selected translation-witness support."""
    rows=[]
    saved=registry.models[model['name']]
    for opponent in config(model,4096)['opponents']:
        groups=defaultdict(Counter)
        with gzip.open(run/opponent/'decisions.jsonl.gz','rt') as stream:
            for line in stream:
                d=json.loads(line)
                if d['arm']!='candidate' or d['logical_player']!=0:continue
                for kind,key in (('exact',d['key']),('selected',d['translation']['selected_key'])):
                    known=saved.entries.get(key) if key is not None else None
                    n=saved.visits.get(key,0) if known is not None else None
                    band='missing' if n is None else '0' if n==0 else '1-9' if n<10 else '10-99' if n<100 else '100+'
                    groups[kind+'/'+d['street']][band]+=1
        for group,bands in groups.items():rows.append({'opponent':opponent,'key_kind_street':group,'decisions':sum(bands.values()),'bands':dict(bands)})
    return rows


def secondary_readout(base,blocks):
    values={};coverage=[];uniform={}
    for label in ('terminal','1b'):
        if read(base/(label+'-replay.json'))['status']!='verified' or not read(base/(label+'-reproduction')/'complete.json')['reproduced_all_hands_and_decisions']:
            raise ValueError('Unverified scripted final')
        for opponent in config(read(OUT/'pairs.json')['terminal-vs-1b'][0],blocks)['opponents']:
            coords={};baseline=[]
            with (base/label/opponent/'hands.jsonl').open() as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['arm']=='baseline':baseline.append(row);continue
                    coord=row['block'],row['rotation']
                    if coord in coords or row['status']!='completed':raise ValueError('Foreign scripted coordinate')
                    coords[coord]=row['candidate_chips']
            if set(coords)!={(b,r) for b in range(blocks) for r in (0,1)}:raise ValueError('Missing scripted block')
            if opponent in uniform and uniform[opponent]!=baseline:raise ValueError('Paired uniform reference differs')
            uniform[opponent]=baseline
            values[label,opponent]=[(coords[b,0]+coords[b,1])/2 for b in range(blocks)]
            for row in read(base/label/opponent/'report.json')['diagnostics']:
                if row['arm']=='candidate' and row['logical_player']==0:coverage.append({'label':label,'opponent':opponent,**row})
    opponents=config(read(OUT/'pairs.json')['terminal-vs-1b'][0],blocks)['opponents']
    put(base/'summary.json',{'status':'verified','descriptive':True,'blocks_per_opponent':blocks,
        'contrasts':{o:interval([a-b for a,b in zip(values['terminal',o],values['1b',o],strict=True)]) for o in opponents},
        'absolute':{label:{o:interval(values[label,o]) for o in opponents} for label in ('terminal','1b')},
        'coverage':coverage,'visits':{label:read(base/(label+'-visits.json')) for label in ('terminal','1b')},
        'all_final_hands_replayed_and_reproduced':True,'translation':asdict(TranslationOptions())})


def secondary(pilot=False):
    models=read(OUT/'pairs.json')['terminal-vs-1b']
    blocks=32 if pilot else 4096
    root=SCRIPT_PILOT_ROOT if pilot else SCRIPT_FINAL_ROOT
    base=OUT/('secondary-pilot' if pilot else 'secondary')
    measurements=[]
    for label,model in zip(('terminal','1b'),models):
        cfg=base/(label+'-config.json');put(cfg,config(model,blocks))
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        tick=perf_counter();registry=LinkedRegistry(scripted_plan(config(model,blocks),'random',blocks,root),sorted_average_rows=True)
        load_seconds=perf_counter()-tick
        tick=perf_counter()
        execute(cfg,base/label,blocks,root,revision,registry=registry)
        checked=audit(base/label,base/(label+'-replay.json'))
        execute(cfg,base/(label+'-reproduction'),blocks,root,revision,registry=registry,reproduce=base/label)
        put(base/(label+'-visits.json'),reached_visits(base/label,registry,model))
        measurements.append({'label':label,'load_seconds':load_seconds,'variable_and_fixed_seconds':perf_counter()-tick,
            'primary':read(base/label/'costs.json'),'reproduction':read(base/(label+'-reproduction')/'costs.json'),
            'replay_seconds':checked['seconds'],'blocks':blocks,'entries':model['entries']})
        del registry;gc.collect()
    put(base/'costs.json',{'measurements':measurements,'outcomes_inspected_for_quote':False})
    if not pilot:secondary_readout(base,blocks)


def freshness(blocks):
    # Physical seeds, not only root names, are compared with all known HU100
    # schedules through merged #215. The prior campaign verifies its17 roots.
    prior=read(ROOT/'docs/reports/hu100-independent-stages-artifacts/freshness.json')
    roots={int(r):n for r,n in prior['roots'].items()}
    settings=config(read(OUT/'indexed-models.json')['1000000000'],8192)
    seen=set()
    for root,count in roots.items():
        doc=frozen_schedule({'models':[settings['model']]},count,root)
        seen.update(b['deal_seeds'][0] for p in doc['panels'].values() for b in p['blocks'])
    new=set()
    pair=read(OUT/'pairs.json')['terminal-vs-1b']
    counts={}
    for root,count in ((PILOT_ROOT,32),(FINAL_ROOT,blocks)):
        for rung in read(OUT/'pairs.json'):
            schedule=build_schedule(make_plan(pair,count,root,rung))
            deals={b.deal_seeds[0] for b in schedule}
            if len(deals)!=count or deals&seen or deals&new:raise ValueError('Fresh direct deal collision')
            new.update(deals);counts[str(root)+'/'+rung]=count
    for root,count in ((SCRIPT_PILOT_ROOT,32),(SCRIPT_FINAL_ROOT,4096)):
        doc=frozen_schedule({'models':[settings['model']]},count,root)
        deals={b['deal_seeds'][0] for p in doc['panels'].values() for b in p['blocks']}
        if deals&seen or deals&new:raise ValueError('Fresh scripted deal collision')
        new.update(deals);counts[str(root)+'/scripted']=len(deals)
    return {'all_pairwise_disjoint':True,'prior_roots':roots,'new':counts,'new_physical_seeds':len(new)}


def freeze(a_seconds):
    all_pairs=read(OUT/'pairs.json')
    costs={p.parent.name:read(p) for p in (OUT/'direct-pilot').glob('*/costs.json')}
    if set(costs)!=set(all_pairs):raise ValueError('Every prospective rung needs a timing pilot')
    fixed=max(sum(m[x]['load_or_validation_seconds'] for x in ('primary','repeat')) for m in costs.values())
    slope=max((m['seconds']-sum(m[x]['load_or_validation_seconds'] for x in ('primary','repeat')))/m['blocks'] for m in costs.values())
    operations=[read(p) for p in (OUT/'guards/operations').glob('*/receipt.json')]
    training=sum(o['seconds'] for o in operations if any(word in o['name'] for word in ('train','export','audit','spec','check')))
    # Pilot compressed raw bytes scale with blocks. Fixed metadata, guard logs,
    # all current originals and an immutable archive copy are reserved separately.
    raw_per_block=max(sum(pin['bytes'] for arm in ('primary','repeat') for pin in m[arm]['files'].values())/m['blocks'] for m in costs.values())
    existing=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    indexed=read(OUT/'indexed-models.json')
    duplicate_models=0
    indexed_inodes=set()
    for target,old in indexed.items():
        row=next(m for m in read(ROOT/'docs/reports/native-hu100-growth-1b-artifacts/model-index.json')['models'] if m['spec']['sha256']==old['sha256'])
        for name in ('checkpoint','average','current'):
            path=OUT/'training'/target/(name+'.gz')
            pin=next(v for key,v in row['audit']['files'].items() if Path(key).name==name+'.gz')
            if path.stat().st_size!=pin['bytes'] or file_hash(path)!=pin['sha256']:
                raise ValueError('Indexed duplicate bytes changed before archive exclusion')
            duplicate_models+=pin['bytes']
            st=path.stat();indexed_inodes.add((st.st_dev,st.st_ino))
    # Linked arena snapshots share already verified training bytes. The ZIP
    # omits indexed models and stores each new model once. Count actual inode
    # ownership here; equal bytes on different inodes remain conservative.
    seen=set();base_archive=0
    for path in OUT.rglob('*'):
        if not path.is_file() or path.is_symlink() or path.name in ('source.tar','scoring-source.tar','executed-source.tar'):
            continue
        st=path.stat();inode=(st.st_dev,st.st_ino)
        if inode in indexed_inodes or inode in seen:continue
        seen.add(inode);base_archive+=st.st_size
    available=shutil.disk_usage(OUT).free
    disk_floor=16*1024**3
    overhead=1024**3  # fixed science metadata/guard logs and their archive copies
    def quote(n,rungs):return 2*rungs*(fixed+slope*n)
    def disk(n,rungs):return base_archive+2*raw_per_block*n*rungs+overhead+disk_floor
    peak=max(read(OUT/'guards/operations'/('direct-pilot-'+r)/'receipt.json')['peak_family_rss_bytes'] for r in all_pairs)
    def hardware_fits(n,rungs):return disk(n,rungs)<=available and peak+2048*n<7*1024**3
    def descriptive_fits(n,rungs):return hardware_fits(n,rungs) and a_seconds+training+quote(n,rungs)+1800<=36000
    blocks=524288
    rungs=['terminal-vs-1b']+[r for r in ('2b-vs-1b','1b-vs-500m') if r in all_pairs]
    # Preserve the target primary before dropping precision or selecting results.
    while len(rungs)>1 and not descriptive_fits(blocks,len(rungs)):
        rungs.pop()
    while blocks>=32 and not hardware_fits(blocks,len(rungs)):
        blocks//=2
    if blocks<32:raise ValueError('Primary cannot fit measured storage/memory reserve')
    sec=read(OUT/'secondary-pilot/costs.json')
    sec_seconds=sec_raw=0
    for m in sec['measurements']:
        p,r=m['primary'],m['reproduction']
        fixed_sec=m['load_seconds']+sum(v['model_load_or_validation_seconds']+v['snapshot_seconds']+v['output_model_hash_seconds'] for v in (p,r))
        scalable=sum(v['play_and_report_seconds']+v['output_raw_hash_seconds']+v['panel_setup_seconds'] for v in (p,r))+m['replay_seconds']
        sec_seconds+=2*(fixed_sec+scalable*4096/m['blocks'])
    for p in (OUT/'secondary-pilot').rglob('*'):
        if p.is_file() and 'models' not in p.parts and p.name in ('hands.jsonl','decisions.jsonl.gz'):
            sec_raw+=p.stat().st_size*4096/32
    combined=a_seconds+training+quote(blocks,len(rungs))+sec_seconds+1800
    full_scope_quote=a_seconds+training+quote(blocks,len(all_pairs))+sec_seconds+1800
    skip_secondary=full_scope_quote>36000 or combined>36000 or disk(blocks,len(rungs))+2*sec_raw>available
    put(OUT/'freshness.json',freshness(blocks))
    put(OUT/'frozen-final.json',{'blocks_per_rung':blocks,'rungs':rungs,'dropped_descriptive_rungs':[r for r in all_pairs if r not in rungs],
        'root':FINAL_ROOT,'scripted_root':SCRIPT_FINAL_ROOT,'scripted_blocks':4096,
        'primary':'terminal-vs-1b','decision':'paired Student-t 95% lower >0',
        'direct_quote_seconds':quote(blocks,len(rungs)),'secondary_quote_seconds':sec_seconds,
        'skip_secondary':skip_secondary,'requested_full_scope_quote_seconds':full_scope_quote,
        'ten_hour_scope_threshold_exceeded':full_scope_quote>36000,'stage_a_quote_seconds':a_seconds,'training_actual_seconds':training,
        'cost_rule':'2x measured fixed cost plus block-scaled pilot cost; hardware/disk-admitted primary;10h scope threshold for descriptive/secondary; no pilot winnings/variance',
        'expected_planning_half_width':1.96*1665/blocks**.5,'science_outcomes_used_to_select_budget':False,
        'storage':{'available_bytes':available,'required_additional_free_bytes':disk(blocks,len(rungs))+(0 if skip_secondary else 2*sec_raw),
            'disk_floor_bytes':disk_floor,'existing_original_bytes':existing,'archive_original_reserve_bytes':base_archive,
            'hash_verified_indexed_model_bytes_excluded_from_zip':duplicate_models,
            'archive_reserve_basis':'unique local inodes; hash-verified indexed model inodes and full historical-source snapshots excluded; different-inode equal bytes conservatively retained','raw_bytes_per_block_per_rung':raw_per_block,
            'raw_and_archive_copies':2,'overhead_bytes':overhead,'final_workspace_bytes':2048*blocks,'pilot_peak_family_bytes':peak},
        'full_pair_pilot_family_peaks':{r:read(OUT/'guards/operations'/('direct-pilot-'+r)/'receipt.json')['peak_family_rss_bytes'] for r in all_pairs}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','check','spec','pairs','secondary-pilot','secondary','freeze'))
    p.add_argument('--target',type=int);p.add_argument('--a-seconds',type=float)
    a=p.parse_args()
    if a.command=='check':check_checkpoint(a.target)
    elif a.command=='spec':spec(a.target)
    elif a.command=='secondary-pilot':secondary(True)
    elif a.command=='freeze':freeze(a.a_seconds)
    else:globals()[a.command]()

if __name__=='__main__':main()
