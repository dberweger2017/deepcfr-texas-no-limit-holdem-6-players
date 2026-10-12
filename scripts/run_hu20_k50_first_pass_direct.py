"""One-use cost pilot, frozen direct play and independently replayed readout.

Invoke each stage under the fixed campaign guard. Plans use the existing direct
harness; pilot statistics deliberately contain no means or strength labels.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from time import time

from scipy.stats import t
from scripts.audit_hu20_cfr_plus_direct import stats
from scripts.audit_hu20_k50_first_pass_direct import replay
from scripts.evaluate_hu20_cfr_plus_direct import report
from scripts.hu20_k50_first_pass_assets import TABLES, check, write
from scripts.run_hu20_k50_first_pass_training import BASE, BINARY, execute
from src.arena.schedule import digest
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.policies import v042
from src.policies.files import file_hash

SEEDS = (2026100601, 2026100602, 2026100603)
ROOTS = {'pilot': 20261012051101, 'primary': 20261012051102, 'secondary': 20261012051103}
FINAL_BLOCKS = 32768
PILOT_BLOCKS = 2048


def read(path):
    return json.loads(path.read_text())


def ready():
    if (BASE/'guards/campaign-failure.json').exists():
        raise ValueError('Terminal campaign failure; preserve all outputs')
    if read(BASE/'treatment-complete.json')['status'] != 'verified':
        raise ValueError('All treatment exports and full audits required')
    if read(BASE/'guards/operations/three-sequential-k50-lineages/receipt.json')['status'] != 'complete':
        raise ValueError('Training guard has not closed successfully')
    if file_hash(BINARY) != read(BASE/'gates-complete.json')['binary_sha256']:
        raise ValueError('Qualified native binary changed')
    roots = read(BASE/'root-reservation.json')
    if roots['status'] != 'fresh-reserved' or roots['roots'] != ROOTS:
        raise ValueError('Frozen roots have no matching freshness reservation')
    for name, (size, sha) in TABLES.items():
        check(BASE/'tables'/name, sha, size)


def model(label, seed):
    receipt = read(BASE/'audits'/f'{label}-{seed}-5b.json')
    if receipt['status'] != 'verified' or receipt['audit']['status'] != 'verified':
        raise ValueError('Unaudited model')
    spec = dict(receipt['model'], name=f'{label}-{seed}-5b', table_keys=receipt['density']['count'])
    check(Path(spec['path']), spec['sha256'], spec['bytes'])
    if spec['seed'] != seed or spec['actual_nodes'] < 5000000000:
        raise ValueError('Incomplete or foreign training lineage')
    return spec


def restore_release():
    owner = json.loads(subprocess.check_output(['gh', 'pr', 'view', '198', '--repo',
        'dberweger2017/deepcfr-texas-no-limit-holdem-6-players', '--json', 'state,mergedAt'], text=True))
    if owner['state'] != 'MERGED':
        raise ValueError('Release owner PR is not merged')
    folder = BASE/'release'
    folder.mkdir(exist_ok=False)
    path = folder/v042.ASSET_NAME
    execute('restore-v042', ['gh','release','download','v0.4.2','--repo',
        'dberweger2017/deepcfr-texas-no-limit-holdem-6-players',
        '--pattern',v042.ASSET_NAME,'--dir',str(folder)], [path], disk_reserve=512*1024**2)
    receipt = check(path,v042.MODEL_SHA256,v042.MODEL_BYTES)
    spec = {'name':'v0.4.2-O10B-seed2026100601','path':str(path),'bytes':v042.MODEL_BYTES,
        'sha256':v042.MODEL_SHA256,'checkpoint_sha256':v042.CHECKPOINT_SHA256,
        'strategy':'average','seed':v042.SEED,'iteration':v042.ITERATION,
        'abstraction':HU20_UNCAPPED_SCHEMA}
    write(BASE/'release-restoration.json',{'status':'verified','owner_pr':owner,
        'asset':receipt,'model':spec,'retrieval':'gh release download v0.4.2 --repo dberweger2017/deepcfr-texas-no-limit-holdem-6-players --pattern '+v042.ASSET_NAME+' --dir NEW_IGNORED_DIR; verify SHA256 and size'})


def plan(contrast, blocks, root):
    release = read(BASE/'release-restoration.json')['model']
    pairs = [{'candidate':model('k50',seed),
              'reference':model('v1',seed) if contrast=='primary' else release} for seed in SEEDS]
    budget = read(BASE/'guards/budget.json')
    return {'root':root,'blocks':blocks,'pairs':pairs,'started_at':budget['started'],
        'max_seconds':budget['deadline']-budget['started'],
        'contrast':contrast,'scope':'HU20 K50 5B first pass, nominal paired Student-t95 conditional on these fixed trained policies',
        'seed_weighting':'three equally weighted seeds and two equally weighted seats within each deal block',
        'card_table_hashes':{name:sha for name,(_,sha) in TABLES.items()}}


def play_plan(path, out):
    frozen = read(path)
    for lineage in (1,2,3):
        output = out/f'direct-lineage-{lineage}.hands.jsonl.gz'
        execute(f'{out.name}-lineage-{lineage}',[sys.executable,'-m','scripts.evaluate_hu20_cfr_plus_direct',
            'play','--plan',str(path),'--policies','.', '--out',str(out),'--lineage',str(lineage)],
            [output,out/f'direct-lineage-{lineage}.result.json'], disk_reserve=2*1024**3)
        result = read(out/f'direct-lineage-{lineage}.result.json')
        if result['status']!='complete' or result['hands']!=2*frozen['blocks'] or result['plan_sha256']!=digest(frozen):
            raise ValueError('Final direct lineage is incomplete; no replay retry or hand rerun')


def block_values(cells, blocks):
    expected={(l,b,r) for l in (1,2,3) for b in range(blocks) for r in (0,1)}
    if set(cells)!=expected:
        raise ValueError('Missing/duplicate paired block coordinate')
    return [math.fsum(cells[l,b,r] for l in (1,2,3) for r in (0,1))/6 for b in range(blocks)]


def project(results, replay_seconds, elapsed, *, sd, trace_bytes):
    # Each independent process loads its models once. Only playing/replaying
    # scales with hands; scaling total wall time would multiply fixed load cost.
    ratio=FINAL_BLOCKS/PILOT_BLOCKS
    playing=math.fsum(r['load_seconds']+r['play_seconds']*ratio for r in results)
    replaying=replay_seconds*ratio
    margin=float(t.ppf(.975,FINAL_BLOCKS-1))*sd/math.sqrt(FINAL_BLOCKS)
    archive_closeout_reserve=1800
    projected_total=elapsed+1.25*(playing+replaying)+archive_closeout_reserve
    return {'primary_aggregate_paired_sd_bb_per_100':sd,'projected_primary_half_width_bb_per_100':margin,
        'projected_final_play_seconds':playing,'projected_final_replay_seconds':replaying,
        'time_headroom_multiplier':1.25,'archive_closeout_reserve_seconds':archive_closeout_reserve,
        'elapsed_campaign_seconds':elapsed,'projected_total_seconds':projected_total,
        'projected_trace_bytes':math.ceil(trace_bytes*ratio*1.1),
        'admitted':margin<=3 and projected_total<=21600,
        'decision_rule':'Stop if primary projected nominal95 half-width >3 BB/100 or total >6 hours. No count extension or reruns.'}


def pilot():
    ready()
    restore_release()
    results=[];replayed={};receipts={};trace_bytes=0
    for contrast in ('primary','secondary'):
        path=BASE/'plans'/f'pilot-{contrast}.json'
        frozen=plan(contrast,PILOT_BLOCKS,ROOTS['pilot'])
        write(path,frozen)
        out=BASE/'direct'/f'pilot-{contrast}'
        play_plan(path,out)
        results.extend(read(out/f'direct-lineage-{l}.result.json') for l in (1,2,3))
        cells,receipt=replay(frozen,out,Path('.'))
        write(out/'replay.json',receipt)
        replayed[contrast]=cells;receipts[contrast]=receipt
        trace_bytes+=sum(f['bytes'] for f in receipt['raw_files'])
    sd=statistics.stdev(block_values(replayed['primary'],PILOT_BLOCKS))
    quote=project(results,sum(r['seconds'] for r in receipts.values()),
                  time()-read(BASE/'guards/budget.json')['started'],sd=sd,trace_bytes=trace_bytes)
    quote.update(status='cost-only-pilot',pilot_blocks_per_seed_per_contrast=PILOT_BLOCKS,
                 pilot_roots=ROOTS['pilot'],pilot_results=results,
                 pilot_replay_receipts={c:{k:v for k,v in r.items() if k!='coverage'} for c,r in receipts.items()},
                 observed_peak_worker_rss_bytes=max(r['peak_rss_bytes'] for r in results),
                 completed=time())
    # Explicit prospective refusal is not a scientific/guard failure. No final
    # plan can be accepted if admitted is false; preserve the cost-only evidence.
    write(BASE/'pilot-quote.json',quote)
    print(json.dumps({k:quote[k] for k in ('admitted','projected_primary_half_width_bb_per_100','projected_total_seconds')}))


def freeze():
    ready()
    quote=read(BASE/'pilot-quote.json')
    if not quote['admitted']:
        raise ValueError('Pilot quote forbids dependent final play')
    plans={}
    for contrast in ('primary','secondary'):
        frozen=plan(contrast,FINAL_BLOCKS,ROOTS[contrast])
        write(BASE/'plans'/f'final-{contrast}.json',frozen)
        plans[contrast]=frozen
    document={'protocol':'docs/hu20-k50-5b-first-pass.md','source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'native_base':'045c793a4abd2e371d4d77de663fd3dd2f00316b','binary_sha256':file_hash(BINARY),
        'plans':plans,'plan_digests':{c:digest(p) for c,p in plans.items()},
        'input_table_hashes':{name:sha for name,(_,sha) in TABLES.items()},
        'root_reservation':read(BASE/'root-reservation.json'),'pilot_quote':quote,
        'primary_decision':'lower95>0: stronger, recommend longer protocol; upper95<0: weaker, recommend stopping full-game K50; otherwise inconclusive, no extension',
        'no_outcome_driven_extension':True,'per_street_quantiles':'Distinct stored reached keys, coverage disclosed; exact global density',
        'created':time()}
    write(BASE/'canonical-final-plan.json',document)


def final_play():
    ready()
    canonical=read(BASE/'canonical-final-plan.json')
    posted=read(BASE/'canonical-plan-posted.json')
    if posted['canonical_sha256']!=file_hash(BASE/'canonical-final-plan.json') or not posted['url'].startswith(
        'https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/238#issuecomment-'):
        raise ValueError('Canonical exact plan must be posted on PR238 before final play')
    if not read(BASE/'pilot-quote.json')['admitted']:
        raise ValueError('Pilot quote forbids final play')
    quote=read(BASE/'pilot-quote.json')
    remaining=1.25*(quote['projected_final_play_seconds']+quote['projected_final_replay_seconds'])+quote['archive_closeout_reserve_seconds']
    if time()+remaining>read(BASE/'guards/budget.json')['deadline']:
        raise ValueError('Current clock no longer admits projected final play/replay/archive')
    reserve=quote['projected_trace_bytes']
    # Retain both final traces and ZIP space, with closeout scratch above floor.
    if shutil.disk_usage(BASE).free-2*reserve-6*1024**3<16*1024**3:
        raise ValueError('Final traces plus archive reserve do not fit')
    for contrast in ('primary','secondary'):
        path=BASE/'plans'/f'final-{contrast}.json'
        if read(path)!=canonical['plans'][contrast] or digest(read(path))!=canonical['plan_digests'][contrast]:
            raise ValueError('Final plan changed after canonical freeze')
        play_plan(path,BASE/'direct'/f'final-{contrast}')
    write(BASE/'final-play-complete.json',{'status':'complete','finished':time(),
        'canonical_sha256':posted['canonical_sha256']})


def final_replay():
    ready()
    if read(BASE/'final-play-complete.json')['status']!='complete':
        raise ValueError('Both final contrasts must finish before replay/readout')
    for contrast in ('primary','secondary'):
        frozen=read(BASE/'plans'/f'final-{contrast}.json')
        out=BASE/'direct'/f'final-{contrast}'
        cells,receipt=replay(frozen,out,Path('.'))
        with gzip.open(out/'replayed-cells.jsonl.gz','xt') as stream:
            for (lineage,block,rotation),value in sorted(cells.items()):
                stream.write(json.dumps([lineage,block,rotation,value])+'\n')
        receipt['replayed_cells']=check(out/'replayed-cells.jsonl.gz',file_hash(out/'replayed-cells.jsonl.gz'))
        write(out/'replay.json',receipt)
    write(BASE/'final-replay-complete.json',{'status':'verified','finished':time(),'hands':393216})


def paired_statistics(cells, blocks):
    result={'overall':stats(block_values(cells,blocks)),
        'lineages':{str(l):stats([(cells[l,b,0]+cells[l,b,1])/2 for b in range(blocks)]) for l in (1,2,3)},
        'positions':{seat:stats([math.fsum(cells[l,b,b%2 if seat=='button' else 1-b%2] for l in (1,2,3))/3
                                for b in range(blocks)]) for seat in ('button','big_blind')},
        'seed_by_seat':{str(SEEDS[l-1]):{seat:stats([cells[l,b,b%2 if seat=='button' else 1-b%2]
                for b in range(blocks)]) for seat in ('button','big_blind')} for l in (1,2,3)}}
    return result


def compare_statistics(independent, reported):
    for dimension in ('overall','lineages','positions'):
        a=independent[dimension];b=reported[dimension]
        pairs=[(a,b)] if dimension=='overall' else [(v,b[k]) for k,v in a.items()]
        for left,right in pairs:
            if left['blocks']!=right['blocks'] or not math.isclose(left['bb_per_100'],right['bb_per_100'],abs_tol=1e-9):
                raise ValueError('Independent paired arithmetic differs')
            if (left['ci95'] is None)!=(right['ci95'] is None):
                raise ValueError('Independent interval degeneracy differs')
            if left['ci95'] is not None and not all(math.isclose(x,y,abs_tol=1e-9) for x,y in zip(left['ci95'],right['ci95'],strict=True)):
                raise ValueError('Independent paired interval differs')


def readout():
    ready()
    if read(BASE/'final-replay-complete.json')['status']!='verified':
        raise ValueError('Every final hand in both contrasts must replay before numerical readout')
    summaries={}
    for contrast in ('primary','secondary'):
        out=BASE/'direct'/f'final-{contrast}'
        frozen=read(BASE/'plans'/f'final-{contrast}.json')
        receipt=read(out/'replay.json')
        if receipt['plan_sha256']!=digest(frozen):
            raise ValueError('Foreign replay receipt')
        for item in receipt['raw_files']:
            check(Path(item['path']),item['sha256'],item['bytes'])
        item=receipt['replayed_cells']
        check(Path(item['path']),item['sha256'],item['bytes'])
        cells={}
        with gzip.open(out/'replayed-cells.jsonl.gz','rt') as source:
            for line in source:
                l,b,r,value=json.loads(line)
                if (l,b,r) in cells:raise ValueError('Duplicate replayed payoff')
                cells[l,b,r]=value
        independent=paired_statistics(cells,FINAL_BLOCKS)
        compare_statistics(independent,report(frozen,out))
        summaries[contrast]=dict(independent,independent_arithmetic_matches=True,
            hands=196608,plan_sha256=digest(frozen),coverage=receipt['coverage'])
    ci=summaries['primary']['overall']['ci95']
    decision='stronger; recommend longer protocol' if ci and ci[0]>0 else (
        'weaker; recommend stopping full-game K50' if ci and ci[1]<0 else 'inconclusive; do not extend')
    write(BASE/'final-readout.json',{'status':'verified','contrasts':summaries,'primary_decision':decision,
        'scope':'Nominal paired Student-t95 intervals conditional on these fixed policies. Owner decides next; no adoption or release.',
        'finished':time()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('pilot','freeze','final-play','final-replay','readout'))
    a=p.parse_args()
    {'pilot':pilot,'freeze':freeze,'final-play':final_play,'final-replay':final_replay,'readout':readout}[a.stage]()


if __name__=='__main__':main()
