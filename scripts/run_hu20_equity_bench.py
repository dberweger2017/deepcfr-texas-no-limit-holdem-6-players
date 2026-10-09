"""Frozen K50 versus v1 bench, using the unchanged native trainer and lock tool."""
import argparse
from copy import deepcopy
import json
from math import ceil
from pathlib import Path
from statistics import mean
import subprocess
import sys
from time import perf_counter

import numpy as np
from scripts.subgame_bench import frozen_inputs, policies, rows, seat_mean
from src.blueprint.abstraction import HU20_EQUITY_SCHEMA
from src.blueprint.equity_buckets import EquityCards
from src.diagnostics.flop_check import atomic_json, factored_key, line_key
from src.diagnostics.global_bucket_validation import BLOCKED, global_labels, project_compact
from src.diagnostics.board_pooling_results import completion
from src.diagnostics.pooling_runtime import run_owned_tool
from src.diagnostics.subgame_bench import FrozenRoot
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/equity-bench'
POOL = Path.home() / 'Local/hu20-board-pooling-20261004'
PLAN = ROOT / 'configs/diagnostics/hu20-board-pooling.json'
BINARY = ROOT / 'native/hu20-trainer/target/release/hu20-trainer'
LOCK = POOL / 'pooling-engineering-05-mac'
SEED = 202610050001


def projected(request, original, labels):
    """Use #190's label transport with the trained schema's scalar card payload."""
    data = project_compact(request, original, {'50': labels, '200': labels})
    maps = {}
    matrix = np.asarray(labels)
    for node in request['nodes']:
        if node['terminal']:
            continue
        table = original['node_tables'][line_key(node['line'])]
        selected = matrix[:1] if node['street'] == 'turn' else matrix[1:]
        template = deepcopy(node['template'])
        template[0] = HU20_EQUITY_SCHEMA
        keys = {str(bucket): factored_key(template, bucket)
                for bucket in sorted(set(map(int, np.unique(selected))) - {BLOCKED})}
        if table in maps and maps[table] != keys:
            raise ValueError('Aliased trained template differs')
        maps[table] = keys
    data['pool_keys'] = {'v1': original['pool_keys']['v1'], 'eq50-fit0': maps}
    data['global_bucket_transport'] = {'aliases': {'eq50-fit0': HU20_EQUITY_SCHEMA},
        'card_payload': 'scalar K50 label; versioned equity schema',
        'unchanged': ['blueprint', 'v1 keys', 'history', 'menus', 'ranges', 'tree']}
    return data


def inputs():
    return frozen_inputs(PLAN, POOL / 'prepared-03', 0)


def prepare():
    plan, records, folds, jobs = inputs()
    cards = EquityCards(OUT / 'tables')
    atomic_json(OUT / 'input-pins.json', {'base': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'tables': cards.sha256, 'native': file_hash(BINARY), 'lock': file_hash(LOCK),
        'corpus': plan['corpus'], 'crossfit': plan['crossfit'], 'jobs': jobs,
        'owning_prs_checked_merged': [149,162,163,169,190,210],
        'restoration': 'tar -xzf ~/Local/Research-Cloud/PR-163-equity-buckets/hu20-equity-buckets-20261005.tar.gz --strip-components 1 hu20-equity-buckets-20261005/{flop,turn,river}-k50.bin'})
    for fold in (0,1):
        roots = [FrozenRoot.from_request(records[j['spot']],json.loads(Path(j['request']).read_text()))
                 for j in jobs if folds[j['spot']] != fold]
        atomic_json(OUT / f'roots-{fold}.json', [r.to_native() for r in roots])


def train(variant, fold, final):
    target = 10_000_000 if final else 3_000_000
    checkpoints = [1_000_000,3_000_000]
    if final and variant == 'equity':
        checkpoints.append(json.loads((OUT/'matched-freeze.json').read_text())['iterations'][str(fold)])
    destination = OUT / ('runs' if final else 'pilot-training') / f'fold-{fold}'
    destination.mkdir(parents=True,exist_ok=True)
    cmd = [str(BINARY),'bench-train','--roots',str(OUT/f'roots-{fold}.json'),'--seed',str(SEED),
        '--iterations',str(target),'--checkpoints',','.join(map(str,checkpoints)),
        '--lineage','2026093001','--variant',variant,'--out',str(destination)]
    if variant == 'equity':
        cmd += ['--card-buckets',str(OUT/'tables')]
    tick = perf_counter()
    subprocess.run(cmd,check=True)
    atomic_json(destination/f'{variant}-timing.json',{'seconds':perf_counter()-tick,'iterations':target,'command':cmd})


def visits(folder):
    result = []
    for name,path in policies(folder):
        if not name.endswith('.current'):
            continue
        doc = json.loads(path.read_text())
        values = sorted(g['roots'] for g in doc['groups'])
        bands = {label:sum(low<=v<high for v in values) for label,low,high in
            [('0',0,1),('1-9',1,10),('10-99',10,100),('100-999',100,1000),('1000+',1000,float('inf'))]}
        result.append({'name':name,'keys':len(values),'visits':sum(values),'mean':mean(values),
            'quantiles':{str(q):float(np.quantile(values,q)) for q in (0,.1,.25,.5,.75,.9,.99,1)},'bands':bands})
    return result


def freeze():
    iterations, details = {}, {}
    for fold in (0,1):
        records = visits(OUT/'pilot-training'/f'fold-{fold}')
        selected = {r['name'].split('/')[1].split('.')[0]:r for r in records if r['name'].startswith('iteration-3000000/')}
        v1,equity = selected['v1'],selected['equity']
        ratio = equity['keys']/v1['keys']
        # Actual traverser work can differ by abstraction. Matching mean visits
        # therefore uses both key count and observed total visits, before scoring.
        matched = ceil(3_000_000*v1['mean']/equity['mean'])
        iterations[str(fold)] = matched
        details[str(fold)] = {'key_ratio':ratio,'mean_visit_ratio':v1['mean']/equity['mean'],
            'v1':v1,'equity':equity,'iterations':matched}
    atomic_json(OUT/'matched-freeze.json',{'iterations':iterations,'details':details,
        'rule':'ceil(3M * v1 mean traverser visits/key / equity mean traverser visits/key); one outcome-blind extrapolation per fold; no later adjustment',
        'scoring_started':False})


def adapters():
    _,_,folds,jobs = inputs()
    cards = EquityCards(OUT/'tables')
    for job in jobs:
        dest = OUT/'prepared'/job['job']
        dest.mkdir(parents=True,exist_ok=False)
        request = json.loads(Path(job['request']).read_text())
        original = json.loads(Path(request['compact_path']).read_text())
        if file_hash(request['compact_path']) != job['compact_sha256']:
            raise ValueError('Compact input differs')
        labels,_ = global_labels(original,{(50,s):cards.tables[s] for s in ('turn','river')},ks=(50,))
        data = projected(request,original,labels['50'])
        atomic_json(dest/'compact.json',data)
        request['compact_path'] = str(dest/'compact.json')
        atomic_json(dest/'request.json',request)
    for fold in (0,1):
        for name,path in policies(OUT/'runs'/f'fold-{fold}'):
            if '.equity' in name:
                raise ValueError('Unexpected policy name')
            if path.stem.startswith('equity.'):
                doc = json.loads(path.read_text())
                if doc['abstraction'] != HU20_EQUITY_SCHEMA or doc['card_tables'] != cards.sha256:
                    raise ValueError('Trained policy schema/tables differ')
                for group in doc['groups']:
                    if group['metric'] != 'equity-k50':
                        raise ValueError('Unexpected source group schema')
                    group['metric'] = 'eq50-fit0'
                atomic_json(OUT/'transport'/f'fold-{fold}'/path.parent.name/path.name,doc)
    for fold in (0,1):
        for _, path in policies(OUT/'pilot-training'/f'fold-{fold}'):
            final = OUT/'runs'/f'fold-{fold}'/path.parent.name/path.name
            if file_hash(final) != file_hash(path):
                raise ValueError('Deterministic 1M/3M training prefix differs')
    atomic_json(OUT/'visits.json',{str(f):visits(OUT/'runs'/f'fold-{f}') for f in (0,1)})


def evaluate(pilot=False):
    _,_,folds,jobs = inputs()
    for job in (jobs[:1] if pilot else jobs):
        fold = folds[job['spot']]
        dest = OUT/('pilot-scoring' if pilot else 'eval')/job['job']
        dest.mkdir(parents=True,exist_ok=False)
        trained = policies(OUT/'runs'/f'fold-{fold}')
        request = json.loads((OUT/'prepared'/job['job']/'request.json').read_text())
        ref = json.loads((OUT/'references/main-06/collect'/job['job']/'reference.json').read_text())
        request.update(pooling_phase='lock-only',max_iterations=0,
            reference_equilibrium_ev_chips=ref['reference_ev_chips'],
            reference_response_sha256=ref['original_response']['sha256'],pooling_measurements=[])
        for name,path in trained:
            equity = path.stem.startswith('equity.')
            transported = OUT/'transport'/f'fold-{fold}'/path.parent.name/path.name if equity else path
            request['pooling_measurements'].append({'metric':name,
                'projection_metric':'eq50-fit0' if equity else 'v1','policy_path':str(transported),'allow_missing':True})
        atomic_json(dest/'request.json',request)
        runtime = run_owned_tool(LOCK,dest/'request.json',dest/'solver',memory_bytes=request['memory_budget_bytes'],
            threads=6,seconds=request['seconds']+300,job_memory_bytes=7*1024**3)
        if runtime['status'] != 'completed':
            raise RuntimeError(runtime['failure'])
        metrics = [r for r in rows(dest/'solver/response.jsonl') if r['event']=='pooling_metric']
        if len(metrics)!=2*len(trained) or {(m['metric'],m['target_solver_seat']) for m in metrics}!={(n,s) for n,_ in trained for s in (0,1)}:
            raise ValueError('Incomplete/duplicate measurements')
        atomic_json(dest/'result.json',{'job':job,'fold':fold,'metrics':metrics,'runtime':runtime})
    if pilot:
        atomic_json(OUT/'scoring-quote.json',{'pilot_job':jobs[0]['job'],'seconds':runtime['elapsed_seconds'],
            'outcomes_used_for_quote':False,'root_count':40,'passes_per_root':1})


def report():
    from random import Random
    _,_,_,jobs = inputs()
    records=[]
    freeze=json.loads((OUT/'matched-freeze.json').read_text())
    for job in jobs:
        bench=json.loads((OUT/'eval'/job['job']/'result.json').read_text())
        collect=json.loads((OUT/'references/main-06/collect'/job['job']/'result.json').read_text())['metrics']
        relock=json.loads((OUT/'references/main-06/relock'/job['job']/'result.json').read_text())['metrics']
        row={'spot':job['spot'],'fold':bench['fold'],'B':seat_mean(collect,'e_bp'),'L':seat_mean(collect,'e_root_v1'),'P':seat_mean(relock,'e_cross_v1')}
        for m in bench['metrics']:
            name=m['metric']
            normalized='matched/'+name.split('/')[1] if name.startswith('iteration-'+str(freeze['iterations'][str(bench['fold'])])+'/') else name
            row[normalized]=seat_mean(bench['metrics'],name)
        records.append(row)
    if any(set(r)!=set(records[0]) for r in records):
        raise ValueError('Board measurements differ')
    rng=Random(202610050002)
    draws=[[rng.randrange(40) for _ in range(40)] for _ in range(2000)]
    def stat(fn):
        vals=sorted(fn([records[i] for i in d]) for d in draws)
        return {'mean':fn(records),'low':vals[49],'high':vals[1949]}
    estimates={k:stat(lambda rs,k=k:mean(r[k] for r in rs)) for k in records[0] if k not in ('spot','fold')}
    q={k:stat(lambda rs,k=k:(mean(r[k] for r in rs)-mean(r['P'] for r in rs))/(mean(r['B'] for r in rs)-mean(r['P'] for r in rs))) for k in estimates if k not in ('B','L','P')}
    contrast=stat(lambda rs:mean(r['matched/equity.average-opponent-sampled']-r['iteration-3000000/v1.average-opponent-sampled'] for r in rs))
    ten=stat(lambda rs:mean(r['iteration-10000000/equity.average-opponent-sampled']-r['iteration-10000000/v1.average-opponent-sampled'] for r in rs))
    outcome='pass' if contrast['high']<-.1 and ten['mean']<0 else 'fail' if contrast['low']>-.1 else 'inconclusive'
    atomic_json(OUT/'summary.json',{'boards':40,'estimates_bb':estimates,'placement_Q':q,'primary':contrast,'ten_million_difference':ten,'outcome':outcome,'rows':records,'global_k50_witness_E':.3917,'bootstrap_seed':202610050002,'draws':2000})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','train','freeze','adapters','pilot','evaluate','report'))
    p.add_argument('--variant',choices=('v1','equity'))
    p.add_argument('--fold',type=int,choices=(0,1))
    p.add_argument('--final',action='store_true')
    a=p.parse_args()
    if a.command=='train':train(a.variant,a.fold,a.final)
    elif a.command=='pilot':evaluate(True)
    else:globals()[a.command]()

if __name__=='__main__':main()
