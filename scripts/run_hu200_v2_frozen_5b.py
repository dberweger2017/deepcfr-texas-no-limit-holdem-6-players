"""Single-menu HU200 experiment; all outcomes sealed through replay/reproduction."""
import argparse
import csv
import fcntl
import gzip
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from time import sleep, time

import psutil

from scripts import evaluate_hu100_direct as direct
from scripts import run_hu100_k50_v2 as memory_gate
from scripts import run_hu200_frozen_evaluation as evaluation
from scripts.archive_hu100_4b_seed_ladder import seal
from scripts.reserve_match_roots import reserve
from src.blueprint.abstraction import HU200_EQUITY_MENU_V2_SCHEMA, HU200_EQUITY_SCHEMA
from src.blueprint.average import AveragePolicy
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.diagnostics.cfr_average import audit, checked_header
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/hu200-v2-frozen-5b'
BINARY = ROOT/'native/hu20-trainer/target/release/hu20-trainer'
PR = 270
SEED = 2026101302
TABLE_BYTES = 1_384_000_000
V1 = '43aa3b3cd3c7365c5be1b177dcc8a7c04e01c8470403c38327aa845922726512'
MAIN = 'e2bcdc63'
# Fixed October 14 22:00 Madrid closeout, 21:30 science end.
END = 1792008000
SCIENCE_END = END-1800
ROOTS = dict(direct_pilot=20261014270001, panel_pilot=20261014270002,
             direct=20261014270003, panel=20261014270004)
put = direct.put
read = evaluation.read
pin = memory_gate.pin
header = memory_gate.header


def evaluation_spec(path, name, schema, tables, receipt):
    meta = header(path)
    if (receipt['status'] != 'verified' or receipt['all_nodes_verified'] != receipt['counts']['entries']
            or receipt['average_sha256'] != file_hash(path)
            or receipt['checkpoint_sha256'] != meta['source_checkpoint_sha256']
            or meta['checkpoint_header']['abstraction'] != schema):
        raise ValueError('Audited evaluation model differs')
    return dict(pin(path), name=name, abstraction=schema, format=meta['format'],
                card_buckets=None if tables is None else str(tables), entries=receipt['all_nodes_verified'],
                iteration=meta['checkpoint_header']['iteration'],
                source_checkpoint_sha256=meta['source_checkpoint_sha256'])


def command(args, allowed=(0,)):
    result = subprocess.run(list(map(str, args)), timeout=max(1, SCIENCE_END-time()))
    if result.returncode not in allowed:
        raise RuntimeError('Worker exit '+str(result.returncode))
    return result.returncode


def prepare():
    owners = {str(n): json.loads(evaluation.gh('pr','view',n,'--json','state,url,headRefOid'))
              for n in (163,250,265,268)}
    if any(x['state'] != 'MERGED' for x in owners.values()):
        raise ValueError('Input owners must be merged before copying')
    original = ROOT/'docs/reports/hu200-k50-first-artifacts/inputs.json'
    provenance = read(original)
    tables = BASE/'tables'; tables.mkdir(exist_ok=False)
    pins = {}
    for street, expected in provenance['tables'].items():
        source = Path(expected['path'])
        if source.stat().st_size != expected['bytes'] or file_hash(source) != EQUITY_K50_SHA256[street]:
            raise ValueError('Pinned table differs')
        target = tables/source.name
        shutil.copy2(source,target)
        pins[street] = dict(original=expected, copy=pin(target), retrieval_command=['cp',str(source),str(target)])
        if pins[street]['copy']['sha256'] != expected['sha256']:
            raise ValueError('Copied table differs')
    v1 = Path.home()/'Local/hu200-k50-frozen-20261013/results/hu200-k50-frozen/frozen/5000000000.average.gz'
    index = read(ROOT/'docs/reports/hu200-k50-frozen-artifacts/model-index.json')
    if v1.stat().st_size != 2756501598 or file_hash(v1) != V1:
        raise ValueError('Frozen v1 5B average differs')
    h = header(v1)
    if h['checkpoint_header']['abstraction'] != HU200_EQUITY_SCHEMA:
        raise ValueError('Wrong baseline schema')
    baseline_audit = read(ROOT/'docs/reports/hu200-k50-frozen-artifacts/5000000000.audit.json')
    put(BASE/'inputs.json',dict(owners=owners, source=evaluation.revision(), tables=pins,
        table_provenance=provenance, baseline=evaluation_spec(v1,'v1-frozen-5b',HU200_EQUITY_SCHEMA,tables,baseline_audit),
        baseline_index=index, baseline_audit=baseline_audit,
        recipe=dict(seed=SEED,stack_bb=200,blinds=[50,100],rake=0,roots_per_seat=1,
                    average_rule='opponent-sampled',weighting='linear',menu='v2',
                    schema=HU200_EQUITY_MENU_V2_SCHEMA)))
    reserve(list(ROOTS.values()), PR, BASE/'root-reservation.json')


def train_args(nodes, out, telemetry, cap, resume=None):
    args = [BINARY,'train','--stack-bb',200,'--seed',SEED,'--roots-per-seat',1,
            '--average-rule','opponent-sampled','--menu','v2','--card-buckets',BASE/'tables',
            '--nodes',nodes,'--max-entries',cap,'--recovery','--out',out,'--telemetry',telemetry]
    if resume:
        args += ['--resume',resume,'--resume-sha256',file_hash(resume)]
    return args


def gates():
    # Reuse the published train/save measurement convention with the HU200 recipe.
    memory_gate.BASE, memory_gate.BINARY = BASE, BINARY
    memory_gate.SEED = SEED
    memory_gate.HU100_EQUITY_MENU_V2_SCHEMA = HU200_EQUITY_MENU_V2_SCHEMA
    memory_gate.train_args = train_args
    memory_gate.gates()


def train():
    gate = read(BASE/'gates-complete.json')
    if pin(BINARY) != gate['binary']:
        raise ValueError('Qualified executable differs')
    dest = BASE/'training'; dest.mkdir(exist_ok=False)
    # Save the exact freeze parent; no update occurs between cap and frozen resume.
    args = train_args(5_000_000_000,dest/'freeze.gz',dest/'cap.telemetry.jsonl',gate['cap'])
    put(BASE/'cap-command.json',dict(args=list(map(str,args))))
    rc = command([*args,'--max-seconds',3600,'--stop-file',BASE/'native-stop.json'],allowed=(3,))
    row = memory_gate.read_lines(dest/'cap.telemetry.jsonl')[-1]
    if row['diagnostics']['entries'] < gate['cap'] or row['stop_requested']:
        raise ValueError('Expected complete-iteration cap stop')
    actual_cap = row['diagnostics']['entries']
    args = train_args(5_000_000_000,dest/'{nodes}.gz',dest/'frozen.telemetry.jsonl',actual_cap,dest/'freeze.gz')
    # Eight hours leaves ample evaluation/archival reserves at the morning start.
    args += ['--freeze-at-cap','--milestones','2000000000','--max-seconds',28800,
             '--stop-file',BASE/'native-stop.json']
    put(BASE/'frozen-command.json',dict(args=list(map(str,args)),freeze_nodes=row['completed_nodes'],
                                      freeze_iteration=row['iteration'],actual_cap=actual_cap))
    command(args)
    rows = memory_gate.read_lines(dest/'frozen.telemetry.jsonl')
    if rows[-1]['completed_nodes'] < 5_000_000_000 or any(r['diagnostics']['entries'] != actual_cap for r in rows):
        raise ValueError('5B not reached or frozen table changed')
    put(BASE/'training-complete.json',dict(status='verified',freeze=row,frozen=rows,cap_returncode=rc))


def exports():
    row = read(BASE/'training-complete.json')['frozen'][-1]
    checkpoint = Path(row['path'])
    if file_hash(checkpoint) != row['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    current = checkpoint.with_suffix('.current.gz'); average = checkpoint.with_suffix('.average.gz')
    command([BINARY,'export',checkpoint,'--current',current,'--average',average,'--zero-mass','uniform'])
    h = header(checkpoint)
    spec = dict(name='hu200-v2-frozen-5b',seed=SEED,iteration=h['iteration'],
                checkpoint_sha256=file_hash(checkpoint),sha256=file_hash(current))
    checked_header(h,spec,expected_schema=HU200_EQUITY_MENU_V2_SCHEMA)
    receipt = audit(checkpoint,current,average,spec,file_hash(average),expected_schema=HU200_EQUITY_MENU_V2_SCHEMA)
    if receipt['all_nodes_verified'] != row['diagnostics']['entries']:
        raise ValueError('Incomplete every-row audit')
    put(BASE/'5b.audit.json',dict(receipt,actual_nodes=row['completed_nodes'],
        files={k:pin(p) for k,p in [('checkpoint',checkpoint),('current',current),('average',average)]}))
    meta = header(average)
    put(BASE/'treatment.json',dict(pin(average),name='v2-frozen-5b',abstraction=HU200_EQUITY_MENU_V2_SCHEMA,
        format=meta['format'],card_buckets=str(BASE/'tables'),entries=row['diagnostics']['entries'],
        iteration=h['iteration'],source_checkpoint_sha256=file_hash(checkpoint)))


def select_counts(costs, remaining):
    options = []
    for blocks, panel in ((131072,16384),(65536,8192)):
        seconds = 1.25*(evaluation.quote(costs['direct'],blocks)+
            sum(evaluation.quote(costs['panel'][op],panel) for op in direct.PANEL_OPPONENTS))
        option = dict(direct=blocks,panel=panel,projected_seconds=seconds,fits=seconds<=remaining)
        options.append(option)
        if option['fits']:
            return option, options
    raise ValueError('Neither pre-outcome count option fits the fixed deadline')


def recommendation(panel):
    flags = {op: row['contrast']['ci95'][1] < -20 for op, row in panel.items()}
    if set(panel) != set(direct.PANEL_OPPONENTS):
        raise ValueError('All seven panel contrasts are required')
    return panel['pot_pressure']['contrast']['ci95'][0] > 0 and not any(flags.values()), flags


def evaluation_stage():
    evaluation.SCIENCE_END = SCIENCE_END
    pair = [read(BASE/'treatment.json'),read(BASE/'inputs.json')['baseline']]
    if (BASE/'baseline-spec-corrected.json').exists():
        pair[1] = read(BASE/'baseline-spec-corrected.json')
    for model in pair:
        if (BASE/('cache-'+model['name']+'.json')).exists():
            continue
        tick = time()
        policy = AveragePolicy(model['path'],model['sha256'],expected_schema=model['abstraction'],
                               card_buckets=Path(model['card_buckets']),sorted_rows=True,cache=BASE/'policy-cache')
        put(BASE/('cache-'+model['name']+'.json'),dict(seconds=time()-tick,description=policy.description,no_hands=True))
        del policy
        import gc
        gc.collect()
    label = 'pilot-direct-cost' if (BASE/'pilot-direct').exists() else 'pilot-direct'
    costs = dict(direct=evaluation.run_pair(BASE,pair,1024,ROOTS['direct_pilot'],label,pilot=True),panel={})
    for op in direct.PANEL_OPPONENTS:
        costs['panel'][op] = evaluation.run_pair(BASE,pair,64,ROOTS['panel_pilot'],'pilot-panel-'+op,op,True)
    put(BASE/'pilot-costs.json',costs)
    counts, options = select_counts(costs,SCIENCE_END-time())
    plan = dict(source=evaluation.revision(),models=pair,roots=ROOTS,counts=counts,
                projection_options=options,pilot=costs,frozen_at=time(),
                translation=None,missing_key_fallback='uniform',zero_mass='uniform',
                opponents=direct.PANEL_OPPONENTS,deadline_utc='2026-10-14T20:00:00Z',
                science_end_utc='2026-10-14T19:30:00Z',projection_multiplier=1.25,
                decision_rule='Recommend Slumbot iff pot_pressure contrast lower95>0 and no panel contrast upper95<-20 BB/100',
                direct_role='Descriptive only; different-menu missing keys prevent using it for the decision')
    put(BASE/'canonical-plan.json',plan)
    original = json.loads(evaluation.gh('pr','view',PR,'--json','body'))['body']
    # Post the complete canonical JSON, including measured costs and paired SDs,
    # so the readback covers the plan whose SHA256 admits final hands.
    body = original+'\n\n## Canonical plan before final hands\n\nSHA256 `'+file_hash(BASE/'canonical-plan.json')+'`.\n\n```json\n'+json.dumps(plan,indent=2,sort_keys=True)+'\n```\n'
    body_file = BASE/'canonical-pr-body.md'; body_file.write_text(body)
    evaluation.gh('pr','edit',PR,'--body-file',body_file)
    if json.loads(evaluation.gh('pr','view',PR,'--json','body'))['body'] != body:
        raise ValueError('Canonical plan readback differs')
    put(BASE/'plan-posted.json',dict(sha256=file_hash(BASE/'canonical-plan.json'),readback_verified=True,at=time()))
    # Panel is the primary benchmark; all verification precedes a global readout.
    for op in direct.PANEL_OPPONENTS:
        evaluation.run_pair(BASE,pair,counts['panel'],ROOTS['panel'],'panel-'+op,op)
    evaluation.run_pair(BASE,pair,counts['direct'],ROOTS['direct'],'direct')
    put(BASE/'all-final-verified.json',dict(status='verified',at=time()))
    summaries = dict(panel={},direct=evaluation.estimate(evaluation.series(BASE/'direct',counts['direct'])['candidate']))
    cells = []
    for folder in [*['panel-'+op for op in direct.PANEL_OPPONENTS],'direct']:
        replay = read(BASE/folder/'replay.json')
        if replay['status'] != 'verified' or not read(BASE/folder/'reproduction/complete.json')['reproduced_all_hands_and_decisions']:
            raise ValueError('Final verification barrier incomplete')
        for phase in ('play','reproduction'):
            for name, expected in read(BASE/folder/phase/'complete.json')['files'].items():
                p = BASE/folder/phase/name
                if p.stat().st_size != expected['bytes'] or file_hash(p) != expected['sha256']:
                    raise ValueError('Verified final bytes changed')
        if folder.startswith('panel-'):
            op = folder[6:]; arms = evaluation.series(BASE/folder,counts['panel'],panel=True)
            summaries['panel'][op] = dict(arms={a:evaluation.estimate(v) for a,v in arms.items()},
                contrast=evaluation.estimate([a-b for a,b in zip(arms['candidate'],arms['baseline'],strict=True)]))
        for key,cell in replay['coverage'].items():
            policy,street = key.split('/'); n = cell['decisions']; missing = cell['lookup'].get('missing-key',0)
            bands = dict(cell['visit_bands']); bands['0'] = bands.get('0',0)-missing; known = n-missing
            if bands['0'] < 0 or sum(bands.values()) != known:
                raise ValueError('Coverage denominator differs')
            cells.append(dict(contrast=folder,policy=policy,street=street,decisions=n,known_decisions=known,
                missing_key_share=missing/n,zero_mass_share=cell['lookup'].get('zero-mass',0)/n,
                **{'known_visit_'+k:bands.get(k,0) for k in ('0','1-9','10-99','100+')},
                **{'known_share_'+k:bands.get(k,0)/known if known else None for k in ('0','1-9','10-99','100+')}))
    passes, flags = recommendation(summaries['panel'])
    summaries.update(status='verified',severe_panel_regressions=flags,
        recommend_slumbot=passes,
        all_final_hands_independently_replayed_and_exactly_reproduced=True,
        canonical_plan_sha256=file_hash(BASE/'canonical-plan.json'))
    put(BASE/'summary.json',summaries)
    with (BASE/'coverage.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)


def archive():
    owner=json.loads(evaluation.gh('pr','view',PR,'--json','state,url,headRefOid'))
    files=[(p,'research/'+p.relative_to(BASE).as_posix()) for p in sorted(BASE.rglob('*'))
           if p.is_file() and not p.is_symlink() and not any(p.is_relative_to(BASE/x) for x in ('tables','policy-cache','guards/operations/archive','continuation-guards/operations/archive'))
           and p.name not in ('archive-receipt.json','average-receipt.json','guard-held.json')]
    files += [(BINARY,'research/bin/hu20-trainer')]
    files += [(ROOT/p,'source/'+p) for p in ('scripts/run_hu200_v2_frozen_5b.py','docs/hu200-v2-frozen-5b-protocol.md')]
    dest=Path.home()/f'Local/Research-Cloud/PR-{PR}-hu200-v2-frozen-5b'
    if shutil.disk_usage(dest.parent).free-sum(p.stat().st_size for p,_ in files)-4*1024**3 < 16*1024**3:
        raise ValueError('Archive would consume disk guard reserve')
    result=seal(files,dest/'hu200-v2-frozen-5b-M4-20261014.zip',dict(source=evaluation.revision(),owner=owner,
        inputs=read(BASE/'inputs.json'),originals_retained=True,excluded_active_archive_guard=True))
    put(BASE/'archive-receipt.json',result)
    model=read(BASE/'treatment.json')
    put(BASE/'average-receipt.json',seal([(Path(model['path']),'averages/5000000000.average.gz')],
        dest/'hu200-v2-frozen-5b-average-M4-20261014.zip',dict(source=evaluation.revision(),owner=owner,model=model)))


def run():
    if subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()!='Apple M4':
        raise ValueError('Only M4 is authorized')
    BASE.mkdir(parents=True,exist_ok=False)
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as one, (Path.home()/'Local/.hu100-m4-research.lock').open('a+') as two:
        for f in (one,two): fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        put(BASE/'locks-acquired.json',dict(at=time(),locks=[one.name,two.name]))
        own={psutil.Process().pid,*[p.pid for p in psutil.Process().parents()]}
        for p in psutil.process_iter(['pid','name','cmdline'],ad_value=None):
            if p.pid in own: continue
            cmd=' '.join((p.info['cmdline'] or [])[1:])
            if p.info['name']=='hu20-trainer' or any(x in cmd for x in ('scripts.run_hu','scripts.evaluate_hu','campaign.py')):
                raise ValueError('Other campaign process active: '+str(p.pid))
        for stage in ('prepare','gates','train','exports','evaluation_stage','archive'):
            if time()>=SCIENCE_END and stage!='archive': raise TimeoutError('Fixed science deadline')
            args=[sys.executable,'-m','scripts.guard_m4_research','--stage',BASE/'guards','--name',stage,
                  *(['--preserve'] if stage=='archive' else []),'--',sys.executable,'-m','scripts.run_hu200_v2_frozen_5b',stage]
            rc=subprocess.run(list(map(str,args))).returncode
            if rc: raise RuntimeError('Guarded stage failed: '+stage)
    put(BASE/'detached-closeout.json',dict(at=time(),within_deadline=time()<=END))


def continue_admission():
    # Only the observed metadata admission failure is recoverable here. No hand
    # may exist, and the original stop/failure evidence is never rewritten.
    failure = (BASE/'pilot-direct.log').read_text()
    if "KeyError: 'entries'" not in failure or list(BASE.rglob('hands.jsonl.gz')):
        raise ValueError('Continuation requires the recorded pre-hand metadata failure')
    original = read(BASE/'inputs.json')
    spec = evaluation_spec(Path(original['baseline']['path']),'v1-frozen-5b',HU200_EQUITY_SCHEMA,
                           BASE/'tables',original['baseline_audit'])
    put(BASE/'baseline-spec-corrected.json',spec)
    put(BASE/'admission-correction.json',dict(at=time(),no_hands_before_correction=True,
        source=evaluation.revision(),original_inputs_sha256=file_hash(BASE/'inputs.json'),
        original_failure_preserved=True,added_fields=['entries','iteration','source_checkpoint_sha256'],
        original_roots_unused=True,scientific_retries=0))
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as one, (Path.home()/'Local/.hu100-m4-research.lock').open('a+') as two:
        for f in (one,two): fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        continuation = BASE/'continuation-guards'; continuation.mkdir(exist_ok=False)
        shutil.copyfile(BASE/'guards/baseline.json',continuation/'baseline.json')
        for stage in ('evaluation_stage','archive'):
            command([sys.executable,'-m','scripts.guard_m4_research','--stage',continuation,'--name',stage,
                     '--',sys.executable,'-m','scripts.run_hu200_v2_frozen_5b',stage])
    put(BASE/'detached-closeout.json',dict(at=time(),within_deadline=time()<=END))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('run','prepare','gates','train','exports','evaluation_stage','archive','continue_admission'))
    args=parser.parse_args()
    if args.stage=='run': run()
    else:
        try: globals()[args.stage]()
        except BaseException as exc:
            put(BASE/(args.stage+'-failure.json'),dict(failure=repr(exc),at=time()))
            raise


if __name__=='__main__': main()
