"""One fixed-seed, M4-only overbet-menu campaign; no failed-hand retries."""
import argparse
import gc
import gzip
import json
import math
from pathlib import Path
import shutil
from statistics import stdev
import subprocess
import sys
from time import perf_counter, time

from scripts import evaluate_hu100_direct as direct
from scripts import evaluate_native_hu100_baseline as scripted
from scripts.audit_native_hu100_baseline import audit as scripted_audit
from scripts.audit_native_hu_checkpoint import inspect
from scripts.guard_hu100_k50_v052 import guard
from scripts.reserve_match_roots import reserve
from scripts.run_hu100_k50_v052 import ReferencedRegistry, configuration, estimate, scripted_values
from src.blueprint.abstraction import HU100_EQUITY_MENU_V2_SCHEMA, HU100_EQUITY_SCHEMA
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/hu100-k50-v2'
BINARY = ROOT/'native/hu20-trainer/target/release/hu20-trainer'
REPO = 'dberweger2017/deepcfr-texas-no-limit-holdem-6-players'
SOURCE_BASE = 'f4049b545fd106c8c3ba93d6045a417ef497306e'
ROOTS = dict(smoke=20261013170101, pilot=20261013170102, primary=20261013170103,
             secondary=20261013170104, scripted=20261013170105)
SEED = 2026100601
ARMS = dict(k50_v2=('k50-v2', None), v052_on=('v052', dict(max_states=512, max_events=128)))
TABLE_BYTES = 1_384_000_000
put = direct.put


def read(path):
    return json.loads(Path(path).read_text())


def source():
    if Path.cwd() != ROOT or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Clean committed campaign checkout required')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def pin(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_hash(path))


def command(args):
    subprocess.run(list(map(str,args)), check=True)


def header(path):
    with gzip.open(path,'rt') as stream:
        return json.loads(next(stream))


def restore(archive, member, destination):
    import hashlib
    from zipfile import ZipFile
    if file_hash(Path(archive['path'])) != archive['sha256']:
        raise ValueError('Indexed archive hash differs')
    with ZipFile(archive['path']) as zipped:
        raw = zipped.read('ARCHIVE-MANIFEST.json')
        if hashlib.sha256(raw).hexdigest() != archive['manifest_sha256']:
            raise ValueError('Indexed manifest hash differs')
        expected = json.loads(raw)['members'][member]
        destination.parent.mkdir(parents=True,exist_ok=True)
        with zipped.open(member) as origin, destination.open('xb') as target:
            shutil.copyfileobj(origin,target,1024**2)
    actual = pin(destination)
    if (actual['bytes'],actual['sha256']) != (expected['bytes'],expected['sha256']):
        raise ValueError('Restored member hash differs')
    return dict(archive=archive,member=member,restored=actual)


def prepare(pr):
    reserve(list(ROOTS.values()), pr, BASE/'root-reservation.json')
    owners = {}
    for n in (163, 246, 254, 256, 258):
        owners[n] = json.loads(subprocess.check_output(['gh','pr','view',str(n),'--repo',REPO,
            '--json','state,mergedAt,url'],text=True))
        if owners[n]['state'] != 'MERGED':
            raise ValueError('Input owner must be merged: '+str(n))
    prior = read(ROOT/'docs/reports/hu100-k50-vs-v052-artifacts/canonical-plan.json')
    tables = BASE/'tables'; tables.mkdir(exist_ok=False)
    table_pins = {}
    for street, expected in prior['tables'].items():
        original = Path(expected['path'])
        if pin(original) != expected:
            raise ValueError('Pinned original table changed')
        dest = tables/original.name
        shutil.copyfile(original, dest)
        table_pins[street] = pin(dest)
        if table_pins[street]['sha256'] != EQUITY_K50_SHA256[street]:
            raise ValueError('Restored table hash differs')
    lineage = read(ROOT/'docs/reports/hu100-k50-2b-artifacts/canonical-plan.json')
    v1 = dict(lineage['pairs']['secondary'][0], name='k50-v1-1b', card_buckets=str(tables))
    v052 = dict(prior['models']['v052'])
    index = read(ROOT/'docs/reports/hu100-k50-2b-artifacts/model-index.json')
    retrieval = {}
    if not Path(v1['path']).exists():
        restored = BASE/'models/1000000000.average.gz'
        retrieval['k50-v1-1b'] = restore(index['archives']['averages'],'averages/1000000000.average.gz',restored)
        v1['path'] = str(restored)
    for spec in (v1, v052):
        if pin(Path(spec['path'])) != {k:spec[k] for k in ('path','bytes','sha256')}:
            raise ValueError('Pinned model changed: '+spec['name'])
    release = Path(v052['path']).parent
    verified = subprocess.check_output([sys.executable,str(release/'verify_hu100_bundle.py'),str(release),
        '--expect-source','3a3bc4218bab96637d6a27a69b73383ee8bb196c','--require-publication'],text=True)
    original_checkpoint = BASE/'models/lineage-fresh20m.gz'
    retrieval['lineage_checkpoint'] = restore(index['archives']['full'],'research/gates/fresh20m.gz',original_checkpoint)
    h = header(original_checkpoint)
    if (h['config']['seed'] != SEED or h['config']['roots_per_seat'] != 1
            or h['average_rule'] != 'opponent-sampled' or 'training_options' in h):
        raise ValueError('Lineage recipe differs')
    put(BASE/'inputs.json', dict(models={'k50-v1-1b':v1, 'v052':v052}, tables=table_pins,
        owners=owners, retrieval=retrieval, release_verification=verified, lineage_header=h, lineage_checkpoint=pin(original_checkpoint),
        lineage_plan_sha256=file_hash(ROOT/'docs/reports/hu100-k50-2b-artifacts/canonical-plan.json'),
        source=source(), recipe=dict(stack_bb=100, small_blind=50, big_blind=100, rake=0, seed=SEED,
        roots_per_seat=1, weighting='linear', average_rule='opponent-sampled', menu='v2',
        schema=HU100_EQUITY_MENU_V2_SCHEMA)))


def train_args(nodes, out, telemetry, cap, resume=None, menu='v2'):
    args = [BINARY,'train','--stack-bb',100,'--seed',SEED,'--roots-per-seat',1,
        '--average-rule','opponent-sampled','--menu',menu,'--card-buckets',BASE/'tables',
        '--nodes',nodes,'--max-entries',cap,'--out',out,'--telemetry',telemetry]
    if resume:
        args += ['--resume',resume,'--resume-sha256',file_hash(resume)]
    return args


def gates():
    dest = BASE/'gates'; dest.mkdir(exist_ok=False)
    command([sys.executable,'-m','scripts.build_native_trainer','--receipt',BASE/'build-receipt.json'])
    # Dedicated child logs and 50 ms samples resolve short train/save phases;
    # the independent campaign guard retains its required five-second readings.
    import psutil
    measurements = {}
    for label, nodes, resume in [('fresh20m',20_000_000,None), ('prefix10m',10_000_000,None),
                               ('resumed20m',20_000_000,dest/'prefix10m.gz')]:
        args = train_args(nodes,dest/(label+'.gz'),dest/(label+'.telemetry.jsonl'),100_000_000,resume)
        samples = []
        with (dest/(label+'.log')).open('x') as log:
            child = subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT)
            process = psutil.Process(child.pid); created = process.create_time()
            while child.poll() is None:
                try:
                    if process.create_time() != created:
                        raise ValueError('Training process identity changed')
                    samples.append(dict(at=time(), rss=process.memory_info().rss))
                except psutil.NoSuchProcess:
                    break
                from time import sleep
                sleep(.05)
            if child.wait():
                raise ValueError('Recovery gate failed')
        put(dest/(label+'.rss.json'), samples)
        row = read_lines(dest/(label+'.telemetry.jsonl'))[-1]
        train = max(s['rss'] for s in samples if s['at'] < row['write_started'])
        save = max([row['process_rss_bytes_after_save'],
                    *[s['rss'] for s in samples if row['write_started'] <= s['at'] <= row['write_finished']]])
        entries = row['diagnostics']['entries']
        # BucketTable owns fully read vectors; their fixed bytes are reserved
        # separately in the owner's formula. Keep all remaining process overhead.
        measurements[label] = dict(train_peak_rss=train, save_peak_rss=save, entries=entries,
                                    fixed_table_bytes=TABLE_BYTES,
                                    bytes_per_entry=(max(train,save)-TABLE_BYTES)/entries)
        resolved = json.loads((dest/(label+'.log')).read_text().splitlines()[0])
        if resolved['options']['menu'] != 'v2' or HU100_EQUITY_MENU_V2_SCHEMA not in json.dumps(resolved):
            raise ValueError('Resolved menu/schema differs')
    fresh, resumed = pin(dest/'fresh20m.gz'), pin(dest/'resumed20m.gz')
    if (fresh['bytes'],fresh['sha256']) != (resumed['bytes'],resumed['sha256']):
        raise ValueError('20M fresh/resume compressed-byte mismatch')
    bpe = measurements['fresh20m']['bytes_per_entry']
    if bpe <= 0:
        raise ValueError('Invalid entry memory measurement')
    cap = math.floor((11_000_000_000-TABLE_BYTES-250_000_000)/(bpe*1.1))
    put(BASE/'gates-complete.json', dict(fresh=fresh, resumed=resumed, measurements=measurements,
        measured_bytes_per_entry=bpe, cap=cap, numerator=9_366_000_000, safety_factor=1.1,
        formula='floor((11000000000-1384000000-250000000)/(measured B/entry*1.1))',
        binary=pin(BINARY), status='verified'))


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def export_audit(checkpoint, directory, schema, name):
    h = header(checkpoint); actual = h['native_state']['completed_nodes']
    current = directory/(str(actual)+'.current.gz'); average = directory/(str(actual)+'.average.gz')
    command([BINARY,'export',checkpoint,'--current',current,'--average',average,'--zero-mass','uniform'])
    receipt = inspect(checkpoint,current,average,100,actual,expected_schema=schema)
    if receipt['audit']['all_nodes_verified'] != receipt['entries']:
        raise ValueError('Incomplete row audit')
    put(directory/(str(actual)+'.audit.json'),receipt)
    return dict(pin(average),name=name,format='holdem-hu100-stored-cfr-average-research-v1',
        abstraction=schema,card_buckets=str(BASE/'tables'),actual_nodes=actual,iteration=h['iteration'],
        entries=receipt['entries'],source_checkpoint_sha256=file_hash(checkpoint))


def training():
    qualified = read(BASE/'gates-complete.json')
    if pin(BINARY) != qualified['binary']:
        raise ValueError('Qualified binary changed')
    dest = BASE/'training'; dest.mkdir(exist_ok=False)
    args = train_args(1_000_000_000,dest/'requested-{nodes}.gz',dest/'telemetry.jsonl',qualified['cap'])
    rc = subprocess.run(list(map(str,[*args,'--milestones','500000000']))).returncode
    rows = read_lines(dest/'telemetry.jsonl')
    if rc not in (0,3) or not rows:
        raise ValueError('Training failed')
    if rc == 3 and (rows[-1]['diagnostics']['entries'] < qualified['cap'] or rows[-1]['stop_requested']):
        raise ValueError('Unexpected capacity-stop reason')
    specs = []
    for row in rows:
        old = Path(row['path']); checkpoint = dest/(str(row['completed_nodes'])+'.checkpoint.gz')
        if pin(old)['sha256'] != row['checkpoint_sha256'] or row['binary_sha256'] != qualified['binary']['sha256']:
            raise ValueError('Training telemetry hash differs')
        old.rename(checkpoint)
        specs.append(export_audit(checkpoint,dest,HU100_EQUITY_MENU_V2_SCHEMA,'k50-v2'))
    put(BASE/'treatment.json',dict(specs=specs,telemetry=rows,cap=qualified['cap'],returncode=rc,
        terminal=specs[-1], reached_1b=specs[-1]['actual_nodes'] >= 1_000_000_000))


def models():
    return dict(read(BASE/'inputs.json')['models'], **{'k50-v2':read(BASE/'treatment.json')['terminal']})


def direct_pair(pair, blocks, root, name, folder, revision):
    registry, original = direct.execute(pair,blocks,root,name,folder/'play',revision)
    replay = direct.audit(folder/'play',folder/'replay.json',cost_only=True)
    registry, repeat = direct.execute(pair,blocks,root,name,folder/'reproduction',revision,registry,folder/'play')
    del registry; gc.collect()
    return dict(blocks=blocks,load_seconds=original['load_or_validation_seconds'],play_seconds=original['play_seconds'],
        replay_seconds=replay['seconds'],reproduction_seconds=repeat['play_seconds'],
        reproduction_validation_seconds=repeat['load_or_validation_seconds'],paired_sd_bb_per_100=replay['paired_sd_bb_per_100'])


def scripted_arm(arm, blocks, root, folder, revision, model_set):
    name, option = ARMS[arm]; cfg = configuration(model_set[name],option,blocks)
    config = folder/(arm+'-config.json'); put(config,cfg)
    tick = perf_counter()
    registry = ReferencedRegistry(scripted.make_plan(cfg,'random',blocks,root),sorted_average_rows=True)
    loaded = perf_counter()-tick
    play = folder/arm/'play'
    original = scripted.execute(config,play,blocks,root,revision,registry=registry)
    replay = scripted_audit(play,folder/(arm+'-replay.json'),cost_only=True)
    repeat = scripted.execute(config,folder/arm/'reproduction',blocks,root,revision,registry=registry,reproduce=play)
    del registry; gc.collect()
    return dict(blocks=blocks,load_seconds=loaded,play_seconds=original['wall_seconds'],replay_seconds=replay['seconds'],
        reproduction_seconds=repeat['wall_seconds'],hands_replayed=replay['hands_replayed'],
        reproduced_all_hands_and_decisions=repeat['reproduced_all_hands_and_decisions'])


def smoke():
    # A tiny v1 witness is sufficient to exercise a mixed-schema match with
    # the actual terminal v2 model before loading the two expensive controls.
    dest = BASE/'smoke'; dest.mkdir(exist_ok=False)
    command(train_args(100_000,dest/'v1.gz',dest/'v1.telemetry.jsonl',100_000_000,menu='v1'))
    witness = export_audit(dest/'v1.gz',dest,HU100_EQUITY_SCHEMA,'v1-witness')
    terminal = models()['k50-v2']
    d = direct_pair([terminal,witness],2,ROOTS['smoke'],'mixed-v2-v1',dest/'direct',source())
    s = scripted_arm('k50_v2',2,ROOTS['smoke'],dest/'scripted',source(),{'k50-v2':terminal})
    put(BASE/'smoke-complete.json',dict(status='verified',direct=d,scripted=s,
        telemetry=True,independent_replay=True,exact_reproduction=True,mixed_v1_v2=True,
        terminal_sha256=terminal['sha256']))


def play(stage):
    revision = source(); model_set = models(); pilot = stage=='pilot'
    if read(BASE/'smoke-complete.json')['terminal_sha256'] != model_set['k50-v2']['sha256']:
        raise ValueError('Smoke model differs')
    if pilot:
        counts = dict(primary=1024,secondary=1024,scripted=64)
    else:
        plan = read(BASE/'canonical-plan.json'); posted = read(BASE/'plan-posted.json')
        if plan['source'] != revision or file_hash(BASE/'canonical-plan.json') != posted['sha256'] or plan['models'] != model_set:
            raise ValueError('Frozen source/model/posting differs')
        counts = plan['counts']
    folder = BASE/stage; quotes = {}
    for contrast, control in [('primary','v052'),('secondary','k50-v1-1b')]:
        quotes[contrast] = direct_pair([model_set['k50-v2'],model_set[control]],counts[contrast],
            ROOTS['pilot'] if pilot else ROOTS[contrast],contrast,folder/contrast,revision)
    for arm in ARMS:
        quotes[arm] = scripted_arm(arm,counts['scripted'],ROOTS['pilot'] if pilot else ROOTS['scripted'],
            folder/'scripted',revision,model_set)
    sd = None
    if pilot:
        sd = {o:stdev([a-b for a,b in zip(scripted_values(folder/'scripted/k50_v2/play',o,64),
            scripted_values(folder/'scripted/v052_on/play',o,64),strict=True)]) for o in scripted.OPPONENTS}
    put(BASE/(stage+'-verified.json'),dict(status='verified',quotes=quotes,paired_scripted_sds=sd,
        outcomes_inspected=False,source=revision,at=time()))


def freeze(pr):
    costs = read(BASE/'pilot-verified.json'); q = costs['quotes']
    elapsed = time()-read(BASE/'budget.json')['started']
    def project(primary):
        seconds = 0
        for c,n in [('primary',primary),('secondary',131072),*[(a,4096) for a in ARMS]]:
            r = q[c]
            seconds += r['load_seconds']+r.get('reproduction_validation_seconds',0)
            seconds += sum(r[k] for k in ('play_seconds','replay_seconds','reproduction_seconds'))*n/r['blocks']
        return elapsed+1.25*seconds+1800
    primary = 131072 if project(262144)>5*3600 else 262144
    if project(primary)>7*3600:
        raise ValueError('Fixed counts exceed whole-campaign deadline')
    reserve(list(ROOTS.values()),pr,BASE/'final-root-reservation.json')
    document = dict(pr=pr,source=source(),models=models(),tables=read(BASE/'inputs.json')['tables'],roots=ROOTS,
        counts=dict(primary=primary,secondary=131072,scripted=4096),arms=ARMS,opponents=scripted.OPPONENTS,
        pilot=costs,requested_projection_seconds=project(262144),projected_total_seconds=project(primary),
        projection_rule='elapsed + 1.25*(measured load/validation and scaled play/replay/reproduction) + 1800s closeout; halve primary once if requested projection exceeds 5h',
        decision_rule='recommend v0.5.3 candidate only if primary nominal paired Student-t95 lower>0 AND no scripted paired95 upper<-20 BB/100; owner decides',
        conditional_on='fixed models and training seed; no seed-population claim',translation_direct=None,
        no_failed_hand_reruns=True,no_outcome_extension=True,all_final_hands_replayed_before_readout=True,
        training=read(BASE/'treatment.json'),guard_rule='5s retained samples;60s sustained RSS>13GiB/critical pressure/free<8%/swap growth>4GB;disk<16GiB immediate;7h deadline')
    put(BASE/'canonical-plan.json',document)


def posted(pr):
    # Freeze is installed in the PR description: comments are reserved for
    # concrete issues with SOMA, as requested by the owner.
    plan = read(BASE/'canonical-plan.json')
    body = BASE/'frozen-pr-body.md'
    body.write_text('Does menu v2 remove K50’s pot-pressure regression? M4-only fixed-seed research; no release, tag, default change or merge.\n\n'
        'Canonical pre-outcome plan, SHA256 `'+file_hash(BASE/'canonical-plan.json')+'`.\n\n```json\n'+json.dumps(plan,indent=2,sort_keys=True)+'\n```\n')
    command(['gh','pr','edit',pr,'--repo',REPO,'--body-file',body])
    value = json.loads(subprocess.check_output(['gh','pr','view',str(pr),'--repo',REPO,'--json','body,url'],text=True))
    if value['body'] != body.read_text():
        raise ValueError('Canonical PR description readback differs')
    put(BASE/'plan-posted.json',dict(url=value['url'],sha256=file_hash(BASE/'canonical-plan.json'),readback_verified=True,at=time()))


def readout():
    plan = read(BASE/'canonical-plan.json'); verified = read(BASE/'final-verified.json')
    if verified['status'] != 'verified' or verified['source'] != plan['source']:
        raise ValueError('All final verification required before readout')
    for complete in (BASE/'final').rglob('complete.json'):
        done = read(complete)
        if done['status'] != 'complete':
            raise ValueError('Incomplete final run')
        pins = done['files'] if 'files' in done else read(complete.parent/'output-files.json')
        for name, expected in pins.items():
            if pin(complete.parent/name) != dict(path=str(complete.parent/name),**expected):
                raise ValueError('Final bytes changed after replay')
    results = {}
    for contrast in ('primary','secondary'):
        n = plan['counts'][contrast]; cells = {}
        with gzip.open(BASE/f'final/{contrast}/play/hands.jsonl.gz','rt') as stream:
            for row in map(json.loads,stream):
                cells[row['block'],row['rotation']] = row['candidate_chips']
        results[contrast] = estimate([(cells[b,0]+cells[b,1])/2 for b in range(n)])
        results[contrast]['coverage'] = read(BASE/f'final/{contrast}/replay.json')['coverage']
    n = plan['counts']['scripted']
    values = {(arm,o):scripted_values(BASE/f'final/scripted/{arm}/play',o,n) for arm in ARMS for o in scripted.OPPONENTS}
    absolute = {arm:{o:estimate(values[arm,o]) for o in scripted.OPPONENTS} for arm in ARMS}
    diffs = {o:estimate([a-b for a,b in zip(values['k50_v2',o],values['v052_on',o],strict=True)]) for o in scripted.OPPONENTS}
    severe = {o:d['ci95'][1]<-20 for o,d in diffs.items()}
    put(BASE/'summary.json',dict(status='verified',**results,scripted_absolute=absolute,scripted_differences=diffs,
        severe_regressions=severe,recommend_candidate=results['primary']['ci95'][0]>0 and not any(severe.values()),
        fraction_of_prior_pot_gap_closed=(diffs['pot_pressure']['bb_per_100']+85)/85,
        gap_closed_scope='descriptive only; different deals from #254',all_hands_replayed_and_reproduced=True,
        canonical_plan_sha256=file_hash(BASE/'canonical-plan.json')))


def archive(pr):
    from scripts.archive_hu100_4b_seed_ladder import seal
    owner = json.loads(subprocess.check_output(['gh','pr','view',str(pr),'--repo',REPO,'--json','state,headRefName,url'],text=True))
    if owner['headRefName'] != subprocess.check_output(['git','branch','--show-current'],text=True).strip():
        raise ValueError('Owning PR branch differs')
    if not (BASE/'summary.json').exists() and not (BASE/'failure.json').exists():
        raise ValueError('Final or stopped evidence required')
    files = []
    for path in sorted(BASE.rglob('*')):
        rel = path.relative_to(BASE)
        if not path.is_file() or path.is_symlink() or rel.parts[0] in ('tables','models') or rel.parts[:2]==('guards','archive') or path.name.startswith('archive-receipt'):
            continue
        files.append((path,'research/'+rel.as_posix()))
    files.append((BINARY,'research/bin/hu20-trainer'))
    for name in subprocess.check_output(['git','diff','--name-only',SOURCE_BASE,'HEAD'],text=True).splitlines():
        path = ROOT/name
        if path.is_file() and path.stat().st_size <= 1024**2:
            files.append((path,'source/'+name))
    cloud = Path.home()/f'Local/Research-Cloud/PR-{pr}-hu100-k50-v2'
    metadata = dict(pr=pr,owner=owner,source=source(),inputs=read(BASE/'inputs.json'),originals_retained=True)
    terminal = read(BASE/'treatment.json')['terminal'] if (BASE/'treatment.json').exists() else None
    if terminal:
        model = Path(terminal['path'])
        put(BASE/'average-archive-receipt.json',seal([(model,'averages/'+model.name)],cloud/'hu100-k50-v2-terminal-average.zip',metadata))
    put(BASE/'archive-receipt.json',seal(files,cloud/'hu100-k50-v2-evidence.zip',metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','gates','training','smoke','pilot','freeze','posted','final','readout','archive','guard'))
    parser.add_argument('--pr',type=int,required=True)
    parser.add_argument('--worker')
    args = parser.parse_args(); BASE.mkdir(parents=True,exist_ok=True)
    if args.stage == 'guard':
        if not (BASE/'budget.json').exists():
            put(BASE/'budget.json',dict(started=time(),deadline=time()+7*3600))
        guard(BASE,args.worker,[sys.executable,'-m','scripts.run_hu100_k50_v2',args.worker,'--pr',str(args.pr)],
            sample_seconds=5,deadline=read(BASE/'budget.json')['deadline'])
    elif args.stage in ('prepare','freeze','posted','archive'):
        globals()[args.stage](args.pr)
    elif args.stage in ('pilot','final'):
        play(args.stage)
    else:
        globals()[args.stage]()


if __name__ == '__main__':
    main()
