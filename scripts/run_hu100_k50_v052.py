"""Frozen direct and three-arm scripted evidence; never trains or promotes."""
import argparse
from dataclasses import asdict
import gc
import gzip
import json
import math
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import sys
from time import perf_counter, time
from zipfile import ZipFile

from scipy.stats import t
from scripts import evaluate_hu100_direct as direct
from scripts import evaluate_native_hu100_baseline as scripted
from scripts.audit_native_hu100_baseline import audit as scripted_audit
from scripts.guard_hu100_k50_v052 import guard
from scripts.reserve_match_roots import reserve
from src.arena.registry import PolicyRegistry
from src.arena.schedule import build_schedule, digest
from src.blueprint.abstraction import HU100_SCHEMA, HU100_EQUITY_SCHEMA
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.policies.files import file_hash
from src.policies.hu100_bundle import RELEASES

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/hu100-k50-vs-v052'
ROOTS = {'pilot':20261013101101,'primary':20261013101102,'scripted':20261013101103}
ARMS = {'k50_off':('k50',None), 'v052_on':('v052',{'max_states':512,'max_events':128}), 'v052_off':('v052',None)}
REPO = 'dberweger2017/deepcfr-texas-no-limit-holdem-6-players'
put = direct.put


def read(path):
    return json.loads(Path(path).read_text())


def revision():
    if Path.cwd()!=ROOT or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Clean committed campaign checkout required')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def pin(path):
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':file_hash(path)}


class ReferencedRegistry(PolicyRegistry):
    def snapshot(self, output):
        # Indexed policies remain read-only, including their link metadata.
        put(output/'model-references.json', {n:pin(m.source_path) for n,m in self.models.items()})


def prepare(pr):
    reserve(list(ROOTS.values()),pr,BASE/'root-reservation.json')
    states = {}
    for number in (163,226,240,246):
        states[number] = json.loads(subprocess.check_output(['gh','pr','view',str(number),'--repo',REPO,'--json','state,mergedAt,url'],text=True))
        if states[number]['state']!='MERGED':
            raise ValueError('Input owner is open: '+str(number))
    index = read(ROOT/'docs/reports/hu100-k50-2b-artifacts/model-index.json')
    asset = next(a for a in index['models'] if a['kind']=='average' and a['actual_nodes']>=2_000_000_000)
    original = Path.home()/'Local/hu100-k50-2b-20261012/results/hu100-k50-2b/training/2000000000.average.gz'
    provenance = {'original_path':str(original),'indexed':asset,'archive':index['archives']['averages']}
    if original.exists() and original.stat().st_size==asset['bytes'] and file_hash(original)==asset['sha256']:
        path = original; provenance['method'] = 'hash-verified read-only original'
    else:
        archive = Path(index['archives']['averages']['path'])
        if file_hash(archive)!=index['archives']['averages']['sha256']:
            raise ValueError('Average archive hash differs')
        member = 'averages/2000000000.average.gz'
        dest = BASE/'models'; dest.mkdir(exist_ok=False)
        with ZipFile(archive) as z:
            manifest = z.read('ARCHIVE-MANIFEST.json')
            import hashlib
            if hashlib.sha256(manifest).hexdigest()!=index['archives']['averages']['manifest_sha256']:
                raise ValueError('Average manifest hash differs')
            entry = json.loads(manifest)['members'][member]
            if entry['bytes']!=asset['bytes'] or entry['sha256']!=asset['sha256']:
                raise ValueError('Average manifest member differs')
            path = dest/'2000000000.average.gz'
            with z.open(member) as src, path.open('xb') as target:
                shutil.copyfileobj(src,target,1024**2)
        provenance.update(method='whole/archive/member verified ZIP restoration',member=member)
    if path.stat().st_size!=2_004_530_099 or file_hash(path)!='dbc6772a1f3a2b5d39dea7b01af163d508211cc5c593a2b8e6fba2ceaab828d8':
        raise ValueError('K50 input mismatch')
    k50_path = path
    tables = BASE/'tables'; tables.mkdir(exist_ok=False)
    archive = Path.home()/'Local/Research-Cloud/PR-163-equity-buckets/hu20-equity-buckets-20261005.tar.gz'
    if file_hash(archive)!='03d4828f030082f213c8ce82e29fa0fa8fd97288bd8035ad0056de85bc0816ca':
        raise ValueError('Table archive mismatch')
    subprocess.run(['tar','-xzf',str(archive),'-C',str(tables),'--strip-components','1',
        *[f'hu20-equity-buckets-20261005/{s}-k50.bin' for s in EQUITY_K50_SHA256]],check=True)
    table_pins = {s:pin(tables/(s+'-k50.bin')) for s in EQUITY_K50_SHA256}
    if any(table_pins[s]['sha256']!=sha for s,sha in EQUITY_K50_SHA256.items()):
        raise ValueError('Table member mismatch')
    dest = BASE/'release-v0.5.2'; dest.mkdir(exist_ok=False)
    subprocess.run(['gh','release','download','v0.5.2','--repo',REPO,'--dir',str(dest)],check=True)
    source = '3a3bc4218bab96637d6a27a69b73383ee8bb196c'
    result = subprocess.check_output([sys.executable,str(dest/'verify_hu100_bundle.py'),str(dest),
        '--expect-source',source,'--require-publication'],text=True)
    item = RELEASES['v0.5.2']['model']; vpath = dest/item['file']
    if file_hash(vpath)!='0ed7f9512996bb3e9535a7258a5d0e4bc8c00f22734d511c642c0e5b95cbd3c0':
        raise ValueError('Release hash mismatch')
    models = {'v052':dict(pin(vpath),name='v052',format=item['format'],abstraction=HU100_SCHEMA,
        entries=item['entries'],iteration=item['iteration'],source_checkpoint_sha256=item['checkpoint_sha256']),
        'k50':dict(pin(k50_path),name='k50',format='holdem-hu100-stored-cfr-average-research-v1',
        abstraction=HU100_EQUITY_SCHEMA,card_buckets=str(tables),entries=asset['entries'],iteration=asset['iterations'],
        source_checkpoint_sha256=next(a['sha256'] for a in index['models'] if a['kind']=='checkpoint' and a['actual_nodes']==asset['actual_nodes']))}
    put(BASE/'inputs.json',{'models':models,'tables':table_pins,'k50_restoration':provenance,
        'release_verification_stdout':result,'release_source':source,'owners':states,'source':revision()})


def configuration(model, option, blocks):
    cfg = read(ROOT/'configs/arena/hu100-playing-baseline-v1.json')
    cfg.update(model=model,action_translation=option,proposed_final_blocks_per_opponent=blocks)
    return cfg


def scripted_values(run, opponent, blocks):
    coords = {}
    with (run/opponent/'hands.jsonl').open() as stream:
        for row in map(json.loads,stream):
            if row['arm']=='candidate':
                key = row['block'],row['rotation']
                if key in coords or row['status']!='completed':
                    raise ValueError('Scripted duplicate/failed coordinate')
                coords[key] = row['candidate_chips']
    if set(coords)!={(b,r) for b in range(blocks) for r in (0,1)}:
        raise ValueError('Scripted incomplete blocks')
    return [(coords[b,0]+coords[b,1])/2 for b in range(blocks)]


def scripted_arm(arm, blocks, root, folder, source, pilot):
    label, option = ARMS[arm]; model = read(BASE/'inputs.json')['models'][label]
    cfg = configuration(model,option,blocks); config = folder/(arm+'-config.json'); put(config,cfg)
    tick = perf_counter()
    registry = ReferencedRegistry(scripted.make_plan(cfg,'random',blocks,root),sorted_average_rows=True)
    load_seconds = perf_counter()-tick
    play = folder/arm/'play'
    original = scripted.execute(config,play,blocks,root,source,registry=registry)
    replay = scripted_audit(play,folder/(arm+'-replay.json'),cost_only=True)
    repeat = scripted.execute(config,folder/arm/'reproduction',blocks,root,source,registry=registry,reproduce=play)
    result = {'load_seconds':load_seconds,'play_seconds':original['wall_seconds'],
        'replay_seconds':replay['seconds'],'reproduction_seconds':repeat['wall_seconds'],'blocks':blocks,
        'hands_replayed':replay['hands_replayed'],'actions_replayed':replay['actions_replayed'],
        'reproduced_all_hands_and_decisions':repeat['reproduced_all_hands_and_decisions']}
    del registry; gc.collect()
    return result


def play(pilot=False):
    source = revision(); models = read(BASE/'inputs.json')['models']; pair = [models['k50'],models['v052']]
    if pilot:
        n, sn, root, sroot = 1024,64,ROOTS['pilot'],ROOTS['pilot']
    else:
        plan = read(BASE/'canonical-plan.json'); posted = read(BASE/'plan-posted.json')
        if plan['source']!=source or file_hash(BASE/'canonical-plan.json')!=posted['sha256'] or plan['inputs_sha256']!=file_hash(BASE/'inputs.json'):
            raise ValueError('Frozen source/inputs/posting differs')
        n,sn,root,sroot = plan['primary_blocks'],4096,ROOTS['primary'],ROOTS['scripted']
    folder = BASE/('pilot' if pilot else 'final'); quotes = {}
    registry,done = direct.execute(pair,n,root,'k50-v052',folder/'direct/play',source)
    replay = direct.audit(folder/'direct/play',folder/'direct/replay.json',cost_only=pilot)
    registry,repeat = direct.execute(pair,n,root,'k50-v052',folder/'direct/reproduction',source,registry,folder/'direct/play')
    quotes['direct'] = {'load_seconds':done['load_or_validation_seconds'],'play_seconds':done['play_seconds'],
        'replay_seconds':replay['seconds'],'reproduction_seconds':repeat['play_seconds'],
        'reproduction_validation_seconds':repeat['load_or_validation_seconds'],'blocks':n}
    if pilot:
        quotes['direct']['paired_sd_bb_per_100'] = replay['paired_sd_bb_per_100']
    del registry; gc.collect()
    for arm in ARMS:
        quotes[arm] = scripted_arm(arm,sn,sroot,folder/'scripted',source,pilot)
    if pilot:
        series = {(arm,o):scripted_values(folder/'scripted'/arm/'play',o,sn) for arm in ARMS for o in scripted.OPPONENTS}
        paired_sd = {o:{arm:stdev([a-b for a,b in zip(series['k50_off',o],series[arm,o],strict=True)])
            for arm in ('v052_on','v052_off')} for o in scripted.OPPONENTS}
    else:
        paired_sd = None
    put(BASE/('pilot-costs.json' if pilot else 'all-final-verified.json'),{'status':'verified','quotes':quotes,
        'source':source,'paired_scripted_sds':paired_sd,'outcomes_inspected':False,'at':time()})


def freeze(pr):
    costs = read(BASE/'pilot-costs.json'); q = costs['quotes']
    def project(n):
        d = q['direct']
        seconds = d['load_seconds']+d['reproduction_validation_seconds']+sum(d[k] for k in ('play_seconds','replay_seconds','reproduction_seconds'))*n/1024
        for arm in ARMS:
            a = q[arm]
            seconds += a['load_seconds']+sum(a[k] for k in ('play_seconds','replay_seconds','reproduction_seconds'))*4096/64
        return 1.25*seconds+1800
    n = 131072 if project(262144)>5*3600 else 262144
    reserve(list(ROOTS.values()),pr,BASE/'final-root-reservation.json')
    models = read(BASE/'inputs.json')['models']
    document = {'pr':pr,'source':revision(),'inputs_sha256':file_hash(BASE/'inputs.json'),'models':models,
        'tables':read(BASE/'inputs.json')['tables'],'roots':ROOTS,'primary_blocks':n,'scripted_blocks_per_opponent':4096,
        'arms':ARMS,'opponents':scripted.OPPONENTS,'pilot':costs,'requested_projection_seconds':project(262144),
        'projected_total_seconds':project(n),'projection_rule':'1.25 times measured fixed and scaled play/replay/reproduction costs plus 1800 seconds closeout; requested >5h reduces primary once to131072',
        'decision_rule':'primary nominal paired Student-t95: lower>0 stronger/recommend v0.5.3 candidate; upper<0 weaker; otherwise inconclusive; owner decides; do not extend',
        'severe_regression_rule':'scripted paired nominal95 upper bound below -20 BB/100',
        'primary_translation':None,'scripted_translation':{'max_states':512,'max_events':128},
        'pairing':'seat-swapped duplicate block;100BB reset;50/100 blinds;no rake;#226 direct streams;#236 scripted streams and shared physical deals across three arms',
        'guards':{'rss_bytes':13*1024**3,'critical_pressure':4,'min_free_percent':8,'swap_growth_bytes':4_000_000_000,'sustained_seconds':60,'immediate_disk_floor_bytes':16*1024**3},
        'no_failed_hand_reruns':True,'no_outcome_extension':True,'no_training_release_or_default_change':True,
        'all_final_hands_independently_replayed_before_readout':True,'root_reservation_sha256':file_hash(BASE/'final-root-reservation.json')}
    put(BASE/'canonical-plan.json',document)
    body = BASE/'plan-comment.md'
    body.write_text('Canonical pre-outcome plan. SHA256 `'+file_hash(BASE/'canonical-plan.json')+'`.\n\n```json\n'+json.dumps(document,indent=2,sort_keys=True)+'\n```\n')
    url = subprocess.check_output(['gh','pr','comment',str(pr),'--repo',REPO,'--body-file',str(body)],text=True).strip()
    comments = json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/issues/{pr}/comments','--paginate'],text=True))
    posted = next(c for c in comments if c['html_url']==url)
    if posted['body']!=body.read_text().rstrip():
        raise ValueError('Canonical PR posting readback differs')
    put(BASE/'plan-posted.json',{'url':url,'comment_id':posted['id'],'sha256':file_hash(BASE/'canonical-plan.json'),'body_sha256':file_hash(body),'readback_verified':True,'at':time()})


def estimate(values):
    center = mean(values); sd = stdev(values); margin = float(t.ppf(.975,len(values)-1))*sd/math.sqrt(len(values))
    return {'bb_per_100':center,'ci95':[center-margin,center+margin],'sd':sd,'blocks':len(values)}


def readout():
    verified = read(BASE/'all-final-verified.json'); frozen = read(BASE/'canonical-plan.json')
    if verified['status']!='verified' or verified['source']!=frozen['source']:
        raise ValueError('Full verification before readout required')
    for complete in (BASE/'final').rglob('complete.json'):
        done = read(complete)
        if done['status']!='complete':
            raise ValueError('Final incomplete receipt')
        if 'files' in done:
            pins = done['files']
        else:
            pins = read(complete.parent/'output-files.json')
        for name, expected in pins.items():
            path = complete.parent/name
            if path.stat().st_size!=expected['bytes'] or file_hash(path)!=expected['sha256']:
                raise ValueError('Final bytes changed after replay: '+str(path))
    folder = BASE/'final/direct'; n = frozen['primary_blocks']; cells = {}; positions = {'button':[],'big_blind':[]}
    models = frozen['models']; pair = [models['k50'],models['v052']]
    buttons = {b.index:b.button for b in build_schedule(direct.make_plan(pair,n,ROOTS['primary'],'k50-v052'))}
    with gzip.open(folder/'play/hands.jsonl.gz','rt') as stream:
        for row in map(json.loads,stream):
            cells[row['block'],row['rotation']] = row['candidate_chips']
            positions['button' if row['rotation']==buttons[row['block']] else 'big_blind'].append(row['candidate_chips'])
    primary = estimate([(cells[b,0]+cells[b,1])/2 for b in range(n)])
    lo,hi = primary['ci95']
    primary.update(decision='stronger' if lo>0 else 'weaker' if hi<0 else 'inconclusive',
        seats={s:estimate(v) for s,v in positions.items()},absolute_returns={
        'k50':{'overall':primary['bb_per_100'],**{s:mean(v) for s,v in positions.items()}},
        'v052':{'overall':-primary['bb_per_100'],'button':-mean(positions['big_blind']),'big_blind':-mean(positions['button'])}},
        coverage=read(folder/'replay.json')['coverage'])
    values = {(arm,o):scripted_values(BASE/'final/scripted'/arm/'play',o,4096) for arm in ARMS for o in scripted.OPPONENTS}
    absolute = {arm:{o:estimate(values[arm,o]) for o in scripted.OPPONENTS} for arm in ARMS}
    differences = {o:{arm:estimate([a-b for a,b in zip(values['k50_off',o],values[arm,o],strict=True)])
        for arm in ('v052_on','v052_off')} for o in scripted.OPPONENTS}
    flags = {o:{arm:d['ci95'][1]<-20 for arm,d in contrasts.items()} for o,contrasts in differences.items()}
    translation = {o:estimate([a-b for a,b in zip(values['v052_on',o],values['v052_off',o],strict=True)]) for o in scripted.OPPONENTS}
    put(BASE/'summary.json',{'status':'verified','primary':primary,'scripted_absolute':absolute,
        'scripted_paired_differences':differences,'severe_regressions':flags,'v052_translation_on_minus_off':translation,
        'all_final_hands_replayed_and_reproduced':True,'canonical_plan_sha256':file_hash(BASE/'canonical-plan.json')})


def archive(pr):
    from scripts.archive_hu100_4b_seed_ladder import seal
    owner = json.loads(subprocess.check_output(['gh','pr','view',str(pr),'--repo',REPO,
        '--json','state,headRefName,headRefOid,url'],text=True))
    if owner['headRefName']!=subprocess.check_output(['git','branch','--show-current'],text=True).strip():
        raise ValueError('Only the authorized owning campaign may be archived')
    if not (BASE/'summary.json').exists() and not (BASE/'failure.json').exists():
        raise ValueError('Complete or explicitly failed evidence required')
    files = []
    for path in sorted(BASE.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(BASE)
        if rel.parts[0] in ('tables','models','release-v0.5.2') or rel.parts[:2]==('guards','archive') or path.name.startswith('archive-receipt'):
            continue
        files.append((path,'research/'+rel.as_posix()))
    for name in subprocess.check_output(['git','diff','--name-only','a9b213bcc86747925f3179628575ee702baca27b','HEAD'],text=True).splitlines():
        path = ROOT/name
        if path.is_file() and path.stat().st_size<=1024**2:
            files.append((path,'source/'+name))
    metadata = {'pr':pr,'owner':owner,'source':revision(),'inputs':read(BASE/'inputs.json'),
        'canonical_plan':read(BASE/'canonical-plan.json') if (BASE/'canonical-plan.json').exists() else None,
        'originals_retained':True,'remote_archive_bytes_downloaded':False,
        'excluded':'indexed models, release bundle and PR163 tables; live archive guard retained separately'}
    cloud = Path.home()/f'Local/Research-Cloud/PR-{pr}-hu100-k50-vs-v052'
    put(BASE/'archive-receipt.json',seal(files,cloud/'hu100-k50-vs-v052-M4-20261013.zip',metadata))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','pilot','freeze','final','readout','archive','guard'))
    p.add_argument('--pr',type=int); p.add_argument('--worker',choices=('prepare','pilot','final','readout','archive'))
    args = p.parse_args(); BASE.mkdir(parents=True,exist_ok=True)
    if args.stage=='guard':
        command = [sys.executable,'-m','scripts.run_hu100_k50_v052',args.worker]
        if args.pr is not None: command += ['--pr',str(args.pr)]
        guard(BASE,args.worker,command)
    elif args.stage in ('prepare','freeze','archive'):
        globals()[args.stage](args.pr)
    elif args.stage in ('pilot','final'):
        play(args.stage=='pilot')
    else:
        readout()


if __name__=='__main__':
    main()
