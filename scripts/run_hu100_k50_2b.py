"""Owner-authorized M4 HU100 K50 campaign; every stage is once-only and guarded."""
import argparse
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
from time import time, sleep
import psutil

from scripts.guard_hu20_equity_scoring import host, machine_identity, stop_owned
from scripts.research_process_family import owned_processes
from scripts.evaluate_hu100_direct import put
from src.policies.files import file_hash
from src.blueprint.abstraction import HU100_SCHEMA, HU100_EQUITY_SCHEMA
from src.blueprint.equity_buckets import EQUITY_K50_SHA256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/hu100-k50-2b'
BINARY = ROOT/'native/hu20-trainer/target/release/hu20-trainer'
TABLES = BASE/'tables'
GIB = 1024**3
SOFT, HARD = 10_500_000_000, 11*GIB
# #226's largest measured train/save bytes per entry, plus 10% and fixed reserves.
BPE = 6_826_049_536/70_466_972
TABLE_BYTES = 12_867_944 + 139_600_524 + 1_231_562_564
RESERVE = 250_000_000
CAP = math.floor((SOFT-TABLE_BYTES-RESERVE)/(1.1*BPE))
SEED = 2026100601
ROOTS = {'pilot':20261012101101,'primary':20261012101102,'secondary':20261012101103}
COUNTS = {'primary':262144,'secondary':131072}
REPO = 'dberweger2017/deepcfr-texas-no-limit-holdem-6-players'
RELEASES = {
 '1b':('v0.5.0','b9c9bd161422e87371a2641ae4c60304c66eed79','verify_v050_bundle.py'),
 '2b':('v0.5.1','dca7547e6eca82ce77f3e42230b82fabc1c29a4c','verify_hu100_bundle.py')}


def read(path):
    return json.loads(Path(path).read_text())


def revision():
    if Path.cwd()!=ROOT: raise ValueError('Campaign checkout required')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Clean committed source required')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def command(args):
    subprocess.run(list(map(str,args)),check=True,cwd=ROOT)


def pin(path):
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':file_hash(path)}


def limits(sample,rss,baseline,now,*,allowed_pressure=(1,),swap_growth_bytes=512*1024**2):
    if now>=baseline['deadline']:return 'seven-hour campaign ceiling'
    if rss>=HARD:return '11 GiB hard family RSS ceiling'
    if rss>=SOFT:return '10.5 GB soft family RSS ceiling'
    if sample['pressure_level'] not in allowed_pressure or sample['free_percent']<15:return 'allowed memory pressure / 15% free'
    if sample['swap_bytes']-baseline['swap_bytes']>swap_growth_bytes:return str(swap_growth_bytes)+' byte swap growth'
    if sample['disk_free_bytes']<16*GIB:return '16 GiB disk floor'
    if not sample['ac']:return 'AC power'
    return None


def guard(name,args,*,preservation_only=False):
    if preservation_only and not (name=='archive' and len(args)==5
            and args[:4]==['.venv/bin/python','-m','planning.archive_campaign','--pr'] and str(args[4]).isdigit()):
        raise ValueError('Preservation permission only admits the archive worker')
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'campaign-failure.json').exists() and not preservation_only:raise ValueError('Terminal failure: preserve partials, no rerun')
    identity=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string','hw.memsize','hw.ncpu'],text=True).splitlines()
    machine_identity('m4',identity)
    known={};child=None;failure=None;peak=0;count=0;started=time()
    folder=BASE/'guards'/name
    folder.mkdir(parents=True,exist_ok=False)
    def interrupted(signum,frame):raise RuntimeError('Supervisor interrupted')
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            sample=host(BASE)
            if not (BASE/'budget.json').exists():
                put(BASE/'budget.json',{'started':started,'deadline':started+7*3600,'swap_bytes':sample['swap_bytes'],'initial_host':sample})
            baseline=read(BASE/'budget.json')
            put(folder/'intent.json',{'command':list(map(str,args)),'source':revision(),'identity':identity,'started':started,'preservation_only':preservation_only})
            if reason:=limits(sample,psutil.Process().memory_info().rss,baseline,time()):
                raise RuntimeError('Admission refused: '+reason)
            # process_iter tolerates unreadable unrelated metadata; owned descendants fail closed.
            for process in psutil.process_iter(['pid','cmdline']):
                try:
                    cmd=' '.join(process.info['cmdline'] or [])
                    if process.pid==os.getpid():continue
                    if any(token in cmd for token in ('hu20-trainer train','scripts.evaluate_hu','scripts.run_hu','scripts.archive_hu')):
                        if str(ROOT) not in cmd:raise RuntimeError('Another research process is active: '+str(process.pid))
                except (psutil.AccessDenied,psutil.NoSuchProcess,psutil.ZombieProcess):pass
            with (folder/'worker.log').open('x') as log,(folder/'resources.jsonl').open('x') as raw:
                child=subprocess.Popen(list(map(str,args)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                known[child.pid]=psutil.Process(child.pid).create_time()
                while True:
                    rss=0
                    for proc in owned_processes(known):
                        try:rss+=proc.memory_info().rss
                        except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
                    sample=host(BASE);peak=max(peak,rss);count+=1
                    raw.write(json.dumps(dict(sample,family_rss_bytes=rss))+'\n');raw.flush()
                    if reason:=limits(sample,rss,baseline,time()):raise RuntimeError('Guard breach: '+reason)
                    if child.poll() is not None:
                        if child.returncode:raise RuntimeError('Worker exit '+str(child.returncode))
                        break
                    sleep(.5)
        except BaseException as exc:
            failure=repr(exc)
            if not (BASE/'campaign-failure.json').exists():put(BASE/'campaign-failure.json',{'operation':name,'failure':failure,'at':time()})
            raise
        finally:
            stop_owned(known)
            if child is not None:child.wait(timeout=5)
            for s,h in handlers.items():signal.signal(s,h)
            put(folder/'receipt.json',{'status':'failed' if failure else 'complete','failure':failure,'started':started,'finished':time(),
                'seconds':time()-started,'peak_family_rss_bytes':peak,'samples':count,'returncode':child.returncode if child else None})


def prepare():
    owner=json.loads(subprocess.check_output(['gh','pr','view','163','--repo',REPO,'--json','state,mergedAt'],text=True))
    if owner['state']!='MERGED':raise ValueError('Bucket owner must be merged before retrieval')
    TABLES.mkdir(parents=True,exist_ok=False)
    archive=Path.home()/'Local/Research-Cloud/PR-163-equity-buckets/hu20-equity-buckets-20261005.tar.gz'
    command(['tar','-xzf',archive,'-C',TABLES,'--strip-components','1',*[f'hu20-equity-buckets-20261005/{s}-k50.bin' for s in EQUITY_K50_SHA256]])
    tables={}
    for s,sha in EQUITY_K50_SHA256.items():
        p=TABLES/(s+'-k50.bin');tables[s]=pin(p)
        if tables[s]['sha256']!=sha:raise ValueError('Pinned table mismatch')
    releases={}
    for label,(tag,source,verifier) in RELEASES.items():
        dest=BASE/'releases'/tag;dest.mkdir(parents=True,exist_ok=False)
        command(['gh','release','download',tag,'--repo',REPO,'--dir',dest])
        command([sys.executable,dest/verifier,dest,'--expect-source',source,'--require-publication'])
        model=next(dest.glob('O*.jsonl.gz'))
        with gzip.open(model,'rt') as f:h=json.loads(next(f))
        spec={**pin(model),'name':'v1-'+label,'format':h['format'],'abstraction':HU100_SCHEMA,
            'source_checkpoint_sha256':h['source_checkpoint_sha256'],'iteration':h['checkpoint_header']['iteration'],
            'entries':41_010_014 if label=='1b' else 54_626_283}
        releases[label]={'tag':tag,'source':source,'verifier':pin(dest/verifier),'spec':spec}
    put(BASE/'inputs.json',{'status':'verified','tables':tables,'owning_pr':owner,'releases':releases})


def train_command(nodes,checkpoint,telemetry,resume=None):
    args=[BINARY,'train','--stack-bb','100','--seed',SEED,'--roots-per-seat','1','--average-rule','opponent-sampled',
        '--nodes',nodes,'--max-entries',CAP,'--card-buckets',TABLES,'--out',checkpoint,'--telemetry',telemetry]
    if resume is not None:args+=['--resume',resume,'--resume-sha256',file_hash(resume)]
    return args


def gates():
    dest=BASE/'gates';dest.mkdir(exist_ok=False)
    command([sys.executable,'-m','scripts.build_native_trainer','--receipt',BASE/'build-receipt.json'])
    for label,nodes,resume in [('fresh20m',20_000_000,None),('prefix10m',10_000_000,None),('resumed20m',20_000_000,dest/'prefix10m.gz')]:
        command(train_command(nodes,dest/(label+'.gz'),dest/(label+'.telemetry.jsonl'),resume))
    left,right=pin(dest/'fresh20m.gz'),pin(dest/'resumed20m.gz')
    if (left['bytes'],left['sha256'])!=(right['bytes'],right['sha256']):raise ValueError('20M recovery byte mismatch; stop')
    put(BASE/'gates-complete.json',{'status':'verified','fresh':left,'resumed':right,'binary':pin(BINARY)})


def treatment():
    from scripts.audit_native_hu_checkpoint import inspect
    if read(BASE/'gates-complete.json')['binary']['sha256']!=file_hash(BINARY):raise ValueError('Qualified binary changed')
    dest=BASE/'training';dest.mkdir(exist_ok=False)
    # One native run retains complete iterations at both targets without loading a new lineage.
    args=[*train_command(2_000_000_000,dest/'{nodes}.gz',dest/'telemetry.jsonl'),'--milestones','1000000000']
    rc=subprocess.run(list(map(str,args)),cwd=ROOT).returncode
    if rc not in (0,3):raise RuntimeError('Native training failed: '+str(rc))
    rows=[json.loads(line) for line in (dest/'telemetry.jsonl').read_text().splitlines()]
    if rc==3 and not (rows and rows[-1]['diagnostics']['entries']>=CAP and not rows[-1]['stop_requested']):
        raise ValueError('Exit 3 is not the declared clean entry-cap stop')
    specs={}
    for target in (1_000_000_000,2_000_000_000):
        checkpoint=dest/(str(target)+'.gz')
        if not checkpoint.exists():continue
        with gzip.open(checkpoint,'rt') as f:h=json.loads(next(f))
        actual=h['native_state']['completed_nodes']
        telemetry=next(r for r in rows if r['checkpoint_sha256']==file_hash(checkpoint))
        if telemetry['binary_sha256']!=file_hash(BINARY):raise ValueError('Telemetry binary changed')
        current=dest/(str(target)+'.current.gz');average=dest/(str(target)+'.average.gz')
        command([BINARY,'export',checkpoint,'--current',current,'--average',average,'--zero-mass','uniform'])
        receipt=inspect(checkpoint,current,average,100,actual,expected_schema=HU100_EQUITY_SCHEMA)
        put(dest/(str(target)+'.audit.json'),receipt)
        spec={**pin(average),'name':'k50-'+('1b' if target==1_000_000_000 else '2b'),'format':'holdem-hu100-stored-cfr-average-research-v1',
            'abstraction':HU100_EQUITY_SCHEMA,'card_buckets':str(TABLES),'actual_nodes':actual,'iteration':h['iteration'],
            'entries':receipt['entries'],'source_checkpoint_sha256':file_hash(checkpoint)}
        if receipt['audit']['all_nodes_verified']!=spec['entries']:raise ValueError('Full audit incomplete')
        if actual>=target:specs['1b' if target==1_000_000_000 else '2b']=spec
    put(BASE/'treatment.json',{'specs':specs,'telemetry':rows,'cap':CAP,'primary_reached':'2b' in specs})


def pairs():
    inputs=read(BASE/'inputs.json');treatment=read(BASE/'treatment.json')
    return {contrast:[treatment['specs'][label],inputs['releases'][label]['spec']]
        for contrast,label in [('primary','2b'),('secondary','1b')] if label in treatment['specs']}


def pilot():
    from scripts import evaluate_hu100_direct as direct
    quotes={};source=revision()
    for contrast,specs in pairs().items():
        out=BASE/'pilot'/contrast
        registry,done=direct.execute(specs,1024,ROOTS['pilot'],contrast,out/'play',source)
        checked=direct.audit(out/'play',out/'replay.json',cost_only=True)
        # Deterministic reproduction is verification, never replacement of a failed hand.
        _,repeat=direct.execute(specs,1024,ROOTS['pilot'],contrast,out/'reproduction',source,registry,out/'play')
        quotes[contrast]={'load_seconds':done['load_or_validation_seconds'],'play_seconds':done['play_seconds'],
            'replay_seconds':checked['seconds'],'reproduction_seconds':repeat['play_seconds'],
            'paired_sd_bb_per_100':checked['paired_sd_bb_per_100'],
            'trace_bytes':sum(p.stat().st_size for p in out.rglob('*.jsonl.gz'))}
        del registry, _
        import gc;gc.collect()
    put(BASE/'pilot-quote.json',{'quotes':quotes,'pilot_blocks':1024,'outcomes_inspected':False,'at':time()})


def freeze(pr, *, reservation_path=None, fixed_counts=False):
    from scipy.stats import t
    quote=read(BASE/'pilot-quote.json');budget=read(BASE/'budget.json');counts=dict(COUNTS)
    active=pairs()
    reservation=read(reservation_path or ROOT/'docs/reports/hu100-k50-2b-artifacts/root-reservation.json')
    if reservation['roots']!=ROOTS or reservation['status']!='fresh-reserved':raise ValueError('Root reservation differs')
    def projection():
        return time()-budget['started']+1800+1.25*sum(q['load_seconds']+(q['play_seconds']+q['replay_seconds']+q['reproduction_seconds'])*counts[c]/1024 for c,q in quote['quotes'].items())
    projected=projection()
    if projected>7*3600 and fixed_counts:raise ValueError('Fixed final counts do not fit the unchanged seven-hour ceiling')
    if projected>7*3600:
        counts={c:65536 for c in COUNTS}
        while projection()>7*3600 and counts['primary']>1024:counts={c:n//2 for c,n in counts.items()}
    if projection()>7*3600:raise ValueError('Cost quote refuses all final play')
    pilot_peak=read(BASE/'guards/pilot/receipt.json')['peak_family_rss_bytes']
    # The short pilot cannot fault every bucket page. Allow both the policy and
    # independent replay mappings to become fully resident in final play.
    memory_forecast={c:pilot_peak+2*TABLE_BYTES+5200*(counts[c]-1024)+100_000_000 for c in active}
    if any(n>=SOFT for n in memory_forecast.values()):raise ValueError('Final schedule memory admission refused')
    trace_forecast=math.ceil(sum(q['trace_bytes']*counts[c]/1024 for c,q in quote['quotes'].items())*1.1)
    import shutil
    if shutil.disk_usage(BASE).free-2*trace_forecast-6*GIB<16*GIB:raise ValueError('Final trace plus archive disk admission refused')
    specs={c:s for c,s in active.items()}
    document={'source':revision(),'pr':pr,'roots':ROOTS,'counts':{c:counts[c] for c in active},'pairs':specs,
        'input_receipt_sha256':file_hash(BASE/'inputs.json'),'tables':EQUITY_K50_SHA256,'binary':pin(BINARY),
        'pilot_quote':quote,'projected_whole_campaign_seconds':projection(),'requested_projection_seconds':projected,'memory_forecast_bytes':memory_forecast,'trace_forecast_bytes':trace_forecast,'root_reservation':reservation,
        'projected_half_widths':{c:float(t.ppf(.975,counts[c]-1))*q['paired_sd_bb_per_100']/math.sqrt(counts[c]) for c,q in quote['quotes'].items()},
        'decision_rule':'primary lower95>0: K50 stronger, recommend K50 for HU200; upper95<0: K50 weaker, recommend v1; otherwise inconclusive, no extension; owner decides',
        'secondary':'descriptive nominal paired 95%', 'translation':None,'no_failed_hand_reruns':True,'created':time()}
    put(BASE/'canonical-plan.json',document)


def final_play():
    from scripts import evaluate_hu100_direct as direct
    frozen=read(BASE/'canonical-plan.json');posted=read(BASE/'plan-posted.json')
    if file_hash(BASE/'canonical-plan.json')!=posted['sha256']:raise ValueError('Unposted/changed final plan')
    if frozen['source']!=revision() or frozen['roots']!=ROOTS:raise ValueError('Frozen source/roots changed')
    quote=frozen['pilot_quote']['quotes']
    remaining=1800+1.25*sum(q['load_seconds']+(q['play_seconds']+q['replay_seconds']+q['reproduction_seconds'])*frozen['counts'][c]/1024 for c,q in quote.items())
    if time()+remaining>read(BASE/'budget.json')['deadline']:raise ValueError('Current clock no longer admits final scope')
    for contrast,specs in frozen['pairs'].items():
        out=BASE/'final'/contrast
        registry,done=direct.execute(specs,frozen['counts'][contrast],ROOTS[contrast],contrast,out/'play',frozen['source'])
        checked=direct.audit(out/'play',out/'replay.json')
        _,repeat=direct.execute(specs,frozen['counts'][contrast],ROOTS[contrast],contrast,out/'reproduction',frozen['source'],registry,out/'play')
        del registry, _
        import gc;gc.collect()
    put(BASE/'all-final-verified.json',{'status':'verified','at':time(),'canonical_sha256':file_hash(BASE/'canonical-plan.json')})


def readout():
    from scipy.stats import t
    from scripts.evaluate_hu100_direct import make_plan
    from src.arena.schedule import build_schedule
    verified=read(BASE/'all-final-verified.json')
    if verified['status']!='verified':raise ValueError('Full independent replay before readout')
    if verified['canonical_sha256']!=file_hash(BASE/'canonical-plan.json'):raise ValueError('Final replay plan differs')
    def stats(values):
        center=statistics.mean(values);sd=statistics.stdev(values);margin=float(t.ppf(.975,len(values)-1))*sd/math.sqrt(len(values))
        return {'bb_per_100':center,'ci95':[center-margin,center+margin],'paired_sd':sd,'blocks':len(values)}
    results={};frozen=read(BASE/'canonical-plan.json')
    for contrast in frozen['pairs']:
        out=BASE/'final'/contrast;cells={};positions={'button':[],'big_blind':[]}
        done=read(out/'play/complete.json')
        for name,expected in done['files'].items():
            actual=pin(out/'play'/name)
            if (actual['bytes'],actual['sha256'])!=(expected['bytes'],expected['sha256']):raise ValueError('Final trace changed after replay')
        buttons={b.index:b.button for b in build_schedule(make_plan(frozen['pairs'][contrast],frozen['counts'][contrast],frozen['roots'][contrast],contrast))}
        with gzip.open(out/'play/hands.jsonl.gz','rt') as f:
            for row in map(json.loads,f):
                cells[row['block'],row['rotation']]=row['candidate_chips']
                positions['button' if row['rotation']==buttons[row['block']] else 'big_blind'].append(row['candidate_chips'])
        n=frozen['counts'][contrast];values=[(cells[b,0]+cells[b,1])/2 for b in range(n)]
        result=stats(values);low,high=result['ci95']
        result.update(seats={seat:stats(v) for seat,v in positions.items()},absolute_returns={'k50':{'overall':result['bb_per_100'],'button':statistics.mean(positions['button']),'big_blind':statistics.mean(positions['big_blind'])},
                'v1':{'overall':-result['bb_per_100'],'button':-statistics.mean(positions['big_blind']),'big_blind':-statistics.mean(positions['button'])}},
            coverage=read(out/'replay.json')['coverage'],decision='stronger' if low>0 else 'weaker' if high<0 else 'inconclusive')
        results[contrast]=result
    put(BASE/'summary.json',{'results':results,'primary_reached':read(BASE/'treatment.json')['primary_reached'],'all_hands_independently_replayed':True,'at':time()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('guard','prepare','gates','treatment','pilot','freeze','final','readout'))
    p.add_argument('--pr',type=int);p.add_argument('--name');p.add_argument('--preservation-only',action='store_true')
    a,extra=p.parse_known_args()
    if a.stage=='guard':guard(a.name,extra[1:] if extra[:1]==['--'] else extra,preservation_only=a.preservation_only)
    elif a.stage=='freeze':freeze(a.pr)
    else:{'prepare':prepare,'gates':gates,'treatment':treatment,'pilot':pilot,'final':final_play,'readout':readout}[a.stage]()

if __name__=='__main__':main()
