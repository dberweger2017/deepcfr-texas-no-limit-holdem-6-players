"""M1 orchestration around unchanged direct/arena play; compress only storage metadata."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PosixPath
import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
from scipy.stats import t

REPO=Path(__file__).resolve().parents[1]
ROOT=Path.home()/'Local/hu20-zero-mass-fallback-20261006'
OLD=Path.home()/'Local/hu20-native-average-20261005'
SEEDS=(2026100601,2026100602,2026100603)
FAMILIES=('A-Tprime-vs-T','B-Oprime-vs-O','secondary-Tprime-vs-R1','secondary-Oprime-vs-R1','secondary-Tprime-vs-O')
FLOOR=8*1024**3

class CompressedPath(PosixPath):
    """The unchanged tools read/write small JSON metadata through a gzip storage boundary."""
    def write_text(self, data, *args, **kwargs):
        target=Path(str(self)+'.gz')
        with gzip.open(target,'xt') as f:return f.write(data)
    def read_text(self, *args, **kwargs):
        with gzip.open(Path(str(self)+'.gz'),'rt') as f:return f.read()

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def filehash(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(p,'xt') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def read(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def guard(deadline=None):
    free=shutil.disk_usage(ROOT).free
    if free<FLOOR:raise OSError(f'Free disk {free} below 8 GiB; stop without deletion')
    if deadline and time.time()>deadline:raise TimeoutError('Aggregate research deadline')
    return free

def worker(kind,plan_path,key=None):
    sys.path.insert(0,str(REPO))
    plan=read(plan_path);out=CompressedPath(plan_path.parent/'run');out.mkdir(exist_ok=True)
    if kind=='direct':
        from scripts.evaluate_hu20_cfr_plus_direct import run
        result=run(plan,ROOT,out,int(key))
        if result['status']!='complete':raise RuntimeError(result)
    elif kind=='direct-audit':
        from scripts.evaluate_hu20_cfr_plus_direct import report
        from scripts.audit_hu20_cfr_plus_direct import audit
        report(plan,out)
        write(plan_path.parent/'run/audit.json.gz',audit(plan,out,digest(plan)))
    elif kind=='arena':
        from scripts.evaluate_hu20_v041_arena import play_model
        result=play_model(plan,key,ROOT,out,rss_limit=6*1024**3)
        if result['status']!='complete':raise RuntimeError(result)
    elif kind=='pressure-audit':
        from scripts.report_audit_hu20_zero_mass_pressure import analyze
        analyze(plan,Path(out),False);analyze(plan,Path(out),True)
    elif kind=='export-audit':
        from src.diagnostics.cfr_average import audit
        spec=plan['spec'];result=audit(Path(plan['checkpoint']),Path(plan['current']),Path(spec['path']),
            dict(spec,sha256=plan['current_sha256']),spec['sha256'])
        write(plan_path.parent/(spec['name']+'.audit.json.gz'),result)
    else:raise ValueError(kind)

def launch(jobs,stage,deadline,max_workers=2):
    children=[];began=time.perf_counter();stop=False
    def jobrun(job):
        kind,plan,key=job
        if stop:raise RuntimeError('Campaign stopped')
        guard(deadline)
        label=plan.parent.name+'-'+kind+'-'+str(key)
        log=ROOT/stage/(label+'.log.gz');log.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(log,'xb') as f:
            p=subprocess.Popen([sys.executable,str(Path(__file__)),'worker',kind,str(plan),str(key)],cwd=REPO,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(p)
            while chunk:=p.stdout.read(65536):f.write(chunk);f.flush()
            if p.wait():raise RuntimeError(f'{label} failed; retained {log}')
        return label
    pool=ThreadPoolExecutor(max_workers=max_workers)
    futures=[pool.submit(jobrun,j) for j in jobs]
    resources=[]
    try:
        while not all(f.done() for f in futures):
            resources.append({'at':time.time(),'free_disk_bytes':guard(deadline),'active_pids':[p.pid for p in children if p.poll() is None]})
            for f in futures:
                if f.done():f.result()
            time.sleep(2)
        for f in futures:f.result()
        write(ROOT/stage/'complete.json.gz',{'seconds':time.perf_counter()-began,'jobs':len(jobs),'at':time.time()})
    finally:
        stop=True
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        pool.shutdown(wait=True,cancel_futures=True)
        write(ROOT/stage/'resources.json.gz',resources)

def prepare():
    guard();historical=json.loads((OLD/'arena/plan.json').read_text())
    assert digest(historical)=='5d096197840097b93e56aca3ccbb480ac3335e9ef546c7faac358233ac18626f'
    write(ROOT/'inputs/pr165-plan.json.gz',historical)
    models={};receipts=[]
    for s in historical['models']:
        if s['arm'] not in ('O','T','C') and s['name']!='R-2026093001':continue
        p=OLD/'policies'/s['path'];assert p.stat().st_size==s['bytes'] and filehash(p)==s['sha256']
        pinned=dict(s,path=str(p));models[s['arm'],s['seed']]=pinned
        receipts.append({'kind':'original-policy','spec':pinned})
        if s['arm'] in ('O','T'):
            cp=OLD/'training'/f"{s['arm']}-{s['seed']}-1000000000.json.gz"
            assert filehash(cp)==s['checkpoint_sha256']
            receipts.append({'kind':'checkpoint','path':str(cp),'sha256':s['checkpoint_sha256'],'bytes':cp.stat().st_size})
    write(ROOT/'input-verification.json.gz',receipts)
    write(ROOT/'models-original.json.gz',[v for k,v in models.items()])
    write(ROOT/'environment.json.gz',{'host':platform.node(),'platform':platform.platform(),'python':sys.version,
        'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'free_disk_bytes':guard(),
        'engine':subprocess.check_output([sys.executable,'-c','import pokers; print(pokers.__file__)'],text=True).strip()})
    for arm in ('T','O'):
        for seed in SEEDS:
            guard();s=models[arm,seed];cp=OLD/'training'/f'{arm}-{seed}-1000000000.json.gz'
            output=ROOT/'policies'/f'{arm}prime-{seed}.average.jsonl.gz';output.parent.mkdir(exist_ok=True)
            current=ROOT/'policies'/f'Tcurrent-{seed}.json.gz' if arm=='T' else Path(models['C',seed]['path'])
            args=[str(Path.home()/'Local/deepcfr-review-pr178/native/hu20-trainer/target/release/hu20-trainer'),'export',str(cp),'--average',str(output),'--zero-mass','current']
            if arm=='T':args+=['--current',str(current)]
            assert not output.exists() and (arm!='T' or not current.exists())
            started=time.time()
            result=subprocess.run(args,capture_output=True,check=True)
            with gzip.open(ROOT/(f'{arm}prime-{seed}.export.log.gz'),'xb') as f:f.write(result.stdout+result.stderr)
            guard();pinned=dict(s,arm=arm+'prime',name=f'{arm}prime-{seed}',path=str(output),sha256=filehash(output),bytes=output.stat().st_size)
            models[arm+'prime',seed]=pinned
            write(ROOT/'export-audits'/f'{arm}prime-{seed}.plan.json.gz',{'spec':pinned,'checkpoint':str(cp),
                'current':str(current),'current_sha256':filehash(current),'export_seconds':time.time()-started})
            launch([('export-audit',ROOT/'export-audits'/f'{arm}prime-{seed}.plan.json.gz',None)],f'export-audit-{arm}-{seed}',time.time()+7200,1)
            print(pinned['name']+' exported and all nodes audited',flush=True)
    write(ROOT/'models.json.gz',list(models.values()))
    files=['scripts/evaluate_hu20_cfr_plus_direct.py','scripts/audit_hu20_cfr_plus_direct.py','scripts/evaluate_hu20_v041_arena.py','scripts/evaluate_hu20_cfr_average.py']
    pr175=json.loads((Path.home()/'Local/hu20-floor-control-20261006/unchanged-tool-hashes.json').read_text())
    hashes={p:filehash(REPO/p) for p in files}
    assert hashes==pr175,'Direct/arena tools changed since #175'
    write(ROOT/'unchanged-tools.json.gz',{'current':hashes,'pr175':pr175})
    print('Preparation complete',flush=True)

def plans():
    models={(m['arm'],m['seed']):m for m in read(ROOT/'models.json.gz')};r=models['R',2026093001]
    pairs={FAMILIES[0]:[{'candidate':models['Tprime',s],'reference':models['T',s]} for s in SEEDS],
        FAMILIES[1]:[{'candidate':models['Oprime',s],'reference':models['O',s]} for s in SEEDS],
        FAMILIES[2]:[{'candidate':models['Tprime',s],'reference':r} for s in SEEDS],
        FAMILIES[3]:[{'candidate':models['Oprime',s],'reference':r} for s in SEEDS],
        FAMILIES[4]:[{'candidate':models['Tprime',s],'reference':models['O',s]} for s in SEEDS]}
    started=time.time()
    for family in FAMILIES:
        write(ROOT/'pilot'/family/'plan.json.gz',{'root':202610063401,'blocks':2048,'pairs':pairs[family],
            'primary_pairs':[1,2,3],'started_at':started,'max_seconds':7200,'scope':family+'; outcome-blind pilot; excluded'})

def freeze():
    costs={};sizing={};counts={}
    for f in FAMILIES:
        p=read(ROOT/'pilot'/f/'plan.json.gz');cells=[];timings=[];rss=[];hand_seconds=[]
        for l in (1,2,3):
            r=read(ROOT/'pilot'/f/'run'/f'direct-lineage-{l}.result.json.gz')
            assert r['status']=='complete' and r['plan_sha256']==digest(p)
            timings.append(r['seconds']);rss.append(r['peak_rss_bytes']);values={}
            with gzip.open(ROOT/'pilot'/f/'run'/f'direct-lineage-{l}.hands.jsonl.gz','rt') as stream:
                for row in map(json.loads,stream):
                    k=row['block'],row['rotation'];assert k not in values and row['root_seed']==p['root']
                    values[k]=row['target_chips'];hand_seconds.append(row['seconds'])
            assert set(values)=={(b,r) for b in range(p['blocks']) for r in (0,1)}
            cells.append([(values[b,0]+values[b,1])/2 for b in range(p['blocks'])])
        sd=[statistics.stdev(v) for v in [*cells,[math.fsum(v)/3 for v in zip(*cells,strict=True)]]]
        n=32768
        while float(t.ppf(.975,n-1))*max(sd)/math.sqrt(n)>3.5:n+=4096
        counts[f]=n;sizing[f]={'block_sd_lineages_and_aggregate':sd}
        costs[f]={'pair_seconds':timings,'peak_rss_bytes':max(rss),'seconds_per_pair_block':math.fsum(hand_seconds)/(3*p['blocks']),
            'startup_seconds':max(0,(sum(timings)-math.fsum(hand_seconds))/3)}
    for f in FAMILIES[2:]:counts[f]=max(counts[FAMILIES[0]],counts[FAMILIES[1]])
    for f in FAMILIES:sizing[f].update(blocks=counts[f],projected_max_half_width=float(t.ppf(.975,counts[f]-1))*max(sizing[f]['block_sd_lineages_and_aggregate'])/math.sqrt(counts[f]))
    play=1.5*sum(c['startup_seconds']+c['seconds_per_pair_block']*counts[f] for f,c in costs.items())
    replay=1.5*sum(c['seconds_per_pair_block']*counts[f] for f,c in costs.items())*2
    quote={'sizing':sizing,'costs':costs,'play_seconds':play,'replay_report_seconds':replay,'with_50_percent_headroom_seconds':1.5*(play+replay),'pilot_outcomes_inspected':False}
    write(ROOT/'quote.json.gz',quote)
    assert quote['with_50_percent_headroom_seconds']<=3600,'#175 one-hour feasibility gate'
    bundle={'root':202610063501,'started_at':time.time(),'max_seconds':7200,'families':{},'orchestration_sha256':filehash(__file__),'workers':2}
    for f in FAMILIES:
        p=read(ROOT/'pilot'/f/'plan.json.gz');p.update(root=bundle['root'],started_at=bundle['started_at'],blocks=counts[f],scope=f+'; frozen final; nominal paired 95%; no release decision')
        write(ROOT/'final'/f/'plan.json.gz',p);bundle['families'][f]={'plan':p,'sha256':digest(p)}
    write(ROOT/'final-plan.json.gz',bundle);write(ROOT/'final-plan-hash.json.gz',{'sha256':digest(bundle)})
    comment='Outcome-blind M1 pilot complete; final sample frozen before scores. No pilot means, intervals or labels inspected.\n\nFresh final root **202610063501**, pilot **202610063401** excluded.\n\n'
    for f in FAMILIES:comment+=f'- {f}: **{counts[f]:,} duplicate blocks × three matched pairings × two seats = {counts[f]*6:,} hands**.\n'
    comment+=f'\nPrimary sizing targets projected ≤3.5 BB/100 (headroom for achieved ≤4), maximum over each lineage and its aggregate. No outcome-driven extension. Largest pilot RSS **{max(c["peak_rss_bytes"] for c in costs.values())/1024**3:.2f} GiB**. Startup-inclusive play **{play/60:.1f} min**; independent replay/report allowance **{replay/60:.1f} min**; **{1.5*(play+replay)/60:.1f} min with 50% headroom**.\n\nM1 only, free, two workers (≤3 limit), unchanged 6-GiB worker guard, ≥8-GiB free disk, two-hour aggregate cap. Canonical bundled plan SHA256 `{digest(bundle)}`. Individual plan hashes and outcome-blind SD/cost quote retained. Every pilot/final action and settlement independently audited; pilot arithmetic follows freezing. Labels and contrasts remain as predeclared. No release decisions.\n'
    with gzip.open(ROOT/'freeze-comment.md.gz','xt') as f:f.write(comment)
    print(json.dumps({'counts':counts,'quote':quote,'sha256':digest(bundle)},indent=2))

def execute(stage,kind):
    pths=[ROOT/stage/f/'plan.json.gz' for f in FAMILIES]
    deadline=min(read(p)['started_at']+read(p)['max_seconds'] for p in pths)
    jobs=[('direct',p,str(l)) for p in pths for l in (1,2,3)] if kind=='play' else [('direct-audit',p,None) for p in pths]
    launch(jobs,stage+'-'+kind,deadline)

def pressure_plan():
    audited=[read(ROOT/'final'/f/'run/audit.json.gz') for f in FAMILIES[:2]]
    if not any(a['primary_overall']['label']=='better' for a in audited):
        write(ROOT/'pressure-not-triggered.json.gz',{'reason':'Neither primary aggregate labeled better; conditional check not run'})
        return
    models=[m for m in read(ROOT/'models.json.gz') if m['arm'] in ('Tprime','T','Oprime','O')]
    panel=next(p for p in read(ROOT/'inputs/pr165-plan.json.gz')['panels'] if p['name']=='native-pressure')
    plan={'root':202610063601,'panels':[dict(panel,blocks=12288)],'stage':'frozen-final','models':models,
        'started_at':time.time(),'max_seconds':7200,'expected_hands':12*2*12288,
        'scope':'predeclared conditional pressure check; all four arms; no release gate','workers':2}
    write(ROOT/'pressure/plan.json.gz',plan)
    print(json.dumps({'pressure_plan_sha256':digest(plan),'blocks':12288,'hands':plan['expected_hands']}))

def pressure_run(kind):
    path=ROOT/'pressure/plan.json.gz';plan=read(path);deadline=plan['started_at']+plan['max_seconds']
    jobs=[('arena',path,m['name']) for m in plan['models']] if kind=='play' else [('pressure-audit',path,None)]
    launch(jobs,'pressure-'+kind,deadline)

if __name__=='__main__':
    a=sys.argv[1:]
    try:
        if a[0]=='worker':worker(a[1],Path(a[2]),a[3])
        elif a[0]=='prepare':prepare()
        elif a[0]=='plans':plans()
        elif a[0]=='freeze':freeze()
        elif a[0]=='pressure-plan':pressure_plan()
        elif a[0]=='pressure':pressure_run(a[1])
        elif a[0] in ('play','audit'):execute(a[1],a[0])
        else:raise ValueError(a)
    except Exception as e:
        write(ROOT/(f'failure-{os.getpid()}-{time.time_ns()}.json.gz'),{'args':a,'type':type(e).__name__,'error':str(e),'at':time.time(),'free_disk_bytes':shutil.disk_usage(ROOT).free})
        raise
