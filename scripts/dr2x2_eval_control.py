"""Exact-owned A/C evaluation rentals, durable leases, verified teardown."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import tomllib
import urllib.parse
import urllib.request

from scripts.dr2x2_control import connections, coordination, estimated_cost, event, remote_json, run
from scripts.hu20_platform_pilot import write
from scripts.mature_cpu_rental_guard import api, check_quote, owned_pods
from src.arena.schedule import digest
from src.diagnostics.saved_hu20 import file_hash


def verify_closed_archive(path):
    with tarfile.open(path,'r:') as archive:
        members={m.name:m for m in archive.getmembers()}
        if len(members)!=len(archive.getmembers()):raise ValueError('Duplicate archive member')
        for n,m in members.items():
            parts=Path(n).parts
            if Path(n).is_absolute() or '..' in parts or not parts or parts[0]!='results' or not(m.isfile() or m.isdir()):
                raise ValueError('Unsafe evaluation archive')
        manifest='results/dr2x2-eval/manifest.json'
        if manifest not in members:
            if 'results/outer-exit.txt' not in members or archive.extractfile(members['results/outer-exit.txt']).read().strip()==b'0':
                raise ValueError('Successful/unclosed setup without worker manifest')
            return {'passed':True,'setup_failed_retained':True,'verified_files':0}
        record=json.load(archive.extractfile(members[manifest]));verified=0
        for relative,item in record['files'].items():
            n='results/dr2x2-eval/'+relative;m=members[n];h=sha256()
            if m.size!=item['bytes']:raise ValueError('Archive member size differs')
            with archive.extractfile(m) as f:
                for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
            if h.hexdigest()!=item['sha256']:raise ValueError('Archive member hash differs')
            verified+=1
        return {'passed':True,'verified_files':verified}


def patch_control(ssh, fields):
    program="import fcntl,json,os;from pathlib import Path;p=Path('/workspace/control.json');lock=p.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);v=json.loads(p.read_text()) if p.exists() else {};v.update(json.loads(input()));t=p.with_suffix('.tmp');t.write_text(json.dumps(v));t.replace(p)"
    subprocess.run(ssh+['python3 -c '+shlex.quote(program)],input=(json.dumps(fields)+'\n').encode(),stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,timeout=30)


def check_credit(key_path, root):
    key=tomllib.loads(key_path.read_text())['apikey']
    request=urllib.request.Request('https://api.runpod.io/graphql?'+urllib.parse.urlencode({'api_key':key}),
            data=json.dumps({'query':'query { myself { clientBalance currentSpendPerHr } }'}).encode(),
            headers={'Content-Type':'application/json','User-Agent':'runpodctl/1.14.5'})
    try:
        with urllib.request.urlopen(request,timeout=20) as response:value=json.loads(response.read())['data']['myself']
    except Exception:raise RuntimeError('Private evaluation credit check failed') from None
    private=key_path.parent/'eval-billing-baseline-private.json';write(private,{'time':time.time(),'account':value});os.chmod(private,0o600)
    enough=float(value['clientBalance'])>=4
    write(root/'credit-check.json',{'sufficient_for_analysis_subcap':enough,'checked':time.time()})
    if not enough:raise ValueError('Private credit cannot cover quoted analysis cap')


def valid_eval_lease(lease):
    matches=[re.fullmatch(r'dr2x2-EVAL-(202609300[123])-(\d+)',n) for n in lease['names']]
    return (len(set(lease['names']))==3 and len(matches)==3 and all(matches)
            and {m.group(1) for m in matches if m}=={'2026093001','2026093002','2026093003'}
            and lease['subcap_usd']==4 and lease['reserve_usd']==1)


def watch(a):
    lease=json.loads((a.root/'lease.json').read_text());names=set(lease['names'])
    if not valid_eval_lease(lease):
        raise ValueError('Exact three evaluation names and approved analysis allocation required')
    saved=a.root/'watchdog-ledger.json'
    ledger=json.loads(saved.read_text()) if saved.exists() else []
    while True:
        try:
            live=owned_pods(api(a.key,'/v2/pods')['pods'],names)
            records=json.loads((a.root/'pods.json').read_text()) if (a.root/'pods.json').exists() else []
            indexed={r['id']:dict(r) for r in ledger if r.get('id')}
            for r in records:
                if r.get('id'):
                    old=indexed.get(r['id'],{});indexed[r['id']]=dict(r)
                    if old.get('terminated'):indexed[r['id']]['terminated']=old['terminated']
            for p in live:
                if p['id'] not in indexed:indexed[p['id']]={'id':p['id'],'name':p['name'],'created_epoch':datetime.fromisoformat(p['createdAt'].replace('Z','+00:00')).timestamp(),'upper_rate':max(.18,float(p.get('cost') or .13)+.05)}
            ledger=list(indexed.values());write(a.root/'watchdog-ledger.json',ledger)
            upper=lease.get('previous_analysis_upper_usd',0)+estimated_cost(ledger,time.time());heartbeat=a.root/'controller-heartbeat.json'
            age=time.time()-json.loads(heartbeat.read_text())['heartbeat'] if heartbeat.exists() else time.time()-lease['started']
            stop=a.root/'budget-stop.json'
            if upper>=3 or age>900:
                if not stop.exists():write(stop,{'time':time.time(),'reason':'analysis cost reserve' if upper>=3 else 'controller stale','upper_cost_usd':upper})
            if stop.exists():
                reason=json.loads(stop.read_text())['reason']
                for r in records:
                    if r.get('endpoint') and not r.get('terminated'):
                        try:patch_control(connections(r,a.root)[0],{'lease_until':time.time(),'stop':reason})
                        except Exception:pass
                if time.time()-json.loads(stop.read_text())['time']>600:
                    for p in live:
                        api(a.key,'/v2/pods/'+p['id'],'DELETE')
                        indexed[p['id']]['terminated']=time.time()
                        event(a.root,'budget-forced-teardown',pod_id=p['id'],latest_verified_state_preserved=True)
                    ledger=list(indexed.values());write(a.root/'watchdog-ledger.json',ledger)
            # An unrecoverable individual transport incident cannot rent forever.
            # Other admitted lineages continue; preserve latest off-pod closed tasks.
            for r in records:
                if (r.get('id') and not indexed.get(r['id'],{}).get('terminated') and r.get('incident_epoch')
                        and time.time()-r['incident_epoch']>600):
                    api(a.key,'/v2/pods/'+r['id'],'DELETE')
                    indexed[r['id']]['terminated']=time.time()
                    event(a.root,'incident-forced-teardown',pod_id=r['id'],latest_verified_tasks=r.get('verified_tasks',[]))
            ledger=list(indexed.values());write(a.root/'watchdog-ledger.json',ledger)
            write(a.root/'watchdog.json',{'status':'armed','pid':os.getpid(),'heartbeat':time.time(),'owned_ids':[p['id'] for p in live],'upper_cost_usd':upper})
            if (a.root/'operator-finished.json').exists() and not live:
                write(a.root/'watchdog.json',{'status':'finished','heartbeat':time.time(),'owned_ids':[],'upper_cost_usd':upper});return
        except Exception as exc:
            write(a.root/'watchdog-error.json',{'time':time.time(),'error_type':type(exc).__name__})
        time.sleep(15)


def execute(a):
    quote=json.loads(a.quote.read_text());plan=json.loads(a.plan.read_text())
    if (quote['analysis_subcap_usd']!=4 or quote['all_in_ceiling_usd']!=16
            or quote['frozen_source']!='6ce14e1513ce1b6f452b59930bf8f2034ad80115'
            or quote['plan_sha256']!=digest(plan) or quote['max_concurrent_pods']!=3):raise ValueError('Frozen quote/scope differs')
    a.root=a.root.resolve();a.root.mkdir(parents=True,exist_ok=False);os.chmod(a.root,0o700)
    if shutil.disk_usage(a.root).free<20*2**30:raise OSError('Off-pod storage capacity insufficient')
    check_credit(a.key,a.root)
    # Recheck all immutable input transport hashes before renting, no model loads.
    hashes={'path':'sha256','checkpoint_path':'checkpoint_sha256','average_path':'average_sha256'}
    for spec in plan['models']:
        for k,h in hashes.items():
            if file_hash(a.inputs/spec[k])!=spec[h]:raise ValueError('Input changed before creation')
    stamp=int(time.time());jobs=[{'seed':s,'name':f'dr2x2-EVAL-{s}-{stamp}','status':'uncreated'} for s in (2026093001,2026093002,2026093003)]
    names={r['name'] for r in jobs}
    if owned_pods(api(a.key,'/v2/pods')['pods'],names):raise ValueError('Owned names existed before arming')
    write(a.root/'lease.json',{'names':sorted(names),'subcap_usd':4,'reserve_usd':1,'started':time.time(),'source':quote['frozen_source'],'plan_sha256':digest(plan),'all_in_ceiling_usd':16,'previous_training_upper_usd':1.0041844655513763,'previous_analysis_upper_usd':quote.get('previous_analysis_upper_usd',0)})
    write(a.root/'plan.json',plan);write(a.root/'quote.json',quote);write(a.root/'pods.json',jobs)
    run(['ssh-keygen','-t','ed25519','-N','','-f',str(a.root/'pod-key'),'-C','dr2x2-evaluation-owned'])
    write(a.root/'controller-heartbeat.json',{'heartbeat':time.time(),'pid':os.getpid()})
    with (a.root/'watchdog.log').open('x') as log:
        guardian=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'watch','--root',str(a.root),'--key',str(a.key)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(30):
        if (a.root/'watchdog.json').exists():break
        if guardian.poll() is not None:raise RuntimeError('Independent watchdog failed to arm')
        time.sleep(1)
    else:raise RuntimeError('Watchdog arming timed out')
    catalog=api(a.key,'/v2/catalog/cpus');write(a.root/'live-catalog.json',catalog)
    flavor=next(c for c in catalog['cpus'] if c['id']=='cpu5m')
    if flavor['price']['securePerVcpu']>.065+1e-9:raise ValueError('Live compute price exceeds quote')
    coordination('Doctor Research A/C evaluation ownership BEFOREcreation: '+json.dumps(sorted(names))+'. One worker/pod, USD4 analysis subcap/reserve1 within approved USD16; Ctraining1.004184 upper retained; no D/newtraining/M4 science. Root '+str(a.root))
    closed=threading.Event();lock=threading.Lock();allocation_lock=threading.Lock()
    def publish():
        with lock:write(a.root/'pods.json',jobs)
    def leases():
        while not closed.is_set():
            write(a.root/'controller-heartbeat.json',{'heartbeat':time.time(),'pid':os.getpid()})
            for row in list(jobs):
                if row.get('endpoint') and not row.get('terminated'):
                    try:
                        reason=json.loads((a.root/'budget-stop.json').read_text())['reason'] if (a.root/'budget-stop.json').exists() else row.get('requested_stop')
                        patch_control(connections(row,a.root)[0],{'lease_until':time.time()+300,'stop':reason})
                    except Exception:pass
            closed.wait(20)
    threading.Thread(target=leases,daemon=True).start()
    def workload(row):
        folder=a.root/'jobs'/str(row['seed']);folder.mkdir(parents=True)
        try:
            # Serialize provider allocations; the scientific workers still run in
            # parallel. A failed POST is reconciled once and never retried here.
            with allocation_lock:
                pod=api(a.key,'/v2/pods','POST',{'name':row['name'],'cloud':'SECURE','cpu':{'id':'cpu5m','vcpuCount':2},'image':'runpod/base:0.7.0-ubuntu2004','disk':30,'dataCenterIds':['EU-RO-1','EUR-IS-1'],'ports':['22/tcp'],'startSsh':True,'env':{'PUBLIC_KEY':(a.root/'pod-key.pub').read_text().strip()}})
            pod=pod.get('pod',pod)
            row.update(id=pod['id'],created_epoch=datetime.fromisoformat(pod['createdAt'].replace('Z','+00:00')).timestamp(),upper_rate=max(.18,float(pod['cost'])+.05),quote=pod,status='provisioning');publish()
            if not check_quote({'cpu_id':'cpu5m','vcpus':2,'ram_gb':16},pod,.13):raise ValueError('Actual CPU/RAM/price exceeds frozen quote')
            coordination('Doctor Research evaluation allocated: '+json.dumps({k:row[k] for k in ('name','id','seed','status')}))
            for _ in range(90):
                pod=api(a.key,'/v2/pods/'+row['id'])
                if pod.get('ssh',{}).get('direct') and pod['status']=='RUNNING':row['endpoint']=pod['ssh']['direct'];break
                time.sleep(10)
            else:raise TimeoutError('SSH provisioning unavailable after15min')
            ssh,scp,address=connections(row,a.root)
            for _ in range(30):
                try:run(ssh+['true']);break
                except Exception:time.sleep(5)
            else:raise TimeoutError('Pod SSH not ready')
            payload=folder/'inputs.tar'
            with tarfile.open(payload,'w',dereference=True) as tar:
                for cell in ('A','C'):tar.add(a.inputs/cell/str(row['seed']),arcname=cell+'/'+str(row['seed']))
            input_hash=file_hash(payload)
            reference=a.references/f"{row['seed']}-attempt-1"/'summary.json'
            run(ssh+['mkdir -p /workspace/results /workspace/inputs'])
            for local,remote in ((payload,'inputs.tar'),(a.plan,'plan.json'),(reference,'reference.json'),(a.setup,'eval-setup.sh'),(a.pod_wrapper,'eval-pod.py')):
                run(scp+[str(local),address+':/workspace/'+remote],timeout=600)
                run(ssh+['printf '+shlex.quote(file_hash(local)+'  /workspace/'+remote+'\n')+' | sha256sum -c -'])
            run(ssh+['tar -xf /workspace/inputs.tar -C /workspace/inputs'])
            patch_control(ssh,{'lease_until':time.time()+300,'stop':None})
            cmd='bash /workspace/eval-setup.sh '+quote['frozen_source']+' '+str(row['seed'])+' '+quote['plan_sha256']+' > /workspace/results/setup.log 2>&1; code=$?; printf "%s\\n" "$code" > /workspace/results/outer-exit.txt'
            row['launch_intent']=time.time();publish()
            run(ssh+['nohup bash -c '+shlex.quote(cmd)+' < /dev/null > /dev/null 2>&1 &'])
            row.update(status='admission',launched=time.time(),input_tar_sha256=input_hash);publish();seen=set();failures=0
            while True:
                try:
                    row['worker']=remote_json(ssh,'/workspace/results/dr2x2-eval/worker.json')
                    parity=remote_json(ssh,'/workspace/results/dr2x2-eval/linux-parity.json')
                    if parity:row['parity']=parity;row['status']='evaluating' if parity['passed'] else 'parity-failed'
                    raw=run(ssh+['if test -d /workspace/results/dr2x2-eval/evaluation; then find /workspace/results/dr2x2-eval/evaluation -name "*.receipt.json" -type f -exec cat {} \\;; fi'])
                    # Receipts are newline-delimited JSON from the canonical writer.
                    for text in raw.splitlines():
                        if not text:continue
                        receipt=json.loads(text);name=receipt['file']
                        if name in seen:continue
                        if Path(name).name!=name:raise ValueError('Unsafe receipt filename')
                        target=folder/'closed-task-backups'/name;target.parent.mkdir(exist_ok=True)
                        run(scp+[address+':/workspace/results/dr2x2-eval/evaluation/'+name,str(target)+'.transfer'],timeout=600)
                        transfer=Path(str(target)+'.transfer')
                        if transfer.stat().st_size!=receipt['bytes'] or file_hash(transfer)!=receipt['sha256']:raise ValueError('Closed task backup differs')
                        with transfer.open('rb') as f:os.fsync(f.fileno())
                        transfer.replace(target);write(target.with_suffix('.receipt.json'),receipt);seen.add(name)
                        row['verified_tasks']=sorted(seen);event(a.root,'closed-task-backup-verified',seed=row['seed'],task=name,hands=receipt['hands']);publish()
                    outer=run(ssh+['if test -f /workspace/results/outer-exit.txt; then cat /workspace/results/outer-exit.txt; fi']).strip()
                    if outer:row['outer_exit']=outer;break
                    failures=0;publish()
                except Exception as exc:
                    failures+=1;row['transport_incident']={'time':time.time(),'consecutive':failures,'error_type':type(exc).__name__};publish()
                    if failures>=20:raise RuntimeError('Transport unavailable10min; protect verified task backups') from exc
                time.sleep(30)
            final=remote_json(ssh,'/workspace/results/dr2x2-eval/finished.json');row['worker_final']=final
            run(ssh+['tar -cf /workspace/final.tar -C /workspace results'],timeout=240)
            expected=run(ssh+['sha256sum /workspace/final.tar']).split()[0];size=int(run(ssh+['stat -c %s /workspace/final.tar']).strip())
            if shutil.disk_usage(a.root).free-size<10*2**30:raise OSError('Off-pod retrieval headroom insufficient')
            transfer=folder/'final.tar.transfer';run(scp+[address+':/workspace/final.tar',str(transfer)],timeout=600)
            if transfer.stat().st_size!=size or file_hash(transfer)!=expected:raise ValueError('Final transport hash/size differs')
            with transfer.open('rb') as f:os.fsync(f.fileno())
            transfer.replace(folder/'final.tar');row['archive_verification']=verify_closed_archive(folder/'final.tar')
            row.update(archive_sha256=expected,archive_bytes=size,archive_path=str(folder/'final.tar'),status='retrieved-complete' if final and final['status']=='complete' and row['outer_exit']=='0' else 'retrieved-incident');publish()
            api(a.key,'/v2/pods/'+row['id'],'DELETE');row['terminated']=time.time();publish()
            event(a.root,'rental-terminated-after-verified-retrieval',seed=row['seed'],pod_id=row['id'],status=row['status'])
            try:coordination('Doctor Research evaluation teardown/hashverified: '+json.dumps({k:row.get(k) for k in ('name','id','seed','status','terminated','archive_sha256')}))
            except Exception:event(a.root,'coordination-update-pending',pod_id=row['id'])
        except Exception as exc:
            row.update(status='operational-incident',failure=str(exc)[:300],requested_stop='Operational incident; preserve partials',incident_epoch=time.time());publish();event(a.root,'operational-incident',seed=row['seed'],error_type=type(exc).__name__)
            if row.get('id') and not row.get('launch_intent'):
                # No science/setup was launched; the provider quote and allocation
                # incident are already retained locally, and no new state is at risk.
                api(a.key,'/v2/pods/'+row['id'],'DELETE');row['terminated']=time.time();publish()
            if not row.get('id'):
                # Reconcile uncertain creation exactly; no replacement or secondPOST.
                for lost in owned_pods(api(a.key,'/v2/pods')['pods'],{row['name']}):
                    row.update(id=lost['id'],created_epoch=datetime.fromisoformat(lost['createdAt'].replace('Z','+00:00')).timestamp(),upper_rate=max(.18,float(lost.get('cost') or .13)+.05));publish()
                    api(a.key,'/v2/pods/'+lost['id'],'DELETE');row['terminated']=time.time();publish()
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(workload,jobs))
    remaining=owned_pods(api(a.key,'/v2/pods')['pods'],names)
    status='evaluation-complete' if all(r.get('status')=='retrieved-complete' for r in jobs) and not remaining else 'incident-needs-recovery'
    write(a.root/'operator-finished.json',{'status':status,'remaining_ids':[p['id'] for p in remaining],'time':time.time(),'upper_cost_usd':quote.get('previous_analysis_upper_usd',0)+estimated_cost(jobs,time.time())})
    event(a.root,status)
    while remaining:
        time.sleep(30);remaining=owned_pods(api(a.key,'/v2/pods')['pods'],names)
    ledger=json.loads((a.root/'watchdog-ledger.json').read_text())
    for row in jobs:
        watched=next((r for r in ledger if r.get('id')==row.get('id')),None)
        if watched and watched.get('terminated') and not row.get('terminated'):
            row.update(terminated=watched['terminated'],forced_teardown=True)
    publish()
    write(a.root/'operator-finished.json',{'status':'evaluation-complete' if all(r['status']=='retrieved-complete' for r in jobs) else 'incomplete-retained',
          'remaining_ids':[],'time':time.time(),'upper_cost_usd':quote.get('previous_analysis_upper_usd',0)+estimated_cost(jobs,time.time())})
    closed.set()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('execute','watch'))
    for name in ('root','key'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('quote','plan','inputs','references','setup','pod-wrapper'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();watch(a) if a.mode=='watch' else execute(a)
