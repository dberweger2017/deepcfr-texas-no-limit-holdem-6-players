"""M1 controller for three independently owned CPU pods and verified retrieval."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

from scripts.hu20_cards_v2_rental_guard import validate
from scripts.mature_cpu_rental_guard import api,check_quote,owned_pods,write
from src.diagnostics.saved_hu20 import file_hash


def run(command,timeout=60):
    return subprocess.check_output(command,stderr=subprocess.STDOUT,text=True,timeout=timeout)


def verify_archive(folder):
    archive=folder/'results.tar'
    expected=(folder/'results.tar.sha256').read_text().split()[0]
    if file_hash(archive)!=expected:raise ValueError('Archive transport hash mismatch')
    with tarfile.open(archive) as tar:
        for item in tar:
            parts=Path(item.name).parts
            if item.name.startswith('/') or '..' in parts or parts[0]!='results' or not(item.isfile() or item.isdir()):
                raise ValueError('Unsafe owned archive member')
        tar.extractall(folder,filter='data')
    root=folder/'results/work';manifest=json.loads((root/'manifest.json').read_text())
    for name,spec in manifest.items():
        p=root/name
        if p.stat().st_size!=spec['bytes'] or file_hash(p)!=spec['sha256']:
            raise ValueError('Retrieved work artifact hash mismatch')
    write(folder/'verified.json',{'archive_sha256':expected,'files':len(manifest),'verified':time.time()})


def execute(args):
    if sys.platform!='darwin':raise ValueError('M1 control host required')
    plan=json.loads(args.plan.read_text());source=run(['git','rev-parse','HEAD']).strip()
    if run(['git','status','--porcelain']).strip():raise ValueError('Frozen clean source required')
    if not plan['budget_approval'].startswith('approved') or plan['provider']['max_total_cost_usd']!=10:
        raise ValueError('Approved $10 plan required')
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False);root.chmod(0o700)
    started=time.time();deadline=started+18000;names=[f'new-guy-hu20-card-v2-{seed}-{int(started)}' for seed in plan['seeds']]
    lease={'names':names,'started':started,'deadline':deadline,'max_hourly_per_pod':.57,'source_sha':source,
           'plan_sha256':sha256(args.plan.read_bytes()).hexdigest()};validate(lease);write(root/'lease.json',lease)
    keyfile=root/'ssh-key';run(['ssh-keygen','-t','ed25519','-N','','-f',str(keyfile),'-C','new-guy-card-v2'])
    public=keyfile.with_suffix('.pub').read_text().strip()
    catalog=api(args.key,'/v2/catalog/cpus');write(root/'live-catalog.json',catalog)
    cpu=next(c for c in catalog['cpus'] if c['id']=='cpu5m')
    if cpu['price']['securePerVcpu']*8>.52:raise ValueError('Live rate exceeds approval')
    with (root/'watchdog.log').open('w') as log:
        watcher=subprocess.Popen([sys.executable,'-m','scripts.hu20_cards_v2_rental_guard','--lease',str(root/'lease.json'),'--key',str(args.key)],
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(30):
        if (root/'watchdog.json').exists():break
        if watcher.poll() is not None:raise ValueError('Independent watchdog failed')
        time.sleep(1)
    if not (root/'watchdog.json').exists():raise ValueError('Independent watchdog not armed')
    rows=[]
    try:
        for seed,name in zip(plan['seeds'],names):
            state=json.loads((root/'watchdog.json').read_text())
            if state['status']!='armed' or time.time()-state['heartbeat']>45:raise ValueError('Unhealthy cutoff watcher')
            row={'name':name,'seed':seed,'status':'creation-pending','attempted':time.time()};rows.append(row);write(root/'pods.json',rows)
            try:
                pod=api(args.key,'/v2/pods','POST',{'name':name,'cloud':'SECURE','cpu':{'id':'cpu5m','vcpuCount':8},
                    'image':'runpod/base:0.7.0-ubuntu2004','disk':40,'ports':['22/tcp'],'startSsh':True,
                    'env':{'PUBLIC_KEY':public},'dataCenterIds':[]})
                pod=pod.get('pod',pod);row.update(id=pod['id'],created_at=pod['createdAt'],actual_cpu=pod.get('cpu'),
                    total_rate=pod.get('cost'),status='provisioning')
                if not check_quote({'cpu_id':'cpu5m','vcpus':8,'ram_gb':64},pod,.57):
                    raise ValueError('Actual quote differs from approved CPU/RAM/rate')
            except Exception as exc:
                row.update(status='creation-failed',reason=str(exc))
                for pod in owned_pods(api(args.key,'/v2/pods')['pods'],{name}):api(args.key,'/v2/pods/'+pod['id'],'DELETE')
            write(root/'pods.json',rows)
        def workload(row):
            folder=root/str(row['seed']);folder.mkdir();ssh=None;remote=None;scp=None
            try:
                until=min(deadline-900,time.time()+900)
                while time.time()<until:
                    pod=api(args.key,'/v2/pods/'+row['id']);endpoint=pod.get('ssh',{}).get('direct')
                    if pod['status']=='RUNNING' and endpoint:break
                    time.sleep(10)
                else:raise TimeoutError('Provisioning readiness cutoff')
                remote=endpoint['username']+'@'+endpoint['host'];options=['-i',str(keyfile),'-o','BatchMode=yes','-o','ConnectTimeout=20',
                    '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(root/'known-hosts')]
                ssh=['ssh',*options,'-p',str(endpoint['port']),remote];scp=['scp',*options,'-P',str(endpoint['port'])]
                row['endpoint']=endpoint;write(folder/'endpoint.json',endpoint)
                for _ in range(30):
                    try:run(ssh+['true']);break
                    except subprocess.CalledProcessError:time.sleep(5)
                else:raise TimeoutError('SSH readiness cutoff')
                run(ssh+['mkdir -p /workspace/baseline /workspace/results'])
                run(scp+['-r',str(args.reference.resolve()),remote+':/workspace/reference'],timeout=300)
                baseline=json.loads(Path('configs/diagnostics/hu20-stackoff-v1.json').read_text())
                spec=next(s for s in baseline['models'] if s['seed']==row['seed'] and s['milestone']==100000000)
                for field,hashfield in [('path','sha256'),('checkpoint_path','checkpoint_sha256')]:
                    path=args.baseline/spec[field]
                    if file_hash(path)!=spec[hashfield]:raise ValueError('Cached baseline hash differs')
                    run(scp+[str(path),remote+':/workspace/baseline/'+spec[field]],timeout=300)
                run(scp+['scripts/hu20_cards_v2_setup.sh',remote+':/workspace/setup.sh'])
                command=('nohup timeout '+str(max(1,int(deadline-time.time()-900)))+' bash /workspace/setup.sh '+shlex.quote(source)+' '+str(deadline-900)+' '+str(row['seed'])+
                         ' > /workspace/results/setup.log 2>&1 < /dev/null & echo $! > /workspace/owned-worker.pid')
                run(ssh+[command]);row.update(status='running',started=time.time());write(folder/'state.json',row)
                while time.time()<deadline-600:
                    # Work is detached; a transient SSH loss never restarts it.
                    try:
                        output=run(ssh+['cat /workspace/results/work/worker.json 2>/dev/null || true'])
                        if output:
                            state=json.loads(output);write(folder/'worker.json',state)
                            if state['status']!='running':break
                        alive=run(ssh+['kill -0 "$(cat /workspace/owned-worker.pid)" 2>/dev/null && echo alive || true']).strip()
                        if not alive:break
                    except (subprocess.SubprocessError,json.JSONDecodeError):pass
                    time.sleep(15)
                row['status']='retrieving';write(folder/'state.json',row)
                run(ssh+['tar -cf /workspace/results.tar -C /workspace results && sha256sum /workspace/results.tar > /workspace/results.tar.sha256'],timeout=300)
                run(scp+[remote+':/workspace/results.tar',str(folder/'results.tar')],timeout=max(1,int(deadline-time.time()-120)))
                run(scp+[remote+':/workspace/results.tar.sha256',str(folder/'results.tar.sha256')])
                verify_archive(folder);row.update(status='verified',retrieved=time.time())
            except Exception as exc:row.update(status='failed',reason=str(exc))
            finally:
                api(args.key,'/v2/pods/'+row['id'],'DELETE');row['terminated']=time.time();write(folder/'state.json',row)
            return row
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(workload,row) for row in rows if row['status']=='provisioning']
            for future in as_completed(futures):future.result();write(root/'pods.json',rows)
    finally:
        for pod in owned_pods(api(args.key,'/v2/pods')['pods'],names):api(args.key,'/v2/pods/'+pod['id'],'DELETE')
        remaining=owned_pods(api(args.key,'/v2/pods')['pods'],names)
        upper=sum(.57*(r.get('terminated',time.time())-r['attempted'])/3600 for r in rows)
        write(root/'pods.json',rows);write(root/'operator-finished.json',{'remaining_owned_ids':[p['id'] for p in remaining],
            'compute_and_disk_upper_usd':upper,'settled_billing':'not yet verified','finished':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','key','root','reference','baseline'):p.add_argument('--'+name,type=Path,required=True)
    execute(p.parse_args())
