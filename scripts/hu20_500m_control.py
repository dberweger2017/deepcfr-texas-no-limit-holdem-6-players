"""Durable RunPod control: price ceiling, hash acknowledgements, quiet events."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time

from scripts.hu20_platform_pilot import canonical, write
from scripts.mature_cpu_rental_guard import api, check_quote, owned_pods
from src.blueprint.windowed import _hash


def run(cmd, timeout=60):
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout)


def estimated_cost(records, now):
    return sum(max(0, r.get('terminated', now)-r['created_epoch']) *
               r['upper_rate'] / 3600 for r in records if r.get('created_epoch'))


def budget_action(cost, ceiling, reserve):
    return 'stop' if cost >= ceiling-reserve else 'continue'


def connections(row, root):
    endpoint = row['endpoint']
    options = ['-i', str(root/'pod-key'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile='+str(root/'known-hosts')]
    address = endpoint['username']+'@'+endpoint['host']
    return (['ssh', *options, '-p', str(endpoint['port']), address],
            ['scp', *options, '-P', str(endpoint['port'])], address)


def remote_json(ssh, path):
    raw = run(ssh+['if test -f '+shlex.quote(path)+'; then cat '+shlex.quote(path)+'; else printf null; fi'])
    return json.loads(raw)


def remote_write(ssh, path, value):
    # JSON travels on stdin; shell interpolation never sees its contents.
    cmd = 'cat > '+shlex.quote(path+'.tmp')+' && mv '+shlex.quote(path+'.tmp')+' '+shlex.quote(path)
    subprocess.run(ssh+[cmd], input=canonical(value), stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, check=True, timeout=30)


def event(root, kind, **fields):
    with (root/'events.jsonl').open('ab') as stream:
        stream.write(canonical(dict(time=time.time(), kind=kind, **fields))+b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def watch(a):
    """Independent provider watchdog survives controller failure."""
    lease = json.loads((a.root/'lease.json').read_text())
    names = set(lease['names'])
    if len(names) != 3 or lease['ceiling_usd'] != 15:
        raise ValueError('Exactly three prospectively owned names / USD15 ceiling')
    while True:
        try:
            pods = owned_pods(api(a.key, '/v2/pods')['pods'], names)
            records = json.loads((a.root/'pods.json').read_text()) if (a.root/'pods.json').exists() else []
            indexed = {r.get('id'): r for r in records}
            # Reconcile uncertain creation responses by exact frozen names.
            for pod in pods:
                if pod['id'] not in indexed:
                    from datetime import datetime
                    records.append(dict(id=pod['id'], name=pod['name'],
                        created_epoch=datetime.fromisoformat(pod['createdAt'].replace('Z', '+00:00')).timestamp(),
                        upper_rate=max(.18, float(pod.get('cost') or .18)+.05)))
            cost = estimated_cost(records, time.time())
            stop = budget_action(cost, 15, 2)
            heartbeat = a.root/'controller-heartbeat.json'
            controller_age = time.time()-json.loads(heartbeat.read_text())['heartbeat'] if heartbeat.exists() else 0
            if stop == 'stop' or controller_age > 900:
                reason = 'budget reserve' if stop == 'stop' else 'controller unattended/stale'
                if not (a.root/'budget-stop.json').exists():
                    write(a.root/'budget-stop.json', dict(reason=reason, time=time.time(), upper_cost_usd=cost))
                    event(a.root, 'independent-stop-request', reason=reason, upper_cost_usd=cost)
                stop_at = json.loads((a.root/'budget-stop.json').read_text())['time']
                for row in records:
                    if row.get('endpoint') and not row.get('terminated'):
                        try:
                            ssh, _, _ = connections(row, a.root)
                            remote_write(ssh, '/workspace/control.json', dict(lease_until=time.time(), stop=reason))
                        except Exception:
                            pass
                # Retrieval gets ten minutes, charged in the reserved USD2.
                if time.time()-stop_at > 600:
                    for pod in pods:
                        api(a.key, '/v2/pods/'+pod['id'], 'DELETE')
                        event(a.root, 'budget-safety-termination', pod_id=pod['id'], reason=reason)
            write(a.root/'watchdog.json', dict(status='armed', pid=os.getpid(), heartbeat=time.time(),
                  upper_cost_usd=cost, current_owned_ids=[p['id'] for p in pods], controller_age_seconds=controller_age))
            if (a.root/'operator-finished.json').exists() and not pods:
                return
        except Exception as exc:
            write(a.root/'watchdog-error.json', dict(time=time.time(), error_type=type(exc).__name__, error=str(exc)[:300]))
        time.sleep(15)


def retrieve_file(row, item, destination, a):
    ssh, scp, address = connections(row, a.root)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/item['name']
    if path.exists() and _hash(path) == item['sha256']:
        return
    temporary = path.with_suffix(path.suffix+'.transfer')
    run(scp+[address+':/workspace/results/worker/training/'+item['name'], str(temporary)], timeout=240)
    if _hash(temporary) != item['sha256'] or temporary.stat().st_size != item['bytes']:
        raise ValueError('Off-pod transport hash or length mismatch')
    temporary.replace(path)


def emergency_archive(row, a, expected, size):
    """Final-archive failover also avoids staging on a full M4."""
    ssh, _, _ = connections(row, a.root)
    m1 = ['ssh', '-i', str(a.root/'backup-key'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
          'dberweger@100.115.183.86']
    target = '/Users/dberweger/Local/hu20-500m-emergency-backups-20261001/'+str(row['seed'])+'-final-'+row['attempt']+'.tar'
    free = int(run(m1+['python3 -c '+shlex.quote('import shutil; print(shutil.disk_usage("/Users/dberweger/Local").free)')]).strip())
    if free-size < 10*2**30:
        raise OSError('M1 final-archive capacity is insufficient')
    with subprocess.Popen(ssh+['cat /workspace/final.tar'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as read:
        subprocess.run(m1+['cat > '+shlex.quote(target+'.transfer')], stdin=read.stdout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=600)
        read.stdout.close()
        if read.wait(timeout=30):
            raise OSError('Final archive stream did not close successfully')
    code = ('import hashlib,pathlib; p=pathlib.Path('+repr(target+'.transfer')+'); h=hashlib.sha256(); '
            'f=p.open("rb"); [h.update(b) for b in iter(lambda:f.read(1048576),b"")]; '
            'assert h.hexdigest()=='+repr(expected)+'; assert p.stat().st_size=='+str(size)+'; '
            'p.replace('+repr(target)+'); print("verified")')
    run(m1+['python3 -c '+shlex.quote(code)], timeout=600)
    event(a.root, 'final-archive-storage-failover', seed=row['seed'], path=target)
    return target


def backup(row, saved, a):
    """No acknowledgement exists until all destination bytes match."""
    seed = str(row['seed'])
    local = a.root/'artifacts'/seed/row['attempt']
    needed = sum(f['bytes'] for f in saved['files'])
    if shutil.disk_usage(a.root).free-needed >= 10*2**30:
        for item in saved['files']:
            retrieve_file(row, item, local, a)
        receipt = dict(saved, destination='M4', destination_path=str(local), verified=time.time())
    else:
        # Stream bytes without staging a full extra copy on a full M4.
        m1 = ['ssh', '-i', str(a.root/'backup-key'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
              'dberweger@100.115.183.86']
        remote = '/Users/dberweger/Local/hu20-500m-emergency-backups-20261001/'+seed+'/'+row['attempt']
        capacity = int(run(m1+['python3 -c '+shlex.quote('import shutil; print(shutil.disk_usage("/Users/dberweger/Local").free)')]).strip())
        if capacity-needed < 10*2**30:
            raise OSError('Neither verified storage host has safe free capacity')
        run(m1+['mkdir -p '+shlex.quote(remote)])
        ssh, _, _ = connections(row, a.root)
        for item in saved['files']:
            target = remote+'/'+item['name']
            with subprocess.Popen(ssh+['cat '+shlex.quote('/workspace/results/worker/training/'+item['name'])],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE) as read:
                subprocess.run(m1+['cat > '+shlex.quote(target+'.transfer')], stdin=read.stdout,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=240)
                read.stdout.close()
                if read.wait(timeout=30):
                    raise OSError('Pod stream failed before destination verification')
            code = ('import hashlib,pathlib; p=pathlib.Path('+repr(target+'.transfer')+'); h=hashlib.sha256(); '
                    'f=p.open("rb"); [h.update(b) for b in iter(lambda:f.read(1048576),b"")]; '
                    'assert h.hexdigest()=='+repr(item['sha256'])+'; assert p.stat().st_size=='+str(item['bytes'])+'; '
                    'p.replace('+repr(target)+'); print("verified")')
            run(m1+['python3 -c '+shlex.quote(code)], timeout=240)
        receipt = dict(saved, destination='M1', destination_path=remote, verified=time.time())
        event(a.root, 'storage-failover', seed=row['seed'], id=saved['id'], destination=remote)
    local.mkdir(parents=True, exist_ok=True)
    write(local/(saved['id']+'.receipt.json'), receipt)
    ssh, _, _ = connections(row, a.root)
    remote_write(ssh, '/workspace/results/worker/training/ack/'+saved['id']+'.json',
                 dict(id=saved['id'], files=saved['files'], destination=receipt['destination'], verified=receipt['verified']))
    event(a.root, 'verified-milestone', seed=row['seed'], id=saved['id'], nodes=saved['completed_nodes'],
          entries=saved['entries'], destination=receipt['destination'])
    # Only superseded verified temporary backups are expendable. Permanent
    # checkpoints, policies, records and all incident evidence remain retained.
    receipts = sorted(local.glob('*.receipt.json'), key=lambda p: json.loads(p.read_text())['iteration'])
    for old_path in receipts[:-2]:
        old = json.loads(old_path.read_text())
        if not old['permanent'] and not old.get('spec') and old['destination'] == 'M4':
            for item in old['files']:
                if item['name'].startswith('checkpoint-'):
                    path = Path(old['destination_path'])/item['name']
                    if path.exists() and _hash(path) == item['sha256']:
                        path.unlink()
    if saved.get('spec'):
        spec = dict(saved['spec'])
        spec.update(checkpoint_path=receipt['destination_path']+'/'+saved['files'][0]['name'],
                    path=receipt['destination_path']+'/'+saved['files'][1]['name'])
        task = dict(spec=spec, destination=receipt['destination'], receipt=receipt)
        queue = a.root/'evaluation-queue'
        queue.mkdir(exist_ok=True)
        write(queue/(seed+'-'+str(spec['milestone'])+'.json'), task)


def execute(a):
    if sys.platform != 'darwin':
        raise ValueError('M4 controller required')
    plan = json.loads(a.plan.read_text())
    source = run(['git', 'rev-parse', 'HEAD']).strip()
    if run(['git', 'status', '--porcelain', '--untracked-files=no']).strip():
        raise ValueError('Frozen tracked source required')
    a.root = a.root.resolve()
    a.root.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(a.root).free < 12*2**30:
        raise OSError('Insufficient initial M4 space')
    names = ['dr-research-500m-'+str(p['seed'])+'-'+str(int(time.time())) for p in plan['parents']]
    write(a.root/'lease.json', dict(names=names, started=time.time(), ceiling_usd=15,
          source=source, plan_sha256=sha256(canonical(plan)).hexdigest()))
    write(a.root/'plan.json', plan)
    # Key for M1 fallback is provisioned before rental, with a verified test.
    run(['ssh-keygen', '-t', 'ed25519', '-N', '', '-f', str(a.root/'pod-key'), '-C', 'dr-research-hu20-500m'])
    shutil.copy2(a.backup_key, a.root/'backup-key')
    os.chmod(a.root/'backup-key', 0o600)
    write(a.root/'controller-heartbeat.json', dict(heartbeat=time.time(), pid=os.getpid()))
    with (a.root/'watchdog.log').open('w') as log:
        guardian = subprocess.Popen([sys.executable, '-m', 'scripts.hu20_500m_control', 'watch',
                    '--root', str(a.root), '--key', str(a.key)], stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
    for _ in range(30):
        if (a.root/'watchdog.json').exists():
            break
        if guardian.poll() is not None:
            raise RuntimeError('Independent spending watchdog did not arm')
        time.sleep(1)
    else:
        raise RuntimeError('Watchdog arming timed out')
    catalog = api(a.key, '/v2/catalog/cpus')
    write(a.root/'live-catalog.json', catalog)
    flavor = next(c for c in catalog['cpus'] if c['id'] == 'cpu5m')
    if flavor['price']['securePerVcpu']*2 > .13+1e-9:
        raise ValueError('Live initial compute price exceeds USD0.13')
    records, lock = [], threading.Lock()
    closing = threading.Event()
    def publish():
        with lock:
            write(a.root/'pods.json', records)
    def leases():
        while not closing.is_set():
            write(a.root/'controller-heartbeat.json', dict(heartbeat=time.time(), pid=os.getpid()))
            for row in list(records):
                if row.get('endpoint') and not row.get('terminated'):
                    try:
                        stop = (json.loads((a.root/'budget-stop.json').read_text())['reason']
                                if (a.root/'budget-stop.json').exists() else row.get('requested_stop'))
                        ssh, _, _ = connections(row, a.root)
                        remote_write(ssh, '/workspace/control.json', dict(lease_until=time.time()+300, stop=stop))
                    except Exception:
                        pass
            closing.wait(20)
    thread = threading.Thread(target=leases, daemon=True)
    thread.start()
    for parent, name in zip(plan['parents'], names):
        row = dict(name=name, seed=parent['seed'], parent=parent, attempt='attempt-1', status='creating')
        records.append(row)
        publish()
        try:
            pod = api(a.key, '/v2/pods', 'POST', dict(name=name, cloud='SECURE',
                cpu=dict(id='cpu5m', vcpuCount=2), image=plan['rental']['image'], disk=30,
                ports=['22/tcp'], startSsh=True,
                env={'PUBLIC_KEY': (a.root/'pod-key.pub').read_text().strip()}))
            pod = pod.get('pod', pod)
            from datetime import datetime
            row.update(id=pod['id'], created_epoch=datetime.fromisoformat(pod['createdAt'].replace('Z', '+00:00')).timestamp(),
                       quote=pod, upper_rate=float(pod['cost'])+.05, status='provisioning')
            if not check_quote(dict(cpu_id='cpu5m', vcpus=2, ram_gb=16), pod, .18):
                raise ValueError('Actual CPU/RAM/total quote differs from frozen initial shape')
        except Exception as exc:
            row.update(status='allocation-incident', failure=str(exc)[:300])
            event(a.root, 'allocation-incident', seed=parent['seed'], reason=str(exc)[:300])
        publish()

    def workload(row):
        try:
            if not row.get('id'):
                return
            for _ in range(90):
                pod = api(a.key, '/v2/pods/'+row['id'])
                if pod.get('ssh', {}).get('direct') and pod['status'] == 'RUNNING':
                    row['endpoint'] = pod['ssh']['direct']
                    break
                time.sleep(10)
            else:
                raise TimeoutError('Provisioning did not expose SSH in 15 minutes')
            ssh, scp, address = connections(row, a.root)
            for _ in range(30):
                try:
                    run(ssh+['true'])
                    break
                except Exception:
                    time.sleep(5)
            else:
                raise TimeoutError('Allocated pod SSH never became ready')
            folder = a.root/str(row['seed'])
            folder.mkdir(exist_ok=True)
            parent = dict(row['parent'], checkpoint_path='/workspace/parent.json.gz')
            write(folder/'parent.json', parent)
            run(ssh+['mkdir -p /workspace/results'])
            run(scp+[row['parent']['checkpoint_path'], address+':/workspace/parent.json.gz'], timeout=240)
            run(scp+[str(folder/'parent.json'), address+':/workspace/parent.json'])
            run(scp+[str(a.references/str(row['seed'])/'direct/result.json'), address+':/workspace/reference.json'])
            run(scp+['scripts/hu20_500m_setup.sh', address+':/workspace/setup.sh'])
            remote_write(ssh, '/workspace/control.json', dict(lease_until=time.time()+300, stop=None))
            run(ssh+['nohup bash -c '+shlex.quote('bash /workspace/setup.sh '+shlex.quote(source)+
                   ' > /workspace/results/setup.log 2>&1; code=$?; printf "%s\\n" "$code" > /workspace/results/outer-exit.txt')+
                   ' < /dev/null > /dev/null 2>&1 &'])
            row.update(status='running', launched=time.time())
            publish()
            verified = set()
            failures = 0
            while True:
                try:
                    worker = remote_json(ssh, '/workspace/results/worker/worker.json')
                    row['worker'] = worker
                    raw = run(ssh+['if test -f /workspace/results/worker/training/saved.jsonl; then cat /workspace/results/worker/training/saved.jsonl; fi'])
                    saves = [json.loads(line) for line in raw.splitlines() if line]
                    for saved in saves:
                        if saved['id'] not in verified:
                            backup(row, saved, a)
                            verified.add(saved['id'])
                            row['latest_verified'] = saved
                            publish()
                    final = remote_json(ssh, '/workspace/results/worker/finished.json')
                    outer = run(ssh+['if test -f /workspace/results/outer-exit.txt; then cat /workspace/results/outer-exit.txt; fi']).strip()
                    if final or outer:
                        row['worker_final'] = final
                        row['outer_exit'] = outer
                        break
                    if (a.root/'budget-stop.json').exists():
                        row['status'] = 'budget-stopping'
                    failures = 0
                    publish()
                except Exception as exc:
                    failures += 1
                    row['connection_incident'] = dict(time=time.time(), consecutive=failures, error_type=type(exc).__name__)
                    publish()
                    if failures == 3:
                        event(a.root, 'transport-incident', seed=row['seed'], error_type=type(exc).__name__)
                    if failures >= 20:
                        raise RuntimeError('Transport unavailable for ten minutes; preserve verified recovery') from exc
                time.sleep(30)
            for _ in range(60):
                if run(ssh+['if test -f /workspace/results/outer-exit.txt; then cat /workspace/results/outer-exit.txt; fi']).strip():
                    break
                time.sleep(2)
            else:
                raise TimeoutError('Worker closed but outer setup log is not closed')
            # Logs are closed before packing. Every transferred byte is verified
            # before deleting this rental; metadata failures cannot erase peers.
            run(ssh+['tar -cf /workspace/final.tar -C /workspace results; sha256sum /workspace/final.tar'], timeout=240)
            expected = run(ssh+['sha256sum /workspace/final.tar']).split()[0]
            size = int(run(ssh+['stat -c %s /workspace/final.tar']).strip())
            if shutil.disk_usage(a.root).free-size < 10*2**30:
                archive = emergency_archive(row, a, expected, size)
            else:
                run(scp+[address+':/workspace/final.tar', str(folder/'final.tar.transfer')], timeout=600)
                if _hash(folder/'final.tar.transfer') != expected:
                    raise ValueError('Final archive transport hash mismatch')
                (folder/'final.tar.transfer').replace(folder/'final.tar')
                archive = str(folder/'final.tar')
            row.update(archive_sha256=expected, archive_bytes=size, archive_path=archive,
                       status='retrieved-complete' if final and final['status']=='complete' else 'retrieved-incident')
            publish()
            api(a.key, '/v2/pods/'+row['id'], 'DELETE')
            row['terminated'] = time.time()
            publish()
            event(a.root, 'rental-terminated-after-retrieval', seed=row['seed'], status=row['status'])
        except Exception as exc:
            row.update(status='operational-incident', failure=str(exc)[:300],
                       requested_stop='Operational incident; protect last complete state')
            event(a.root, 'operational-incident', seed=row['seed'], reason=str(exc)[:300])
            # Preserve the rental until bounded agent recovery or the independent
            # budget reserve forces shutdown. Other lineages continue unchanged.
            publish()

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(workload, records))
    remaining = owned_pods(api(a.key, '/v2/pods')['pods'], names)
    status = 'training-complete' if all(r['status']=='retrieved-complete' for r in records) and not remaining else 'incident-needs-recovery'
    write(a.root/'operator-finished.json', dict(status=status, time=time.time(), remaining_ids=[p['id'] for p in remaining],
          upper_cost_usd=estimated_cost(records, time.time())))
    if not remaining:
        closing.set()
    event(a.root, status)
    while remaining:
        time.sleep(30)
        remaining = owned_pods(api(a.key, '/v2/pods')['pods'], names)
    closing.set()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('run', 'watch'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--key', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--references', type=Path)
    p.add_argument('--backup-key', type=Path)
    a = p.parse_args()
    if a.phase == 'watch':
        watch(a)
    else:
        execute(a)
