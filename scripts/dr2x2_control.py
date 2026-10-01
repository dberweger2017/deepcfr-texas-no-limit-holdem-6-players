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
import tarfile
import time
import tomllib
import urllib.parse
import urllib.request

from scripts.hu20_platform_pilot import canonical, write
from scripts.mature_cpu_rental_guard import api, check_quote, owned_pods
from src.blueprint.windowed import _hash


def run(cmd, timeout=60):
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout)


def estimated_cost(records, now):
    return sum(max(0, r.get('terminated', now)-r['created_epoch']) *
               r['upper_rate'] / 3600 for r in records if r.get('created_epoch') is not None)


def verify_archive(path):
    """Rehash every closed worker artifact from the retrieved tar without extraction."""
    with tarfile.open(path,'r:') as archive:
        members={member.name:member for member in archive.getmembers()}
        for name,member in members.items():
            parts=Path(name).parts
            if Path(name).is_absolute() or '..' in parts or not parts or parts[0]!='results' or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe final archive member')
        if 'results/worker/manifest.json' not in members:
            exit_member=members.get('results/outer-exit.txt')
            if exit_member is None or archive.extractfile(exit_member).read().strip()==b'0':
                raise ValueError('Successful/unfinished setup has no closed worker manifest')
            if any(n.startswith('results/worker/') for n in members):
                raise ValueError('Partial worker must be sealed before teardown')
            return dict(passed=True,setup_failure_evidence_retained=True,verified_files=0)
        manifest=json.load(archive.extractfile(members['results/worker/manifest.json']))
        for relative,item in manifest['files'].items():
            member=members['results/worker/'+relative]
            if member.size!=item['bytes']:
                raise ValueError('Closed artifact size mismatch')
            digest=sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda:stream.read(1048576),b''):
                    digest.update(chunk)
            if digest.hexdigest()!=item['sha256']:
                raise ValueError('Closed artifact hash mismatch')
        return dict(passed=True,verified_files=len(manifest['files']))


def budget_action(cost, ceiling, reserve):
    return 'stop' if cost >= ceiling-reserve else 'continue'


def verify_credit(key_path, out):
    key = tomllib.loads(key_path.read_text())['apikey']
    url = 'https://api.runpod.io/graphql?'+urllib.parse.urlencode({'api_key':key})
    query = {'query':'query { myself { clientBalance currentSpendPerHr } }'}
    request = urllib.request.Request(url, data=json.dumps(query).encode(),
              headers={'Content-Type':'application/json', 'User-Agent':'runpodctl/1.14.5'})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read())['data']['myself']
    except Exception:
        raise RuntimeError('Private campaign credit verification failed') from None
    sufficient = float(result['clientBalance']) >= 16
    write(out, dict(checked=time.time(), sufficient_for_ceiling=sufficient,
                   unrelated_current_spend_nonzero=float(result['currentSpendPerHr'])>0))
    if not sufficient:
        raise ValueError('Verified account credit cannot cover authorized ceiling')
    return result


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
    if len(names) != 3 or lease['ceiling_usd'] != 16 or not all(n.startswith('dr2x2-C-') for n in names):
        raise ValueError('Exactly three C-owned names / approved USD16 ceiling')
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
                        upper_rate=max(lease['rates_by_name'][pod['name']], float(pod.get('cost') or lease['rates_by_name'][pod['name']])+.05)))
            ledger_path=a.root/'watchdog-cost-records.json'
            previous=json.loads(ledger_path.read_text()) if ledger_path.exists() else []
            indexed={r['id']:dict(r) for r in previous if r.get('id')}
            for row in records:
                if row.get('id'):
                    old=indexed.get(row['id'],{})
                    indexed[row['id']]=dict(row)
                    if old.get('terminated'):
                        indexed[row['id']]['terminated']=old['terminated']
            records=list(indexed.values())
            write(ledger_path,records)
            cost = estimated_cost(records, time.time())
            stop = budget_action(cost, lease['training_subcap_usd'], lease['shutdown_reserve_usd'])
            heartbeat = a.root/'controller-heartbeat.json'
            controller_age = time.time()-json.loads(heartbeat.read_text())['heartbeat'] if heartbeat.exists() else time.time()-lease['started']
            if stop == 'stop' or controller_age > 900:
                reason = 'training budget reserve' if stop == 'stop' else 'controller unattended/stale'
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
                        for record in records:
                            if record.get('id')==pod['id']:
                                record['terminated']=time.time()
                        write(a.root/'watchdog-cost-records.json',records)
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
    with temporary.open('rb') as saved:
        os.fsync(saved.fileno())
    temporary.replace(path)
    directory = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def backup(row, saved, a):
    """All destination hashes precede ACK; unverified backups are never rotated."""
    local = a.root/'artifacts'/row['cell']/str(row['seed'])/row['attempt']
    needed = sum(f['bytes'] for f in saved['files'])
    if shutil.disk_usage(a.root).free-needed < 10*2**30:
        raise OSError('M1 backup headroom insufficient; preserve pod copies')
    for item in saved['files']:
        if Path(item['name']).name != item['name']:
            raise ValueError('Unsafe backup name')
        retrieve_file(row, item, local, a)
    receipt = dict(saved, destination='M1', destination_path=str(local), verified=time.time())
    write(local/(saved['id']+'.receipt.json'), receipt)
    ssh, _, _ = connections(row, a.root)
    remote_write(ssh, '/workspace/results/worker/training/ack/'+saved['id']+'.json',
                 dict(id=saved['id'], files=saved['files'], destination='M1', verified=receipt['verified']))
    event(a.root, 'verified-milestone', cell=row['cell'], seed=row['seed'], id=saved['id'],
          nodes=saved['completed_nodes'], entries=saved['entries'], destination='M1')
    receipts = sorted(local.glob('*.receipt.json'), key=lambda p: json.loads(p.read_text())['iteration'])
    for old_path in receipts[:-2]:
        old = json.loads(old_path.read_text())
        if not old['permanent']:
            for item in old['files']:
                if item['name'].startswith('checkpoint-'):
                    path = local/item['name']
                    if path.exists():
                        if _hash(path) != item['sha256']:
                            raise ValueError('Refuse to rotate changed verified backup')
                        path.unlink()


def coordination(message):
    """Append ownership only; never inspect or allocate #136 M4 compute."""
    command = 'cat >> /tmp/DR_RESEARCH_M4_COORDINATION.txt'
    subprocess.run(['ssh','-o','HostName=100.122.216.94','-o','BatchMode=yes','-o','ConnectTimeout=15',
                    'm4',command], input=('\n'+message+'\n').encode(),
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=30)


def execute(a):
    if sys.platform != 'darwin':
        raise ValueError('M1 controller required')
    plan = json.loads(a.plan.read_text())
    source = run(['git', 'rev-parse', 'HEAD']).strip()
    if run(['git', 'status', '--porcelain', '--untracked-files=no']).strip():
        raise ValueError('Frozen tracked source required')
    a.root = a.root.resolve()
    a.root.mkdir(parents=True, exist_ok=False)
    os.chmod(a.root,0o700)
    if shutil.disk_usage(a.root).free < 40*2**30:
        raise OSError('Insufficient M1 capacity for retained artifacts and headroom')
    funds = verify_credit(a.key, a.root/'credit-check.json')
    private = a.key.parent/'campaign-billing-baseline-private.json'
    write(private, dict(time=time.time(), account=funds))
    os.chmod(private, 0o600)
    if plan['schema']!='dr2x2-c-only-100m-campaign-v1' or len(plan['jobs'])!=3 or any(p['cell']!='C' for p in plan['jobs']) or not plan['owner_approved_budget']['approval'].startswith('APPROVED'):
        raise ValueError('Owner-approved budget required')
    jobs = [dict(p, name='dr2x2-'+p['cell']+'-'+str(p['seed'])+'-'+str(int(time.time()))) for p in plan['jobs']]
    names = [p['name'] for p in jobs]
    rates = {p['name']:plan['cells'][p['cell']]['max_total_usd_per_hour'] for p in jobs}
    write(a.root/'lease.json', dict(names=names, started=time.time(), ceiling_usd=16, training_subcap_usd=4, shutdown_reserve_usd=1, rates_by_name=rates, max_concurrent_pods=3,
          source=source, plan_sha256=sha256(canonical(plan)).hexdigest()))
    write(a.root/'plan.json', plan)
    # Only the dedicated public SSH key enters owned pods.
    run(['ssh-keygen', '-t', 'ed25519', '-N', '', '-f', str(a.root/'pod-key'), '-C', 'dr2x2-owned-cd'])
    write(a.root/'controller-heartbeat.json', dict(heartbeat=time.time(), pid=os.getpid()))
    with (a.root/'watchdog.log').open('w') as log:
        guardian = subprocess.Popen([sys.executable, '-m', 'scripts.dr2x2_control', 'watch',
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
    if flavor['price']['securePerVcpu'] > .065+1e-9:
        raise ValueError('Live initial compute price exceeds USD0.13')
    reference_inventory=json.loads((a.references/'manifest.json').read_text())['files']
    if json.loads((a.references/'manifest.json').read_text())['source']!=source:
        raise ValueError('M1 references are not the frozen production source')
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
    def create(parent):
        name = parent['name']
        shape = plan['cells'][parent['cell']]
        row = dict(name=name, cell=parent['cell'], seed=parent['seed'], parent={k:v for k,v in parent.items() if k!='name'}, attempt='attempt-1', status='creating')
        records.append(row)
        publish()
        try:
            pod = api(a.key, '/v2/pods', 'POST', dict(name=name, cloud='SECURE',
                cpu=dict(id='cpu5m', vcpuCount=shape['vcpus']), image=plan['rental']['image'], disk=30,
                dataCenterIds=plan['rental']['data_center_ids'],
                ports=['22/tcp'], startSsh=True,
                env={'PUBLIC_KEY': (a.root/'pod-key.pub').read_text().strip()}))
            pod = pod.get('pod', pod)
            from datetime import datetime
            row.update(id=pod['id'], created_epoch=datetime.fromisoformat(pod['createdAt'].replace('Z', '+00:00')).timestamp(),
                       quote=pod, upper_rate=max(shape['max_total_usd_per_hour'],float(pod['cost'])+.05), status='provisioning')
            if not check_quote(shape, pod, shape['max_compute_usd_per_hour']):
                raise ValueError('Actual CPU/RAM/total quote differs from frozen initial shape')
        except Exception as exc:
            row.update(status='allocation-incident', failure=str(exc)[:300])
            event(a.root, 'allocation-incident', seed=parent['seed'], reason=str(exc)[:300])
            # A failed quote or uncertain create is never a training license.
            for lost in owned_pods(api(a.key, '/v2/pods')['pods'], {name}):
                from datetime import datetime
                row.update(id=lost['id'],
                    created_epoch=datetime.fromisoformat(lost['createdAt'].replace('Z', '+00:00')).timestamp(),
                    upper_rate=max(shape['max_total_usd_per_hour'],float(lost.get('cost') or shape['max_total_usd_per_hour'])+.05))
                api(a.key, '/v2/pods/'+lost['id'], 'DELETE')
                row.update(id=lost['id'], terminated=time.time())
        publish()
        coordination('Doctor Research dr2x2 ownership: '+json.dumps({k:row.get(k) for k in ('name','id','cell','seed','status','terminated')},sort_keys=True))
        return row

    def workload(row):
        try:
            if not row.get('id') or row['status'] != 'provisioning':
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
            folder = a.root/'jobs'/row['cell']/str(row['seed'])
            folder.mkdir(parents=True,exist_ok=True)
            parent = dict(row['parent'])
            write(folder/'parent.json', parent)
            run(ssh+['mkdir -p /workspace/results'])
            run(scp+[str(folder/'parent.json'), address+':/workspace/parent.json'])
            reference=a.references/row['cell']/str(row['seed'])/'direct/result.json'
            if _hash(reference)!=reference_inventory[str(reference.relative_to(a.references))]:
                raise ValueError('Reference hash changed before transfer')
            run(scp+[str(reference), address+':/workspace/reference.json'])
            run(ssh+['printf '+shlex.quote(_hash(reference)+'  /workspace/reference.json\n')+' | sha256sum -c -'])
            run(scp+['scripts/dr2x2_setup.sh', address+':/workspace/setup.sh'])
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
                raise OSError('M1 final-archive headroom insufficient; preserve pod')
            run(scp+[address+':/workspace/final.tar', str(folder/'final.tar.transfer')], timeout=600)
            if _hash(folder/'final.tar.transfer') != expected or (folder/'final.tar.transfer').stat().st_size != size:
                raise ValueError('Final archive transport hash/length mismatch')
            with (folder/'final.tar.transfer').open('rb') as stream:
                os.fsync(stream.fileno())
            (folder/'final.tar.transfer').replace(folder/'final.tar')
            archive = str(folder/'final.tar')
            row['archive_verification']=verify_archive(Path(archive))
            write(folder/'retrieval-verification.json',row['archive_verification'])
            row.update(archive_sha256=expected, archive_bytes=size, archive_path=archive,
                       status='retrieved-complete' if final and final['status']=='complete' else 'retrieved-incident')
            publish()
            api(a.key, '/v2/pods/'+row['id'], 'DELETE')
            row['terminated'] = time.time()
            publish()
            event(a.root, 'rental-terminated-after-retrieval', cell=row['cell'], seed=row['seed'], status=row['status'])
            try:
                coordination('Doctor Research dr2x2 teardown/hash-verified retrieval: '+json.dumps({k:row.get(k) for k in ('name','id','cell','seed','status','archive_sha256','terminated')},sort_keys=True))
            except Exception:
                event(a.root,'coordination-note-update-pending',pod_id=row['id'])
        except Exception as exc:
            row.update(status='operational-incident', failure=str(exc)[:300],
                       requested_stop='Operational incident; protect last complete state',incident_epoch=time.time())
            event(a.root, 'operational-incident', seed=row['seed'], reason=str(exc)[:300])
            # Preserve the rental until bounded agent recovery or the independent
            # budget reserve forces shutdown. Other lineages continue unchanged.
            publish()

    coordination('Doctor Research APPROVED C-only pivot; operational USD4 training subcap within USD16 ceiling, D deferred before creation. Frozen owned names: '+json.dumps(names)+'. Max3Cpods/one worker; M4 PR136 scientific reservation unchanged. C16GB, separate ledger/root '+str(a.root))
    try:
        for wave in plan['waves']:
            if (a.root/'budget-stop.json').exists():
                raise RuntimeError('Budget safety stop before next wave')
            rows=[]
            for spec in wave:
                job=next(p for p in jobs if p['cell']==spec['cell'] and p['seed']==spec['seed'])
                rows.append(create(job))
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(workload, rows))
            if not all(r.get('terminated') and r['status']=='retrieved-complete' for r in rows):
                raise RuntimeError('Wave incident; preserve evidence, no automatic failed-lineage retry')
    except Exception as exc:
        event(a.root,'campaign-incident',reason=str(exc)[:300])

    remaining = owned_pods(api(a.key, '/v2/pods')['pods'], names)
    status = 'training-complete' if len(records)==3 and all(r['status']=='retrieved-complete' for r in records) and not remaining else 'incident-needs-recovery'
    write(a.root/'operator-finished.json', dict(status=status, time=time.time(), remaining_ids=[p['id'] for p in remaining],
          upper_cost_usd=estimated_cost(records, time.time())))
    if not remaining:
        closing.set()
    event(a.root, status)
    while remaining:
        time.sleep(30)
        remaining = owned_pods(api(a.key, '/v2/pods')['pods'], names)
    closing.set()
    latest=json.loads((a.root/'watchdog-cost-records.json').read_text()) if (a.root/'watchdog-cost-records.json').exists() else records
    write(a.root/'final-cost-estimate.json',dict(upper_cost_usd=estimated_cost(latest,time.time()),records=latest,settled_invoice=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('run', 'watch'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--key', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--references', type=Path)
    a = p.parse_args()
    if a.phase == 'watch':
        watch(a)
    else:
        execute(a)
