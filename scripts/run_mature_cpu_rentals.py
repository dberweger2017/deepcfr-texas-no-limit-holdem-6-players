"""M4 coordinator: six isolated rentals, serial archive verification, no M1 work."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from scripts.mature_cpu_rental_guard import api, check_quote, owned_pods, write


def run(command, log=None, timeout=60):
    return subprocess.run(command, check=True, stdout=log or subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=timeout, text=True).stdout


def execute(args):
    if sys.platform != 'darwin':
        raise ValueError('M4 control/retrieval host required')
    plan = json.loads(args.plan.read_text())
    if plan['budget_approval'].startswith('pending') or plan['max_total_cost_usd'] != 4:
        raise ValueError('Approved six-class/four-dollar plan required')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = run(['git', 'rev-parse', 'HEAD']).strip()
    if run(['git', 'status', '--porcelain']).strip():
        raise ValueError('Frozen clean driver source required')
    started = time.time()
    deadline = started + 7200
    configs = plan['configurations']
    names = [f'doctor-research-mature-{c["cpu_id"]}-{int(started)}' for c in configs]
    lease = root / 'lease.json'
    write(lease, {'names': names, 'started': started, 'deadline': deadline,
                 'max_total_cost_usd': 4, 'driver_source': source,
                 'plan_sha256': sha256(args.plan.read_bytes()).hexdigest()})
    (root / 'plan.json').write_bytes(args.plan.read_bytes())
    keyfile = root / 'ssh-key'
    run(['ssh-keygen', '-t', 'ed25519', '-N', '', '-f', str(keyfile), '-C', 'doctor-research-mature-pilot'])
    public_key = keyfile.with_suffix('.pub').read_text().strip()
    watchdog_log = (root / 'watchdog.log').open('w')
    watchdog = subprocess.Popen([sys.executable, '-m', 'scripts.mature_cpu_rental_guard',
        '--lease', str(lease), '--key', str(args.key)], stdout=watchdog_log,
        stderr=subprocess.STDOUT, start_new_session=True)
    watchdog_log.close()
    for _ in range(30):
        if (root / 'watchdog.json').exists():
            break
        if watchdog.poll() is not None:
            raise ValueError('Shutdown watchdog failed before provisioning')
        time.sleep(1)
    if not (root / 'watchdog.json').exists():
        raise ValueError('Shutdown watchdog not armed')
    catalog = api(args.key, '/v2/catalog/cpus')
    write(root / 'live-catalog.json', catalog)
    flavors = {c['id']: c for c in catalog['cpus']}
    # Total price includes a conservative $0.05/h disk allowance per pod.
    maximum_rate = sum(c['max_compute_usd_per_hour'] + .05 for c in configs)
    if maximum_rate * 2 > 4:
        raise ValueError('Two-hour maximum exceeds approved total cap')
    records = []
    note = Path('/tmp/DR_RESEARCH_M4_COORDINATION.txt')
    with note.open('a') as stream:
        stream.write(f'\nDr Research six-class RunPod CLAIM: coordinator {os.getpid()}, root {root}, '
                     f'cutoff {deadline}; network/control only until serial guarded archive verification. '
                     'Up to six Linux trainers; no heavy M1 work.\n')
    for config, name in zip(configs, names):
        guard = json.loads((root / 'watchdog.json').read_text())
        if guard['status'] != 'armed' or time.time() - guard['heartbeat'] > 45:
            raise ValueError('Independent watchdog unhealthy')
        row = dict(config, name=name, status='creation-pending', attempted=time.time())
        records.append(row)
        write(root / 'pods.json', records)
        current = flavors[config['cpu_id']]
        if current['price']['securePerVcpu'] * config['vcpus'] > config['max_compute_usd_per_hour']:
            row.update(status='unavailable', reason='Live rate exceeds frozen ceiling')
            write(root / 'pods.json', records)
            continue
        try:
            pod = api(args.key, '/v2/pods', 'POST', {
                'name': name, 'cloud': 'SECURE',
                'cpu': {'id': config['cpu_id'], 'vcpuCount': config['vcpus']},
                'image': 'runpod/base:0.7.0-ubuntu2004', 'disk': 30,
                'ports': ['22/tcp'], 'startSsh': True, 'env': {'PUBLIC_KEY': public_key}})
            if 'pod' in pod:
                pod = pod['pod']
            row.update(id=pod['id'], created_at=pod['createdAt'],
                       actual_cpu=pod.get('cpu'), total_rate=pod.get('cost'),
                       image=pod['image'], data_center=pod.get('dataCenterId'), status='provisioning')
            write(root / 'pods.json', records)
            if not check_quote(config, pod, config['max_compute_usd_per_hour'] + .05):
                api(args.key, '/v2/pods/' + pod['id'], 'DELETE')
                row.update(status='rejected', reason='Actual shape or total rate differs')
        except Exception as error:
            row.update(status='creation-failed', reason=str(error))
            # Never retry a lost creation response. Reconcile exact name and
            # terminate it; the independently armed watcher covers outages.
            for pod in owned_pods(api(args.key, '/v2/pods')['pods'], {name}):
                row['id'] = pod['id']
                api(args.key, '/v2/pods/' + pod['id'], 'DELETE')
        write(root / 'pods.json', records)

    def workload(row):
        folder = root / row['cpu_id']
        folder.mkdir()
        try:
            limit = min(deadline - 900, time.time() + 900)
            while time.time() < limit:
                pod = api(args.key, '/v2/pods/' + row['id'])
                endpoint = pod.get('ssh', {}).get('direct')
                if endpoint and pod['status'] == 'RUNNING':
                    break
                time.sleep(10)
            else:
                raise TimeoutError('No direct SSH within fixed provisioning allowance')
            row.update(data_center=pod.get('dataCenterId'), actual_cpu=pod.get('cpu'),
                       total_rate=pod['cost'], connected=time.time())
            host, port = endpoint['host'], str(endpoint['port'])
            user = endpoint.get('user', 'root')
            ssh = ['ssh', '-i', str(keyfile), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                   '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile=' + str(root / 'known-hosts'),
                   '-p', port, user + '@' + host]
            scp = ['scp', '-i', str(keyfile), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                   '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile=' + str(root / 'known-hosts'),
                   '-P', port]
            # Readiness can lag provider RUNNING; retries are read-only probes.
            for _ in range(30):
                try:
                    run(ssh + ['true'])
                    break
                except subprocess.CalledProcessError:
                    time.sleep(5)
            else:
                raise TimeoutError('Direct SSH never became ready')
            run(ssh + ['mkdir -p /workspace/results'])
            run(scp + [plan['parent']['checkpoint_path'], user + '@' + host + ':/workspace/parent.json.gz'], timeout=300)
            run(scp + ['scripts/mature_cpu_linux_setup.sh', user + '@' + host + ':/workspace/setup.sh'], timeout=60)
            row.update(status='running', workload_started=time.time())
            with (folder / 'connection.log').open('w') as log:
                remaining = max(1, int(deadline - time.time() - 600))
                command = ('timeout ' + str(remaining) + ' bash /workspace/setup.sh '
                           + shlex.quote(source) + ' ' + str(deadline)
                           + ' > /workspace/results/setup.log 2>&1; '
                           + 'code=$?; printf "%s\\n" "$code" > /workspace/results/outer-exit.txt; exit "$code"')
                finished = subprocess.run(ssh + [command], stdout=log, stderr=subprocess.STDOUT,
                                          timeout=remaining + 60)
            row.update(workload_exit=finished.returncode, workload_finished=time.time(), status='retrieving')
            # Raw archive is opaque on M1. Pack/hash on its Linux owner.
            run(ssh + ['tar -cf /workspace/results.tar -C /workspace results && sha256sum /workspace/results.tar > /workspace/results.tar.sha256'], timeout=300)
            run(scp + [user + '@' + host + ':/workspace/results.tar', str(folder / 'results.tar')], timeout=300)
            run(scp + [user + '@' + host + ':/workspace/results.tar.sha256', str(folder / 'results.tar.sha256')], timeout=60)
            row.update(status='retrieved', retrieved=time.time())
            return row
        except Exception as error:
            row.update(status='failed', reason=str(error), failed=time.time())
            return row

    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            pending = {pool.submit(workload, row): row for row in records if row['status'] == 'provisioning'}
            for future in as_completed(pending):
                row = future.result()
                write(root / 'pods.json', records)
                if row['status'] == 'retrieved':
                    folder = root / row['cpu_id']
                    # Exactly one M4 CPU-heavy verifier at a time.
                    with (folder / 'verification.log').open('w') as log:
                        command = [sys.executable, '-m', 'scripts.mature_cpu_verify_transport',
                                   '--folder', str(folder), '--reference', str(args.reference),
                                   '--deadline', str(deadline)]
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=max(1, min(600, deadline-time.time())))
                    row.update(verification_exit=result.returncode,
                               status='verified' if result.returncode == 0 else 'verification-failed')
                    if result.returncode != 0:
                        # No further gameplay is needed to investigate parity.
                        for pod in owned_pods(api(args.key, '/v2/pods')['pods'], names):
                            api(args.key, '/v2/pods/' + pod['id'], 'DELETE')
                        write(root / 'pods.json', records)
                        break
                if row.get('id'):
                    api(args.key, '/v2/pods/' + row['id'], 'DELETE')
                    row['terminated'] = time.time()
                write(root / 'pods.json', records)
    finally:
        for pod in owned_pods(api(args.key, '/v2/pods')['pods'], names):
            api(args.key, '/v2/pods/' + pod['id'], 'DELETE')
        remaining = owned_pods(api(args.key, '/v2/pods')['pods'], names)
        write(root / 'pods.json', records)
        write(root / 'operator-finished.json', {'finished': time.time(), 'remaining_owned_ids': [p['id'] for p in remaining],
              'status': 'complete' if len(records) == 6 and all(r['status'] == 'verified' for r in records) and not remaining else 'incomplete'})
        with note.open('a') as stream:
            stream.write(f'\nDr Research six-class RunPod RELEASE: coordinator {os.getpid()}, '
                         f'{len(remaining)} owned pods remain; retained root {root}.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('plan', 'key', 'root', 'reference'):
        parser.add_argument('--' + name, type=Path, required=True)
    execute(parser.parse_args())
