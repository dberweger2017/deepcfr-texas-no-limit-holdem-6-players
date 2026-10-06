"""Retrieve the completed three-pod attempt, verify all bytes, then terminate it.

This command never stops active poker work. Recovery reuses verified chunks and
termination receipts; the relay stays alive until every pod has been retrieved.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from time import time

from scripts.hu20_autonomous_arena import read_status, pod_done, route
from scripts.hu20_search_arena_control import durable_json
from scripts.hu20_search_evidence import file_hash, verify_archives
from scripts.monitor_hu20_search_arena import ssh

OWNED = frozenset(('gdyfqg9817qme0', 'ne7ui0na5wd27u', 'yig5a8bfutpxjg'))
REMOTE = '/workspace/closeout-attempt-2'


def fleet(ledger):
    pods = ledger['pods']
    if len(pods) != 3 or {p['id'] for p in pods} != OWNED or ledger.get('attempt') != 2:
        raise ValueError('Closeout is bound to the owner-handed-over attempt-2 fleet')
    return pods


def data(reply):
    if 'error' in reply or reply.get('result', {}).get('isError'):
        raise RuntimeError('RunPod MCP error: ' + json.dumps(reply))
    result = reply['result']
    if 'structuredContent' in result:
        return result['structuredContent']
    return json.loads(next(c['text'] for c in result['content'] if c['type'] == 'text'))


def retrieve_manifest(manifest, destination, fetch, root, ledger):
    """Check actual free disk before each chunk, retain compressed evidence."""
    limit = ledger['retrieval_limit_bytes']
    reserve = ledger['retrieval_free_reserve_bytes']
    size = sum(row['bytes'] for row in manifest['archives'])
    if size > limit:
        raise OSError('Per-pod archive size exceeds the ledger retrieval limit')
    for row in manifest['archives']:
        name = row['path']
        if Path(name).name != name or not 0 <= row['bytes'] <= 10**9:
            raise ValueError('Invalid transfer chunk')
        path = destination / name
        if path.exists() and path.stat().st_size == row['bytes'] and file_hash(path) == row['sha256']:
            continue
        if shutil.disk_usage(root).free < row['bytes'] + reserve:
            raise OSError('Insufficient real free disk before ' + name)
        fetch(name, path)
        if path.stat().st_size != row['bytes'] or file_hash(path) != row['sha256']:
            raise ValueError('Archive differs after transfer')
    verified = verify_archives(destination)
    durable_json(destination / 'retrieval-verified.json', {
        **verified, 'archive_manifest_sha256': file_hash(destination / 'manifest.json'), 'at': time()})
    return verified


def retrieve(pod, ledger, root):
    destination = root / 'retrieved' / pod['id']
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'retrieval-verified.json').exists():
        proof = json.loads((destination / 'retrieval-verified.json').read_text())
        if proof['archive_manifest_sha256'] != file_hash(destination / 'manifest.json'):
            raise ValueError('Retrieved manifest changed')
        verify_archives(destination)
        return
    # Freeze operational files after all workers have finished. No finalization
    # or deletion runs here: original worker manifest hashes remain valid.
    command = f"""set -eu
if test ! -f {REMOTE}/manifest.json; then
  test ! -d {REMOTE}
  mkdir -p /workspace/evidence/attempt-2-operations
  for f in controller-pod.log ledger-pod.json paid-approval.json launch-autonomous.sh; do
    cp /workspace/$f /workspace/evidence/attempt-2-operations/$f
  done
  cd /workspace/repo
  /workspace/venv/bin/python -m scripts.hu20_search_evidence pack /workspace/evidence {REMOTE}
fi
sha256sum {REMOTE}/manifest.json
"""
    out = ssh(pod, command, timeout=5400)
    expected = re.search(r'([0-9a-f]{64})  ' + re.escape(REMOTE + '/manifest.json'), out)[1]
    host, _ = route(pod, ledger)
    remote = REMOTE
    if host['id'] != pod['id']:
        remote = '/workspace/relay/' + pod['id'] + '-closeout'
        ssh(host, 'mkdir -p ' + remote)
        relay_ssh = f"ssh -o BatchMode=yes -p {host['ssh_port']}"
        ssh(pod, 'rsync -a --partial -e ' + shlex.quote(relay_ssh) + ' ' + REMOTE + '/ root@' +
            host['ssh_host'] + ':' + remote + '/', timeout=3600)
    def fetch(name, path):
        temporary = path.with_name(path.name + '.partial')
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-P', str(host['ssh_port']),
                        'root@' + host['ssh_host'] + ':' + remote + '/' + name, str(temporary)],
                       check=True, timeout=1800)
        temporary.replace(path)
    fetch('manifest.json', destination / 'manifest.json')
    if file_hash(destination / 'manifest.json') != expected:
        raise ValueError('Manifest differs after transfer')
    manifest = json.loads((destination / 'manifest.json').read_text())
    retrieve_manifest(manifest, destination, fetch, root, ledger)


def terminate_all(root, ledger, call):
    pods = fleet(ledger)
    # Reverify every archive/member before the first destructive operation.
    for pod in pods:
        folder = root / 'retrieved' / pod['id']
        proof = json.loads((folder / 'retrieval-verified.json').read_text())
        if not proof.get('verified') or proof['archive_manifest_sha256'] != file_hash(folder / 'manifest.json'):
            raise ValueError('Missing/changed retrieval proof')
        verify_archives(folder)
    for pod in sorted(pods, key=lambda p: p['id'] == ledger['relay_pod_id']):
        folder = root / 'retrieved' / pod['id']
        if pod.get('terminated_at'):
            continue
        get = call('tools/call', {'name': 'get-pod', 'arguments': {'id': pod['id']}}, 400)
        durable_json(folder / 'pre-termination-readback.json', get)
        # A delete may have succeeded before a connection loss. Only an actual
        # 404 plus our durable deletion-intent receipt allows that recovery.
        missing = get.get('result', {}).get('isError') and '404' in json.dumps(get)
        intent = folder / 'termination-intent.json'
        if missing and not intent.exists():
            raise RuntimeError('Pod disappeared before verified termination')
        if not missing:
            actual = data(get)
            actual = actual.get('pod', actual)
            if (actual['id'] != pod['id'] or actual['name'] != pod['name'] or
                actual['gpu']['id'] != pod['gpu_id']):
                raise ValueError('Owned pod identity differs')
            durable_json(intent, {'id': pod['id'], 'at': time(), 'owner_handoff': pod['owner_handoff']})
            result = call('tools/call', {'name': 'delete-pod', 'arguments': {'id': pod['id']}}, 401)
            durable_json(folder / 'termination.json', result)
            if 'error' in result or result.get('result', {}).get('isError'):
                raise RuntimeError('Termination failed')
        check = call('tools/call', {'name': 'get-pod', 'arguments': {'id': pod['id']}}, 402)
        durable_json(folder / 'termination-readback.json', check)
        if not (check.get('result', {}).get('isError') and '404' in json.dumps(check)):
            raise RuntimeError('Termination lacks get-pod 404')
        pod['terminated_at'] = time()
        pod['retrieval_verified'] = True
        durable_json(root / 'ledger.json', ledger)
    cursor = None
    seen = set()
    pages = []
    while True:
        args = {'cursor': cursor} if cursor else {}
        reply = call('tools/call', {'name': 'list-pods', 'arguments': args}, 403)
        pages.append(reply)
        durable_json(root / 'final-list-pods.json', pages)
        value = data(reply)
        if OWNED.intersection(p['id'] for p in value['pods']):
            raise RuntimeError('Owned pod still listed')
        if not value['pagination']['hasNextPage']:
            break
        cursor = value['pagination']['nextCursor']
        if not cursor or cursor in seen:
            raise ValueError('Invalid list-pods pagination')
        seen.add(cursor)
    durable_json(root / 'CLOSEOUT_COMPLETE.json', {'at': time(), 'pods': sorted(OWNED), 'verified': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--mcp-helper', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    ledger = json.loads((root / 'ledger.json').read_text())
    pods = fleet(ledger)
    spec = importlib.util.spec_from_file_location('closeout_mcp', args.mcp_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    for pod in pods:
        if not pod.get('terminated_at') and not (root / 'retrieved' / pod['id'] / 'retrieval-verified.json').exists():
            status = read_status(pod)
            if pod_done(status, len(pod['workers'])) not in ('complete', 'incomplete'):
                raise ValueError('Pod still has active work; never interrupt it for closeout')
    for pod in sorted(pods, key=lambda p: p['id'] == ledger['relay_pod_id']):
        retrieve(pod, ledger, root)
    terminate_all(root, ledger, helper.call)


if __name__ == '__main__':
    main()
