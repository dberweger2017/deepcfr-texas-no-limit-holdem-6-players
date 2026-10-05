"""One-time launch of the named, owner-approved fleet after every actual-host gate.

This does not allocate pods, restart workers, or alter the frozen science. Run
beside the durable M4 supervisor; an existing launch receipt forbids dispatch.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shlex
from time import sleep, time

from scripts.hu20_search_arena_control import durable_json
from scripts.monitor_hu20_search_arena import RemoteControl, pod_status, ssh


def upload(pod, path, data):
    encoded=base64.b64encode(data).decode()
    command="python3 - <<'UPLOAD'\nimport base64,os\nfrom pathlib import Path\n"
    command+=f"p=Path({path!r});p.parent.mkdir(parents=True,exist_ok=True)\n"
    command+=f"data=base64.b64decode({encoded!r})\n"
    command+="temp=p.with_name(p.name+'.upload')\nwith temp.open('wb') as stream:\n stream.write(data);stream.flush();os.fsync(stream.fileno())\nos.replace(temp,p)\nUPLOAD"
    ssh(pod,command)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--controller',required=True)
    parser.add_argument('--pod',action='append',required=True)
    args=parser.parse_args();root=args.root.resolve()
    receipt=root/'ARENA_LAUNCH.json'
    if any((root/('ARENA_LAUNCH_'+p+'.json')).exists() for p in args.pod):raise ValueError('Preserve existing launch; workers cannot restart')
    token=(root/'control-token').read_text().strip()
    control=RemoteControl(args.controller,token,root/'ledger.json')
    ledger=json.loads((root/'ledger.json').read_text())
    pods=[p for p in ledger['pods'] if not p.get('terminated_at')]
    approval=json.loads((root/'four-pod-owner-approval.json').read_text())
    expected={'m5pxmipuqyjtoo':[0,1],'mislfsw9a0bmva':[2,3],'6oqjxlfdjk0bzp':[4],'xtu3jr3utxqasx':[5]}
    assert {p['id']:p['workers'] for p in pods}==expected
    assert approval['owner_approved'] and ledger['hard_ceiling_usd']==25
    assert set(args.pod)<=set(expected)
    pods=[p for p in pods if p['id'] in args.pod]
    while True:
        state=control.request({'op':'check'})
        if state['status'] not in ('preflight','running'):raise RuntimeError('Launch controller gate closed')
        live=json.loads((root/'live-status.json').read_text())
        if time()-live['at']>90:raise RuntimeError('Durable supervisor is not reporting fresh state')
        with ThreadPoolExecutor(max_workers=4) as pool:statuses=list(pool.map(pod_status,pods))
        if any(s['preflight_failed'] or s['resource_failures'] for s in statuses):
            control.request({'op':'stop','reason':'Actual-host preflight failed; no arena dispatch'})
            raise RuntimeError('Actual-host preflight failed; preserve evidence and ask owner')
        if all(s['preflight_passed'] for s in statuses):break
        sleep(15)
    lock=(root/'dispatch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    ledger=json.loads((root/'ledger.json').read_text())
    pods=[p for p in ledger['pods'] if not p.get('terminated_at') and p['id'] in args.pod]
    if any((root/('ARENA_LAUNCH_'+p['id']+'.json')).exists() for p in pods):raise ValueError('A selected pod was already dispatched')
    if control.request({'op':'check'})['status'] not in ('preflight','running'):raise RuntimeError('Arena dispatch is stopped')
    for pod in pods:
        output=ssh(pod,"python3 - <<'HOST'\nimport json,hashlib\nfrom pathlib import Path\nr=Path('/workspace/evidence')\n"
            "checker=json.loads((r/'checker/reference.json').read_text())\n"
            "replay=json.loads((r/'replay/summary.json').read_text())\n"
            "admission=json.loads((r/'host-admission.json').read_text())\n"
            "retention=json.loads((r/'parity-retention-check.json').read_text())\n"
            "reference=json.loads((r/'reference-retention-check.json').read_text())\n"
            "assert checker['status']=='passed' and checker['cross_platform_parity']=='passed'\n"
            "assert replay['requests']==replay['passed']==96 and replay['max_profile_difference']==0\n"
            "assert (r/'source.txt').read_text().strip()=='6a5ff44ec3e5d313be594d177244c3579081ff00'\n"
            "print('HU20_GATE='+json.dumps({'checker':checker,'replay':replay,'admission':admission,'retention':retention,'reference_retention':reference}))\nHOST",timeout=45)
        import re
        gates=json.loads(re.search(r'HU20_GATE=(\{[^\n]+\})',output)[1])
        durable_json(root/(pod['id']+'-verified-gates.json'),gates)
        pod['parity_retention_passed']=True
        pod['parity_sha256']=hashlib.sha256(json.dumps(gates,sort_keys=True).encode()).hexdigest()
    quote=json.loads((root/'approved-actual-fleet-quote.json').read_text())
    forecast=quote['without_5080'];hourly=sum(p['hourly_usd'] for p in ledger['pods'] if not p.get('terminated_at'))
    charge=control.charge()
    maximum=charge+forecast['maximum_worker_seconds']/3600*hourly
    expected_cost=charge+forecast['expected_including_past_usd']-forecast['past_cap_counted_charge_upper_usd']-1.5
    durable_json(root/'launch-cost-gate.json',{'at':time(),'maximum_usd':maximum,'expected_usd':expected_cost,'counted_charge_upper_usd':charge,'sleep_excluded_usd':ledger['owner_excluded_charge_usd'],'hard_ceiling_usd':25})
    if maximum>25:
        control.request({'op':'stop','reason':'Recomputed actual fleet maximum exceeds $25; ask owner'})
        raise RuntimeError('Recomputed maximum exceeds $25; ask owner before dispatch')
    ledger['start_required_pods']=['m5pxmipuqyjtoo','mislfsw9a0bmva']
    ledger['partial_start_owner_approval']='chat: start both 3070s after their parity, do not hold for proxy-only hosts'
    durable_json(root/'ledger.json',ledger)
    controller=next(p for p in ledger['pods'] if p['id']=='m5pxmipuqyjtoo')
    upload(controller,'/workspace/ledger.json',(root/'ledger.json').read_bytes())
    # Final client changes are separately hashed, leaving 6a5ff44 science intact.
    hashes={}
    for pod in pods:
        for name in ['hu20_search_arena_control.py','run_hu20_search_arena_guarded.py','hu20_search_evidence.py']:
            data=(root/'ops/scripts'/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest()
            upload(pod,'/workspace/repo/scripts/'+name,data)
        upload(pod,'/workspace/control-token',token.encode())
        ssh(pod,'chmod 600 /workspace/control-token')
        upload(pod,'/workspace/evidence/launch-operations.json',json.dumps(hashes,sort_keys=True).encode())
        # Actual Linux clients must authenticate through the public HTTPS route.
        command="cd /workspace/repo && /workspace/venv/bin/python - <<'HOST'\nfrom pathlib import Path\nfrom scripts.hu20_search_arena_control import ControlClient\n"
        command+=f"x=ControlClient({args.controller!r},{pod['id']!r},{pod['workers'][0]},Path('/workspace/control-token').read_text()).request('check')\nassert x['status'] in ('preflight','running'),x\nHOST"
        ssh(pod,command,timeout=45)
    # Record start admission before creating any process. Partial dispatch is never retried.
    for pod in pods:durable_json(root/('ARENA_LAUNCH_'+pod['id']+'.json'),{'at':time(),'status':'dispatching','workers':pod['workers'],'source':'6a5ff44','maximum_usd':maximum,'expected_usd':expected_cost})
    if control.request({'op':'check'})['status']=='preflight':control.request({'op':'start'})
    for pod in pods:
        gate=json.loads((root/(pod['id']+'-verified-gates.json')).read_text())
        paid={'owner_approved':True,'arena_plan_sha256':'a6126470862a7576a14d1d2967e7f03d242d0934fd4a0a62066902865666ced2',
              'search_config_sha256':hashlib.sha256(json.dumps(gate['checker']['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
              'selected_settings_parity':'passed','quote_sha256':hashlib.sha256((root/'approved-actual-fleet-quote.json').read_bytes()).hexdigest(),
              'parity_sha256':pod['parity_sha256'],'worker_seconds':int(pod['production_stop_at']-time()),'rss_limit_bytes':pod['rss_limit_bytes'],
              'owner_approval':'chat: four named pods, six workers, unchanged science/retention/stops',
              'sleep_excluded_usd':ledger['owner_excluded_charge_usd'],'dispatch_stop_usd':21,'hard_ceiling_usd':25}
        upload(pod,'/workspace/paid-approval.json',json.dumps(paid,sort_keys=True).encode())
        for worker in pod['workers']:
            command='set -eu\nmkdir -p /workspace/evidence/arena\n'
            command+=f'test ! -e /workspace/evidence/arena/worker-{worker}.pid\ntest ! -e /workspace/evidence/arena/worker-{worker}\n'
            shell='cd /workspace/repo; . /workspace/venv/bin/activate; export PYTHONPATH=/workspace/repo OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; '
            shell+=f'export HU20_CONTROL_URL={shlex.quote(args.controller)} HU20_POD_ID={pod["id"]} HU20_WORKER_INDEX={worker}; export HU20_CONTROL_TOKEN=$(cat /workspace/control-token); '
            shell+='exec python -m scripts.run_hu20_search_arena_guarded --plan /workspace/bundle/arena-plan.json --inputs /workspace/bundle/inputs --phase arena --binary /workspace/hu20-turn-search-tool/harness/target/release/hu20-exact-flop-tool --search-config /workspace/bundle/config.json --paid-approval /workspace/paid-approval.json '
            shell+=f'--worker-index {worker} --worker-count 6 --out /workspace/evidence/arena/worker-{worker}'
            command+=f'setsid bash -c {shlex.quote(shell)} > /workspace/evidence/arena/worker-{worker}.launch.log 2>&1 < /dev/null &\necho $! > /workspace/evidence/arena/worker-{worker}.pid\n'
            ssh(pod,command,timeout=45)
    for pod in pods:durable_json(root/('ARENA_LAUNCH_'+pod['id']+'.json'),{'at':time(),'status':'launched','workers':pod['workers'],'source':'6a5ff44','operations_sha256':hashes,'maximum_usd':maximum,'expected_usd':expected_cost})


if __name__=='__main__':main()
