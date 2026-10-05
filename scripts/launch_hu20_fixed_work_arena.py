"""Dispatch approved fixed work once, after both actual-host gates and cost admission.

Provisioning is separate. The creation ledger, source CI, source-bound references
and original chat-approved quote are checked before any scientific coordinate.
"""
import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
import re
import shlex
from time import time

from src.arena.schedule import digest
from scripts.hu20_search_arena_control import durable_json
from scripts.launch_hu20_mixed_arena import upload
from scripts.monitor_hu20_search_arena import RemoteControl, ssh, PROTECTED
from scripts.quote_hu20_fixed_work_arena import admitted_workers

SOURCE = '249a79b892313f90d5e280b0ea079ed4c38941e2'
PROTOCOL = 'hu20-fixed50-no-fallback-v1'
PLAN = '8a4e939f7058e0a85d281ef1a069e90bf0b63465f3f115c0f095ac888216e42e'
QUOTE = 'ffea2f314a39a1d6372e37feb6c0b0b42f6cb26df5418faf7cef5d7d2fce5a95'


def actual_quote(original, hosts, reference_replays, charge, now=None):
    """Same forecast rules, scaled by the slowest matched actual-host request."""
    scale = original['timing']['stage4_host_cost_scale']
    for host in hosts:
        for row in host['replay']['rows']:
            if not row['solve'].startswith('stage4/'): continue
            ref = reference_replays[int(row['solve'].split('/')[1])]
            if ref['cpu_seconds'] > 1:
                scale = max(scale, row['solver_seconds']/(ref['cpu_seconds']/6))
    n = len(hosts);workers = sum(len(h['workers']) for h in hosts)
    assert n == 2 and workers == 6
    rates = sum(h['hourly_usd'] for h in hosts)
    reserves = sum(original['reserves_seconds_per_pod'].values())
    mean = (17100*(original['timing']['mean_reference_cpu_seconds']*scale/6+1.5)+12000)/workers
    maximum = (17100*(original['timing']['p95_reference_cpu_seconds']*scale/6+4)+36000)/workers
    expected_seconds=ceil(mean+reserves);maximum_seconds=ceil(1.5*(maximum+reserves))
    return {'at':time() if now is None else now,'work_protocol':PROTOCOL,'host_cost_scale':scale,
            'expected_seconds':expected_seconds,'maximum_forecast_seconds':maximum_seconds,
            'expected_cost_usd':ceil((charge+expected_seconds/3600*rates)*100)/100,
            'maximum_forecast_cost_usd':ceil((charge+maximum_seconds/3600*rates)*100)/100,
            'charge_upper_usd_at_gate':charge,'hard_ceiling_usd':25,'dispatch_stop_usd':21,
            'not_a_solve_deadline':True,'completion_guarantee':False}


def validate_gate(gate):
    h=gate['admission'];r=gate['replay'];c=gate['checker']
    assert gate['source']==SOURCE and admitted_workers(h['quota_cpus'],h['admitted_ram_bytes'])>=3
    assert h['quota_cpus']>=18 and h['admitted_ram_bytes']>=32*10**9
    assert c['status']=='passed' and c.get('cross_platform_parity')=='passed'
    assert c['reference_max_strategy_absolute_difference']==0 and c['reference_max_value_chips_absolute_difference']==0
    assert r['requests']==r['passed']==120 and r['max_profile_difference']==0 and r['exact_scientific_outputs']
    assert {x['solve'] for x in r['rows'] if x['solve'].startswith('stage4/')}=={f'stage4/{i:02d}' for i in range(24)}
    assert gate['retention']['requests']>=120 and gate['reference_retention']['requests']==120


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--controller',required=True)
    a=p.parse_args();root=a.root.resolve()
    if (root/'ARENA_LAUNCH.json').exists():raise ValueError('Never restart scientific work')
    owner=json.loads((root/'owner-approval.json').read_text());ci=json.loads((root/'CI_GREEN.json').read_text())
    assert owner['owner_approved'] and owner['source']==SOURCE and owner['protocol']==PROTOCOL
    assert owner['quote_sha256']==QUOTE and owner['plan_sha256']==PLAN
    assert ci['head_sha']==SOURCE and ci['conclusion']=='success'
    assert hashlib.sha256((root/'approved-quote-original.json').read_bytes()).hexdigest()==QUOTE
    assert digest(json.loads((root/'plan.json').read_text()))==PLAN
    ledger=json.loads((root/'ledger.json').read_text());pods=ledger['pods']
    assert ledger['work_protocol']==PROTOCOL and ledger['hard_ceiling_usd']==25
    assert len(pods)==2 and sorted(w for h in pods for w in h['workers'])==list(range(6))
    assert not any(h['id'] in PROTECTED or h.get('terminated_at') or h['compute_hourly_usd']>.13 for h in pods)
    control=RemoteControl(a.controller,(root/'control-token').read_text().strip(),root/'ledger.json')
    assert control.request({'op':'check'})['status']=='preflight'
    live=json.loads((root/'live-status.json').read_text());assert time()-live['at']<90
    for pod in pods:
        output=ssh(pod,"python3 - <<'HOST'\nimport json\nfrom pathlib import Path\nr=Path('/workspace/evidence')\n"
            "assert (r/'PREFLIGHT_PASSED').exists() and not (r/'PREFLIGHT_FAILED').exists()\n"
            "print('HU20_GATE='+json.dumps({'source':(r/'source.txt').read_text().strip(),'admission':json.loads((r/'host-admission.json').read_text()),'checker':json.loads((r/'checker/reference.json').read_text()),'replay':json.loads((r/'replay/summary.json').read_text()),'retention':json.loads((r/'parity-retention-check.json').read_text()),'reference_retention':json.loads((r/'reference-retention-check.json').read_text())}))\nHOST")
        gate=json.loads(re.search(r'HU20_GATE=(\{[^\n]+\})',output)[1]);validate_gate(gate)
        durable_json(root/(pod['id']+'-verified-gates.json'),gate)
        pod.update(parity_retention_passed=True,parity_sha256=digest(gate),replay=gate['replay'])
        times=sorted(x['solver_seconds'] for x in gate['replay']['rows']);pod['host_replay_p99_seconds']=times[ceil(len(times)*.99)-1]
    quote=actual_quote(json.loads((root/'approved-quote-original.json').read_text()),pods,
                       json.loads((root/'cost-only-results.json').read_text()),control.charge())
    durable_json(root/'actual-host-cost-gate.json',quote)
    if quote['maximum_forecast_cost_usd']>25:
        control.request({'op':'stop','reason':'Actual-host maximum exceeds $25; owner decision required'})
        raise RuntimeError('Actual-host maximum exceeds $25; no arena dispatch')
    durable_json(root/'ledger.json',ledger)
    controller=next(h for h in pods if h['id']==ledger['controller_pod_id'])
    upload(controller,'/workspace/ledger.json',(root/'ledger.json').read_bytes())
    for pod in pods:
        paid={'owner_approved':True,'work_protocol':PROTOCOL,'arena_plan_sha256':PLAN,
              'search_config_sha256':digest(json.loads((root/'config.json').read_text())),
              'selected_settings_parity':'passed','quote_sha256':QUOTE,'parity_sha256':pod['parity_sha256'],
              'rss_limit_bytes':9*1024**3,'host_replay_p99_seconds':pod['host_replay_p99_seconds'],
              'owner_approval':'chat: go ahead for revision 3','dispatch_stop_usd':21,'hard_ceiling_usd':25}
        upload(pod,'/workspace/paid-approval.json',json.dumps(paid,sort_keys=True).encode())
        upload(pod,'/workspace/control-token',(root/'control-token').read_bytes());ssh(pod,'chmod 600 /workspace/control-token')
        command="cd /workspace/repo; /workspace/venv/bin/python - <<'PY'\nfrom scripts.hu20_search_arena_control import ControlClient\nfrom pathlib import Path\n"
        command+=f"assert ControlClient({a.controller!r},{pod['id']!r},{pod['workers'][0]},Path('/workspace/control-token').read_text()).request('check')['status']=='preflight'\nPY"
        ssh(pod,command)
    durable_json(root/'ARENA_LAUNCH.json',{'at':time(),'status':'dispatching','source':SOURCE,'plan_sha256':PLAN,'cost':quote})
    assert control.request({'op':'start'})['status']=='running'
    for pod in pods:
        for w in pod['workers']:
            assert control.request({'op':'check'})['status']=='running'
            shell='cd /workspace/repo; . /workspace/venv/bin/activate; export PYTHONPATH=/workspace/repo OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; '
            shell+=f'export HU20_CONTROL_URL={shlex.quote(a.controller)} HU20_POD_ID={pod["id"]} HU20_WORKER_INDEX={w}; export HU20_CONTROL_TOKEN=$(cat /workspace/control-token); '
            shell+='exec python -m scripts.run_hu20_search_arena_guarded --plan /workspace/bundle/arena-plan.json --inputs /workspace/bundle/inputs --phase arena --binary /workspace/hu20-turn-search-tool/harness/target/release/hu20-exact-flop-tool --search-config /workspace/bundle/config.json --paid-approval /workspace/paid-approval.json '
            shell+=f'--worker-index {w} --worker-count 6 --out /workspace/evidence/arena/worker-{w}'
            command=f'set -eu\nmkdir -p /workspace/evidence/arena\ntest ! -e /workspace/evidence/arena/worker-{w}.pid\ntest ! -e /workspace/evidence/arena/worker-{w}\n'
            command+=f'setsid bash -c {shlex.quote(shell)} > /workspace/evidence/arena/worker-{w}.launch.log 2>&1 < /dev/null &\necho $! > /workspace/evidence/arena/worker-{w}.pid\n'
            ssh(pod,command)
    durable_json(root/'ARENA_LAUNCH.json',{'at':time(),'status':'launched','source':SOURCE,'plan_sha256':PLAN,'cost':quote})
    print(json.dumps(quote))


if __name__=='__main__':main()
