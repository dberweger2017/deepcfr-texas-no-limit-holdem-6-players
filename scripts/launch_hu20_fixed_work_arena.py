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


def actual_quote(original, hosts, reference_replays, charge, now=None, *, hard_ceiling=25, dispatch_stop=21):
    """Same forecast rules, scaled by the slowest matched actual-host request."""
    ratios=[]
    scale = original['timing']['stage4_host_cost_scale']
    for host in hosts:
        for row in host['replay']['rows']:
            if not row['solve'].startswith('stage4/'): continue
            ref = reference_replays[int(row['solve'].split('/')[1])]
            if ref['cpu_seconds'] > 1:
                ratios.append(row['solver_seconds']/(ref['cpu_seconds']/6))
    if not ratios:raise ValueError('Complete matched host timing required')
    scale=max(ratios)
    n = len(hosts);workers = sum(len(h['workers']) for h in hosts)
    if not n or not workers:raise ValueError('An admitted fleet is required')
    rates = sum(h['hourly_usd'] for h in hosts)
    # Setup and parity are completed and counted in charge, not billed twice.
    reserves = sum(v for k,v in original['reserves_seconds_per_pod'].items()
                   if k not in ('setup_build','actual_host_parity'))
    mean = (17100*(original['timing']['mean_reference_cpu_seconds']*scale/6+1.5)+12000)/workers
    maximum = (17100*(original['timing']['p95_reference_cpu_seconds']*scale/6+4)+36000)/workers
    expected_seconds=ceil(mean+reserves);maximum_seconds=ceil(1.5*(maximum+reserves))
    return {'at':time() if now is None else now,'work_protocol':PROTOCOL,'host_cost_scale':scale,
            'expected_seconds':expected_seconds,'maximum_forecast_seconds':maximum_seconds,
            'expected_cost_usd':ceil((charge+expected_seconds/3600*rates)*100)/100,
            'maximum_forecast_cost_usd':ceil((charge+maximum_seconds/3600*rates)*100)/100,
            'charge_upper_usd_at_gate':charge,'hard_ceiling_usd':hard_ceiling,'dispatch_stop_usd':dispatch_stop,
            'setup_parity_basis':'actual allocation charge already counted; remaining reserves only',
            'not_a_solve_deadline':True,'completion_guarantee':False}


def validate_gate(gate, minimum_workers=3, minimum_ram_bytes=32*10**9):
    h=gate['admission'];r=gate['replay'];c=gate['checker']
    assert gate['source']==SOURCE and admitted_workers(h['quota_cpus'],h['admitted_ram_bytes'])>=minimum_workers
    assert h['quota_cpus']>=6*minimum_workers and h['admitted_ram_bytes']>=minimum_ram_bytes
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
    ledger=json.loads((root/'ledger.json').read_text())
    pods=[h for h in ledger['pods'] if not h.get('terminated_at')]
    ops_ci=json.loads((root/'OPS_CI_GREEN.json').read_text())
    assert ops_ci['conclusion']=='success' and ops_ci['head_sha']==ledger['operations_source']
    ceiling=ledger['hard_ceiling_usd'];dispatch=ledger['dispatch_stop_usd']
    assert ledger['work_protocol']==PROTOCOL and ceiling==owner['hard_ceiling_usd'] and dispatch<=ceiling-4
    fleet_quote=json.loads((root/'approved-fleet-quote.json').read_text()) if owner.get('equivalent_fleet_delegation') else None
    if fleet_quote:
        assert ceiling==20 and owner['equivalent_fleet_delegation']
        assert hashlib.sha256((root/'approved-fleet-quote.json').read_bytes()).hexdigest()==owner['fleet_quote_sha256']
        assert len(pods)==fleet_quote['pods']
        if fleet_quote.get('admission_only_until_measured_quote'):
            assert fleet_quote['admission_allowance_usd']<=.50
        else:assert fleet_quote['maximum_forecast_cost_usd']<=ceiling
        assert all(h['gpu_id']==fleet_quote['gpu_id'] and h['gpu_count']==fleet_quote['gpu_count'] for h in pods)
    else:assert len(pods)==2 and ceiling==25
    assert not any(h['id'] in PROTECTED or h.get('terminated_at')
                   or h['compute_hourly_usd']>(fleet_quote['maximum_compute_hourly_usd'] if fleet_quote else .13) for h in pods)
    control=RemoteControl(a.controller,(root/'control-token').read_text().strip(),root/'ledger.json')
    assert control.request({'op':'check'})['status']=='preflight'
    live=json.loads((root/'live-status.json').read_text());assert time()-live['at']<90
    for pod in pods:
        output=ssh(pod,"python3 - <<'HOST'\nimport json\nfrom pathlib import Path\nr=Path('/workspace/evidence')\n"
            "assert (r/'PREFLIGHT_PASSED').exists() and not (r/'PREFLIGHT_FAILED').exists()\n"
            "print('HU20_GATE='+json.dumps({'source':(r/'source.txt').read_text().strip(),'admission':json.loads((r/'host-admission.json').read_text()),'checker':json.loads((r/'checker/reference.json').read_text()),'replay':json.loads((r/'replay/summary.json').read_text()),'retention':json.loads((r/'parity-retention-check.json').read_text()),'reference_retention':json.loads((r/'reference-retention-check.json').read_text())}))\nHOST")
        gate=json.loads(re.search(r'HU20_GATE=(\{[^\n]+\})',output)[1])
        validate_gate(gate,fleet_quote['minimum_workers_per_pod'] if fleet_quote else 3,
                      fleet_quote['minimum_ram_bytes'] if fleet_quote else 32*10**9)
        durable_json(root/(pod['id']+'-verified-gates.json'),gate)
        pod.update(parity_retention_passed=True,parity_sha256=digest(gate),replay=gate['replay'])
        if fleet_quote:
            count=admitted_workers(gate['admission']['quota_cpus'],gate['admission']['admitted_ram_bytes'])
            first=sum(len(h['workers']) for h in pods[:pods.index(pod)])
            pod['workers']=list(range(first,first+count))
        times=sorted(x['solver_seconds'] for x in gate['replay']['rows']);pod['host_replay_p99_seconds']=times[ceil(len(times)*.99)-1]
    quote=actual_quote(json.loads((root/'approved-quote-original.json').read_text()),pods,
                       json.loads((root/'cost-only-results.json').read_text()),control.charge(),
                       hard_ceiling=ceiling,dispatch_stop=dispatch)
    durable_json(root/'actual-host-cost-gate.json',quote)
    if quote['maximum_forecast_cost_usd']>ceiling:
        control.request({'op':'stop','reason':f'Actual-host maximum exceeds ${ceiling}; no arena dispatch'})
        raise RuntimeError(f'Actual-host maximum exceeds ${ceiling}; no arena dispatch')
    durable_json(root/'ledger.json',ledger)
    controller=next(h for h in pods if h['id']==ledger['controller_pod_id'])
    upload(controller,'/workspace/ledger.json',(root/'ledger.json').read_bytes())
    for pod in pods:
        paid={'owner_approved':True,'work_protocol':PROTOCOL,'arena_plan_sha256':PLAN,
              'search_config_sha256':digest(json.loads((root/'config.json').read_text())),
              'selected_settings_parity':'passed','quote_sha256':QUOTE,'parity_sha256':pod['parity_sha256'],
              'rss_limit_bytes':9*1024**3,'host_replay_p99_seconds':pod['host_replay_p99_seconds'],
              'owner_approval':owner.get('owner_chat','chat: go ahead for revision 3'),'dispatch_stop_usd':dispatch,'hard_ceiling_usd':ceiling}
        upload(pod,'/workspace/paid-approval.json',json.dumps(paid,sort_keys=True).encode())
        upload(pod,'/workspace/control-token',(root/'control-token').read_bytes());ssh(pod,'chmod 600 /workspace/control-token')
        command="cd /workspace/repo; /workspace/venv/bin/python - <<'PY'\nfrom scripts.hu20_search_arena_control import ControlClient\nfrom pathlib import Path\n"
        command+=f"assert ControlClient({a.controller!r},{pod['id']!r},{pod['workers'][0]},Path('/workspace/control-token').read_text()).request('check')['status']=='preflight'\nPY"
        ssh(pod,command)
    durable_json(root/'ARENA_LAUNCH.json',{'at':time(),'status':'dispatching','source':SOURCE,'plan_sha256':PLAN,'cost':quote})
    assert control.request({'op':'start'})['status']=='running'
    for pod in pods:
        worker_count=sum(len(h['workers']) for h in pods)
        for w in pod['workers']:
            assert control.request({'op':'check'})['status']=='running'
            shell='cd /workspace/repo; . /workspace/venv/bin/activate; export PYTHONPATH=/workspace/repo OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; '
            shell+=f'export HU20_CONTROL_URL={shlex.quote(a.controller)} HU20_POD_ID={pod["id"]} HU20_WORKER_INDEX={w}; export HU20_CONTROL_TOKEN=$(cat /workspace/control-token); '
            shell+='exec python -m scripts.run_hu20_search_arena_guarded --plan /workspace/bundle/arena-plan.json --inputs /workspace/bundle/inputs --phase arena --binary /workspace/hu20-turn-search-tool/harness/target/release/hu20-exact-flop-tool --search-config /workspace/bundle/config.json --paid-approval /workspace/paid-approval.json '
            shell+=f'--worker-index {w} --worker-count {worker_count} --out /workspace/evidence/arena/worker-{w}'
            command=f'set -eu\nmkdir -p /workspace/evidence/arena\ntest ! -e /workspace/evidence/arena/worker-{w}.pid\ntest ! -e /workspace/evidence/arena/worker-{w}\n'
            command+=f'setsid bash -c {shlex.quote(shell)} > /workspace/evidence/arena/worker-{w}.launch.log 2>&1 < /dev/null &\necho $! > /workspace/evidence/arena/worker-{w}.pid\n'
            ssh(pod,command)
    durable_json(root/'ARENA_LAUNCH.json',{'at':time(),'status':'launched','source':SOURCE,'plan_sha256':PLAN,'cost':quote})
    print(json.dumps(quote))


if __name__=='__main__':main()
