"""Read-only: collect each held host's passed gates and print the measured actual-host quote.

Dispatch is separate (launch_hu20_fixed_work_arena). This allocates nothing and starts no work.
"""
import argparse
import json
import re
from pathlib import Path
from time import time

from scripts.launch_hu20_fixed_work_arena import actual_quote, validate_gate
from scripts.monitor_hu20_search_arena import RemoteControl, ssh
from scripts.quote_hu20_fixed_work_arena import admitted_workers
from src.arena.schedule import digest

READ = ("python3 - <<'HOST'\nimport json\nfrom pathlib import Path\nr=Path('/workspace/evidence')\n"
        "assert (r/'PREFLIGHT_PASSED').exists() and not (r/'PREFLIGHT_FAILED').exists()\n"
        "print('HU20_GATE='+json.dumps({'source':(r/'source.txt').read_text().strip(),"
        "'admission':json.loads((r/'host-admission.json').read_text()),"
        "'checker':json.loads((r/'checker/reference.json').read_text()),"
        "'replay':json.loads((r/'replay/summary.json').read_text()),"
        "'retention':json.loads((r/'parity-retention-check.json').read_text()),"
        "'reference_retention':json.loads((r/'reference-retention-check.json').read_text())}))\nHOST")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--controller', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    ledger = json.loads((root/'ledger.json').read_text())
    pods = [h for h in ledger['pods'] if not h.get('terminated_at')]
    control = RemoteControl(a.controller, (root/'control-token').read_text().strip(), root/'ledger.json')
    first = 0
    gates = {}
    for pod in pods:
        out = ssh(pod, READ)
        gate = json.loads(re.search(r'HU20_GATE=(\{[^\n]+\})', out)[1])
        validate_gate(gate, 2, 22*10**9)
        count = admitted_workers(gate['admission']['quota_cpus'], gate['admission']['admitted_ram_bytes'])
        pod.update(replay=gate['replay'], workers=list(range(first, first+count)))
        times = sorted(x['solver_seconds'] for x in gate['replay']['rows'])
        pod['host_replay_p99_seconds'] = times[-(-len(times)*99//100)-1]
        first += count
        gates[pod['id']] = gate
    original = json.loads((root/'approved-quote-original.json').read_text())
    quote = actual_quote(original, pods, json.loads((root/'cost-only-results.json').read_text()),
                         control.charge(), hard_ceiling=ledger['hard_ceiling_usd'],
                         dispatch_stop=ledger['dispatch_stop_usd'])
    rate = sum(h['hourly_usd'] for h in pods)
    result = {'quote': quote, 'workers': {h['id']: h['workers'] for h in pods}, 'fleet_hourly_usd': rate,
              'per_pod_quota_cpus': {k: g['admission']['quota_cpus'] for k, g in gates.items()},
              'gate_digests': {k: digest(g) for k, g in gates.items()}, 'at': time()}
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
