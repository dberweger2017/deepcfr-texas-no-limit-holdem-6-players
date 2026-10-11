"""Machine/phase metadata only; never decode policies, scores or worker logs."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from scripts.guard_hu20_equity_scoring import GIB, host, violation


def competing_research():
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True).splitlines()
    inventory = {}
    for row in rows:
        pid, parent, command = row.strip().split(None, 2)
        inventory[int(pid)] = (int(parent), command)
    ancestors = {os.getpid()}
    pid = os.getpid()
    while pid in inventory and inventory[pid][0] not in ancestors:
        pid = inventory[pid][0]
        ancestors.add(pid)
    terms = (' -m scripts.', 'hu20-trainer', 'lock-evaluator', 'pooling-engineering')
    return [{'pid': pid, 'command': command} for pid, (_, command) in inventory.items()
            if pid not in ancestors and any(term in command for term in terms)]


def metadata(base, swap_ceiling_bytes=10_000_000_000):
    sample = host(base if base.exists() else Path.cwd())
    busy = competing_research()
    refusal = violation(sample, 0, swap_ceiling_bytes=swap_ceiling_bytes)
    if sample['free_percent']*16*GIB/100 < 9*GIB:
        refusal = refusal or '9 GiB admission headroom'
    operations = {}
    for phase in ('restore', 'prepare', 'pilot', 'evaluate', 'report'):
        folder = base/'operations'/phase
        receipt = folder/'receipt.json'
        operations[phase] = (json.loads(receipt.read_text()) if receipt.exists()
                             else {'status': 'running-or-interrupted' if folder.exists() else 'not-claimed'})
    failure = base/'campaign-failure.json'
    return {'host': sample, 'competing_research': busy,
            'resource_admission_refusal': refusal, 'operations': operations,
            'completed_roots': len(list((base/'work/eval').glob('*/result.json'))),
            'evaluation_complete_marker': (base/'evaluation-complete.json').exists(),
            'campaign_failure': json.loads(failure.read_text()) if failure.exists() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(metadata(args.base.resolve()), indent=2))


if __name__ == '__main__':
    main()
