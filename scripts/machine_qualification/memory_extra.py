"""Selected 112 GiB coverage, preserving actual child wait statuses on stop.

The outer on-host guard owns the process group and the absolute deadline.
This parent handles its termination signal only to record observed outcomes;
it does not extend the guard's grace period or restart a child.
"""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import threading

root = Path(os.environ['QUALIFICATION_ROOT'])
mask = os.environ.get('MEMTESTER_TEST_MASK')
if mask != '0x1ec00':
    raise SystemExit('Only the predeclared remaining-pattern mask is authorized here')
soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
if soft != resource.RLIM_INFINITY and soft < 14336 * 1024**2:
    raise SystemExit('Insufficient inherited memlock; refuse allocation shrinking')
out = root / 'memory-extra-instances'
out.mkdir(exist_ok=False)
received = []
wakeup = threading.Event()
def on_signal(number, frame):
    received.append(number)
    wakeup.set()
for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, on_signal)

processes, results = [], []
for index in range(8):
    stream = (out / f'instance-{index}.log').open('w')
    proc = subprocess.Popen(['stdbuf', '-oL', '-eL', '/usr/sbin/memtester', '14336M', '1'],
                            stdout=stream, stderr=subprocess.STDOUT)
    processes.append((index, proc, stream))
(out / 'processes.json').write_text(json.dumps([
    {'instance': i, 'pid': p.pid, 'requested_mib': 14336, 'test_mask': mask}
    for i, p, _ in processes], indent=2) + '\n')
failure = False
while processes:
    for item in list(processes):
        index, proc, stream = item
        code = proc.poll()
        if code is None:
            continue
        stream.close()
        results.append({'instance': index, 'pid': proc.pid, 'exit_code': code,
                        'termination_signal': -code if code < 0 else None,
                        'requested_mib': 14336, 'test_mask': mask})
        # Write wait statuses before reading/hashing potentially bulky logs.
        pending = out / 'results.pending.json'
        pending.write_text(json.dumps(sorted(results, key=lambda r: r['instance']), indent=2) + '\n')
        pending.replace(out / 'results.json')
        processes.remove(item)
        if code and not received and not failure:
            failure = True
            for _, other, _ in processes:
                if other.poll() is None:
                    other.terminate()
            wakeup.set()
    wakeup.wait(0.01 if received or failure else 1)
    wakeup.clear()
if received:
    raise SystemExit(128 + received[0])
if failure:
    raise SystemExit('A memtester instance failed; all other children stopped')
print(json.dumps(results), flush=True)
