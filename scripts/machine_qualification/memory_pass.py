"""One parallel userspace pass; preserve every instance's result, stop on failure."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(os.environ['QUALIFICATION_ROOT'])
output = root / os.environ.get('QUALIFICATION_MEMORY_LABEL', 'memory-instances')
output.mkdir(exist_ok=False)
# 112 GiB total, leaving about 13.7 GiB for the OS, guard and owner session.
processes = []
results = []
for index in range(8):
    stream = (output / f'instance-{index}.log').open('w')
    proc = subprocess.Popen(['stdbuf', '-oL', '-eL', '/usr/sbin/memtester', '14336M', '1'],
                            stdout=stream, stderr=subprocess.STDOUT)
    processes.append((index, proc, stream))
while processes:
    for item in list(processes):
        index, proc, stream = item
        code = proc.poll()
        if code is None:
            continue
        stream.close()
        results.append({'instance': index, 'pid': proc.pid, 'exit_code': code,
                        'requested_mib': 14336})
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        processes.remove(item)
        if code:
            # The outer watchdog owns and terminates the whole group on this exit.
            raise SystemExit(f'memtester instance {index} failed with exit {code}')
    time.sleep(1)
print(json.dumps(results), flush=True)
