"""Three bounded integer-CPU verification steps, each with its own cool start."""
import datetime
import os
from pathlib import Path
import subprocess
import time

import watchdog  # Deployed as watchdog.py alongside this entry point if used.

root = Path(os.environ['QUALIFICATION_ROOT']).resolve()
sensor = watchdog.package_sensor()
for workers in (1, 4, 8):
    # The deadline remains authoritative during cooldown as well as load.
    deadline = datetime.datetime.fromisoformat(os.environ['QUALIFICATION_DEADLINE_UTC'].replace('Z', '+00:00')).timestamp()
    while watchdog.read_int(sensor) >= 60000:
        if time.time() >= deadline - 180:
            raise SystemExit('Not enough deadline budget for a cool CPU step')
        time.sleep(2)
    command = ['python3', str(root / 'watchdog-overnight.py'), '--output', str(root / f'cpu-{workers}'),
        '--seconds', '180', '--sample-seconds', '0.5', '--',
        'stress-ng', '--cpu', str(workers), '--cpu-method', 'int64', '--verify',
        '--timeout', '60s', '--metrics-brief']
    subprocess.run(command, check=True)
    print(f'CPU {workers}-worker verification passed', flush=True)
