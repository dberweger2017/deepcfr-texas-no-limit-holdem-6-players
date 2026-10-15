"""Bounded direct-I/O fio on one new scratch file; never accepts a device path."""
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path(os.environ['QUALIFICATION_ROOT']).resolve()
output = root / 'disk-results'
output.mkdir(exist_ok=False)
scratch = output / 'fio-scratch.bin'
if scratch.exists() or shutil.disk_usage(output).free < 40 * 1024**3:
    raise SystemExit('Need a new scratch path and at least 40 GiB free')
common = ['fio', f'--filename={scratch}', '--size=8G', '--direct=1',
          '--ioengine=libaio', '--numjobs=1', '--group_reporting=1',
          '--output-format=json', '--end_fsync=1']
# Initialize the full extent before any read measurement.
cases = [('initialize', 'write', '1M', 16, False),
         ('sequential-write', 'write', '1M', 16, True),
         ('sequential-read', 'read', '1M', 16, True),
         ('random-read-q1', 'randread', '4k', 1, True),
         ('random-write-q1', 'randwrite', '4k', 1, True),
         ('random-read-q32', 'randread', '4k', 32, True),
         ('random-write-q32', 'randwrite', '4k', 32, True)]
commands = []
for name, mode, block, depth, timed in cases:
    command = [*common, f'--name={name}', f'--rw={mode}', f'--bs={block}',
               f'--iodepth={depth}', f'--output={output / (name + ".json")}']
    if timed:
        command.extend(['--time_based=1', '--runtime=30', '--ramp_time=3'])
    commands.append(command)
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    subprocess.run(command, check=True)
    receipt = json.loads((output / (name + '.json')).read_text())
    if any(job['error'] for job in receipt['jobs']):
        raise SystemExit(f'fio reports an error in {name}')
    print(f'{name} complete', flush=True)
# A separate CRC32C write/read verification is not included in throughput rates.
command = [*common, '--name=verify', '--rw=write', '--bs=128k', '--iodepth=8',
           '--verify=crc32c', '--do_verify=1', '--verify_fatal=1',
           f'--output={output / "verify.json"}']
commands.append(command)
(output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
subprocess.run(command, check=True)
receipt = json.loads((output / 'verify.json').read_text())
if any(job['error'] for job in receipt['jobs']):
    raise SystemExit('fio verification error')
print('All disk scratch tests passed; scratch retained.', flush=True)
