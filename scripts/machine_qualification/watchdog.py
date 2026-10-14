"""Run one bounded qualification job with an on-host, fail-closed thermal guard.

The process group belongs to this supervisor, so loss of SSH does not remove
the guard. Callers must detach the supervisor itself and limit workload threads.
"""

import argparse
import datetime as dt
import glob
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time


def read_int(path):
    return int(Path(path).read_text().strip())


def package_sensor():
    for name in glob.glob('/sys/class/hwmon/hwmon*/name'):
        if Path(name).read_text().strip() != 'coretemp':
            continue
        for label in Path(name).parent.glob('temp*_label'):
            if label.read_text().strip() == 'Package id 0':
                return label.with_name(label.name.replace('_label', '_input'))
    raise RuntimeError('No coretemp package sensor; refusing load')


def memory():
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        fields[key] = int(value.split()[0]) * 1024
    return fields


def sample(sensor):
    mem = memory()
    clocks = [read_int(p) for p in glob.glob(
        '/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq')]
    counters = {p: read_int(p) for p in glob.glob(
        '/sys/devices/system/cpu/cpu[0-9]*/thermal_throttle/*_throttle_count')}
    return {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'monotonic': time.monotonic(), 'package_c': read_int(sensor) / 1000,
            'clock_khz': clocks, 'throttle_counts': counters,
            'available_bytes': mem['MemAvailable'],
            'swap_used_bytes': mem['SwapTotal'] - mem['SwapFree']}


def group_rss(pgid):
    total = 0
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if os.getpgid(int(proc.name)) == pgid:
                total += int((proc / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            continue
    return total


def terminate_group(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        if sig == signal.SIGTERM:
            time.sleep(0.5)
    proc.wait()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sensor = package_sensor()
    first = sample(sensor)
    if first['package_c'] >= 70:
        raise RuntimeError('Package must cool below 70 C before a job starts')
    env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               OPENBLAS_NUM_THREADS='8', CARGO_BUILD_JOBS='4', RAYON_NUM_THREADS='8')
    command = ['taskset', '-c', '0-7', *args.command]
    (args.output / 'command.json').write_text(json.dumps(
        {'command': command, 'cwd': os.getcwd(), 'sensor': str(sensor),
            'max_seconds': args.seconds, 'temperature_limit_c': 85,
         'max_rss_gib': args.max_rss_gib, 'min_disk_gib': args.min_disk_gib,
         'cpu_affinity': '0-7', 'environment_overrides': {
             k: env[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                 'OPENBLAS_NUM_THREADS', 'CARGO_BUILD_JOBS', 'RAYON_NUM_THREADS')}}, indent=2))
    started = time.monotonic()
    reason = None
    peak_rss = 0
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda number, frame: interrupted.append(number))
    with (args.output / 'output.log').open('w') as output, \
            (args.output / 'resources.jsonl').open('w', buffering=1) as readings:
        proc = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            while True:
                try:
                    row = sample(sensor)
                    row['family_rss_bytes'] = group_rss(proc.pid)
                    row['free_disk_bytes'] = shutil.disk_usage(Path.cwd()).free
                    peak_rss = max(peak_rss, row['family_rss_bytes'])
                    readings.write(json.dumps(row) + '\n')
                    if row['package_c'] >= 85:
                        reason = 'package temperature >=85 C'
                    elif row['available_bytes'] < 6 * 1024**3:
                        reason = 'available RAM below 6 GiB'
                    elif row['swap_used_bytes'] - first['swap_used_bytes'] > 256 * 1024**2:
                        reason = 'swap grew by more than 256 MiB'
                    elif args.max_rss_gib and row['family_rss_bytes'] > args.max_rss_gib * 1024**3:
                        reason = 'family RSS above declared limit'
                    elif row['free_disk_bytes'] < args.min_disk_gib * 1024**3:
                        reason = 'free disk below declared floor'
                except Exception as exc:
                    reason = f'sensor/resource read failed: {exc!r}'
                if interrupted:
                    reason = f'supervisor signaled: {interrupted}'
                if time.monotonic() - started > args.seconds:
                    reason = 'wall-time budget exceeded'
                if reason:
                    terminate_group(proc)
                    break
                if proc.poll() is not None:
                    break
                time.sleep(2)
        finally:
            # A supervisor exception must not leave an unsupervised load behind.
            terminate_group(proc)
        code = proc.wait()
    result = {'exit_code': code, 'stop_reason': reason,
              'elapsed_seconds': time.monotonic() - started,
              'sampled_peak_family_rss_bytes': peak_rss,
              'status': 'passed' if code == 0 and reason is None else 'failed'}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--max-rss-gib', type=float, default=0)
    parser.add_argument('--min-disk-gib', type=float, default=16)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ['--']:
        args.command.pop(0)
    if not args.command or args.seconds <= 0:
        parser.error('A command and a positive wall-time budget are required')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
