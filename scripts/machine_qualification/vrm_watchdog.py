"""Exclusive four-CPU supervisor while cygnus-x1's VRM heatsink is absent.

Owns the workload process group on the host. A VRM stop is a failed attempt,
including when the child saves successfully; it is never permission to resume.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from scripts.machine_qualification import watchdog as base


class ThermalWindow:
    def __init__(self):
        self.since = None

    def update(self, now, temperature):
        if temperature < 95:
            self.since = None
        elif self.since is None:
            self.since = now
        return self.since is not None and now - self.since >= 30


def motherboard_sensors(root=Path('/sys/class/hwmon')):
    matches = []
    for name in root.glob('*/name'):
        if name.read_text().strip() == 'nct6687':
            for label in name.parent.glob('temp*_label'):
                if label.read_text().strip() == 'Thermistor 15':
                    temperature = label.with_name(label.name.replace('_label', '_input'))
                    fan = name.parent / 'fan1_input'
                    base.read_int(temperature)
                    if not fan.exists():
                        raise RuntimeError('Missing nct6687 fan1; refusing load')
                    base.read_int(fan)
                    matches.append((temperature, fan))
    if len(matches) != 1:
        raise RuntimeError('Need exactly one nct6687 Thermistor 15 and fan1; refusing load')
    return matches[0]


def request_stop(path, reason):
    with path.open('x') as out:
        out.write(reason + '\n')


def run(args):
    # The same lock covers tests, exports, audits and play, not only training.
    with Path('/tmp/cfr-cygnus-exclusive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return supervised(args)


def supervised(args):
    package = base.package_sensor()
    vrm, fan = motherboard_sensors()
    def reading():
        row = base.sample(package)
        row.update(vrm_c=base.read_int(vrm) / 1000, fan1_rpm=base.read_int(fan))
        if not -20 <= row['vrm_c'] <= 150 or row['fan1_rpm'] < 0:
            raise RuntimeError('Invalid motherboard reading')
        return row
    first = reading()
    if first['package_c'] >= 70 or first['vrm_c'] >= 85:
        raise RuntimeError('Must cool below CPU 70 C and VRM 85 C before load')
    stop = args.stop_file.resolve()
    if stop.exists():
        raise RuntimeError('Stop file already exists; refusing load')
    stop.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', CARGO_BUILD_JOBS='4', RAYON_NUM_THREADS='4',
               NPLAYER_STOP_FILE=str(stop))
    command = ['taskset', '-c', '0-3', *args.command]
    (args.output / 'command.json').write_text(json.dumps({
        'command': command, 'cwd': os.getcwd(), 'cpu_limit_c': 85,
        'vrm_sensor': str(vrm), 'fan1_sensor': str(fan), 'vrm_limit_c': 95,
        'vrm_continuous_seconds': 30, 'clean_stop_file': str(stop),
        'clean_stop_grace_seconds': args.clean_stop_grace_seconds,
        'seconds': args.seconds, 'max_rss_gib': args.max_rss_gib,
        'min_disk_gib': args.min_disk_gib, 'cpu_affinity': '0-3',
        'thread_limit': 4, 'lock': '/tmp/cfr-cygnus-exclusive.lock'}, indent=2) + '\n')
    started = time.monotonic()
    window = ThermalWindow()
    requested_at = None
    reason = None
    hard_reason = None
    peak = 0
    interrupted = []
    previous = {sig: signal.signal(sig, lambda number, frame: interrupted.append(number))
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        with (args.output / 'output.log').open('w') as output, \
                (args.output / 'resources.jsonl').open('w', buffering=1) as readings:
            proc = subprocess.Popen(command, env=env, stdout=output,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    now = time.monotonic()
                    try:
                        row = reading()
                        row['family_rss_bytes'] = base.group_rss(proc.pid)
                        row['free_disk_bytes'] = shutil.disk_usage(Path.cwd()).free
                        peak = max(peak, row['family_rss_bytes'])
                        hot = window.update(now, row['vrm_c'])
                        row['vrm_hot_seconds'] = 0 if window.since is None else now - window.since
                        row['clean_stop_requested'] = requested_at is not None
                        readings.write(json.dumps(row) + '\n')
                        if row['package_c'] >= 85:
                            hard_reason = 'package temperature >=85 C'
                        elif row['available_bytes'] < 6 * 1024**3:
                            hard_reason = 'available RAM below 6 GiB'
                        elif row['swap_used_bytes'] - first['swap_used_bytes'] > 256 * 1024**2:
                            hard_reason = 'swap grew by more than 256 MiB'
                        elif row['family_rss_bytes'] > args.max_rss_gib * 1024**3:
                            hard_reason = 'family RSS above declared limit'
                        elif row['free_disk_bytes'] < args.min_disk_gib * 1024**3:
                            hard_reason = 'free disk below declared floor'
                        if hot and requested_at is None:
                            reason = 'VRM >=95 C for 30 continuous seconds'
                            request_stop(stop, reason)
                            requested_at = now
                    except Exception as exc:
                        hard_reason = f'sensor/resource read failed: {exc!r}'
                    if interrupted:
                        hard_reason = f'supervisor signaled: {interrupted}'
                    if now - started >= args.seconds:
                        hard_reason = 'wall-time budget exceeded'
                    if requested_at is not None and now - requested_at >= args.clean_stop_grace_seconds:
                        hard_reason = 'clean-stop grace exhausted'
                    if hard_reason or proc.poll() is not None:
                        break
                    time.sleep(1)
            finally:
                # Also kills descendants if a wrapper exits before its workload.
                base.terminate_group(proc)
            code = proc.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    result = {'exit_code': code, 'stop_reason': reason, 'hard_stop_reason': hard_reason,
              'elapsed_seconds': time.monotonic() - started,
              'sampled_peak_family_rss_bytes': peak,
              'status': 'passed' if code == 0 and not reason and not hard_reason else 'failed'}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return int(result['status'] != 'passed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--stop-file', required=True, type=Path)
    p.add_argument('--seconds', required=True, type=float)
    p.add_argument('--max-rss-gib', default=80, type=float)
    p.add_argument('--min-disk-gib', default=32, type=float)
    p.add_argument('--clean-stop-grace-seconds', default=180, type=float)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.command[:1] == ['--']:
        a.command.pop(0)
    if not a.command or min(a.seconds, a.max_rss_gib, a.clean_stop_grace_seconds) <= 0:
        p.error('Need a command and positive time/RSS/grace limits')
    raise SystemExit(run(a))


if __name__ == '__main__':
    main()
