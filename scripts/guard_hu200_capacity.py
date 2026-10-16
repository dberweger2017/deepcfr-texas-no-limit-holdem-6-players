"""Own a bounded HU200 capacity workload on bb128; never signal other campaigns."""
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


class VrmWindow:
    """Require continuous observed heat; an observation gap breaks continuity."""
    def __init__(self):
        self.since = None
        self.previous = None

    def update(self, now, temperature):
        if self.previous is not None and now - self.previous > 3:
            self.since = None
        self.previous = now
        if temperature < 95:
            self.since = None
        elif self.since is None:
            self.since = now
        return self.since is not None and now - self.since >= 30


def motherboard_sensors(root=Path('/sys/class/hwmon')):
    matches = []
    for name in root.glob('*/name'):
        if name.read_text().strip() != 'nct6687':
            continue
        for label in name.parent.glob('temp*_label'):
            if label.read_text().strip() == 'Thermistor 15':
                vrm = label.with_name(label.name.replace('_label', '_input'))
                fan = name.parent / 'fan1_input'
                base.read_int(vrm)
                base.read_int(fan)
                matches.append((vrm, fan))
    if len(matches) != 1:
        raise RuntimeError('Need exactly one Thermistor 15 and fan1')
    return matches[0]


def stop_files(jobs, reason):
    for job in jobs:
        path = Path(job['stop_file'])
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open('x') as stream:
                stream.write(reason + '\n')
        except FileExistsError:
            pass


def supervise(plan, output):
    if os.uname().nodename != 'cygnus-x1':
        raise RuntimeError('Only cygnus-x1 is authorized')
    output.mkdir(parents=True, exist_ok=False)
    jobs = plan['jobs']
    if not jobs or sum(len(job['cpus']) for job in jobs) > 8:
        raise ValueError('Need jobs within eight physical cores')
    assigned = [cpu for job in jobs for cpu in job['cpus']]
    if len(set(assigned)) != len(assigned) or any(cpu not in range(8) for cpu in assigned):
        raise ValueError('Physical CPU sets must be disjoint within 0..7')
    if len({job['stop_file'] for job in jobs}) != len(jobs):
        raise ValueError('Each job needs its own stop file')
    if any(Path(job['stop_file']).exists() for job in jobs):
        raise ValueError('Stop files must be fresh')
    package = base.package_sensor()
    vrm, fan = motherboard_sensors()

    def sample():
        row = base.sample(package)
        row.update(vrm_c=base.read_int(vrm) / 1000, fan1_rpm=base.read_int(fan))
        row['fan_rpm'] = {p.stem.removesuffix('_input'): base.read_int(p) for p in fan.parent.glob('fan*_input')}
        row['fan_rpm']['fan1'] = row['fan1_rpm']
        if not -20 <= row['vrm_c'] <= 150 or row['fan1_rpm'] < 0:
            raise ValueError('Invalid motherboard reading')
        row['free_disk_bytes'] = shutil.disk_usage(output).free
        return row

    first = sample()
    if first['package_c'] >= 70 or first['vrm_c'] >= 85:
        raise RuntimeError('Admission requires CPU <70 C and VRM <85 C')
    if first['available_bytes'] < 6 * 1024**3 or first['free_disk_bytes'] < 50_000_000_000:
        raise RuntimeError('RAM/disk admission failed')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    processes, logs, identities = [], [], []
    interrupted = []
    previous = {sig: signal.signal(sig, lambda number, frame: interrupted.append(number))
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    reason = hard = None
    requested = None
    window = VrmWindow()
    started = time.monotonic()
    try:
        for job in jobs:
            log = (output / (job['name'] + '.log')).open('x')
            logs.append(log)
            threads = str(len(job['cpus']))
            env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                       OPENBLAS_NUM_THREADS=threads, RAYON_NUM_THREADS=threads,
                       MKL_CBWR='AVX2,STRICT', PYTHONUNBUFFERED='1')
            command = ['taskset', '-c', ','.join(map(str, job['cpus'])), *job['command']]
            processes.append(subprocess.Popen(command, env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True))
            identities.append(Path(f'/proc/{processes[-1].pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
        with (output / 'resources.jsonl').open('x', buffering=1) as readings:
            while True:
                now = time.monotonic()
                try:
                    row = sample()
                    rss = [base.group_rss(proc.pid) for proc in processes]
                    row.update(job_rss_bytes=dict(zip((j['name'] for j in jobs), rss)),
                               family_rss_bytes=sum(rss), clean_stop_requested=requested is not None)
                    hot = window.update(now, row['vrm_c'])
                    row['vrm_hot_seconds'] = 0 if window.since is None else now - window.since
                    readings.write(json.dumps(row) + '\n')
                    if row['package_c'] >= 85:
                        hard = 'CPU >=85 C'
                    elif row['available_bytes'] < 6 * 1024**3:
                        hard = 'Available RAM <6 GiB'
                    elif row['swap_used_bytes'] - plan['swap_baseline_bytes'] > 256 * 1024**2:
                        hard = 'Swap growth >256 MiB'
                    elif row['free_disk_bytes'] < 50_000_000_000:
                        hard = 'Disk below 50 GB floor'
                    elif sum(rss) > plan['aggregate_rss_gib'] * 1024**3:
                        hard = 'Aggregate family RSS cap'
                    elif any(value > job['rss_gib'] * 1024**3 for value, job in zip(rss, jobs)):
                        hard = 'Per-job family RSS cap'
                    if hot and requested is None:
                        reason = 'VRM >=95 C for 30 continuous observed seconds'
                        stop_files(jobs, reason)
                        requested = now
                except Exception as exc:
                    hard = f'Sensor/resource failure: {exc!r}'
                if interrupted:
                    hard = f'Supervisor signal {interrupted}'
                if now - started >= plan['max_seconds']:
                    hard = 'Wall-time cap'
                if requested is not None and now - requested >= plan['clean_save_grace_seconds']:
                    hard = 'Clean-save grace exhausted'
                if hard:
                    stop_files(jobs, hard)
                    break
                codes = [proc.poll() for proc in processes]
                allowed_codes = (0, 3) if requested is not None else (0,)
                if any(code is not None and code not in allowed_codes for code in codes):
                    hard = 'A workload failed; stopping all owned work'
                    stop_files(jobs, hard)
                    break
                if all(code is not None for code in codes):
                    break
                time.sleep(1)
    finally:
        owned = []
        for proc, identity in zip(processes, identities):
            # A finished leader's PID may have been reused during another long job.
            try:
                current = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            except FileNotFoundError:
                continue
            if current == identity:
                owned.append(proc)
        # Broadcast first: one group's grace must not delay stopping the others.
        for proc in owned:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc in owned:
            base.terminate_group(proc)
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    codes = [proc.wait() for proc in processes]
    result = dict(status='passed' if not reason and not hard and all(c == 0 for c in codes) else 'failed',
                  exit_codes=codes, stop_reason=reason, hard_stop_reason=hard,
                  elapsed_seconds=time.monotonic() - started)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    with Path('/tmp/cfr-cygnus-exclusive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = supervise(json.loads(args.plan.read_text()), args.out)
    print(json.dumps(result), flush=True)
    raise SystemExit(result['status'] != 'passed')


if __name__ == '__main__':
    main()
