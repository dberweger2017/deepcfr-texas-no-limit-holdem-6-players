"""Measure a fixed blueprint play task; never infer search or playing strength."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value, *_ = line.split()
        values[key.rstrip(':')] = int(value) * 1024
    return values


def temperatures():
    result = {}
    for folder in Path('/sys/class/hwmon').glob('hwmon*'):
        for label in folder.glob('temp*_label'):
            name = label.read_text().strip()
            if name in ('Package id 0', 'Thermistor 15'):
                result[name] = int(label.with_name(
                    label.name.replace('_label', '_input')).read_text()) / 1000
    if set(result) != {'Package id 0', 'Thermistor 15'}:
        raise RuntimeError('Required temperature sensors unavailable')
    return result


def sample(out, swap_baseline):
    mem = memory()
    temp = temperatures()
    disk = os.statvfs(out)
    available = disk.f_bavail * disk.f_frsize
    row = dict(at=time.time(), temperatures_c=temp,
               available_ram_bytes=mem['MemAvailable'],
               swap_bytes=mem['SwapTotal'] - mem['SwapFree'],
               free_disk_bytes=available)
    if temp['Package id 0'] >= 75 or temp['Thermistor 15'] >= 85:
        raise RuntimeError('Conservative pilot thermal stop')
    if mem['MemAvailable'] < 6 * 1024**3:
        raise RuntimeError('Pilot available-memory stop')
    if row['swap_bytes'] - swap_baseline > 256 * 1024**2:
        raise RuntimeError('Pilot swap-growth stop')
    if available - 512 * 1024**2 < 50_000_000_000:
        raise RuntimeError('Pilot disk-reservation stop')
    return row


def summarize_trace(path, expected_hands):
    """Count decisions without reading rewards or selecting profitable hands."""
    with path.open() as stream:
        header = json.loads(next(stream))
        if header['format'] != 'nplayer-privileged-policy-replay-v1':
            raise ValueError('Unexpected trace format')
        if header['hands'] != expected_hands or len(header['policies']) != 3:
            raise ValueError('Unexpected hand or player count')
        streets = {}
        hands = 0
        for line in stream:
            hand = json.loads(line)
            if hand['hand'] != hands:
                raise ValueError('Missing, duplicate or unordered hand')
            if len(hand['states']) != len(hand['actors']) + 1:
                raise ValueError('Incomplete decision states')
            if len(hand['actions']) != len(hand['actors']):
                raise ValueError('Incomplete decision actions')
            for state in hand['states'][:-1]:
                street = state['street']
                streets[street] = streets.get(street, 0) + 1
            hands += 1
    if hands != expected_hands:
        raise ValueError('Incomplete trace')
    return dict(hands=hands, decisions_by_street=streets)


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    mem = memory()
    swap = mem['SwapTotal'] - mem['SwapFree']
    first = sample(out, swap)
    if first['temperatures_c']['Package id 0'] >= 70 or first['temperatures_c']['Thermistor 15'] >= 80:
        raise RuntimeError('Pilot temperature admission refused')
    if sha256(args.binary) != args.binary_sha256:
        raise ValueError('Qualified binary hash mismatch')
    if args.model.stat().st_size != 705_933_433:
        raise ValueError('Unexpected Phase 2 model size')
    trace = out / 'hands.jsonl'
    command = [str(args.binary), 'play', str(args.tables), '1000', str(args.seed),
               str(trace)] + [str(args.model), args.model_sha256] * 3
    admission = dict(command=command, cpu=args.cpu, hands=1000, seed=args.seed,
                     model_sha256=args.model_sha256,
                     model_hash_verified_by='PolicyTable::load before play',
                     binary_sha256=args.binary_sha256,
                     wall_fail_safe_seconds=600, address_space_bytes=16 * 1024**3,
                     initial_sample=first, input_archive_member='training/2026101541/1000m-average.gz')
    (out / 'admission.json').write_text(json.dumps(admission, indent=2) + '\n')

    def child_limits():
        os.sched_setaffinity(0, {args.cpu})
        os.nice(19)
        resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3,) * 2)

    env = dict(os.environ, RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    process = None
    samples = []
    child_start = time.monotonic()
    try:
        with (out / 'stdout.json').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
            process = subprocess.Popen(['ionice', '-c', '3', *command], stdout=stdout,
                stderr=stderr, env=env, start_new_session=True, preexec_fn=child_limits)
            with (out / 'resources.jsonl').open('w') as log:
                while True:
                    row = sample(out, swap)
                    try:
                        status = Path(f'/proc/{process.pid}/status').read_text()
                        row['child_rss_bytes'] = next(int(line.split()[1]) * 1024
                            for line in status.splitlines() if line.startswith('VmRSS:'))
                    except (OSError, StopIteration):
                        row['child_rss_bytes'] = 0
                    samples.append(row)
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    if time.monotonic() - start >= 600:
                        raise RuntimeError('Pilot wall fail-safe reached')
                    if trace.exists() and trace.stat().st_size > 256 * 1024**2:
                        raise RuntimeError('Pilot trace-size stop')
                    if process.poll() is not None:
                        break
                    time.sleep(1)
        child_seconds = time.monotonic() - child_start
        if process.returncode:
            raise RuntimeError(f'Native play failed: exit {process.returncode}')
        native = json.loads((out / 'stdout.json').read_text())
        trace_summary = summarize_trace(trace, 1000)
        trace_hash = sha256(trace)
        if trace_hash != native['trace_sha256']:
            raise ValueError('Native trace hash differs from readback')
        if sum(trace_summary['decisions_by_street'].values()) != sum(native['decisions_by_street']):
            raise ValueError('Native decision counts differ from trace')
        result = dict(status='complete-cost-only', **trace_summary,
            native_process_observed_seconds=child_seconds,
            timing_scope='process startup, model/table loading, play, native transition replay, trace writing; one-second polling plus sensor/read overhead',
            total_measurement_seconds=time.monotonic() - start,
            sampled_peak_child_rss_bytes=max(s['child_rss_bytes'] for s in samples),
            trace_bytes=trace.stat().st_size, trace_sha256=trace_hash,
            native_summary=native, search_attempts=0,
            missing_search_prerequisite='Current K50 three-player search adapter is not qualified')
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        (out / 'failure.json').write_text(json.dumps(dict(error=repr(error),
            elapsed_seconds=time.monotonic() - start), indent=2) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
