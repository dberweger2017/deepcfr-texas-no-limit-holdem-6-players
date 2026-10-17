"""Bounded first cost pilot for PR301; the fixed scientific harness stays unchanged."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from scripts import schedule_hu200_capacity as q
from scripts import continue_hu200_capacity as storage
from scripts.run_hu200_capacity_training import pin, put
from src.blueprint import policy_cache
from src.policies.files import file_hash

PHASE = 'evaluation-first-pilot-20261017'
BUDGET = 258_400_000_000
RESERVATIONS = {'cache-large':18_500_000_000, 'cache-twin':5_200_000_000,
                'play':250_000_000, 'replay':64_000_000, 'reproduce':250_000_000,
                'verify':64_000_000, 'quote':64_000_000}


def evaluation_fits(owned, free, reserved, extra):
    return (owned + reserved + extra + 15_000_000_000 + q.FLOOR <= BUDGET
            and free - reserved - extra >= q.FLOOR)


def seal_clock(path, now, source):
    # Exclusive creation consumes this attempt. No reset, credit or retry path.
    put(path, dict(started_epoch=now, deadline_epoch=now + 86400,
                   source=source, maximum_seconds=86400, automatic_retry=False))
    return q.read(path)


def intact(path, expected):
    if path.is_symlink() or path.stat().st_size != expected['bytes'] or file_hash(path) != expected['sha256']:
        raise ValueError('Required model bytes changed')


def restore_twin(base):
    storage.mount_checked()
    spec = q.read(base / 'twins-restored.json')['models'][0]['spec']
    target = Path(spec['path'])
    if not target.is_relative_to(base) or target.exists():
        raise ValueError('Restore target is not fresh and owned')
    source = storage.HDD / 'research/PR-301-hu200-capacity-cygnus/inactive-models-20261017' / target.relative_to(base)
    intact(source, spec)
    before = source.stat()
    if shutil.disk_usage(base).free - spec['bytes'] - 30_000_000_000 < q.FLOOR:
        raise ValueError('Restoration and first cache pair do not fit')
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=8*1024**2)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    intact(target, spec)
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if (source.stat().st_size, source.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise ValueError('HDD source changed while restoring')
    put(base / PHASE / 'restore-twin.json', dict(status='verified', source=str(source),
        retained_source=True, destination=pin(target), utc=storage.now()))


def validate(base, config_path, readback):
    from scripts.evaluate_hu200_capacity import checked_config
    cfg = checked_config(config_path)
    if base != q.BASE or cfg['seed'] != 2026101302 or cfg['kind'] != '20B':
        raise ValueError('Only this campaign first cost pilot is admitted')
    expected = q.read(Path('docs/reports/hu200-capacity-cygnus-artifacts/training-complete-verification-20261017.json'))
    if pin(base / 'training-complete.json')['sha256'] != next(
            x['sha256'] for x in expected['pins'] if x['path'] == 'training-complete.json'):
        raise ValueError('Training completion changed')
    for seed, audit in q.read(base / 'training-complete.json')['audits'].items():
        if audit['status'] != 'verified' or audit['audit']['all_nodes_verified'] != audit['native']['diagnostics']['entries']:
            raise ValueError('All final full-row audits must pass')
    roots = q.read(base / PHASE / 'roots.json')
    if not roots['fresh'] or roots['pr'] != 301 or roots['matches'] or time.time()-roots['at'] > 3600:
        raise ValueError('Fresh live root reservation required')
    if not {202610160301,202610160302}.issubset(roots['roots']):
        raise ValueError('First pilot roots were not reserved')
    publication = q.read(readback)
    if (not publication['exact_body_readback'] or publication['state'] != 'OPEN'
            or publication['config_sha256'] != file_hash(config_path)
            or publication['source'] != cfg['source']):
        raise ValueError('Pinned PR declaration readback missing')
    for spec in cfg['specs']:
        if not Path(spec['path']).is_relative_to(base):
            raise ValueError('Active model must be on owned NVMe')
        intact(Path(spec['path']), spec)
        if policy_cache.folder(Path(cfg['cache']),spec['sha256'],spec['abstraction']).exists():
            raise ValueError('Cache construction already attempted')
    if not q.read(base / PHASE / 'restore-twin.json')['retained_source']:
        raise ValueError('Retained HDD source is required')
    return cfg


class Pilot(q.Queue):
    def __init__(self, base, config_path, readback):
        self.base, self.config_path = base, config_path
        self.config = validate(base, config_path, readback)
        self.phase = base / PHASE
        if (self.phase/'evaluation-failed.json').exists() or (base/'evaluation-clock.json').exists():
            raise ValueError('Evaluation already attempted; no automatic retry')
        self.root = base / 'guards' / PHASE
        self.root.mkdir(exist_ok=False)
        self.clock = seal_clock(base/'evaluation-clock.json', time.time(), self.config['source'])
        self.deadline = self.clock['deadline_epoch']
        self.admission = q.read(base/'admission.json')
        self.active, self.done, self.tasks = {}, set(), {}
        self.next_counter, self.last_counter = 0, None
        self.cpu_deltas, self.disk_busy_ms, self.last_disk = {}, 0, None
        previous = []
        for name in RESERVATIONS:
            self.task(name, 'train', previous, seed=2026101302)
            previous = [name]

    def folder(self, task):
        return Path(self.config['cache'] if task['name'].startswith('cache-') else self.config['output'])

    def command(self, task):
        return [sys.executable, '-m', 'scripts.pilot_hu200_capacity', 'stage',
                '--base',str(self.base),'--config',str(self.config_path),
                '--name',task['name'],'--stop',str(self.phase/(task['name']+'.stop'))]

    def scientific_command(self, task):
        name = task['name']
        command = [sys.executable, '-m', 'scripts.evaluate_hu200_capacity']
        if name.startswith('cache-'):
            return command + ['cache','--config',str(self.config_path),'--index',str(0 if name=='cache-large' else 1)]
        if name == 'quote':
            return command + ['quote','--config',str(self.config_path),'--out',str(self.phase/'cost-quote.json')]
        return command + [name,'--config',str(self.config_path),'--pilot']

    def launch(self, task, row, totals):
        if self.active or row['package_c'] >= 68 or row['vrm_c'] >= 83:
            return False
        name = task['name']; reserve = RESERVATIONS[name]
        if not evaluation_fits(totals,row['free_disk_bytes'],0,reserve):
            return False
        stop = self.phase / (name+'.stop')
        if stop.exists():
            raise ValueError('Stage stop file is not fresh')
        cpus = [0,1]
        plan = dict(name=name,kind='pilot-'+name,cpus=cpus,rss_gib=38,
            stop_file=str(stop),command=self.command(task),reserve_bytes=reserve,
            original_deadline=self.deadline)
        put(self.root/(name+'.plan.json'),plan)
        log = (self.root/(name+'.log')).open('x')
        env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            RAYON_NUM_THREADS='2',MKL_CBWR='AVX2,STRICT',PYTHONUNBUFFERED='1')
        before = q.owned_bytes(self.folder(task))
        proc = subprocess.Popen(['taskset','-c','0,1',*plan['command']],env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.active[name] = dict(**plan,proc=proc,identity=q.process(proc.pid)['start'],log=log,
            started=time.monotonic(),folder=self.folder(task),before_bytes=before)
        print(json.dumps(dict(event='started',job=plan,pid=proc.pid)),flush=True)
        return True

    def finish(self, name):
        job = self.active[name]
        if job['proc'].poll() != 0:
            raise RuntimeError(f'{name} failed; no automatic scientific retry')
        put(self.root/(name+'.result.json'),dict(status='passed',exit_code=0,
            elapsed_seconds=time.monotonic()-job['started'],cpus=job['cpus']))
        job['log'].close();del self.active[name];self.done.add(name)

    def decide_batch(self, row, totals):
        if not evaluation_fits(totals,row['free_disk_bytes'],sum(self.reservation(j) for j in self.active.values()),0):
            raise RuntimeError('Original258.4GB evaluation/output reservation exhausted')

    def publish(self):
        pass

    def complete(self):
        put(self.phase/'pilot-complete.json',dict(status='verified-cost-only-pilot',
            quote=pin(self.phase/'cost-quote.json'),config=pin(self.config_path),
            clock=self.clock,finished_utc=storage.now(),final_hands_started=False,outcomes_inspected=False))


def run_stage(base, config_path, name, stop):
    if name not in RESERVATIONS or stop != base/PHASE/(name+'.stop') or stop.exists():
        raise ValueError('Fresh owned stage stop required')
    job = Pilot.__new__(Pilot)
    job.base, job.config_path, job.phase = base, config_path, base/PHASE
    proc = subprocess.Popen(job.scientific_command({'name':name}))
    while proc.poll() is None:
        if stop.exists():
            # Cache/play workers have no resumable stop operation. End only the
            # owned child, preserve its partials, and report this attempt failed.
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()
            raise RuntimeError('Guard requested stop; partial attempt retained')
        time.sleep(0.25)
    if proc.returncode != 0:
        raise RuntimeError(f'Fixed harness exit {proc.returncode}; no retry')


def deliver(base, event):
    value = urllib.parse.urlsplit((base/'campaign-events-url.txt').read_text().strip())
    if value.scheme != 'https' or value.hostname != 'relay.t3.codes' or not value.path.startswith('/v1/hooks/'):
        raise ValueError('Unrecognized existing event endpoint')
    payload = dict(event=event,experiment='PR301',observed_utc=storage.now())
    request = urllib.request.Request(value.geturl(),data=json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(request,timeout=10) as response:
            put(base/PHASE/'event-delivery.json',dict(payload=payload,http_status=response.status))
    except Exception as exc:
        print(json.dumps(dict(event=event,delivery='failed',error_type=type(exc).__name__)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('restore','run','stage'))
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--readback',type=Path)
    parser.add_argument('--name',choices=tuple(RESERVATIONS))
    parser.add_argument('--stop',type=Path)
    args=parser.parse_args()
    if args.base != q.BASE or os.uname().nodename != 'cygnus-x1':
        raise ValueError('Only the owned PR301 root on cygnus-x1')
    if args.phase=='stage':
        run_stage(args.base,args.config,args.name,args.stop);return
    if args.phase=='restore':
        restore_twin(args.base);return
    with Path('/tmp/cfr-cygnus-exclusive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            Pilot(args.base,args.config,args.readback).run()
        except BaseException as exc:
            failure=args.base/PHASE/'evaluation-failed.json'
            if not failure.exists():
                put(failure,dict(status='failed',error=repr(exc),automatic_retry=False,utc=storage.now()))
            deliver(args.base,'evaluation_failed')
            raise
        deliver(args.base,'first_evaluation_pilot_cost_verified')


if __name__=='__main__':
    main()
