"""Owner-started continuous self-play with rolling recovery and a stop file."""

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from scripts.local_fullgame import GIB, used_bytes, write_json
from src.arena.schedule import build_schedule
from src.game.hand import Table
from src.holdem.collection import collection_seed
from src.holdem.experiment import Experiment, _append, _checkpoint, evaluate, manifest
from src.holdem.training import HoldemTrainer

ROOT = Path(__file__).resolve().parents[1]


def pin_comparison(out, iteration=1024):
    """Hard links preserve published comparison artifacts without copying bytes."""
    for directory in out.glob('scenario-*'):
        target = directory / 'pinned' / str(iteration)
        marker = directory / f'training-{iteration}.json'
        if marker.exists():
            target.mkdir(parents=True, exist_ok=True)
            for path in (marker.with_suffix('.pt'), marker):
                if not (target/path.name).exists():
                    try:
                        os.link(path, target/path.name)
                    except FileExistsError:
                        pass
        # Exports are atomically published by save_average; metadata stays in place.
        export = directory / f'average-{iteration}.pt'
        if export.exists():
            target.mkdir(parents=True, exist_ok=True)
            if not (target/export.name).exists():
                try:
                    os.link(export, target/export.name)
                except FileExistsError:
                    pass


class AttachedWorker:
    """Supervise an existing worker without restarting its training state."""

    def __init__(self, pid, out):
        command = subprocess.check_output(['ps', '-ww', '-p', str(pid), '-o', 'command='], text=True)
        if '-m scripts.continuous_holdem --worker' not in command or str(out) not in command:
            raise ValueError('PID does not match the requested continuous worker')
        self.pid, self.out, self.returncode = pid, out, None

    def poll(self):
        if self.returncode is None:
            state = subprocess.run(['ps', '-p', str(self.pid), '-o', 'stat='],
                                   capture_output=True, text=True)
            if state.returncode or state.stdout.strip().startswith('Z'):
                status = json.loads((self.out/'status.json').read_text())
                self.returncode = 0 if status.get('state') == 'stopped' else 1
        return self.returncode

    def wait(self, timeout=None):
        start = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic()-start >= timeout:
                raise subprocess.TimeoutExpired('attached worker', timeout)
            time.sleep(0.1)
        return self.returncode


def compact_outputs(directory, keep=2):
    """Retire only this run's older recovery files after publishing a new one."""
    markers = sorted(directory.glob('training-*.json'),
                     key=lambda p: int(p.stem.split('-')[1]))
    for marker in markers[:-keep]:
        data = json.loads(marker.read_text())
        path = marker.with_suffix('.pt')
        _append(directory / 'retention.jsonl', {**data, 'path': path.name,
                                               'reason': 'rolling_recovery_keep_two'})
        path.unlink()
        marker.unlink()
    exports = sorted(directory.glob('average-*.pt'),
                     key=lambda p: int(p.stem.split('-')[1]))
    for path in exports[:-keep]:
        _append(directory / 'retention.jsonl', {'path': path.name,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'reason': 'rolling_exports_keep_two'})
        path.unlink()
    for path in directory.glob('outcomes-*.json'):
        target = path.with_suffix('.json.gz')
        temporary = target.with_suffix('.tmp')
        with path.open('rb') as source, gzip.open(temporary, 'wb', compresslevel=1) as out:
            shutil.copyfileobj(source, out)
        def checksum(stream):
            h = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                h.update(chunk)
            return h.hexdigest()
        with path.open('rb') as original, gzip.open(temporary, 'rb') as recovered:
            if checksum(original) != checksum(recovered):
                raise ValueError('Compressed outcome verification failed')
        temporary.replace(target)
        path.unlink()


def worker(plan_path, out, stop_after=None):
    recipe = json.loads(plan_path.read_text())
    campaign = recipe.pop('campaign', None)
    limit = campaign['iterations'] if campaign else None
    keep = 1 if campaign else 2
    plan = Experiment.from_dict({**recipe, 'iterations': 1, 'max_seconds': 1800})
    if len(plan.seeds) != 1 or len(plan.scenarios) != 1 or plan.reference:
        raise ValueError('Continuous baseline needs one seed/scenario and no reference')
    out.mkdir(parents=True, exist_ok=False)
    provenance = manifest(plan)
    provenance.pop('plan')
    provenance.update(format='holdem-continuous-v1', recipe=recipe,
                      iteration_limit=limit, wall_time_limit=None, campaign=campaign,
                      stop_after_for_verification=stop_after,
                      retention={'recovery_checkpoints': keep, 'policy_exports': keep,
                                 'outcomes': 'all retained losslessly compressed'})
    write_json(out/'manifest.json', provenance)
    seed, scenario = plan.seeds[0], plan.scenarios[0]
    directory = out/f'scenario-0-seed-{seed}'
    directory.mkdir()
    table = Table(tuple(f'player-{i}' for i in range(len(scenario.stacks))),
                  scenario.stacks, small_blind=scenario.small_blind,
                  big_blind=scenario.big_blind, chip_unit=scenario.chip_unit)
    trainer = HoldemTrainer(table, replace(plan.training, seed=seed))
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    evaluation_deals = {d for b in build_schedule(plan.arena(scenario)) for d in b.deal_seeds}
    if campaign:
        final_plan = replace(plan, blocks=campaign['final_blocks'],
                             evaluation_seed=campaign['final_seed'], benchmarks=())
        final_deals = {d for b in build_schedule(final_plan.arena(scenario)) for d in b.deal_seeds}
        if evaluation_deals & final_deals:
            raise ValueError('Validation/final deal overlap')
        evaluation_deals |= final_deals
    def status(state, phase, **extra):
        write_json(out/'status.json', dict(state=state, phase=phase, seed=seed,
                   iteration=trainer.iteration, pid=os.getpid(), updated_unix=time.time(), **extra))
    def checkpoint():
        status('running', 'checkpoint', phase_deadline_unix=time.time()+1800)
        _checkpoint(trainer, directory, provenance)
        pin_comparison(out)
        compact_outputs(directory, keep=keep)
    try:
        while not stopped and not (out/'STOP').exists() and (limit is None or trainer.iteration < limit):
            iteration = trainer.iteration+1
            if any(collection_seed(seed, iteration, role, sample, 'deal') in evaluation_deals
                   for role in range(len(scenario.stacks))
                   for sample in range(plan.training.traversals_per_player)):
                raise ValueError('Training/evaluation deal overlap')
            status('running', 'training', phase_deadline_unix=time.time()+plan.training.max_seconds+30)
            try:
                trainer.step()
            finally:
                if trainer.last_timing is not None:
                    _append(directory/'training-timing.jsonl', trainer.last_timing)
            _append(directory/'iteration-reports.jsonl', asdict(trainer.reports[-1]))
            if iteration % plan.save_every == 0:
                checkpoint()
            evaluate_now = (iteration in campaign['evaluate_at'] if campaign else iteration % plan.evaluate_every == 0)
            if evaluate_now and not stopped and not (out/'STOP').exists():
                status('running', 'evaluation', phase_deadline_unix=time.time()+1830)
                evaluation_plan = (replace(plan, benchmarks=plan.benchmarks if iteration in campaign['random_at'] else ())
                                   if campaign else plan)
                evaluate(trainer, scenario, evaluation_plan, directory, provenance, time.perf_counter()+1800)
                pin_comparison(out)
                compact_outputs(directory, keep=keep)
            if stop_after is not None and iteration >= stop_after:
                stopped = True
        if trainer.iteration and not (directory/f'training-{trainer.iteration}.json').exists():
            checkpoint()
        complete = bool(campaign and trainer.iteration == limit and not stopped and not (out/'STOP').exists())
        if complete:
            status('running', 'final_evaluation', phase_deadline_unix=time.time()+3630)
            final_directory = directory/'final'
            final_directory.mkdir()
            original = directory/f'average-{limit}.pt'
            os.link(original, final_directory/original.name)
            evaluate(trainer, scenario, final_plan, final_directory, provenance,
                     time.perf_counter()+3600, reuse_export=True)
            compact_outputs(final_directory, keep=1)
        status('complete' if complete else 'stopped', 'idle', reason='iteration_limit' if complete else 'requested_stop')
        write_json(out/'result.json', {'complete': complete, 'stopped': not complete,
                   'iteration': trainer.iteration, 'promoted': False})
    except BaseException as exc:
        status('failed', 'idle', error=repr(exc))
        write_json(out/'failure.json', {'error': repr(exc), 'iteration': trainer.iteration})
        raise


def supervise(plan, out, stop_after=None, attach_worker=None):
    campaign = json.loads(plan.read_text()).get('campaign')
    output_limit = (campaign['output_limit_gib'] if campaign else 8)*GIB
    if out.exists() and attach_worker is None:
        raise ValueError('Use a new output directory')
    if attach_worker is None and shutil.disk_usage(out.parent).free < 20*GIB:
        raise RuntimeError('Need 20 GiB free at launch')
    command = [sys.executable, '-m', 'scripts.continuous_holdem', '--worker',
               '--plan', str(plan), '--out', str(out)]
    if stop_after is not None:
        command += ['--stop-after', str(stop_after)]
    record_path = out.with_name(out.name+'-supervisor.json')
    log_path = out.with_name(out.name+('-adopt.log' if attach_worker else '-worker.log'))
    with log_path.open('xb') as log:
        child = (AttachedWorker(attach_worker, out) if attach_worker is not None else
                 subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True))
        record = {'state': 'running', 'supervisor_pid': os.getpid(), 'worker_pid': child.pid,
                  'command': command, 'iteration_limit': campaign['iterations'] if campaign else None, 'wall_time_limit': None,
                  'output_limit_bytes': output_limit,
                  'memory_limit_bytes': None, 'attached_worker': attach_worker,
                  'pinned_iteration': 1024}
        write_json(record_path, record)
        def stop(signum, frame):
            if out.exists():
                (out/'STOP').touch()
            else:
                os.killpg(child.pid, signal.SIGTERM)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        started = time.monotonic()
        state = {}
        try:
            while child.poll() is None:
                measured = subprocess.run(['ps','-p',str(child.pid),'-o','rss='],
                                          capture_output=True,text=True)
                if measured.returncode and child.poll() is None:
                    raise RuntimeError('Cannot measure worker RSS')
                rss = int(measured.stdout.strip() or 0)*1024
                all_processes = subprocess.check_output(['ps','-ww','-axo','rss=,command='],text=True)
                combined = sum(int(line.split(None,1)[0])*1024 for line in all_processes.splitlines()
                               if '-m scripts.continuous_holdem --worker' in line)
                free = shutil.disk_usage(out.parent).free
                stored = used_bytes(out) if out.exists() else 0
                state = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                pin_comparison(out)
                if free < 12*GIB or stored > output_limit:
                    raise RuntimeError('disk_limit')
                if time.time() > state.get('phase_deadline_unix', float('inf')):
                    raise RuntimeError('phase_deadline')
                if not state and time.monotonic()-started > 300:
                    raise RuntimeError('startup_deadline')
                record.update(rss_bytes=rss, combined_rss_bytes=combined, free_bytes=free,
                              stored_bytes=stored, peak_rss_bytes=max(rss,record.get('peak_rss_bytes',0)))
                write_json(record_path, record)
                time.sleep(5)
            record.update(state='stopped' if child.returncode == 0 else 'failed', returncode=child.returncode)
        except BaseException as exc:
            record.update(state='failed', error=repr(exc))
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if out.exists():
                write_json(out/'failure.json', {'error':repr(exc), 'last_boundary':state.get('iteration')})
                write_json(out/'status.json', {**state,'state':'failed','error':repr(exc)})
            raise
        finally:
            write_json(record_path, record)
    if child.returncode:
        raise RuntimeError(f'worker_exit_{child.returncode}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--attach-worker', type=int, help='Adopt a verified existing worker PID')
    p.add_argument('--stop-after', type=int, help='Finite boundary for verification only')
    args=p.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        p.error('stop-after must be positive')
    out=args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if args.attach_worker is not None:
            p.error('worker and attach-worker are mutually exclusive')
        worker(args.plan.resolve(), out, args.stop_after)
    else:
        supervise(args.plan.resolve(), out, args.stop_after, args.attach_worker)


if __name__ == '__main__':
    main()
