"""Two owner-bounded HU100 stages using existing recovery, audits and arena tools."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import time
from zipfile import ZipFile, ZIP_STORED

from scripts.hu20_scaling_supervise import run as supervise
from scripts.native_hu_followup_limits import memory_snapshot, unsafe_memory
from scripts.run_native_hu100_learning_curves import BASELINE, quote
from scripts.run_tp20_campaign import swap_bytes
from src.policies.files import file_hash

ROOT = Path('/Users/dberweger/Local/hu100-growth-20261008')
ENTRY_STOP = 6510774
PARENT_NODES = 11042440
TARGET = 20000000
SAVE = 39.10
TOOLS = 670.50

def read(p):
    return json.loads(p.read_text())

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(p)

def claim(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())

@contextmanager
def lock(path):
    import fcntl
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def identity():
    if platform.system() != 'Darwin' or Path.cwd() != ROOT:
        raise ValueError('Only the isolated M4 root is authorized')
    chip = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    if chip != 'Apple M4':
        raise ValueError('Only Apple M4 is authorized')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Committed clean tracked source required')
    return source

def snapshot(directory, deadline):
    memory = memory_snapshot()
    swap = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True)
    power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
    listing = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,%cpu=,rss=,comm='], text=True)
    heavy = []
    for line in listing.splitlines():
        fields = line.split(None, 4)
        if (len(fields) == 5 and int(fields[0]) != os.getpid()
            and (int(fields[3]) > 512 * 1024 or float(fields[2]) > 50)
            and any(k in fields[4].lower() for k in ('python', 'hu20-trainer', 'cargo', 'rustc', 'node', 'java'))):
            heavy.append(line)
    s = {'at': time(), 'deadline': deadline, 'memory': memory, 'swap': swap,
         'free_disk_bytes': shutil.disk_usage(directory).free, 'power': power, 'competing': heavy}
    s['refused'] = (time() >= deadline or unsafe_memory(memory, admission=True, rss_gib=10)
        or swap_bytes(swap) - swap_bytes(BASELINE) > .5 * 1024**3
        or s['free_disk_bytes'] < 15.5 * 1024**3 or 'AC Power' not in power or bool(heavy))
    return s

class Campaign:
    def __init__(self, stage, continuation=False):
        self.out = ROOT / 'results' / stage
        self.out.mkdir(parents=True, exist_ok=True)
        self.source = identity()
        self.binary = ROOT / 'bin/hu20-trainer'
        self.pins = {'source': self.source, 'binary_sha256': file_hash(self.binary),
                     'protocol_sha256': file_hash(ROOT / 'docs/native-hu100-growth.md')}
        for name in ('qualification', 'source-review'):
            r = read(ROOT / 'results' / (name + '.json'))
            if r['status'] != 'passed' or r['source'] != self.source:
                raise ValueError('Exact-source qualification and independent review required')
        self.state_path = self.out / 'state.json'
        if self.state_path.exists():
            self.state = read(self.state_path)
            if not continuation or self.state['status'] != 'admission-refused' or self.state['pins'] != self.pins:
                raise ValueError('No retry or changed-source continuation')
            if time() >= self.state['deadline']:
                raise ValueError('Original deadline expired')
            claim(self.out / 'continuation-claim.json', {'at': time(), 'pins': self.pins,
                   'original_state_sha256': file_hash(self.state_path), 'deadline': self.state['deadline']})
            claim(self.out / 'original-refusal-state.json', self.state)
        else:
            if continuation:
                raise ValueError('Continuation requires an existing refused stage')
            self.state = {'status': 'running', 'started': time(), 'deadline': time() + 1800,
                          'pins': self.pins, 'operations': {}}
            claim(self.state_path, self.state)
        self.deadline = self.state['deadline']
        self.state['status'] = 'running'; write(self.state_path, self.state)

    def operation(self, name, command, *, required=10, train=False, end=None):
        old = self.state['operations'].get(name)
        if old:
            if old['command'] != command or old['status'] != 'complete':
                raise ValueError('Attempted/changed operation must never repeat: ' + name)
            for path, expected in old.get('outputs', {}).items():
                p = Path(path)
                if not p.is_file() or file_hash(p) != expected:
                    raise ValueError('Completed operation output changed: ' + path)
            if file_hash(self.out / (name + '-guard/campaign.json')) != old['guard_sha256']:
                raise ValueError('Completed guard receipt changed')
            return old
        s = snapshot(self.out, self.deadline)
        s['required_seconds'] = required
        s['refused'] |= self.deadline - time() < required
        claim(self.out / f'{name}-admission-{time_ns()}.json', s)
        if s['refused']:
            self.state.update(status='admission-refused', refused_operation=name)
            write(self.state_path, self.state)
            raise AdmissionRefused(name)
        intent = {'command': command, 'status': 'attempted', 'at': time()}
        claim(self.out / (name + '-intent.json'), intent)
        self.state['operations'][name] = intent; write(self.state_path, self.state)
        g = supervise([{'name': name, 'command': command}], self.out / (name + '-guard'),
            end or self.deadline, swap_before=BASELINE, require_ac=True, rss_gib=10,
            disk_gib=15.5, swap_gib=.5, system_memory_guard=True,
            **({'stop_file': Path(command[command.index('--stop-file') + 1]),
                'soft_rss_gib': 5.63, 'save_reserve_seconds': SAVE} if train else {}))
        attempts = g['attempts']
        a = attempts[0] if attempts else {}
        # The supervisor can refuse fresh admission after our snapshot: no PID means no work.
        if not a.get('pid') and not g['failure']:
            del self.state['operations'][name]
            intent_path = self.out / (name + '-intent.json')
            intent_path.rename(self.out / (name + '-refused-intent.json'))
            (self.out / (name + '-guard')).rename(self.out / (name + '-refused-guard'))
            self.state.update(status='admission-refused', refused_operation=name)
            write(self.state_path, self.state)
            raise AdmissionRefused(name)
        accepted = (g['status'] == 'complete' or train and a.get('exit_code') == 3
                    and not a.get('guard_failure') and not g['failure'])
        intent.update(status='complete' if accepted else 'failed', seconds=time() - intent['at'],
                      guard_sha256=file_hash(self.out / (name + '-guard/campaign.json')),
                      exit_code=a.get('exit_code'), sampled_peak_family_rss_bytes=a.get('peak_aggregate_job_rss_bytes'))
        if accepted:
            paths = []
            for flag in ('--out', '--current', '--average', '--telemetry'):
                if flag in command:
                    p = Path(command[command.index(flag) + 1])
                    paths.extend([p] if p.is_file() else [f for f in p.rglob('*') if f.is_file()])
            if '--retrieve' in command:
                paths.extend(self.out / n for n in ('parent-checkpoint.gz', 'parent-average.gz', 'retrieval.json', 'parent-pr-status.json'))
            if '--seal' in command:
                paths.extend([self.out / 'archive-receipt.json', Path(command[command.index('--destination') + 1])])
            intent['outputs'] = {str(p): file_hash(p) for p in paths}
        self.state['operations'][name] = intent; write(self.state_path, self.state)
        if not accepted:
            raise ValueError('Terminal child/guard failure: ' + name)
        return intent

    def module(self, name, module, args, **kw):
        return self.operation(name, [sys.executable, '-m', module, *map(str, args)], **kw)

    def seal(self, dest):
        self.module('archive', 'scripts.run_native_hu100_growth',
                    ['--seal', self.out, '--destination', dest], required=120)
        self.state.update(status='complete', finished=time(), elapsed_seconds=time()-self.state['started'])
        write(self.state_path, self.state)

class AdmissionRefused(ValueError):
    pass

def time_ns():
    from time import time_ns as now
    return now()

def retrieve(out):
    index = read(ROOT / 'docs/reports/native-recovery-hu100-artifacts/followup-model-index.json')
    archive = Path.home() / 'Local/Research-Cloud/PR-197-native-recovery-HU100/native-recovery-hu100-followup-20261008.zip'
    # Live status is read through the public API: no credential transfer to M4.
    import urllib.request
    url = 'https://api.github.com/repos/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pulls/197'
    with urllib.request.urlopen(url, timeout=20) as response:
        status = json.load(response)
    write(out / 'parent-pr-status.json', {'url': url, 'at': time(), 'merged': status['merged'], 'merge_commit_sha': status['merge_commit_sha']})
    if not status['merged'] or archive.stat().st_size != index['archive_bytes'] or file_hash(archive) != index['archive_sha256']:
        raise ValueError('Parent PR/archive integrity failure')
    model = index['models'][-1]; restored = []
    with ZipFile(archive) as z:
        manifest = z.read(index['manifest_member'])
        if hashlib.sha256(manifest).hexdigest() != index['manifest_sha256']:
            raise ValueError('Parent manifest mismatch')
        manifest_rows = {r['path']: r for r in json.loads(manifest)['members']}
        for kind in ('checkpoint', 'average'):
            asset = model['assets'][kind]; path = out / ('parent-' + kind + '.gz')
            if manifest_rows.get(asset['path']) != asset:
                raise ValueError('Indexed asset differs from embedded member manifest')
            with z.open(asset['path']) as src, path.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if path.stat().st_size != asset['bytes'] or file_hash(path) != asset['sha256']:
                raise ValueError('Restored member integrity failure')
            restored.append({'kind': kind, **asset, 'local_path': str(path)})
    write(out / 'retrieval.json', {'status': 'verified', 'archive_id': index['archive_id'],
          'archive_sha256': index['archive_sha256'], 'manifest_sha256': index['manifest_sha256'],
          'archive_path': str(archive), 'remote_byte_download': False, 'members': restored,
          'transport': 'native synced accepted archive; read may hydrate Drive bytes'})

def telemetry(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 1:
        raise ValueError('Exactly one atomic save receipt required')
    r = rows[0]
    p = Path(r['path'])
    if p.stat().st_size != r['checkpoint_bytes'] or file_hash(p) != r['checkpoint_sha256']:
        raise ValueError('Atomic checkpoint receipt differs')
    return r

def train(c, label, parent, parent_sha, nodes, max_seconds, end):
    folder = c.out / label
    folder.mkdir(exist_ok=True)
    checkpoint = folder / 'checkpoint.gz'
    command = [str(c.binary), 'train', '--stack-bb', '100', '--resume', str(parent), '--resume-sha256', parent_sha,
        '--nodes', str(nodes), '--seed', '2026100601', '--roots-per-seat', '1', '--average-rule', 'opponent-sampled',
        '--max-entries', str(ENTRY_STOP), '--max-seconds', str(max_seconds), '--stop-file', str(folder / 'stop.json'),
        '--telemetry', str(folder / 'telemetry.jsonl'), '--out', str(checkpoint)]
    old = c.state['operations'].get(label + '-train')
    if old:
        command = old['command']
    c.operation(label + '-train', command, required=SAVE + 30, train=True, end=end)
    r = telemetry(folder / 'telemetry.jsonl')
    if r['binary_sha256'] != c.pins['binary_sha256']:
        raise ValueError('Training binary receipt mismatch')
    return folder, r

def export_audit(c, label, folder, minimum):
    c.operation(label + '-export', [str(c.binary), 'export', str(folder / 'checkpoint.gz'),
        '--current', str(folder / 'current.gz'), '--average', str(folder / 'average.gz'), '--zero-mass', 'uniform'], required=120)
    c.module(label + '-audit', 'scripts.audit_native_hu_checkpoint', ['--checkpoint', folder / 'checkpoint.gz',
        '--current', folder / 'current.gz', '--average', folder / 'average.gz', '--stack-bb', 100,
        '--target-nodes', minimum, '--out', folder / 'audit.json'], required=240)
    return read(folder / 'audit.json')

def growth_quote(pilot, operations, remaining):
    entries = pilot['diagnostics']['entries']; scale = ENTRY_STOP / entries
    training = max(0.001, pilot['elapsed_seconds_including_writes'] - pilot['write_seconds'])
    fixed_load = max(0., operations['pilot-train']['seconds'] - pilot['elapsed_seconds_including_writes'])
    rate = (pilot['completed_nodes'] - PARENT_NODES) / training
    save = max(SAVE, 2 * pilot['write_seconds'] * scale)
    tools = max(TOOLS, 2 * (operations['pilot-export']['seconds'] + operations['pilot-audit']['seconds']) * scale)
    train_seconds = 2 * ((TARGET - pilot['completed_nodes']) / rate + fixed_load)
    family = operations['pilot-train']['sampled_peak_family_rss_bytes']
    memory = 2.2 * family * scale
    required = train_seconds + save + tools + 180
    return {'status': 'admitted' if remaining >= required and memory <= 10 * 1024**3 else 'refused',
        'remaining_seconds': remaining, 'required_seconds': required, 'training_seconds': train_seconds,
        'save_seconds': save, 'tool_seconds': tools, 'closeout_seconds': 180,
        'forecast_family_rss_bytes': memory, 'entry_stop': ENTRY_STOP, 'outcomes_inspected': False}

def stage1(c, dest):
    c.module('retrieve', 'scripts.run_native_hu100_growth', ['--retrieve', c.out], required=180)
    parent = c.out / 'parent-checkpoint.gz'
    pilot_dir, pilot = train(c, 'pilot', parent, file_hash(parent), PARENT_NODES + 100000, 120,
                              c.deadline - TOOLS - 180)
    export_audit(c, 'pilot', pilot_dir, PARENT_NODES + 1)
    if (pilot['completed_nodes'] < PARENT_NODES + 100000
            or pilot['diagnostics']['entries'] >= ENTRY_STOP or pilot['stop_requested']):
        write(c.out / 'result.json', {'status': 'pilot-capacity-stop', 'terminal': pilot, 'terminal_folder': str(pilot_dir)})
    else:
        qpath = c.out / 'growth-quote.json'
        if not qpath.exists():
            claim(qpath, growth_quote(pilot, c.state['operations'], c.deadline - time()))
        q = read(qpath)
        qhash = file_hash(qpath)
        if c.state.get('quote_sha256', qhash) != qhash: raise ValueError('Growth quote changed')
        c.state['quote_sha256'] = qhash; write(c.state_path, c.state)
        if ('terminal-train' in c.state['operations'] or
                q['status'] == 'admitted' and c.deadline - time() >= q['required_seconds']):
            end = c.deadline - q['tool_seconds'] - 180
            folder, terminal = train(c, 'terminal', pilot_dir / 'checkpoint.gz', pilot['checkpoint_sha256'],
                TARGET, q['training_seconds'], end)
            audit = export_audit(c, 'terminal', folder, PARENT_NODES + 1)
            write(c.out / 'result.json', {'status': 'target-complete' if terminal['completed_nodes'] >= TARGET else 'controlled-stop',
                'terminal': terminal, 'terminal_folder': str(folder), 'audit_status': audit['status'],
                'entry_overshoot': max(0, terminal['diagnostics']['entries'] - ENTRY_STOP),
                'parent_nodes': PARENT_NODES, 'additional_nodes': terminal['completed_nodes'] - PARENT_NODES})
        else:
            write(c.out / 'result.json', {'status': 'quote-refused', 'terminal': pilot, 'terminal_folder': str(pilot_dir), 'audit_status': 'verified'})
    c.seal(dest)

def models():
    s1 = ROOT / 'results/stage1'
    if read(s1 / 'state.json')['status'] != 'complete':
        raise ValueError('Stage1 closeout required')
    result = read(s1 / 'result.json'); folder = Path(result['terminal_folder'])
    a = read(folder / 'audit.json'); terminal = telemetry(folder / 'telemetry.jsonl')
    if a['status'] != 'verified' or terminal['completed_nodes'] <= PARENT_NODES:
        raise ValueError('Advanced verified terminal required')
    # Admission pins the exact audited/restored bytes, not self-declared model metadata.
    for path, expected in a['files'].items():
        p = Path(path)
        if p.stat().st_size != expected['bytes'] or file_hash(p) != expected['sha256']:
            raise ValueError('Stage1 audited member changed')
    for member in read(s1 / 'retrieval.json')['members']:
        p = Path(member['local_path'])
        if p.stat().st_size != member['bytes'] or file_hash(p) != member['sha256']:
            raise ValueError('Stage1 restored parent changed')
    paths = [s1 / 'parent-average.gz', folder / 'average.gz']
    specs = []
    for path in paths:
        with gzip.open(path, 'rt') as f:
            h = json.loads(next(f))
        specs.append({'name': 'HU100-average-' + str(h['checkpoint_header']['native_state']['completed_nodes']),
            'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_hash(path), 'format': h['format'],
            'actual_nodes': h['checkpoint_header']['native_state']['completed_nodes'], 'entries': h['entries'],
            'iteration': h['checkpoint_header']['iteration'], 'source_checkpoint_sha256': h['source_checkpoint_sha256']})
    if specs[-1]['source_checkpoint_sha256'] != terminal['checkpoint_sha256']:
        raise ValueError('Average lineage differs')
    return specs

def stage2(c, dest):
    from scripts.report_native_hu100_learning_curves import frozen_schedule
    settings_path = c.out / 'settings.json'
    settings = {**read(ROOT / 'configs/arena/hu100-playing-baseline-v1.json'), 'models': models(),
        'pilot_root': 2026100820411, 'final_root': 2026100820412}
    settings.pop('model')
    if settings_path.exists():
        if read(settings_path) != settings:
            raise ValueError('Frozen models/config changed')
    else:
        claim(settings_path, settings)

    def panels(prefix, blocks, root):
        plays, audits, repeats = [], [], []
        for i, model in enumerate(settings['models']):
            n = str(model['actual_nodes']); config = c.out / f'{prefix}-{n}-config.json'
            single = {k: v for k, v in settings.items() if k != 'models'}; single['model'] = model
            if not config.exists(): claim(config, single)
            if read(config) != single: raise ValueError('Panel config changed')
            args = ['--config', config, '--source', c.source, '--blocks', blocks, '--root', root]
            ref = c.out / prefix / str(settings['models'][0]['actual_nodes'])
            target = c.out / prefix / n
            c.module(prefix + '-' + n + '-play', 'scripts.evaluate_native_hu100_baseline',
                [*args, '--out', target, *(['--reference-run', ref] if i else [])], required=120)
            audit = c.out / f'{prefix}-{n}-audit.json'
            c.module(prefix + '-' + n + '-audit', 'scripts.audit_native_hu100_baseline',
                ['--run', target, '--out', audit], required=120)
            repeat = c.out / (prefix + '-reproduction') / n
            repeat_ref = c.out / (prefix + '-reproduction') / str(settings['models'][0]['actual_nodes'])
            c.module(prefix + '-' + n + '-reproduce', 'scripts.evaluate_native_hu100_baseline',
                [*args, '--out', repeat, '--reproduce', target, *(['--reference-run', repeat_ref] if i else [])], required=120)
            plays.append(read(target / 'complete.json')); audits.append(read(audit)); repeats.append(read(repeat / 'complete.json'))
        def costs(rows):
            return {'wall_seconds': sum(r['wall_seconds'] for r in rows), 'blocks_per_opponent': blocks,
                'panel_costs': [p for r in rows for p in r['panel_costs']]}
        return costs(plays), {'seconds': sum(r['seconds'] for r in audits)}, costs(repeats)

    pilot, replay, reproduction = panels('pilot', 16, settings['pilot_root'])
    freeze = c.out / 'frozen-final.json'
    if not freeze.exists():
        q = quote(pilot, replay, reproduction, c.deadline - time())
        # Existing quote's measured costs apply unchanged; actual distinct hands are 30/block.
        q.update(final_hands=q['blocks_per_opponent'] * 30, source=c.source, final_root=settings['final_root'],
                 deadline=c.deadline, settings_sha256=file_hash(settings_path))
        schedule = c.out / 'frozen-schedule.json'
        if q['blocks_per_opponent']:
            document = frozen_schedule(settings, q['blocks_per_opponent'], settings['final_root'])
            freshness(settings, document)
            claim(schedule, document); q['schedule_sha256'] = file_hash(schedule)
        claim(freeze, q)
    q = read(freeze)
    fhash = file_hash(freeze)
    if c.state.get('freeze_sha256', fhash) != fhash: raise ValueError('Final freeze changed')
    c.state['freeze_sha256'] = fhash; write(c.state_path, c.state)
    if q['settings_sha256'] != file_hash(settings_path) or q['deadline'] != c.deadline or q['source'] != c.source:
        raise ValueError('Freeze/source/deadline differs')
    if q['blocks_per_opponent'] >= 32:
        if file_hash(c.out / 'frozen-schedule.json') != q['schedule_sha256']:
            raise ValueError('Frozen physical schedule changed before play')
        # On continuation only remaining never-started operations run, under the original clock.
        if not any(k.startswith('final-') for k in c.state['operations']) and c.deadline - time() < q['predicted_seconds'] + 240:
            raise ValueError('Fresh final time-fit failure')
        panels('final', q['blocks_per_opponent'], settings['final_root'])
        c.module('report', 'scripts.report_native_hu100_growth', ['--run', c.out], required=120)
    else:
        write(c.out / 'result.json', {'status': 'no-final-budget', 'freeze': q})
    c.seal(dest)

def freshness(settings, document):
    from scripts.report_native_hu100_learning_curves import frozen_schedule
    old = read(ROOT / 'configs/arena/hu100-learning-curves-v1.json')
    roots = [(old, old['pilot_root'], 16), (old, old['final_root'], 2048)]
    baseline = read(ROOT / 'configs/arena/hu100-playing-baseline-v1.json')
    roots.extend([(dict(baseline, models=[baseline['model']]), baseline['pilot_root'], 16),
                  (dict(baseline, models=[baseline['model']]), baseline['final_root'], 2048),
                  (settings, settings['pilot_root'], 16)])
    seeds = {b['deal_seeds'][0] for p in document['panels'].values() for b in p['blocks']}
    for config, root, blocks in roots:
        other = frozen_schedule(config, blocks, root)
        prior = {b['deal_seeds'][0] for p in other['panels'].values() for b in p['blocks']}
        if seeds & prior:
            raise ValueError('Fresh physical-deal collision')

def seal(root, destination):
    paths = {str(p.relative_to(root)): p for p in root.rglob('*') if p.is_file()
             and p.name not in ('phase.lock', 'state.json')
             and not any(part.startswith('archive') for part in p.relative_to(root).parts)}
    # Freeze pre-archive operation state; the live supervisor updates its own receipts.
    state = read(root / 'state.json')
    state['status'] = 'science-complete-awaiting-archive'
    state['operations'].pop('archive', None)
    write(root / 'science-closeout.json', state)
    paths['science-closeout.json'] = root / 'science-closeout.json'
    paths['source.tar'] = ROOT / 'source.tar'; paths['bin/hu20-trainer'] = ROOT / 'bin/hu20-trainer'
    for name in ('qualification', 'source-review', 'environment'):
        paths[name + '.json'] = ROOT / 'results' / (name + '.json')
    # Stage2 model snapshots retain exact bytes; Stage1 is independently restorable.
    members = [{'path': name, 'bytes': p.stat().st_size, 'sha256': file_hash(p)} for name, p in sorted(paths.items())]
    encoded = json.dumps({'source_root': str(root), 'members': members, 'originals_retained': True}, sort_keys=True).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = root / 'archive.partial.zip'
    with ZipFile(temp, 'x', compression=ZIP_STORED, allowZip64=True) as z:
        for m in members:
            z.write(paths[m['path']], 'research/' + m['path'])
        z.writestr('ARCHIVE-MANIFEST.json', encoded)
    with ZipFile(temp) as z:
        for m in members:
            h = hashlib.sha256(); size = 0
            with z.open('research/' + m['path']) as f:
                while chunk := f.read(8 * 1024**2): h.update(chunk); size += len(chunk)
            if size != m['bytes'] or h.hexdigest() != m['sha256']:
                raise ValueError('Archive readback mismatch')
    digest = file_hash(temp)
    # A new destination is never overwritten. Keep the nonsynced local original.
    with temp.open('rb') as src, destination.open('xb') as dst: shutil.copyfileobj(src, dst)
    if file_hash(destination) != digest:
        raise ValueError('Native archive copy mismatch')
    write(root / 'archive-receipt.json', {'status': 'verified', 'archive': str(destination), 'bytes': temp.stat().st_size,
        'sha256': digest, 'manifest_sha256': hashlib.sha256(encoded).hexdigest(), 'members_verified': len(members),
        'finished': time(), 'originals_retained': True, 'cloud_acceptance': 'pending', 'remote_bytes_downloaded': False})

def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=('stage1', 'stage2'))
    p.add_argument('--continue-unstarted', action='store_true'); p.add_argument('--destination', type=Path)
    p.add_argument('--retrieve', type=Path); p.add_argument('--seal', type=Path)
    a = p.parse_args()
    if a.retrieve: retrieve(a.retrieve); return
    if a.seal: seal(a.seal, a.destination); return
    if not a.stage or not a.destination: p.error('--stage and --destination required')
    ROOT.joinpath('results').mkdir(exist_ok=True)
    with lock(ROOT / 'results/phase.lock'):
        if a.stage == 'stage2':
            # Only compact integrity receipts are read before the separate timer.
            if read(ROOT / 'results/stage1/state.json')['status'] != 'complete':
                raise ValueError('Stage1 closeout required before Stage2')
        c = Campaign(a.stage, a.continue_unstarted)
        try:
            (stage1 if a.stage == 'stage1' else stage2)(c, a.destination)
        except AdmissionRefused:
            raise
        except BaseException as e:
            c.state.update(status='failed', error=repr(e), stopped=time()); write(c.state_path, c.state)
            raise

if __name__ == '__main__': main()
