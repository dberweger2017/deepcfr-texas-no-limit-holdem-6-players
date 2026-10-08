"""One reviewed 50M-total HU100 campaign; reuse guarded recovery and paired tools."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import time
from zipfile import ZipFile

from scripts import run_native_hu100_growth as g
from scripts.benchmark_hu_export_audit import measure
from scripts.native_hu100_model_metadata import audited_average_spec
from src.policies.files import file_hash

ROOT = Path('/Users/dberweger/Local/hu100-growth-50m-20261008')
MODULE = 'scripts.run_native_hu100_growth_50m'
PROTOCOL = 'docs/native-hu100-growth-50m.md'
BINARY = 'results/runtime/hu20-trainer'
BINARY_SHA = 'b547173b838d896888973f4cf283ba66a60b61781a29a7ed719ebcbadb448f2c'
PARENT_NODES = 20001470
TARGET = 50000000
PILOT_ROOT = 2026100850411
FINAL_ROOT = 2026100850412
ARCHIVE_ID = '1hC98K7KSU-pFEdPD7lfFMT17EcxpW_Re'
ARCHIVE_SHA = 'bb42f78802d552e4d790e491ca23bbf3a6970f1a233230bbcf409fd116b2a896'
MANIFEST_SHA = '4424b8066fbf7638db09ddc8936fceae94676080e628124c05ec20da117d9a9b'


def indexed_parent():
    return g.read(ROOT / 'docs/reports/native-hu100-growth-artifacts/terminal-audit.json')


def capacity(audit, resources, free_disk_bytes, *, pilot_peak=None, pilot_entries=None):
    """Entry admission, bounded to twice measured entries; guards stay unchanged."""
    entries = audit['entries']
    peak = resources['operations']['terminal-train']['sampled_peak_family_rss_bytes']
    bytes_per_entry = peak / entries
    if pilot_peak is not None:
        bytes_per_entry = max(bytes_per_entry, pilot_peak / pilot_entries)
    asset_bytes = sum(v['bytes'] for v in audit['files'].values()) / entries
    memory_stop = math.floor(5.63 * 1024**3 / (2.2 * bytes_per_entry))
    # Twelve asset-equivalents and four fixed GiB reserve input/pilot/terminal,
    # local+native archive copies and later raw arena/model evidence.
    disk_stop = max(0, math.floor((free_disk_bytes - (15.5 + 4) * 1024**3) / (12 * asset_bytes)))
    entry_stop = min(2 * entries, memory_stop, disk_stop)
    return {'entry_stop': entry_stop, 'parent_entries': entries,
        'memory_entry_stop': memory_stop, 'disk_entry_stop': disk_stop,
        'extrapolation_entry_stop': 2 * entries, 'rss_bytes_per_entry': bytes_per_entry,
        'memory_allowance': 2.2, 'soft_rss_gib': 5.63, 'hard_rss_gib': 10,
        'disk_floor_gib': 15.5, 'fixed_evidence_reserve_gib': 4,
        'disk_asset_equivalents': 12, 'combined_asset_bytes_per_entry': asset_bytes,
        'free_disk_bytes': free_disk_bytes, 'outcomes_inspected': False,
        'scope': 'conservative admission estimate; no demonstrated RAM exhaustion or 50M fit'}


class Campaign(g.Campaign):
    def __init__(self, stage, continuation=False):
        super().__init__(stage, continuation, root=ROOT, protocol=PROTOCOL, binary_relative=BINARY)
        if self.pins['binary_sha256'] != BINARY_SHA:
            raise ValueError('Exact frozen native binary required')

    def operation(self, name, command, **kw):
        if (g.identity(ROOT) != self.source or file_hash(self.binary) != self.pins['binary_sha256']
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], text=True).strip()):
            raise ValueError('Clean source and unchanged ignored runtime binary required')
        return super().operation(name, command, **kw)

    def seal(self, dest):
        self.module('archive', MODULE, ['--seal', self.out, '--destination', dest], required=120)
        self.state.update(status='complete', finished=time(), elapsed_seconds=time()-self.state['started'])
        g.write(self.state_path, self.state)


def retrieve(out):
    import urllib.request
    url = 'https://api.github.com/repos/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pulls/203'
    with urllib.request.urlopen(url, timeout=20) as response:
        status = json.load(response)
    g.claim(out / 'parent-pr-status.json', {'url': url, 'at': time(), 'merged': status['merged'],
                                         'merge_commit_sha': status['merge_commit_sha']})
    if not status['merged']:
        raise ValueError('Parent PR203 must currently be merged')
    archive = Path.home() / 'Local/Research-Cloud/PR-203-HU100-growth-playing-gains/hu100-growth-stage1-M4-20261008.zip'
    index = g.read(ROOT / 'docs/reports/native-hu100-growth-artifacts/stage1-archive.json')
    audit = indexed_parent()
    if (archive.stat().st_size != index['bytes'] or file_hash(archive) != ARCHIVE_SHA
        or index['sha256'] != ARCHIVE_SHA or index['manifest_sha256'] != MANIFEST_SHA):
        raise ValueError('Indexed parent archive differs')
    restored = []
    with ZipFile(archive) as z:
        raw = z.read('ARCHIVE-MANIFEST.json')
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
            raise ValueError('Parent member manifest differs')
        members = {m['path']: m for m in json.loads(raw)['members']}
        for kind in ('checkpoint', 'average', 'audit'):
            member = 'terminal/' + ('audit.json' if kind == 'audit' else kind + '.gz')
            expected = members[member]
            if kind != 'audit':
                pinned = next(v for p, v in audit['files'].items() if Path(p).name == kind + '.gz')
                if {k: expected[k] for k in ('bytes', 'sha256')} != pinned:
                    raise ValueError('Parent member differs from verified full audit')
            target = out / ('parent-audit.json' if kind == 'audit' else 'parent-' + kind + '.gz')
            with z.open('research/' + member) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if target.stat().st_size != expected['bytes'] or file_hash(target) != expected['sha256']:
                raise ValueError('Parent restoration hash/size differs')
            restored.append({'kind': kind, **expected, 'archive_member': 'research/' + member, 'local_path': str(target)})
    if g.read(out / 'parent-audit.json') != audit:
        raise ValueError('Archived audit differs from independently reviewed indexed receipt')
    audited_average_spec(out / 'parent-average.gz', audit,
        checkpoint_sha256=audit['audit']['checkpoint_sha256'], actual_nodes=PARENT_NODES)
    g.claim(out / 'retrieval.json', {'status': 'verified', 'archive_id': ARCHIVE_ID,
        'archive_path': str(archive), 'archive_sha256': ARCHIVE_SHA, 'manifest_sha256': MANIFEST_SHA,
        'members': restored, 'remote_bytes_downloaded': False,
        'transport': 'native synced archive read; hydration possible, no connector byte download'})


def quote(pilot, measurements, operations, entry_stop, remaining):
    entries = pilot['diagnostics']['entries']; scale = entry_stop / entries
    training = max(.001, pilot['elapsed_seconds_including_writes'] - pilot['write_seconds'])
    load = max(0., operations['pilot-train']['seconds'] - pilot['elapsed_seconds_including_writes'])
    rate = (pilot['completed_nodes'] - PARENT_NODES) / training
    save = max(g.SAVE, 2 * pilot['write_seconds'] * scale)
    tools = max(g.TOOLS, 2 * (operations['pilot-export']['seconds'] + operations['pilot-audit']['seconds']) * scale)
    train_seconds = 2 * (max(0, TARGET-pilot['completed_nodes']) / rate + load)
    rss = 2.2 * measurements['sampled_peak_family_rss_bytes'] * scale
    required = train_seconds + save + tools + 180
    return {'status': 'admitted' if remaining >= required and rss <= 5.63 * 1024**3 else 'refused',
        'entry_stop': entry_stop, 'remaining_seconds': remaining, 'required_seconds': required,
        'training_seconds': train_seconds, 'save_seconds': save, 'tool_seconds': tools,
        'closeout_seconds': 180, 'forecast_family_rss_bytes': rss, 'outcomes_inspected': False}


def bind_receipt(c, key, path):
    digest = file_hash(path)
    if c.state.get(key, digest) != digest:
        raise ValueError('Frozen capacity receipt changed')
    c.state[key] = digest; g.write(c.state_path, c.state)


def stage1(c, dest):
    initial_path = c.out / 'initial-capacity.json'
    if not initial_path.exists():
        g.claim(initial_path, capacity(indexed_parent(),
            g.read(ROOT / 'docs/reports/native-hu100-growth-artifacts/stage1-resources.json'), shutil.disk_usage(c.out).free))
    bind_receipt(c, 'initial_capacity_sha256', initial_path)
    initial = g.read(initial_path)
    if initial['entry_stop'] <= initial['parent_entries']:
        raise ValueError('No growth capacity admitted; never-started science')
    c.module('retrieve', MODULE, ['--retrieve', c.out], required=180)
    parent = c.out / 'parent-checkpoint.gz'
    pilot_dir, pilot = g.train(c, 'pilot', parent, file_hash(parent), PARENT_NODES+100000, 120,
        c.deadline-g.TOOLS-180, entry_stop=initial['entry_stop'], measure_module=MODULE)
    g.export_audit(c, 'pilot', pilot_dir, PARENT_NODES+1)
    measurements = g.read(pilot_dir / 'family-measurement.json')
    cap_path = c.out / 'pilot-capacity.json'
    if not cap_path.exists():
        revised = capacity(indexed_parent(), g.read(ROOT / 'docs/reports/native-hu100-growth-artifacts/stage1-resources.json'),
            shutil.disk_usage(c.out).free, pilot_peak=measurements['sampled_peak_family_rss_bytes'],
            pilot_entries=pilot['diagnostics']['entries'])
        revised['entry_stop'] = min(initial['entry_stop'], revised['entry_stop'])
        g.claim(cap_path, revised)
    bind_receipt(c, 'capacity_sha256', cap_path)
    cap = g.read(cap_path); entry_stop = cap['entry_stop']
    terminal, folder = pilot, pilot_dir
    if (pilot['completed_nodes'] >= PARENT_NODES+100000 and not pilot['stop_requested']
        and pilot['diagnostics']['entries'] < entry_stop):
        qpath = c.out / 'growth-quote.json'
        if not qpath.exists():
            g.claim(qpath, quote(pilot, measurements, c.state['operations'], entry_stop, c.deadline-time()))
        q = g.read(qpath)
        qhash = file_hash(qpath)
        if c.state.get('quote_sha256', qhash) != qhash: raise ValueError('Frozen growth quote changed')
        c.state['quote_sha256'] = qhash; g.write(c.state_path, c.state)
        if ('terminal-train' in c.state['operations'] or
            q['status'] == 'admitted' and c.deadline-time() >= q['required_seconds']):
            end = c.deadline-q['tool_seconds']-180
            folder, terminal = g.train(c, 'terminal', pilot_dir / 'checkpoint.gz', pilot['checkpoint_sha256'],
                TARGET, q['training_seconds'], end, entry_stop=entry_stop, measure_module=MODULE)
            g.export_audit(c, 'terminal', folder, PARENT_NODES+1)
    audit = g.read(folder / 'audit.json')
    applied_entry_stop = initial['entry_stop'] if folder == pilot_dir else entry_stop
    g.write(c.out / 'result.json', {'status': 'target-complete' if terminal['completed_nodes'] >= TARGET else 'controlled-stop',
        'terminal': terminal, 'terminal_folder': str(folder), 'audit_status': audit['status'],
        'parent_nodes': PARENT_NODES, 'additional_nodes': terminal['completed_nodes']-PARENT_NODES,
        'entry_stop': applied_entry_stop, 'postpilot_entry_ceiling': entry_stop,
        'entry_overshoot': max(0, terminal['diagnostics']['entries']-applied_entry_stop),
        'target_overshoot': max(0, terminal['completed_nodes']-TARGET),
        'capacity_sha256': file_hash(cap_path), 'initial_capacity_sha256': file_hash(initial_path)})
    c.seal(dest)


def models():
    run = ROOT / 'results/stage1'
    if g.read(run / 'state.json')['status'] != 'complete': raise ValueError('Stage1 closeout required')
    result = g.read(run / 'result.json'); folder = Path(result['terminal_folder'])
    terminal = g.telemetry(folder / 'telemetry.jsonl')
    if terminal['completed_nodes'] <= PARENT_NODES: raise ValueError('Advanced terminal required')
    for path, receipt in g.read(folder / 'audit.json')['files'].items():
        if Path(path).stat().st_size != receipt['bytes'] or file_hash(Path(path)) != receipt['sha256']:
            raise ValueError('Stage1 audited member changed')
    for receipt in g.read(run / 'retrieval.json')['members']:
        path = Path(receipt['local_path'])
        if path.stat().st_size != receipt['bytes'] or file_hash(path) != receipt['sha256']:
            raise ValueError('Restored parent changed')
    return [audited_average_spec(run / 'parent-average.gz', g.read(run / 'parent-audit.json'),
                checkpoint_sha256=indexed_parent()['audit']['checkpoint_sha256'], actual_nodes=PARENT_NODES),
            audited_average_spec(folder / 'average.gz', g.read(folder / 'audit.json'),
                checkpoint_sha256=terminal['checkpoint_sha256'], actual_nodes=terminal['completed_nodes'])]


def stage2(c, dest):
    settings = {**g.read(ROOT / 'configs/arena/hu100-playing-baseline-v1.json'), 'models': models(),
                'pilot_root': PILOT_ROOT, 'final_root': FINAL_ROOT}
    settings.pop('model')
    # Include #203's fresh panels in addition to the existing #197/#200/pilot checks.
    prior = [(settings, 2026100820411, 16), (settings, 2026100820412, 2048)]
    g.stage2(c, dest, settings=settings, prior_roots=prior)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=('stage1', 'stage2'))
    p.add_argument('--continue-unstarted', action='store_true'); p.add_argument('--destination', type=Path)
    p.add_argument('--retrieve', type=Path); p.add_argument('--seal', type=Path)
    p.add_argument('--measure-training', action='store_true'); p.add_argument('--deadline', type=float)
    p.add_argument('--measurement', type=Path); p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.measure_training:
        command = a.command[1:] if a.command[:1] == ['--'] else a.command
        result = measure(command, ROOT, a.measurement.parent, 'training', os.getppid(), a.deadline,
                         accepted_returncodes=(0, 3))
        g.claim(a.measurement, result); sys.exit(result['returncode'])
    if a.retrieve: retrieve(a.retrieve); return
    if a.seal:
        import urllib.request
        url = 'https://api.github.com/repos/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pulls/204'
        with urllib.request.urlopen(url, timeout=20) as response:
            current = json.load(response)
        status_path = a.seal / 'owning-pr-status-before-seal.json'
        g.claim(status_path, {'url': url, 'at': time(), 'state': current['state'], 'merged': current['merged']})
        if current['state'] != 'open' or current['merged']:
            raise ValueError('This owner-authorized open PR204 archive only')
        logs = {'qualification-logs/' + p.name: p for p in (ROOT / 'results').glob('qualification*.log')}
        g.seal(a.seal, a.destination, source_root=ROOT, binary_relative=BINARY, extra_paths=logs)
        return
    if not a.stage or not a.destination: p.error('--stage and --destination required')
    ROOT.joinpath('results').mkdir(exist_ok=True)
    with g.lock(ROOT / 'results/phase.lock'):
        if a.stage == 'stage2' and g.read(ROOT / 'results/stage1/state.json')['status'] != 'complete':
            raise ValueError('Stage1 closeout required before Stage2 timer')
        c = Campaign(a.stage, a.continue_unstarted)
        try: (stage1 if a.stage == 'stage1' else stage2)(c, a.destination)
        except g.AdmissionRefused: raise
        except BaseException as e:
            c.state.update(status='failed', error=repr(e), stopped=time()); g.write(c.state_path, c.state); raise

if __name__ == '__main__': main()
