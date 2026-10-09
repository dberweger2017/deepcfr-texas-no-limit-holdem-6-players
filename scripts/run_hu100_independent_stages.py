"""One-use, independently admitted HU100 training and evaluation stages."""
import argparse
from dataclasses import asdict
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, time
from zipfile import ZipFile, ZIP_STORED

from scripts.hu100_qualification_guard import Campaign, CapacityStop, GIB, put, read
from scripts import run_hu100_seed_qualification as previous
from scripts.evaluate_native_hu100_baseline import execute, make_plan
from scripts.audit_native_hu100_baseline import audit
from scripts.native_hu100_model_metadata import audited_average_spec
from scripts.report_native_hu100_learning_curves import frozen_schedule
from scripts.report_hu100_seed_qualification import report
from src.arena.registry import PolicyRegistry, artifact_suffix
from src.blueprint.average import TranslationOptions
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/hu100-independent-stages'
MODULE = 'scripts.run_hu100_independent_stages'
OLD_PARTIAL = Path.home() / 'Local/hu100-seed-qualification-20261009/results/hu100-seed-qualification'
BINARY = OUT / 'bin/hu20-trainer'
EARLY, TERMINAL, ENTRY_CAP = previous.EARLY, previous.TERMINAL, previous.ENTRY_CAP
SEEDS = previous.SEEDS
CALIBRATION_ROOTS = {32: 2026100911011, 512: 2026100911012}
FINAL_ROOT = 2026100911013
CLOSEOUT = 1800
# The new task supplies the budget and specifically authorizes this fresh campaign.
# These ceilings are inherited; failure never resets either clock or baseline.


def config(model, translated=False):
    value = previous.config(model, translated)
    value.update(pilot_root=CALIBRATION_ROOTS[32], final_root=FINAL_ROOT)
    return value


class LinkedRegistry(PolicyRegistry):
    """One physical canonical model, with audited hardlink snapshot identities."""
    def snapshot(self, output):
        target = output / 'models'
        target.mkdir()
        links = []
        for model in self.models.values():
            origin = model.source_path.resolve()
            destination = target / (model.spec.sha256 + artifact_suffix(model.spec.format))
            before = origin.stat()
            if file_hash(origin) != model.spec.sha256:
                raise ValueError('Canonical snapshot changed')
            os.link(origin, destination)
            after = destination.stat()
            if (after.st_dev, after.st_ino, after.st_size) != (before.st_dev, before.st_ino, before.st_size):
                raise ValueError('Snapshot link identity differs')
            if file_hash(destination) != model.spec.sha256:
                raise ValueError('Linked snapshot bytes differ')
            links.append({'canonical': str(origin), 'snapshot': str(destination),
                'device': after.st_dev, 'inode': after.st_ino, 'bytes': after.st_size,
                'sha256': model.spec.sha256, 'immutable_by_contract': True})
        put(output / 'snapshot-links.json', links)


def checked_put(path, value):
    if path.exists():
        if read(path) != json.loads(json.dumps(value)):
            raise ValueError('Existing preparation metadata differs: ' + str(path))
    else:
        put(path, value)


def prepared():
    index = read(ROOT / 'docs/reports/native-hu100-growth-1b-artifacts/model-index.json')
    assets = []
    for label, nodes in (('early', EARLY), ('terminal', 1_000_002_065)):
        record = next(m for m in index['models'] if m['audit']['native_state']['completed_nodes'] == nodes)
        receipt = record['audit']
        origin = next(Path(p) for p in receipt['files'] if Path(p).name == 'average.gz')
        pin = receipt['files'][str(origin)]
        if origin.stat().st_size != pin['bytes'] or file_hash(origin) != pin['sha256']:
            raise ValueError('Merged #207 input changed')
        folder = OUT / 'inputs' / label
        folder.mkdir(parents=True, exist_ok=True)
        average = folder / 'average.gz'
        if not average.exists():
            os.link(origin, average)
        if average.stat().st_size != pin['bytes'] or file_hash(average) != pin['sha256']:
            raise ValueError('Existing retrieved average differs')
        checked_put(folder / 'audit.json', receipt)
        spec = audited_average_spec(folder / 'average.gz', receipt,
            checkpoint_sha256=receipt['audit']['checkpoint_sha256'], actual_nodes=nodes)
        checked_put(folder / 'spec.json', spec)
        assets.append({'owning_pr': 207, 'status': 'MERGED live before retrieval', 'original': str(origin),
            'member': record['archive_members']['average'], **pin})
    BINARY.parent.mkdir(exist_ok=True)
    if not BINARY.exists():
        shutil.copy2(previous.OLD_ROOT / 'native/hu20-trainer/target/release/hu20-trainer', BINARY)
    if file_hash(BINARY) != previous.BINARY_SHA:
        raise ValueError('Reviewed binary differs')
    frozen = 'bd0e7a417064f736091dc2b667954b50becb4b69'
    trees = {}
    for tree in ('native/hu20-trainer', 'native/hu20-buckets'):
        old = subprocess.check_output(['git', 'rev-parse', frozen + ':' + tree], text=True).strip()
        # The executed binary is #207's, not main's optional equity-bucket trainer.
        trees[tree] = old
    native_source = OUT / 'native-execution-source.tar'
    with native_source.open('xb') as stream:
        subprocess.run(['git','archive',frozen,'native/hu20-trainer','native/hu20-buckets'],stdout=stream,check=True)
    put(OUT/'native-execution-source.json', {'revision':frozen,'trees':trees,
        'source_archive_sha256':file_hash(native_source),'binary_sha256':previous.BINARY_SHA,
        'provenance':'exact #207 frozen native source and its indexed binary; inactive current native source is not executed'})
    partial = OLD_PARTIAL / 'training/2026100901/pilot'
    pin = {'bytes': 15192117, 'sha256': '3f5fbc769bf59363b7558af61b59e413cfff7b8472c02c6d4e3aa4b66f11d0ec'}
    path = partial / 'checkpoint.gz'
    receipt = read(partial / 'audit.json')
    if (path.stat().st_size != pin['bytes'] or file_hash(path) != pin['sha256']
            or receipt['status'] != 'verified' or receipt['audit']['checkpoint_sha256'] != pin['sha256']
            or receipt['native_state']['completed_nodes'] != 1000373 or receipt['iteration'] != 642):
        raise ValueError('#211 partial provenance differs')
    with gzip.open(path, 'rt') as f:
        header = json.loads(next(f))
    put(OUT / 'partial-header.json', header)
    folder = OUT / 'inputs/partial'
    folder.mkdir()
    shutil.copy2(path, folder / 'checkpoint.gz')
    if file_hash(folder / 'checkpoint.gz') != pin['sha256']:
        raise ValueError('Partial retrieval changed')
    put(folder / 'audit.json', receipt)
    assets.append({'owning_pr': 211, 'status': 'MERGED live before retrieval', 'original': str(path),
        'archive_id': '1MkVUkiRMZE_7UK790J7Pu6ObfPw8gMtr',
        'member': 'research/training/2026100901/pilot/checkpoint.gz', **pin})
    with (OUT / 'source.tar').open('xb') as f:
        subprocess.run(['git', 'archive', 'HEAD'], stdout=f, check=True)
    put(OUT / 'environment.json', {'python': sys.version,
        'packages': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
        'native_binary_sha256': previous.BINARY_SHA, 'native_trees': trees})
    put(OUT / 'retrieval.json', {'assets': assets, 'originals_unchanged': True,
        'read_scope': 'retained nonsynced originals verified against merged model/index; no archive hydration',
        'pr207_archive_id': '1iowJoQBQqB3tLRnU6GD0JcniF0qDIvgj',
        'pr207_archive_sha256': previous.ARCHIVE_SHA, 'pr207_manifest_sha256': previous.MANIFEST_SHA})


def freshness():
    settings = config(read(OUT / 'inputs/early/spec.json'))
    roots = dict(previous.PRIOR_ROOTS + [(previous.TRANSLATION_PILOT, 16),
        (previous.TRANSLATION_FINAL, 2048), (2026100810011, 16), (2026100810012, 2048),
        (previous.PILOT_ROOT, 16), (previous.FINAL_ROOT, 8192),
        *[(root, blocks) for blocks, root in CALIBRATION_ROOTS.items()], (FINAL_ROOT, 8192)])
    seen = {}
    for root, blocks in roots.items():
        schedule = frozen_schedule({'models': [settings['model']]}, blocks, root)
        deals = {b['deal_seeds'][0] for p in schedule['panels'].values() for b in p['blocks']}
        if any(deals & old for old in seen.values()):
            raise ValueError('Physical deal collision')
        seen[root] = deals
    put(OUT / 'freshness.json', {'roots': roots, 'all_pairwise_disjoint': True})


def run_options(model, options, blocks, root, prefix, source, registry=None):
    settings = config(model)
    load_seconds = 0
    if registry is None:
        tick = perf_counter()
        registry = LinkedRegistry(make_plan(settings, 'random', blocks, root))
        load_seconds = perf_counter() - tick
    costs = []
    for label, translated in options:
        cfg = OUT / prefix / (label + '-config.json')
        put(cfg, config(model, translated))
        first = '2026100601-early' if prefix == 'final' else 'old-early'
        reference = OUT / prefix / first
        repeat_reference = OUT / (prefix + '-reproduction') / first
        run = OUT / prefix / label
        repeat = OUT / (prefix + '-reproduction') / label
        execute(cfg, run, blocks, root, source, registry=registry,
            reference_run=reference if label != first else None)
        audit(run, OUT / prefix / (label + '-audit.json'))
        execute(cfg, repeat, blocks, root, source, registry=registry, reproduce=run,
            reference_run=repeat_reference if label != first else None)
        primary, reproduction = [read(p / 'costs.json') for p in (run, repeat)]
        replay = read(OUT / prefix / (label + '-audit.json'))
        fixed = sum(c['model_load_or_validation_seconds'] + c['snapshot_seconds'] +
            c['output_model_hash_seconds'] for c in (primary, reproduction))
        variable = sum(c['play_and_report_seconds'] + c['output_raw_hash_seconds'] + c['panel_setup_seconds']
            for c in (primary, reproduction)) + replay['seconds'] - replay['model_hash_seconds']
        costs.append({'label': label, 'blocks': blocks, 'fixed_seconds': fixed + replay['model_hash_seconds'],
            'variable_seconds': variable, 'primary': primary, 'reproduction': reproduction,
            'replay_seconds': replay['seconds'], 'replay_model_hash_seconds': replay['model_hash_seconds']})
    return registry, {'load_seconds': load_seconds, 'arms': costs}


def calibration(source):
    measurements = []
    for endpoint in ('early', 'terminal'):
        model = read(OUT / 'inputs' / endpoint / 'spec.json')
        registry = None
        for blocks, root in CALIBRATION_ROOTS.items():
            options = [('old-' + endpoint, False)]
            if endpoint == 'terminal':
                options.append(('old-terminal-on', True))
            registry, cost = run_options(model, options, blocks, root, 'calibration-' + str(blocks), source, registry)
            measurements.append({'endpoint': endpoint, 'blocks': blocks, **cost})
        del registry
        gc.collect()
    put(OUT / 'calibration-costs.json', {'measurements': measurements,
        'outcomes_inspected': False, 'fixed_loads_per_model': 1,
        'sample_sizes': list(CALIBRATION_ROOTS), 'sample_ratio': 16})


def evaluation(source, seed, endpoint, blocks):
    model = read(OUT / 'models.json')[str(seed)][endpoint]
    options = [(f'{seed}-{endpoint}', False)]
    if endpoint == 'terminal':
        options.append((f'{seed}-terminal-on', True))
    _, costs = run_options(model, options, blocks, FINAL_ROOT, 'final', source)
    put(OUT / f'evaluation-cost-{seed}-{endpoint}.json', costs)


def training_quote():
    old = read(ROOT / 'docs/reports/native-hu100-growth-1b-artifacts/training-result.json')
    ops = read(ROOT / 'docs/reports/native-hu100-growth-1b-artifacts/resources.json')['operations']
    saves = old['saves']
    entries = math.ceil(7_643_261*1.1) + ENTRY_CAP
    train = 2 * (saves[-1]['elapsed_seconds_including_writes'] - sum(x['write_seconds'] for x in saves))
    # Native last-save elapsed is cumulative; all fixed saves are charged separately.
    save = 2 * max(x['write_seconds']/x['diagnostics']['entries'] for x in saves) * entries
    tool = 2 * max(sum(o['seconds'] for o in ops if o['name'] in
        (f"save-{x['requested_nodes']}-export", f"save-{x['requested_nodes']}-audit"))/x['diagnostics']['entries'] for x in saves) * entries
    return {'per_seed_training_seconds': train, 'per_seed_save_seconds': save,
        'per_seed_export_audit_seconds': tool, 'per_seed_seconds': train + save + tool,
        'factor': '2x #207 cumulative nonsave cost and max save/tool seconds per entry at unchanged entry cap',
        'early_entries_forecast': math.ceil(7_643_261*1.1), 'terminal_entries_forecast': ENTRY_CAP}


def evaluation_quote(blocks, measurements, models):
    arms = [arm for m in measurements for arm in m['arms']]
    # Independent small/large phase costs give a conservative slope without
    # ever attributing fixed load/snapshot/model hashing to hand count.
    slope = max(a['variable_seconds']/a['blocks'] for a in arms)
    fixed = max(a['fixed_seconds'] for a in arms)
    terminal_load = max(m['load_seconds'] for m in measurements if m['endpoint'] == 'terminal')
    early_load = max(m['load_seconds'] for m in measurements if m['endpoint'] == 'early')
    old_entries = {'early': 7643261, 'terminal': 41010014}
    loads = sum((early_load if label == 'early' else terminal_load) *
        models[str(seed)][label]['entries']/old_entries[label] for seed in SEEDS for label in ('early', 'terminal'))
    play = 3 * (loads + 9*fixed + 9*slope*blocks)
    oldops = read(ROOT / 'docs/reports/native-hu100-growth-1b-artifacts/resources.json')['operations']
    oldreport = next(x['seconds'] for x in oldops if x['name'] == 'strict-report')
    report_seconds = 3*oldreport*max(1.5*blocks/2048, sum(m['entries'] for pair in models.values() for m in pair.values()) / 126_000_000)
    return {'blocks': blocks, 'load_seconds': loads, 'fixed_arm_seconds': 9*fixed,
        'per_block_all_nine_arms_seconds': 9*slope, 'play_replay_reproduction_seconds': play,
        'report_seconds': report_seconds, 'seconds': play + report_seconds,
        'factor': '3x fixed loads/setup/snapshot/model hashing plus 3x max measured per-block play/replay/reproduction/raw hashing; strict report separate'}


def command(folder, seed, nodes, resume=None):
    previous.BINARY = BINARY
    return previous.fixture_command(folder, seed, nodes, resume=resume)


def tools(campaign, folder, label, nodes, quote):
    campaign.run(label + '-export', [BINARY, 'export', folder / 'checkpoint.gz', '--current',
        folder / 'current.gz', '--average', folder / 'average.gz', '--zero-mass', 'uniform'],
        reserve=CLOSEOUT + quote*2/3, quote=quote/3)
    campaign.run(label + '-audit', [sys.executable, '-m', 'scripts.audit_native_hu_checkpoint',
        '--checkpoint', folder / 'checkpoint.gz', '--current', folder / 'current.gz', '--average',
        folder / 'average.gz', '--stack-bb', 100, '--target-nodes', nodes, '--out', folder / 'audit.json'], quote=quote*2/3)


def spec(folder):
    a = read(folder / 'audit.json')
    return audited_average_spec(folder / 'average.gz', a,
        checkpoint_sha256=a['audit']['checkpoint_sha256'], actual_nodes=a['native_state']['completed_nodes'])


def unique_bytes(root):
    seen = set()
    total = 0
    for path in root.rglob('*'):
        if path.is_file():
            stat = path.stat()
            identity = stat.st_dev, stat.st_ino
            if identity not in seen:
                total += stat.st_size
                seen.add(identity)
    return total


def archive_quote(byte_count):
    # Both the existing canonical data and new endpoints need a future archive
    # copy, even though their existing originals already reduced current free.
    return 2*120.41695427894592*byte_count/20_517_119_304


def seal():
    pr = read(ROOT / 'planning/owning-pr.json')
    destination = Path.home() / 'Local/Research-Cloud' / f"PR-{pr['number']}-HU100-independent-stages"
    destination.mkdir(exist_ok=True)
    snap = OUT / 'archive-monitor-snapshot.jsonl'
    with (OUT / 'continuous-resources.jsonl').open('rb') as src, snap.open('xb') as dst:
        for line in src:
            if not line.endswith(b'\n'):
                break
            dst.write(line)
    exclusions = {'continuous-resources.jsonl', 'archive-receipt.json'}
    paths = [p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in exclusions
        and 'operations/archive' not in str(p.relative_to(OUT))]
    members, aliases, inodes = {}, {}, {}
    for p in paths:
        stat = p.stat()
        inode = (stat.st_dev, stat.st_ino)
        name = 'research/' + str(p.relative_to(OUT))
        pin = {'bytes': stat.st_size, 'sha256': file_hash(p)}
        if inode in inodes:
            aliases[name] = {'canonical_member': inodes[inode], **pin}
        else:
            inodes[inode] = name
            members[name] = {'local_path': str(p), **pin}
    manifest = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'members': members, 'hardlink_aliases': aliases,
        'restore_aliases': 'After extracting canonical members, create aliases with os.link(canonical, alias); verify each alias size/SHA256. Never overwrite.',
        'mutable_lifecycle': 'Archive guard/resources/native/cloud/final-review receipts remain compact Git metadata.'}
    path = destination / 'hu100-independent-stages-M4-20261009.zip'
    raw = json.dumps(manifest, indent=2, sort_keys=True).encode() + b'\n'
    started = time()
    with ZipFile(path, 'x', compression=ZIP_STORED, allowZip64=True) as z:
        z.writestr('ARCHIVE-MANIFEST.json', raw)
        for name, pin in members.items():
            z.write(pin['local_path'], name)
    with ZipFile(path) as z:
        if z.read('ARCHIVE-MANIFEST.json') != raw:
            raise ValueError('Archive manifest differs')
        for name, pin in members.items():
            h = hashlib.sha256(); count = 0
            with z.open(name) as f:
                for block in iter(lambda: f.read(1024**2), b''):
                    h.update(block); count += len(block)
            if count != pin['bytes'] or h.hexdigest() != pin['sha256']:
                raise ValueError('Archive member differs: ' + name)
    put(OUT / 'archive-receipt.json', {'path': str(path), 'bytes': path.stat().st_size,
        'sha256': file_hash(path), 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'verified_members': len(members), 'verified_aliases': len(aliases), 'seconds': time()-started,
        'remote_bytes_verified': False, 'native_upload_status': 'pending separate check',
        'cloud_acceptance': 'pending independent metadata check'})


def campaign(continue_preparation=False):
    revision = previous.source()
    review = read(ROOT / 'planning/independent-source-review.json')
    if review['source'] != revision or review['status'] != 'passed':
        raise ValueError('Independent exact-source review required')
    inventory = subprocess.check_output(['ps', '-axo', 'pid,ppid,command'], text=True)
    own = {os.getpid(), os.getppid()}
    competing = [line for line in inventory.splitlines()[1:] if len(line.split(None, 2)) == 3
        and int(line.split(None, 2)[0]) not in own and any(k in line.split(None, 2)[2] for k in
        ('hu20-trainer train', 'scripts.evaluate_native_hu100', 'scripts.run_native_hu100', 'scripts.run_hu100', 'cargo build'))]
    if competing:
        raise ValueError('Competing research: ' + repr(competing))
    c = Campaign.resume_preparation(ROOT, OUT, revision) if continue_preparation else Campaign(ROOT, OUT, revision)
    suffix = '-readmission' if continue_preparation else ''
    put(OUT / ('process-admission' + suffix + '.json'), {'inventory': inventory, 'competing': competing,
        'exclusive_lock': str(Path.home() / 'Local/.hu100-m4-research.lock')})
    put(OUT / ('source-review' + suffix + '.json'), review)
    status = 'incomplete'
    try:
        c.run('prepare-readmission' if continue_preparation else 'prepare', [sys.executable, '-m', MODULE, 'prepare'], quote=60)
        c.run('freshness', [sys.executable, '-m', MODULE, 'freshness'], quote=60)
        # Exact resume compatibility is checked on #211's actual partial, not
        # merely on a synthetic fixture with a different seed or old loader.
        fixture = OUT / 'recovery-fixture'
        for label, resume in (('resumed', OUT / 'inputs/partial/checkpoint.gz'), ('direct', None)):
            folder = fixture / label
            c.run('compatibility-' + label, command(folder, SEEDS[1], 2_000_000, resume),
                stop_file=folder / 'stop.json', quote=120)
        a, b = [file_hash(fixture / x / 'checkpoint.gz') for x in ('resumed', 'direct')]
        if a != b:
            raise ValueError('Actual partial exact resume compatibility failed')
        put(fixture / 'equivalence.json', {'status': 'verified', 'seed': SEEDS[1], 'requested_nodes': 2000000,
            'direct_sha256': b, 'resumed_sha256': a, 'actual_partial_used': True})
        c.run('calibration', [sys.executable, '-m', MODULE, 'calibration', '--source', revision], quote=1800)
        tq = training_quote()
        models = {str(SEEDS[0]): {label: read(OUT / 'inputs' / label / 'spec.json') for label in ('early', 'terminal')}}
        # Admit each useful training lineage with audited recovery and closeout,
        # independently of the later evaluation quote.
        for seed in SEEDS[1:]:
            free = shutil.disk_usage(OUT).free
            model_bytes = (665_193_414*1.1 + 3_996_276_196*ENTRY_CAP/41_010_014)
            retained = unique_bytes(OUT) + GIB
            archive_seconds = archive_quote(retained + model_bytes)
            required = retained + 2*model_bytes + 3*GIB + int(15.5*GIB)
            admission = {'seed': seed, **tq, 'remaining_seconds': c.remaining(), 'disk_free_bytes': free,
                'required_free_bytes': required, 'retained_archive_bytes': retained,
                'projected_archive_seconds': archive_seconds, 'closeout_reserve_seconds': CLOSEOUT,
                'independent_of_evaluation': True,
                'pass': c.remaining() >= tq['per_seed_seconds'] + CLOSEOUT and free >= required and archive_seconds <= CLOSEOUT}
            put(OUT / f'training-admission-{seed}.json', admission)
            if not admission['pass']:
                raise CapacityStop(f'Training stage {seed} refused own time/disk quote')
            endpoints = {}
            parent = OUT / 'inputs/partial/checkpoint.gz' if seed == SEEDS[1] else None
            for label, nodes in (('early', EARLY), ('terminal', TERMINAL)):
                folder = OUT / 'training' / str(seed) / label
                fraction = (tq['early_entries_forecast'] if label == 'early' else ENTRY_CAP) / (tq['early_entries_forecast'] + ENTRY_CAP)
                tools_quote = tq['per_seed_export_audit_seconds'] * fraction
                train_quote = tq['per_seed_training_seconds'] * (EARLY/TERMINAL if label == 'early' else 1-EARLY/TERMINAL) + tq['per_seed_save_seconds']*fraction
                c.run(f'{seed}-{label}-train', command(folder, seed, nodes, parent),
                    reserve=CLOSEOUT+tools_quote, quote=train_quote, stop_file=folder/'stop.json', accepted=(0,3))
                rows = previous.telemetry(folder); row = rows[-1]
                if file_hash(folder/'checkpoint.gz') != row['checkpoint_sha256']:
                    raise ValueError('Save hash differs from telemetry')
                tools(c, folder, f'{seed}-{label}', row['completed_nodes'], tools_quote)
                endpoints[label] = spec(folder)
                put(folder / 'endpoint.json', {'seed': seed, 'target': nodes, 'actual_nodes': row['completed_nodes'],
                    'complete': row['completed_nodes'] >= nodes, 'telemetry': row, 'model': endpoints[label]})
                if row['completed_nodes'] < nodes:
                    raise CapacityStop(f'Seed {seed} incomplete at {row["completed_nodes"]} nodes')
                parent = folder/'checkpoint.gz'
            models[str(seed)] = endpoints
            put(OUT / f'models-after-{seed}.json', models)
        put(OUT / 'models.json', models)
        measured = read(OUT / 'calibration-costs.json')['measurements']
        choices = [evaluation_quote(blocks, measured, models) for blocks in (8192,4096)]
        # Physical snapshot aliases take no additional model bytes. Raw traces
        # use twice the larger calibration bytes/block, without savings.
        raw_per_block = max(sum(p.stat().st_size for root in
            (OUT/f'calibration-{n}', OUT/f'calibration-{n}-reproduction') for p in root.rglob('*')
            if p.is_file() and 'models' not in p.parts)/n for n in CALIBRATION_ROOTS)
        retained = unique_bytes(OUT) + GIB
        for choice in choices:
            choice.update(remaining_seconds=c.remaining(), disk_free_bytes=shutil.disk_usage(OUT).free)
            # Calibration has three arms; qualification has nine.
            choice['required_free_bytes'] = int(15.5*GIB)+3*GIB+retained+2*raw_per_block*choice['blocks']*3*2
            choice['projected_archive_seconds'] = archive_quote(retained + 2*raw_per_block*choice['blocks']*3)
            choice['time_pass'] = choice['seconds']+CLOSEOUT <= c.remaining() and choice['projected_archive_seconds'] <= CLOSEOUT
            choice['disk_pass'] = choice['disk_free_bytes'] >= choice['required_free_bytes']
        put(OUT / 'evaluation-admission.json', {'choices': choices, 'outcomes_inspected': False,
            'raw_bytes_per_block_three_arms': raw_per_block, 'training_independently_completed': True})
        selected = next((q for q in choices if q['time_pass'] and q['disk_pass']), None)
        if selected is None:
            raise CapacityStop('Training completed; full qualification evaluation deferred by independent time/disk admission')
        blocks = selected['blocks']
        put(OUT / 'frozen-comparisons.json', {'source': revision, 'blocks': blocks, 'seeds': SEEDS,
            'early_target': EARLY, 'terminal_target': TERMINAL, 'growth_alpha': .05/6, 'translation_alpha': .05/3,
            'practical_lower_bb100': 10, 'pilot_roots': CALIBRATION_ROOTS, 'final_root': FINAL_ROOT,
            'quote_sha256': file_hash(OUT/'evaluation-admission.json'), 'outcomes_inspected': False,
            'families': 'separate FWER .05, not union .05'})
        put(OUT/'frozen-schedule.json', frozen_schedule({'models':[models[str(SEEDS[0])]['early']]}, blocks, FINAL_ROOT))
        put(OUT/'frozen-final.json', {'source': revision, 'blocks_per_opponent': blocks, 'final_root': FINAL_ROOT,
            'schedule_sha256':file_hash(OUT/'frozen-schedule.json'), 'comparisons_sha256':file_hash(OUT/'frozen-comparisons.json'),
            'models_sha256':file_hash(OUT/'models.json')})
        c.report_quote = selected['report_seconds']
        for seed in SEEDS:
            for endpoint in ('early','terminal'):
                c.run(f'final-{seed}-{endpoint}', [sys.executable,'-m',MODULE,'evaluate','--source',revision,
                    '--seed',seed,'--endpoint',endpoint,'--blocks',blocks],
                    quote=selected['play_replay_reproduction_seconds']*(1 if endpoint=='early' else 2)/9)
        c.run('strict-report', [sys.executable,'-m',MODULE,'report'])
        status = 'science-complete'
    except CapacityStop as exc:
        put(OUT/'capacity-stop.json', {'at':time(), 'reason':str(exc), 'status':'incomplete'})
        status = 'capacity-stop'
    except BaseException as exc:
        c.latch(repr(exc))
        raise
    finally:
        if c.failure is None:
            put(OUT/'science-closeout.json', {'source':revision,'status':status,'at':time(),'deadline':c.deadline})
            c.run('archive', [sys.executable,'-m',MODULE,'seal'], reserve=0,quote=CLOSEOUT)
            c.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('campaign','prepare','freshness','calibration','evaluate','report','seal'))
    p.add_argument('--continue-preparation', action='store_true')
    p.add_argument('--source'); p.add_argument('--seed',type=int); p.add_argument('--blocks',type=int)
    p.add_argument('--endpoint',choices=('early','terminal'))
    a=p.parse_args()
    if a.stage=='campaign': campaign(a.continue_preparation)
    elif a.stage=='prepare': prepared()
    elif a.stage=='freshness': freshness()
    elif a.stage=='calibration': calibration(a.source)
    elif a.stage=='evaluate': evaluation(a.source,a.seed,a.endpoint,a.blocks)
    elif a.stage=='report': report(OUT, settings_factory=config)
    else: seal()

if __name__=='__main__': main()
