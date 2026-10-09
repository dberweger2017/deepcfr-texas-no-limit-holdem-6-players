"""One-use M4 campaign: cost pilot, fixed seeds, paired play and local closeout."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import sleep, time
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

from scripts.hu100_qualification_guard import Campaign, CapacityStop, GIB, put, read
from scripts.native_hu100_model_metadata import audited_average_spec
from scripts.report_native_hu100_learning_curves import frozen_schedule
from scripts.run_hu100_action_translation import PRIOR_ROOTS
from src.blueprint.average import TranslationOptions
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'scripts.run_hu100_seed_qualification'
OUT = ROOT / 'results/hu100-seed-qualification'
OLD_ROOT = Path.home() / 'Local/hu100-1b-growth-20261008'
OLD_ZIP = Path.home() / 'Local/Research-Cloud/PR-207-hu100-1b/hu100-1b-campaign-M4-retry-20261009.zip'
ARCHIVE_SHA = 'ba3e82d8fa79be32d445c54eb240069c4a717cb75af9713f3eaf7ac86364fddf'
MANIFEST_SHA = '609d4899a363b384aa58b0daaef2ff85f2bcae7b135c2a4e3604813fc4182d58'
BINARY_SHA = '7650ad60bbf2437622ea3c39d37c7d56686d00bac11680e44a6e47dab509a262'
BINARY = OUT / 'bin/hu20-trainer'
SEEDS = (2026100601, 2026100901, 2026100902)
EARLY, TERMINAL = 39_438_279, 1_000_000_000
PILOT_ROOT, FINAL_ROOT = 2026100910011, 2026100910012
ENTRY_CAP = 57_658_644
CLOSEOUT = 1800


def source():
    if Path.cwd() != ROOT:
        raise ValueError('Campaign checkout required')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Committed clean source required')
    for key, value in (('machdep.cpu.brand_string', 'Apple M4'), ('hw.memsize', str(16*GIB)), ('hw.ncpu', '10')):
        if subprocess.check_output(['sysctl', '-n', key], text=True).strip() != value:
            raise ValueError('Only the 10-core/16-GiB M4 is authorized')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def config(spec, translated=False):
    result = read(ROOT / 'configs/arena/hu100-playing-baseline-v1.json')
    result.update(model=spec, pilot_root=PILOT_ROOT, final_root=FINAL_ROOT,
        proposed_final_blocks_per_opponent=8192,
        action_translation=asdict(TranslationOptions()) if translated else None)
    return result


def fixture_command(folder, seed, nodes, *, resume=None):
    folder.mkdir(parents=True)
    command = [BINARY, 'train', '--stack-bb', '100', '--seed', seed,
        '--roots-per-seat', 1, '--average-rule', 'opponent-sampled', '--nodes', nodes,
        '--max-entries', ENTRY_CAP, '--out', folder / 'checkpoint.gz',
        '--telemetry', folder / 'telemetry.jsonl', '--stop-file', folder / 'stop.json']
    if resume:
        command += ['--resume', resume, '--resume-sha256', file_hash(resume)]
    return command


def telemetry(folder):
    return [json.loads(s) for s in (folder / 'telemetry.jsonl').read_text().splitlines()]


def tools(campaign, folder, label, nodes):
    campaign.run(label + '-export', [BINARY, 'export', folder / 'checkpoint.gz',
        '--current', folder / 'current.gz', '--average', folder / 'average.gz', '--zero-mass', 'uniform'])
    campaign.run(label + '-audit', [sys.executable, '-m', 'scripts.audit_native_hu_checkpoint',
        '--checkpoint', folder / 'checkpoint.gz', '--current', folder / 'current.gz',
        '--average', folder / 'average.gz', '--stack-bb', 100, '--target-nodes', nodes,
        '--out', folder / 'audit.json'])


def spec(folder):
    a = read(folder / 'audit.json')
    return audited_average_spec(folder / 'average.gz', a,
        checkpoint_sha256=a['audit']['checkpoint_sha256'], actual_nodes=a['native_state']['completed_nodes'])


def retrieve():
    if OLD_ZIP.stat().st_size != 20_517_119_304 or file_hash(OLD_ZIP) != ARCHIVE_SHA:
        raise ValueError('Indexed #207 archive changed')
    with ZipFile(OLD_ZIP) as z:
        raw = z.read('ARCHIVE-MANIFEST.json')
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
            raise ValueError('Indexed manifest changed')
        manifest = json.loads(raw)
        members = {m['path']: m for m in manifest['members']}
        restored = []
        for label, prefix in (('early', 'research/gate'), ('terminal', 'research/training/1000000000')):
            folder = OUT / 'inputs' / label
            folder.mkdir(parents=True)
            for name in ('average.gz', 'audit.json'):
                member = prefix + '/' + name
                expected = members[member]
                path = folder / name
                with z.open(member) as src, path.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 1024**2)
                if path.stat().st_size != expected['bytes'] or file_hash(path) != expected['sha256']:
                    raise ValueError('Restored member differs: ' + member)
                restored.append({'path': str(path), 'member': member, **expected})
            put(folder / 'spec.json', spec(folder))
    BINARY.parent.mkdir()
    shutil.copy2(OLD_ROOT / 'native/hu20-trainer/target/release/hu20-trainer', BINARY)
    if file_hash(BINARY) != BINARY_SHA:
        raise ValueError('Reviewed native binary changed')
    # The inherited binary is valid only while all of its native sources match
    # the frozen scientific source that produced #207's exact recovery gates.
    frozen = 'bd0e7a417064f736091dc2b667954b50becb4b69'
    for tree in ('native/hu20-trainer', 'native/hu20-buckets'):
        old = subprocess.check_output(['git', '-C', str(OLD_ROOT), 'rev-parse', frozen + ':' + tree], text=True).strip()
        new = subprocess.check_output(['git', 'rev-parse', 'HEAD:' + tree], text=True).strip()
        if old != new:
            raise ValueError('Native source compatibility changed: ' + tree)
    with (OUT / 'source.tar').open('xb') as f:
        subprocess.run(['git', 'archive', 'HEAD'], stdout=f, check=True)
    put(OUT / 'environment.json', {'python': sys.version, 'executable': sys.executable,
        'packages': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
        'native_binary_sha256': BINARY_SHA, 'binary_origin_source': frozen})
    put(OUT / 'retrieval.json', {'archive': str(OLD_ZIP), 'archive_sha256': ARCHIVE_SHA,
        'manifest_sha256': MANIFEST_SHA, 'members': restored,
        'owning_pr_207_status': 'MERGED, live check before retrieval',
        'cloud_acceptance': 'owner handoff; local verified archive used',
        'originals_unchanged': True})


def freshness(settings):
    roots = dict(PRIOR_ROOTS + [(2026100810011, 16), (2026100810012, 2048),
        (PILOT_ROOT, 16), (FINAL_ROOT, 8192)])
    previous = {}
    for root, blocks in roots.items():
        doc = frozen_schedule({'models': [settings['model']]}, blocks, root)
        seeds = {b['deal_seeds'][0] for p in doc['panels'].values() for b in p['blocks']}
        if any(seeds & old for old in previous.values()):
            raise ValueError('Physical deal collision')
        previous[root] = seeds
    put(OUT / 'freshness.json', {'status': 'verified', 'roots': roots,
        'all_pairwise_disjoint': True, 'pilot_outcomes_inspected': False})


def panel(campaign, prefix, label, model, blocks, root, translated=False, first=None):
    folder = OUT / prefix
    cfg = folder / (label + '-config.json')
    put(cfg, config(model, translated))
    args = ['--config', cfg, '--source', campaign.source, '--blocks', blocks, '--root', root]
    campaign.run(prefix + '-' + label + '-play', [sys.executable, '-m', 'scripts.evaluate_native_hu100_baseline',
        *args, '--out', folder / label, *(['--reference-run', folder / first] if first else [])])
    campaign.run(prefix + '-' + label + '-audit', [sys.executable, '-m', 'scripts.audit_native_hu100_baseline',
        '--run', folder / label, '--out', folder / (label + '-audit.json')])
    campaign.run(prefix + '-' + label + '-reproduce', [sys.executable, '-m', 'scripts.evaluate_native_hu100_baseline',
        *args, '--out', OUT / (prefix + '-reproduction') / label, '--reproduce', folder / label,
        *(['--reference-run', OUT / (prefix + '-reproduction') / first] if first else [])])


def quote(campaign):
    # No policy payoffs or pilot chip rows are used for cost admission.
    pilot = OUT / 'training' / str(SEEDS[1]) / 'pilot'
    r = telemetry(pilot)[-1]
    entries = r['diagnostics']['entries']
    train_rate = max(.001, r['elapsed_seconds_including_writes'] - r['write_seconds']) / r['completed_nodes']
    old_training = read(ROOT / 'docs/reports/native-hu100-growth-1b-artifacts/training-result.json')
    old_ops = read(ROOT / 'docs/reports/native-hu100-growth-1b-artifacts/resources.json')['operations']
    # Entry-scaled timings use the slower fresh measurement or #207's measured
    # large-table cost. This prevents a tiny cache-resident pilot understating it.
    train_seconds = max(6 * train_rate * 2e9, 2 * old_training['main_operation']['seconds'] * 2)
    old_saves = old_training['saves']
    save_per_entry = max(r['write_seconds'] / entries,
        max(x['write_seconds'] / x['diagnostics']['entries'] for x in old_saves))
    tool_per_entry = (read(OUT / 'operations/pilot-seed-export/receipt.json')['seconds'] +
        read(OUT / 'operations/pilot-seed-audit/receipt.json')['seconds']) / entries
    for item in old_saves:
        n = item['requested_nodes']
        measured = sum(x['seconds'] for x in old_ops if x['name'] in (f'save-{n}-export', f'save-{n}-audit'))
        tool_per_entry = max(tool_per_entry, measured / item['diagnostics']['entries'])
    endpoint_entries = 2 * (math.ceil(7_643_261 * 1.1) + ENTRY_CAP)
    save_seconds = 2 * save_per_entry * endpoint_entries
    tool_seconds = 2 * tool_per_entry * endpoint_entries
    load_costs, scalable = {}, 0
    for label in ('old-early', 'old-terminal', 'old-terminal-on'):
        records = [read(OUT / p / label / 'complete.json') for p in ('pilot', 'pilot-reproduction')]
        load_costs[label] = sum(x['model_load_seconds'] for x in records)
        scalable += sum(max(0, x['wall_seconds'] - x['model_load_seconds']) for x in records)
        scalable += read(OUT / 'pilot' / (label + '-audit.json'))['seconds']
    model_scale = ENTRY_CAP / 41_010_014
    loads = load_costs['old-early'] * 3.2 + (load_costs['old-terminal'] + load_costs['old-terminal-on']) * (1 + 2 * model_scale)
    planning = read(ROOT / 'docs/reports/hu100-seed-qualification-artifacts/storage-planning-20261009.json')
    choices = []
    elapsed = time() - campaign.started
    for blocks in (8192, 4096):
        play_seconds = 3 * (loads + scalable * blocks / 16 * 3)
        total = elapsed + train_seconds + save_seconds + tool_seconds + play_seconds + CLOSEOUT
        disk_plan = next(x for x in planning['quotes'] if x['blocks'] == blocks)
        # Current free already excludes retained pilot inputs/snapshots. Credit
        # only these new campaign bytes, never an expected owner cleanup.
        retained = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        required_remaining = disk_plan['rounded_planning_target_free_bytes'] - retained
        free = shutil.disk_usage(OUT).free
        choices.append({'blocks': blocks, 'seconds_from_start': total,
            'remaining_cost_seconds': total - elapsed, 'disk_free_bytes': free,
            'remaining_required_free_bytes': required_remaining,
            'timing_pass': total <= 21_600, 'disk_pass': free >= required_remaining,
            'training_seconds': train_seconds, 'saves_seconds': save_seconds,
            'export_audit_seconds': tool_seconds, 'play_replay_reproduction_seconds': play_seconds,
            'local_closeout_reserve_seconds': CLOSEOUT})
    put(OUT / 'measured-quote.json', {'source': campaign.source, 'at': time(),
        'choices': choices, 'pilot_outcomes_inspected': False,
        'fixed_load_costs_seconds': load_costs, 'pilot_scalable_seconds': scalable,
        'training_quote_factor': 'max(6x fresh nonsave pilot, 2x #207 train/save per seed)',
        'save_tools_factor': '2x maximum measured seconds/entry, at frozen entry ceiling',
        'evaluation_factor': '3x fixed loads plus block-scaled play/replay/reproduction',
        'disk_basis': 'reviewed 110/130 GiB complete quote, credit retained new bytes only',
        'deadline': campaign.deadline})
    return choices


def pack():
    dest = Path.home() / 'Local/Research-Cloud/PR-211-hu100-seed-qualification/hu100-seed-qualification-M4-20261009.zip'
    dest.parent.mkdir(parents=True, exist_ok=True)
    members = []
    # The active resource stream/logs have an explicit frozen prefix snapshot;
    # later local archive guard/readback receipts are separate lifecycle evidence.
    snap = OUT / 'archive-monitor-snapshot.jsonl'
    with (OUT / 'continuous-resources.jsonl').open('rb') as src, snap.open('xb') as dst:
        remaining = os.fstat(src.fileno()).st_size
        while remaining:
            data = src.read(min(1024**2, remaining))
            if not data:
                raise ValueError('Monitor prefix snapshot truncated')
            dst.write(data)
            remaining -= len(data)
    for path in sorted(OUT.rglob('*')):
        rel = path.relative_to(OUT)
        if not path.is_file() or rel.as_posix() == 'continuous-resources.jsonl' or rel.parts[:2] == ('operations', 'archive'):
            continue
        members.append({'path': 'research/' + rel.as_posix(), 'bytes': path.stat().st_size,
            'sha256': file_hash(path), 'original_path': str(path)})
    manifest = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'members': members, 'mutable_lifecycle': 'Archive resources/readback/upload/evidence review remain separate compact receipts; guard stream archived through explicit prefix snapshot.'}
    raw = json.dumps(manifest, sort_keys=True).encode()
    with ZipFile(dest, 'x', allowZip64=True) as z:
        z.writestr('ARCHIVE-MANIFEST.json', raw, compress_type=ZIP_DEFLATED)
        for m in members:
            path = Path(m['original_path'])
            z.write(path, m['path'], compress_type=ZIP_STORED if path.suffix == '.gz' else ZIP_DEFLATED)
    with ZipFile(dest) as z:
        for m in members:
            h = hashlib.sha256()
            count = 0
            with z.open(m['path']) as f:
                for block in iter(lambda: f.read(1024**2), b''):
                    count += len(block)
                    h.update(block)
            if count != m['bytes'] or h.hexdigest() != m['sha256']:
                raise ValueError('Archive member readback failed: ' + m['path'])
        if z.read('ARCHIVE-MANIFEST.json') != raw:
            raise ValueError('Embedded manifest readback failed')
    put(OUT / 'archive-receipt.json', {'path': str(dest), 'bytes': dest.stat().st_size,
        'sha256': file_hash(dest), 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'members_verified': len(members), 'all_member_sizes_sha256_verified': True,
        'cloud_acceptance': 'pending; local verification only', 'originals_retained': True})


def main_campaign():
    revision = source()
    review = read(ROOT / 'planning/hu100-seed-source-review.json')
    if review['source'] != revision or review['status'] != 'passed':
        raise ValueError('Independent exact-source review required')
    inventory = subprocess.check_output(['ps', '-axo', 'pid,ppid,command'], text=True)
    own = {os.getpid(), os.getppid()}
    competing = []
    for line in inventory.splitlines()[1:]:
        fields = line.split(None, 2)
        if len(fields) == 3 and int(fields[0]) not in own and any(k in fields[2] for k in
                ('hu20-trainer train', 'scripts.evaluate_native_hu100', 'scripts.run_native_hu100', 'scripts.run_hu100', 'cargo build')):
            competing.append(line)
    if competing:
        raise ValueError('Competing research: ' + repr(competing))
    campaign = Campaign(ROOT, OUT, revision)
    put(OUT / 'process-admission.json', {'at': time(), 'inventory': inventory, 'competing': competing,
        'exclusive_lock': str(Path.home() / 'Local/.hu100-m4-research.lock')})
    put(OUT / 'source-review.json', review)
    status = 'incomplete'
    try:
        campaign.run('retrieve', [sys.executable, '-m', MODULE, 'retrieve'])
        early, terminal = [read(OUT / 'inputs' / n / 'spec.json') for n in ('early', 'terminal')]
        campaign.run('freshness', [sys.executable, '-m', MODULE, 'freshness'])
        # Recovery qualification uses a separate fixture seed, not a restart or
        # additional training of any of the three campaign lineages.
        fixture = OUT / 'recovery-fixture'
        for label, nodes, resume in (('split', 10_000, None), ('resumed', 20_000, fixture / 'split/checkpoint.gz'), ('direct', 20_000, None)):
            command = fixture_command(fixture / label, 2026100999, nodes, resume=resume)
            campaign.run('recovery-' + label, command, stop_file=fixture / label / 'stop.json')
        if file_hash(fixture / 'resumed/checkpoint.gz') != file_hash(fixture / 'direct/checkpoint.gz'):
            raise ValueError('Bounded recovery equivalence failed')
        put(fixture / 'equivalence.json', {'status': 'verified', 'seed': 2026100999,
            'direct_sha256': file_hash(fixture / 'direct/checkpoint.gz'), 'resumed_sha256': file_hash(fixture / 'resumed/checkpoint.gz')})
        pilot = OUT / 'training' / str(SEEDS[1]) / 'pilot'
        campaign.run('pilot-seed-train', fixture_command(pilot, SEEDS[1], 1_000_000), stop_file=pilot / 'stop.json', accepted=(0, 3))
        tools(campaign, pilot, 'pilot-seed', telemetry(pilot)[-1]['completed_nodes'])
        if telemetry(pilot)[-1]['completed_nodes'] < 1_000_000:
            raise CapacityStop('First seed pilot capacity stop')
        for label, model, translated in (('old-early', early, False), ('old-terminal', terminal, False), ('old-terminal-on', terminal, True)):
            panel(campaign, 'pilot', label, model, 16, PILOT_ROOT, translated, first='old-early' if label != 'old-early' else None)
        choices = quote(campaign)
        put(OUT / 'awaiting-admission.json', {'at': time(), 'quote_sha256': file_hash(OUT / 'measured-quote.json'),
            'reason': 'M1 must publish measured quote and frozen count before main training', 'deadline': campaign.deadline})
        marker = OUT / 'launch-admission.json'
        while not marker.exists():
            if campaign.remaining() < min(x['remaining_cost_seconds'] for x in choices):
                raise CapacityStop('No remaining budget for admitted main campaign')
            campaign.check()
            sleep(.2)
        admission = read(marker)
        if admission['source'] != revision or admission['quote_sha256'] != file_hash(OUT / 'measured-quote.json'):
            raise ValueError('Launch admission changed source or quote')
        blocks = admission['blocks']
        selected = next(x for x in choices if x['blocks'] == blocks)
        free = shutil.disk_usage(OUT).free
        if not selected['timing_pass'] or free < selected['remaining_required_free_bytes']:
            raise CapacityStop('Fresh final time/disk admission refused')
        if campaign.remaining() < selected['remaining_cost_seconds']:
            raise CapacityStop('Waiting consumed quoted main budget')
        campaign.tool_quote = selected['export_audit_seconds']/4
        campaign.panel_quote = selected['play_replay_reproduction_seconds']/9
        put(OUT / 'frozen-comparisons.json', {'source': revision, 'blocks': blocks,
            'seeds': SEEDS, 'early_target': EARLY, 'terminal_target': TERMINAL,
            'growth_alpha': .05/6, 'translation_alpha': .05/3, 'practical_lower_bb100': 10,
            'pilot_root': PILOT_ROOT, 'final_root': FINAL_ROOT, 'quote_sha256': admission['quote_sha256'],
            'families': 'separate FWER .05, not joint .05', 'outcomes_inspected': False})
        models = {str(SEEDS[0]): {'early': early, 'terminal': terminal}}
        for seed in SEEDS[1:]:
            seed_root = OUT / 'training' / str(seed)
            previous = pilot / 'checkpoint.gz' if seed == SEEDS[1] else None
            endpoints = {}
            for label, nodes in (('early', EARLY), ('terminal', TERMINAL)):
                folder = seed_root / label
                command = fixture_command(folder, seed, nodes, resume=previous)
                campaign.run(f'seed-{seed}-{label}-train', command, reserve=CLOSEOUT + selected['export_audit_seconds'],
                    quote=selected['training_seconds']/4, stop_file=folder / 'stop.json', accepted=(0, 3))
                rows = telemetry(folder)
                row = rows[-1]
                if file_hash(folder / 'checkpoint.gz') != row['checkpoint_sha256']:
                    raise ValueError('Native save telemetry hash differs')
                tools(campaign, folder, f'seed-{seed}-{label}', row['completed_nodes'])
                endpoints[label] = spec(folder)
                put(folder / 'endpoint.json', {'seed': seed, 'requested_nodes': nodes,
                    'actual_nodes': row['completed_nodes'], 'completed': row['completed_nodes'] >= nodes,
                    'entry_ceiling': ENTRY_CAP, 'telemetry': row, 'model': endpoints[label]})
                if row['completed_nodes'] < nodes:
                    raise CapacityStop(f'Seed {seed} incomplete at capacity: {row["completed_nodes"]}')
                previous = folder / 'checkpoint.gz'
            models[str(seed)] = endpoints
        put(OUT / 'models.json', models)
        schedule = frozen_schedule({'models': [early]}, blocks, FINAL_ROOT)
        put(OUT / 'frozen-schedule.json', schedule)
        put(OUT / 'frozen-final.json', {'source': revision, 'blocks_per_opponent': blocks,
            'final_root': FINAL_ROOT, 'schedule_sha256': file_hash(OUT / 'frozen-schedule.json'),
            'comparisons_sha256': file_hash(OUT / 'frozen-comparisons.json'), 'models_sha256': file_hash(OUT / 'models.json')})
        first = str(SEEDS[0]) + '-early'
        for seed in SEEDS:
            for label, translated in (('early', False), ('terminal', False), ('terminal-on', True)):
                name = str(seed) + '-' + label
                panel(campaign, 'final', name, models[str(seed)]['terminal' if translated else label],
                    blocks, FINAL_ROOT, translated, first=first if name != first else None)
        campaign.run('strict-report', [sys.executable, '-m', 'scripts.report_hu100_seed_qualification'])
        status = 'science-complete'
    except CapacityStop as exc:
        put(OUT / 'capacity-stop.json', {'at': time(), 'reason': str(exc), 'source': revision,
            'status': 'incomplete', 'not_completed_1b_seeds': True})
        status = 'capacity-stop'
    except BaseException as exc:
        campaign.latch(repr(exc))
        raise
    finally:
        if campaign.failure is None:
            put(OUT / 'science-closeout.json', {'source': revision, 'status': status, 'at': time(),
                'deadline': campaign.deadline, 'no_retry_or_deeper_training': True})
            campaign.run('archive', [sys.executable, '-m', MODULE, 'pack'], reserve=0, quote=CLOSEOUT)
            campaign.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('campaign', 'retrieve', 'freshness', 'pack'))
    stage = p.parse_args().stage
    if stage == 'campaign':
        main_campaign()
    elif stage == 'retrieve':
        retrieve()
    elif stage == 'freshness':
        freshness(config(read(OUT / 'inputs/early/spec.json')))
    else:
        pack()


if __name__ == '__main__':
    main()
