"""Fixed-window M4 evaluation; outcomes stay sealed until complete reproduction."""
import argparse
import csv
import fcntl
import gc
import gzip
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
import signal
import shutil
from zipfile import ZipFile
import hashlib
from time import time

import psutil
from scipy.stats import t

from scripts import evaluate_hu100_direct as direct
from scripts.archive_hu100_4b_seed_ladder import seal
from scripts.reserve_match_roots import reserve
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.blueprint.average import AveragePolicy, ZERO_MASS_RULES
from src.policies.files import file_hash

REPO = 'dberweger2017/deepcfr-texas-no-limit-holdem-6-players'
BASELINE = '60bd37c979e018f11f136370f468cc41b8e1435550531707e89d45f28709ec7c'
ENTRIES = 85_373_907
KEY_SHA256 = '0318ed5894960151e5e55f8fde538bdfe1eda1d3ec496877eaf735f34ef4bf3c'
# October 14, 2026 UTC; Madrid is UTC+02:00 on this date.
OPEN, LAST_START, SCIENCE_END, END = 1791932400, 1791946800, 1791963000, 1791964800
ROOTS = {'direct_pilot': 20261014091501, 'panel_pilot': 20261014091502,
         'direct': 20261014091503, 'panel': 20261014091504}
LADDER = ((262144, 131072, 16384), (262144, 32768, 16384),
          (262144, 0, 16384), (131072, 0, 16384), (131072, 0, 8192))


def read(path):
    return json.loads(Path(path).read_text())


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def gh(*args):
    return subprocess.check_output(['gh', *map(str, args), '--repo', REPO], text=True)


def put(path, value):
    direct.put(Path(path), value)


def quote(cost, count):
    """Keep fixed loads separate; audit scaling conservatively includes table hashes."""
    if not count:
        return 0.0
    fixed = cost['primary']['load_or_validation_seconds'] + cost['repeat']['load_or_validation_seconds']
    scaled = (cost['primary']['play_seconds'] + cost['repeat']['play_seconds'] + cost['audit_seconds'])
    return fixed + count * scaled / cost['blocks']


def choose_counts(costs, remaining):
    options = []
    for primary, curve, panel in LADDER:
        seconds = 1.25 * (quote(costs['primary'], primary) + quote(costs['curve'], curve)
            + sum(quote(costs['panel'][op], panel) for op in direct.PANEL_OPPONENTS))
        option = dict(primary=primary, curve=curve, panel=panel, projected_seconds=seconds,
                      fits=seconds <= remaining)
        options.append(option)
        if option['fits']:
            return option, options
    raise ValueError('Minimum authorized counts cannot fit before closeout reserve')


def admission(config, out):
    if not OPEN <= time() < LAST_START:
        raise ValueError('Outside the fixed M4 admission window')
    live = json.loads(gh('pr', 'view', 265, '--json', 'state,body,comments,headRefOid,url'))
    statement = config['release_statement']
    # A human reading the current PR selects an affirmative statement. The literal
    # readback prevents substituting an ETA or the earlier promise to release.
    texts = [live['body'], *[c['body'] for c in live['comments']]]
    if not config.get('release_confirmed') or not statement or not any(statement in s for s in texts):
        raise ValueError('Explicit live PR265 agent release statement required')
    found = []
    own = {psutil.Process().pid, *[p.pid for p in psutil.Process().parents()]}
    for process in psutil.process_iter(['pid', 'ppid', 'create_time', 'cmdline'], ad_value=None):
        if process.info['pid'] in own:
            continue
        arguments = process.info['cmdline'] or []
        # An existing venv can serve a different campaign. The interpreter's
        # location alone does not make that process PR265-owned.
        if arguments and '/.venv/' in arguments[0]:
            arguments = arguments[1:]
        command = ' '.join(arguments)
        if 'hu200-k50-frozen-20261013' in command or 'planning/hu200-k50-frozen/' in command:
            found.append(process.info)
    receipt = {'pr265': live, 'affirmative_release_statement': statement,
               'occupancy': found, 'at': time()}
    put(out/'admission.json', receipt)
    if found:
        raise ValueError('PR265 processes still running; never signal them')


def prepare(config, out):
    # Three 85M-key caches, a restored baseline and traces need separate disk
    # headroom above the immediate guard floor; source models are never removed.
    if shutil.disk_usage(out).free < 36*1024**3:
        raise ValueError('Insufficient disk for three caches and protected headroom')
    owners = {str(n): json.loads(gh('pr', 'view', n, '--json', 'state,url,headRefOid'))
              for n in (163, 250, 265)}
    put(out/'input-owners-current.json', owners)
    models = {}
    for label in ('baseline', 'final', 'five'):
        item = config['models'][label]
        audit = read(item['audit_path'])
        checked = audit.get('audit', audit)
        path = Path(item['path']).expanduser().resolve()
        pin = audit['files']['average']
        if not path.exists():
            # Only a missing baseline is restored here. The owner's specified
            # open-PR candidates are read-only dependencies, never moved.
            if label != 'baseline' or owners['250']['state'] != 'MERGED':
                raise ValueError('Required audited candidate is absent')
            retrieval = item['retrieval']
            archive_path = Path(retrieval['archive_path']).expanduser()
            if file_hash(archive_path) != retrieval['archive_sha256']:
                raise ValueError('Baseline archive SHA256 differs')
            path.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive_path) as archive:
                raw = archive.read('ARCHIVE-MANIFEST.json')
                if hashlib.sha256(raw).hexdigest() != retrieval['manifest_sha256']:
                    raise ValueError('Baseline manifest SHA256 differs')
                member = retrieval['member_path']
                member_pin = json.loads(raw)['members'][member]
                if member_pin['bytes'] != pin['bytes'] or member_pin['sha256'] != item['sha256']:
                    raise ValueError('Baseline archive member pin differs')
                with archive.open(member) as source, path.open('xb') as target:
                    shutil.copyfileobj(source, target, 1024**2)
        if (checked['status'] != 'verified' or checked['all_nodes_verified'] != ENTRIES
                or checked['counts']['entries'] != ENTRIES or path.stat().st_size != pin['bytes']
                or file_hash(path) != pin['sha256'] or pin['sha256'] != item['sha256']
                or checked['average_sha256'] != item['sha256']):
            raise ValueError('Pinned every-row audited average differs: ' + label)
        with gzip.open(path, 'rt') as stream:
            metadata = json.loads(stream.readline())
        header = metadata['checkpoint_header']
        nodes = header['native_state']['completed_nodes']
        if (header['abstraction'] != HU200_EQUITY_SCHEMA or header['average_rule'] != 'opponent-sampled'
                or nodes != audit['actual_nodes'] or metadata['source_checkpoint_sha256'] != checked['checkpoint_sha256']
                or metadata['zero_mass_rule'] != ZERO_MASS_RULES['uniform']):
            raise ValueError('HU200 lineage/header/parent differs: ' + label)
        if label == 'baseline' and (nodes != 752593175 or item['sha256'] != BASELINE):
            raise ValueError('Wrong baseline')
        if label == 'five' and not 5_000_000_000 <= nodes < 5_001_000_000:
            raise ValueError('Wrong audited 5B milestone')
        if label != 'baseline':
            keys = audit['keys']; frozen = header['frozen_table']
            if (keys['entries'] != ENTRIES or keys['sha256'] != KEY_SHA256
                    or frozen['entries'] != ENTRIES or frozen['since_iteration'] != 493496):
                raise ValueError('Frozen key set or parent iteration differs')
        table_dir = Path(item['card_buckets']).expanduser().resolve()
        tables = {street: {'path': str(table_dir/(street+'-k50.bin')),
                          'sha256': file_hash(table_dir/(street+'-k50.bin'))}
                  for street in EQUITY_K50_SHA256}
        if any(tables[s]['sha256'] != sha for s, sha in EQUITY_K50_SHA256.items()):
            raise ValueError('Pinned K50 tables differ')
        spec = dict(name='frozen-'+label, path=str(path), sha256=item['sha256'], bytes=pin['bytes'],
                    format=metadata['format'], abstraction=header['abstraction'], entries=ENTRIES,
                    iteration=header['iteration'], source_checkpoint_sha256=checked['checkpoint_sha256'],
                    card_buckets=str(table_dir))
        models[label] = dict(spec=spec, actual_nodes=nodes, header=header, tables=tables,
                             audit=audit, audit_sha256=file_hash(item['audit_path']),
                             retrieval=item['retrieval'])
    headers = [m['header'] for m in models.values()]
    for header in headers:
        if (header['config']['seed'] != 2026101302 or header['table']['stacks'] != [20000, 20000]
                or header['config']['roots_per_seat'] != 1):
            raise ValueError('Fixed recipe differs')
        if header.get('training_options') is not None:
            raise ValueError('Nonlinear/default training options are outside this lineage')
    for field in ('abstraction', 'average_rule', 'table', 'identity'):
        if any(h[field] != headers[0][field] for h in headers[1:]):
            raise ValueError('Recipe identity differs: ' + field)
    if models['final']['actual_nodes'] < models['five']['actual_nodes']:
        raise ValueError('Final predates 5B')
    put(out/'inputs.json', {'source': revision(), 'models': models, 'configuration': config})
    return models


def pair_worker(out, specs, blocks, root, opponent, pilot, cache):
    registry, primary = direct.execute(specs, blocks, root, 'frozen-lineage', out/'play', revision(),
        stack_bb=200, opponent=opponent, cache=cache)
    checked = direct.audit(out/'play', out/'replay.json', report=False, cost_only=pilot)
    _, repeat = direct.execute(specs, blocks, root, 'frozen-lineage', out/'reproduction', revision(),
        registry, out/'play', stack_bb=200, opponent=opponent, cache=cache)
    if not repeat['reproduced_all_hands_and_decisions']:
        raise ValueError('Reproduction incomplete')
    put(out/'costs.json', {'primary': primary, 'repeat': repeat, 'audit_seconds': checked['seconds'],
        'blocks': blocks, 'pilot': pilot,
        **({'paired_sd_bb_per_100': checked['paired_sd_bb_per_100']} if pilot else {})})
    del registry
    gc.collect()


def run_pair(base, pair, blocks, root, label, opponent=None, pilot=False):
    specs = base/(label+'-specs.json')
    put(specs, pair)
    command = [sys.executable, '-m', 'scripts.run_hu200_frozen_evaluation', 'pair',
               '--out', str(base/label), '--specs', str(specs), '--blocks', str(blocks),
               '--root', str(root), '--cache', str(base/'policy-cache')]
    if pilot:
        command.append('--pilot')
    if opponent:
        command.extend(['--opponent', opponent])
    with (base/(label+'.log')).open('x') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=max(1, SCIENCE_END-time()))
    return read(base/label/'costs.json')


def freeze(base, config, costs):
    selected, ladder = choose_counts(costs, SCIENCE_END-time())
    inputs = read(base/'inputs.json')
    plan = dict(source=revision(), inputs_sha256=file_hash(base/'inputs.json'), models=inputs['models'],
        roots=ROOTS, counts=selected, projection_ladder=ladder, pilot=costs, frozen_at=time(),
        science_end_utc='2026-10-14T07:30:00Z', deadline_utc='2026-10-14T08:00:00Z',
        cache=str(base/'policy-cache'), translation=None, missing_key_fallback='uniform',
        zero_mass='uniform', opponents=direct.PANEL_OPPONENTS, projection_multiplier=1.25,
        rule='Recommend Slumbot only if primary nominal paired Student-t95 lower>0 and no panel upper<-20 BB/100',
        learning_curve='Paired final-minus-5B returns against 752M on overlapping shared blocks; not a final-vs-5B duel',
        guards={'sample_seconds':5, 'sustained_seconds':60, 'rss_gib':13, 'critical_pressure':4,
                'minimum_free_percent':8, 'swap_growth_bytes':4_000_000_000, 'disk_floor_gib':16})
    put(base/'canonical-plan.json', plan)
    body = gh('pr', 'view', config['pr'], '--json', 'body')
    original = json.loads(body)['body']
    posted = original + '\n\n## Canonical plan before final hands\n\nSHA256 `' + file_hash(base/'canonical-plan.json') + '`.\n\n```json\n' + json.dumps(plan, sort_keys=True, indent=2) + '\n```\n'
    body_file = base/'canonical-pr-body.md'
    body_file.write_text(posted)
    gh('pr', 'edit', config['pr'], '--body-file', body_file)
    if json.loads(gh('pr', 'view', config['pr'], '--json', 'body'))['body'] != posted:
        raise ValueError('Canonical PR description readback differs')
    put(base/'plan-posted.json', {'sha256': file_hash(base/'canonical-plan.json'),
         'body_sha256': file_hash(body_file), 'readback_verified': True, 'at': time()})
    return plan


def series(folder, blocks, panel=False):
    rows = {}
    with gzip.open(folder/'play/hands.jsonl.gz', 'rt') as stream:
        for row in map(json.loads, stream):
            key = row['block'], row['rotation'], row['arm']
            if key in rows or row['status'] != 'completed':
                raise ValueError('Duplicate or failed final row')
            rows[key] = row['candidate_chips']
    arms = ('candidate', 'baseline') if panel else ('candidate',)
    if set(rows) != {(b, r, a) for b in range(blocks) for r in (0, 1) for a in arms}:
        raise ValueError('Final coordinates incomplete')
    values = {a: [(rows[b, 0, a]+rows[b, 1, a])/2 for b in range(blocks)] for a in arms}
    return values


def estimate(values):
    center, sd = mean(values), stdev(values)
    margin = float(t.ppf(.975, len(values)-1))*sd/sqrt(len(values))
    return dict(bb_per_100=center, ci95=[center-margin, center+margin], sd=sd, blocks=len(values))


def readout(base, plan):
    # No mean is computed until every contrast has its replay and exact repeat receipt.
    if read(base/'all-final-verified.json')['status'] != 'verified':
        raise ValueError('All final hands must verify before readout')
    counts = plan['counts']
    folders = ['primary', *(['curve'] if counts['curve'] else []), *['panel-'+o for o in direct.PANEL_OPPONENTS]]
    for folder in folders:
        replay = read(base/folder/'replay.json')
        repeat = read(base/folder/'reproduction/complete.json')
        if replay['status'] != 'verified' or not repeat['reproduced_all_hands_and_decisions']:
            raise ValueError('Final verification barrier incomplete')
        for arm in ('play', 'reproduction'):
            done = read(base/folder/arm/'complete.json')
            for name, pin in done['files'].items():
                path = base/folder/arm/name
                if path.stat().st_size != pin['bytes'] or file_hash(path) != pin['sha256']:
                    raise ValueError('Final bytes changed after verification')
    primary = series(base/'primary', counts['primary'])['candidate']
    summaries = {'primary': estimate(primary), 'panel': {}}
    if counts['curve']:
        curve = series(base/'curve', counts['curve'])['candidate']
        summaries['five_minus_baseline'] = estimate(curve)
        summaries['final_minus_five_against_baseline'] = estimate(
            [a-b for a, b in zip(primary[:len(curve)], curve, strict=True)])
    for op in direct.PANEL_OPPONENTS:
        arms = series(base/('panel-'+op), counts['panel'], panel=True)
        summaries['panel'][op] = {'arms': {a: estimate(v) for a, v in arms.items()},
            'contrast': estimate([a-b for a, b in zip(arms['candidate'], arms['baseline'], strict=True)])}
    flags = {op: row['contrast']['ci95'][1] < -20 for op, row in summaries['panel'].items()}
    summaries.update(status='verified', severe_panel_regressions=flags,
        recommend_slumbot=summaries['primary']['ci95'][0] > 0 and not any(flags.values()),
        all_final_hands_independently_replayed_and_exactly_reproduced=True,
        canonical_plan_sha256=file_hash(base/'canonical-plan.json'))
    put(base/'summary.json', summaries)
    cells = []
    for folder in folders:
        for key, cell in read(base/folder/'replay.json')['coverage'].items():
            policy, street = key.split('/')
            n = cell['decisions']; lookup = cell['lookup']; bands = cell['visit_bands']
            missing = lookup.get('missing-key', 0); known = n-missing
            known_bands = dict(bands); known_bands['0'] = bands.get('0', 0)-missing
            if known_bands['0'] < 0 or sum(known_bands.values()) != known:
                raise ValueError('Coverage denominator differs')
            cells.append(dict(contrast=folder, policy=policy, street=street, decisions=n,
                missing_key_share=missing/n, zero_mass_share=lookup.get('zero-mass', 0)/n,
                known_decisions=known, **{'known_visit_'+k: known_bands.get(k, 0) for k in ('0','1-9','10-99','100+')},
                **{'known_share_'+k: known_bands.get(k, 0)/known if known else None for k in ('0','1-9','10-99','100+')},
                **{'all_visit_'+k: bands.get(k, 0) for k in ('0','1-9','10-99','100+')}))
    with (base/'coverage.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)


def science(config, base):
    models = prepare(config, base)
    reserve(list(ROOTS.values()), config['pr'], base/'root-reservation.json')
    pair = [models['final']['spec'], models['baseline']['spec']]
    five_pair = [models['five']['spec'], models['baseline']['spec']]
    for label, model in models.items():
        path = base/('cache-'+label+'-spec.json')
        put(path, model['spec'])
        subprocess.run([sys.executable, '-m', 'scripts.run_hu200_frozen_evaluation', 'cache',
                        '--out', str(base/('cache-'+label+'.json')), '--specs', str(path),
                        '--cache', str(base/'policy-cache')], check=True, timeout=max(1, SCIENCE_END-time()))
    costs = {'primary': run_pair(base, pair, 1024, ROOTS['direct_pilot'], 'pilot-primary', pilot=True),
             'panel': {}}
    # The owner authorized 1,024 direct pilot blocks total. The 5B pair uses
    # that measured cost as a declared planning proxy, without extra hands.
    costs['curve'] = dict(costs['primary'], planning_proxy='final-vs-752M direct pilot; 5B cache already built')
    for op in direct.PANEL_OPPONENTS:
        costs['panel'][op] = run_pair(base, pair, 64, ROOTS['panel_pilot'], 'pilot-panel-'+op, op, True)
    put(base/'pilot-costs.json', costs)
    plan = freeze(base, config, costs)
    if file_hash(base/'canonical-plan.json') != read(base/'plan-posted.json')['sha256']:
        raise ValueError('Plan changed after posting')
    run_pair(base, pair, plan['counts']['primary'], ROOTS['direct'], 'primary')
    if plan['counts']['curve']:
        run_pair(base, five_pair, plan['counts']['curve'], ROOTS['direct'], 'curve')
    for op in direct.PANEL_OPPONENTS:
        run_pair(base, pair, plan['counts']['panel'], ROOTS['panel'], 'panel-'+op, op)
    put(base/'all-final-verified.json', {'status': 'verified', 'source': revision(), 'at': time()})
    readout(base, plan)


def archive(config, base):
    owner = json.loads(gh('pr', 'view', config['pr'], '--json', 'state,url,headRefName,headRefOid'))
    files = [(p, 'research/'+p.relative_to(base).as_posix()) for p in sorted(base.rglob('*'))
             if p.is_file() and not p.is_symlink() and not p.is_relative_to(base/'policy-cache')
             and not p.is_relative_to(base/'guards/operations/archive')
             and p.name not in ('archive-receipt.json', 'guard-held.json')]
    # Inputs are indexed dependencies. This archive includes immutable guard readings
    # after science exits, with the active archive guard retained separately.
    for path in ('scripts/run_hu200_frozen_evaluation.py', 'docs/hu200-frozen-evaluation-protocol.md'):
        files.append((Path(path), 'source/'+path))
    destination = Path.home()/f"Local/Research-Cloud/PR-{config['pr']}-hu200-frozen-evaluation/hu200-frozen-evaluation-M4-20261014.zip"
    result = seal(files, destination, {'source': revision(), 'owner': owner, 'originals_retained': True,
        'inputs': read(base/'inputs.json') if (base/'inputs.json').exists() else config,
        'remote_archive_bytes_downloaded': False, 'active_archive_guard_excluded': True})
    put(base/'archive-receipt.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('run', 'science', 'pair', 'cache', 'archive'))
    parser.add_argument('--config', type=Path); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--specs', type=Path); parser.add_argument('--blocks', type=int)
    parser.add_argument('--root', type=int); parser.add_argument('--cache', type=Path)
    parser.add_argument('--opponent', choices=direct.PANEL_OPPONENTS); parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args(); base = args.out.resolve()
    if args.stage == 'cache':
        spec = read(args.specs); started = time()
        model = AveragePolicy(spec['path'], spec['sha256'], expected_schema=spec['abstraction'],
                              card_buckets=Path(spec['card_buckets']), sorted_rows=True, cache=args.cache)
        put(base, {'name':spec['name'], 'sha256':spec['sha256'], 'seconds':time()-started,
                   'description':model.description, 'no_hands_played':True})
        return
    if args.stage == 'pair':
        pair_worker(base, read(args.specs), args.blocks, args.root, args.opponent, args.pilot, args.cache)
        return
    config = read(args.config)
    if args.stage in ('science', 'archive'):
        def deadline(signum, frame):
            raise TimeoutError('Fixed campaign deadline reached')
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, max(.01, (SCIENCE_END if args.stage == 'science' else END)-time()))
        try:
            if args.stage == 'science': science(config, base)
            else: archive(config, base)
        except BaseException as exc:
            # Capture creation times before signaling only this worker's children.
            from scripts.guard_m4_research import terminate
            owned = {p.pid: p.create_time() for p in psutil.Process().children(recursive=True)}
            terminate(owned)
            put(base/(args.stage+'-failure.json'), {'failure': repr(exc), 'at': time()})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return
    if subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip() != 'Apple M4':
        raise ValueError('Only M4 is authorized')
    base.mkdir(parents=True, exist_ok=False)
    admission(config, base)
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as shared, \
            Path(config['pr265_lock']).open('r+') as prior:
        fcntl.flock(prior, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
        put(base/'locks-acquired.json', {'at': time(), 'shared': shared.name, 'pr265': prior.name})
        for stage in ('science', 'archive'):
            command = [sys.executable, '-m', 'scripts.guard_m4_research', '--stage', str(base/'guards'),
                '--name', stage, *(['--preserve'] if stage == 'archive' else []), '--', sys.executable,
                '-m', 'scripts.run_hu200_frozen_evaluation', stage, '--config', str(args.config.resolve()), '--out', str(base)]
            subprocess.run(command, check=stage == 'archive')
    put(base/'detached-closeout.json', {'at': time(), 'within_deadline': time() <= END,
                                     'summary_exists': (base/'summary.json').exists()})


if __name__ == '__main__':
    main()
