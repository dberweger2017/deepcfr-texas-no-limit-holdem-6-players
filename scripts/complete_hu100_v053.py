"""Complete only #286's eight never-started contrasts using its pinned source.

This operational adapter does not implement poker, replay, sampling or statistics.
The frozen checkout owns those functions; existing final evidence is read-only.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PLAN_SHA = 'f61d0140cfdb0dc39610ed62ec369afb1ab724b4768f0db691f4cd23268d7a20'
SOURCE = 'ed090caecde29569a5c7cd44e49f6dc434145e9c'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(16*1024**2):
            h.update(chunk)
    return h.hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def remaining(plan):
    return [*[f'panel-2026101502-{o}' for o in plan['opponents']], 'direct-2026101502']


def quote(plan):
    rows = []
    for label in remaining(plan):
        c = plan['pilots'][label]
        n = plan['counts']['panel' if label.startswith('panel-') else 'direct']
        rows.append(dict(label=label, blocks=n, root=plan['roots'][label]['final'],
                         fixed_seconds=c['fixed_seconds'],
                         scaled_block_seconds=c['block_work_seconds']*n/c['blocks']))
    # All policy caches were retired at terminal closeout. Both needed builds
    # are one-off work, using their own measured pilot costs without scaling.
    caches = [c for c in plan['one_off_preparation']['pilot_cache_builds']
              if c['source_sha256'] in (plan['models']['2026101502']['sha256'], plan['release']['sha256'])]
    prep = plan['one_off_preparation']['input_preparation']['whole_seconds']
    total = prep + sum(r['fixed_seconds']+r['scaled_block_seconds'] for r in rows) + sum(c['seconds'] for c in caches)
    return dict(comparisons=rows, one_off_input_restoration_and_verification_seconds=prep,
                one_off_cache_builds=[{k:c[k] for k in ('source_sha256','seconds')} for c in caches],
                fixed_seconds=sum(r['fixed_seconds'] for r in rows),
                scaled_block_seconds=sum(r['scaled_block_seconds'] for r in rows),
                planning_margin=1.25, closeout_seconds=1800,
                wall_cap_seconds=math.ceil(1.25*total+1800))


def verify_files(folder, expected_hands):
    replay = read(folder/'replay.json')
    repeat = read(folder/'reproduction/complete.json')
    assert replay['status'] == 'verified' and repeat['reproduced_all_hands_and_decisions']
    assert replay['hands_replayed'] == repeat['hands'] == expected_hands
    rows = {}
    for stage in ('play','reproduction'):
        complete = read(folder/stage/'complete.json')
        assert complete['status'] == 'complete' and complete['hands'] == expected_hands
        for name, expected in complete['files'].items():
            p = folder/stage/name
            assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256']
            rows[f'{stage}/{name}'] = dict(expected, mtime_ns=p.stat().st_mtime_ns)
    assert read(folder/'reached-key-coverage.json')['status'] == 'verified'
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('controller','pair'))
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--label')
    a = p.parse_args()
    # Imports must come from the exact scientific checkout, even though this
    # adapter and its declaration are committed on a new current-main branch.
    scientific = a.out/'frozen-source'
    os.chdir(scientific)
    sys.path.insert(0, str(scientific))
    from scripts import evaluate_hu100_v053_seeds as e
    from scripts.train_hu100_v2_frozen_5b import is_campaign_process
    from src.blueprint.equity_buckets import EQUITY_K50_SHA256
    from src.blueprint.policy_cache import folder as cache_folder
    import psutil
    e.BINARY = a.old/'native/hu20-trainer/target/release/hu20-trainer'
    oldbase = a.old/'results/hu100-v053-seeds'
    plan = read(oldbase/'canonical-plan.json')
    assert sha(oldbase/'canonical-plan.json') == PLAN_SHA
    assert subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip() == SOURCE
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], text=True).strip()
    out = a.out
    env = read(out/'operational-envelope.json')
    assert env['canonical_plan_sha256'] == PLAN_SHA and env['quote'] == quote(plan)
    assert time.time() < env['deadline']-1800
    e.CACHE = out/'policy-cache'
    specs = [dict(plan['models']['2026101502'], path=str(out/'inputs/2026101502.average.gz')), plan['release']]
    if a.stage == 'pair':
        assert a.label in remaining(plan)
        row = next(r for r in env['quote']['comparisons'] if r['label'] == a.label)
        opponent = a.label.split('-',2)[2] if a.label.startswith('panel-') else None
        e.pair_worker(out/'final'/a.label, specs, row['blocks'], row['root'], opponent, False)
        return
    locks = []
    try:
        for name in ('.hu20-m4-research.lock','.hu100-m4-research.lock'):
            stream = (Path.home()/'Local'/name).open('a+')
            fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
            locks.append(stream)
        own = {os.getpid(), *[x.pid for x in psutil.Process().parents()]}
        for proc in psutil.process_iter(['pid','name','cmdline'], ad_value=None):
            if proc.pid not in own and is_campaign_process(proc.info['name'],proc.info['cmdline']):
                raise ValueError('Another research campaign is active: '+str(proc.pid))
        assert subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() == 'Apple M4'
        assert shutil.disk_usage(out).free >= 30*1024**3
        owners = read(out/'input-owners.json')
        assert time.time()-owners['at'] < 600
        assert all(r['state'] == 'MERGED' for r in owners['owners'].values())
        posted = read(out/'envelope-posted.json')
        assert posted['readback_verified'] and posted['sha256'] == sha(out/'operational-envelope.json')
        assert sha(Path(__file__)) == posted['adapter_sha256']
        assert sha(e.BINARY) == '4f6abcbb954c3c744448c2a3542ab3164e0e1a015cb05ecebeda077525991cd4'
        tables = {s:sha(e.TABLES/(s+'-k50.bin')) for s in EQUITY_K50_SHA256}
        assert tables == EQUITY_K50_SHA256
        assert not (out/'final').exists()
        oldcache = oldbase/'policy-cache'
        assert not oldcache.exists(), 'Unexpected cache state; retain and inspect dependencies'
        put(out/'cache-retirement-state.json', dict(at=time.time(), old_cache_path=str(oldcache),
            absent_before_this_task=True, removed_paths=[], release_cache_also_absent=True,
            provenance=str(oldbase/'partial-cache-removal.json'),
            rebuild='AveragePolicy(pinned path, sha256, expected_schema, card_buckets, sorted_rows=True, cache=OUT/policy-cache)',
            input_pins=plan['models'], release=plan['release']))
        prior = {}
        for label in e.labels():
            if label in remaining(plan):
                assert not (oldbase/'final'/label).exists(), 'Never rerun an existing comparison'
            else:
                n = plan['counts']['panel' if label.startswith('panel-') else 'direct']
                prior[label] = verify_files(oldbase/'final'/label, n*(4 if label.startswith('panel-') else 2))
        put(out/'prior-nine-verified.json',dict(at=time.time(), comparisons=prior, no_outcomes_read=True))
        archive = read(a.old/'docs/reports/hu100-v053-three-seeds-artifacts/2026101502-average-archive-receipt.json')
        assert shutil.disk_usage(out).free-2*specs[0]['bytes']-14*1024**3 >= 20*1024**3
        restored = e.restore(archive,'averages/2026101502/5000000000.average.gz',Path(specs[0]['path']))
        assert sha(Path(specs[0]['path'])) == specs[0]['sha256']
        assert sha(Path(specs[1]['path'])) == specs[1]['sha256']
        put(out/'input-restoration.json',dict(restored=restored, specs=specs, tables=tables,
            source=SOURCE, adapter_sha256=sha(Path(__file__)), owners=owners))
        # Six-action upper bound and a small manifest reserve, plus the release
        # cache, leave at least four GiB above the unchanged sixteen-GiB floor.
        cache_bytes = sum(s['entries']*(35+8*(6 if s.get('card_buckets') else 4))+1024**2 for s in specs)
        free = shutil.disk_usage(out).free
        assert free-cache_bytes >= 20*1024**3
        put(out/'disk-admission.json',dict(at=time.time(), free_bytes=free, projected_cache_bytes=cache_bytes,
            projected_free_after_build_bytes=free-cache_bytes, minimum_projected_free_bytes=20*1024**3))
        for spec in specs:
            tick = time.time()
            model = e.AveragePolicy(spec['path'],spec['sha256'],expected_schema=spec['abstraction'],
                card_buckets=Path(spec['card_buckets']) if spec.get('card_buckets') else None,
                sorted_rows=True,cache=e.CACHE)
            put(out/f"cache-{spec['name']}.json",dict(seconds=time.time()-tick, source_sha256=spec['sha256'],
                path=str(cache_folder(e.CACHE,spec['sha256'],spec['abstraction'])), description=model.description))
            del model
            e.gc.collect()
        assert shutil.disk_usage(out).free >= 20*1024**3
        for row in env['quote']['comparisons']:
            seconds = env['deadline']-1800-time.time()
            assert seconds > 0
            cmd = [sys.executable, str(Path(__file__).resolve()), 'pair', '--old', str(a.old),
                   '--out', str(out), '--label', row['label']]
            tick = time.time()
            with (out/(row['label']+'.log')).open('x') as log:
                done = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=seconds)
            put(out/(row['label']+'.operation.json'),dict(command=cmd,started=tick,finished=time.time(),returncode=done.returncode))
            if done.returncode:
                raise ValueError('Comparison stopped; no retry: '+row['label'])
        # A composite read-only view preserves original evidence and allows the
        # frozen global verifier to inspect all seventeen without rerunning nine.
        view = out/'readout-view'
        (view/'final').mkdir(parents=True)
        for label in e.labels():
            source = (out if label in remaining(plan) else oldbase)/'final'/label
            (view/'final'/label).symlink_to(source, target_is_directory=True)
        (view/'canonical-plan.json').symlink_to(oldbase/'canonical-plan.json')
        for label, before in prior.items():
            n = plan['counts']['panel' if label.startswith('panel-') else 'direct']
            assert verify_files(oldbase/'final'/label,n*(4 if label.startswith('panel-') else 2)) == before
        put(view/'all-final-verified.json',dict(status='verified',at=time.time(),labels=e.labels(),
            canonical_plan_sha256=PLAN_SHA, completed_now=remaining(plan), reused_unchanged=list(prior)))
        e.BASE = view
        coverage = e.verify_final_files()
        put(out/'global-barrier.json',dict(status='verified',at=time.time(),comparisons=e.labels(),
            canonical_plan_sha256=PLAN_SHA, original_final_hands=1703936, prior_nine_unchanged=True))
        summary = dict(panels={},direct={},coverage=coverage,status='verified',canonical_plan_sha256=PLAN_SHA,
                       all_final_replayed_and_reproduced=True)
        for seed in (2026100601,*e.SEEDS):
            summary['direct'][seed] = e.estimate(e.series(view/'final'/f'direct-{seed}',plan['counts']['direct'])['candidate'])
            if seed in e.SEEDS:
                panel = {}
                for opponent in e.direct.PANEL_OPPONENTS:
                    arms = e.series(view/'final'/f'panel-{seed}-{opponent}',plan['counts']['panel'],panel=True)
                    panel[opponent] = dict(arms={k:e.estimate(v) for k,v in arms.items()},
                        contrast=e.estimate([x-y for x,y in zip(arms['candidate'],arms['baseline'],strict=True)]))
                outcome, flags = e.recommendation(panel)
                summary['panels'][seed] = dict(opponents=panel,historical_pr272_rule_outcome=outcome,
                    historical_upper_below_minus20_flags=flags,release_pass_fail_gate=False)
        put(out/'summary.json',summary)
        put(out/'science-complete.json',dict(at=time.time(),all_17_verified=True,release_cache_retained=True))
    except BaseException as exc:
        put(out/('failure-'+str(time.time_ns())+'.json'),dict(at=time.time(),failure=repr(exc)))
        raise
    finally:
        for stream in locks:
            stream.close()


if __name__ == '__main__':
    main()
