"""Preserve a terminal campaign without reading poker returns or restarting work."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
from time import time
from zipfile import ZipFile

from scripts.archive_hu100_4b_seed_ladder import seal
from scripts.evaluate_hu100_direct import PANEL_OPPONENTS, put
from scripts.evaluate_hu100_v053_seeds import labels
from scripts.report_hu100_v053_seeds import table, interval
from scripts.report_m4_resources import summarize
from scripts.train_hu100_v053_seeds import BASE, ROOT, BINARY, SEEDS, read, require_disk

FOLDER = Path.home()/'Local/Research-Cloud/PR-286-hu100-v053-three-seeds'
ASSETS = ROOT/'docs/reports/hu100-v053-three-seeds-artifacts'
MODEL_NAMES = ('stage1-requested-500000000.checkpoint.gz',
               'stage2-requested-5000000000.checkpoint.gz', '5000000000.current.gz')


def seal_models(files, destination, metadata, expected):
    receipt=seal(files,destination,metadata)
    # Reuse the archive's first source hash and required member readback; do not
    # hash multi-GB models again merely to compare against their existing audit.
    with ZipFile(destination) as archive:
        members=json.loads(archive.read('ARCHIVE-MANIFEST.json'))['members']
    for member,pin in expected.items():
        if any(members[member][key]!=pin[key] for key in ('bytes','sha256')):
            raise ValueError('Archive model differs from existing verified training/audit pin')
    receipt['matches_existing_model_audit_pins']=True
    return receipt


def verification(folder):
    """Read completion metadata only; no traces, coverage counts or estimates."""
    paths = [folder/'costs.json', folder/'replay.json',
             folder/'reproduction/complete.json', folder/'reached-key-coverage.json']
    if not all(p.exists() for p in paths):
        return dict(status='not-completed', present=folder.exists())
    costs, replay, repeat, reached = map(read, paths)
    if (replay['status'] != 'verified' or not repeat['reproduced_all_hands_and_decisions']
            or reached['status'] != 'verified'):
        return dict(status='incomplete')
    return dict(status='replayed-and-reproduced', blocks=costs['blocks'],
                hands=replay['hands_replayed'], selected_coverage_verified=True,
                fixed_seconds=costs['fixed_seconds'], block_work_seconds=costs['block_work_seconds'])


def inventory():
    if not (BASE/'controller-failure.json').exists():
        raise ValueError('Terminal scientific failure required; never a normal readout path')
    if (BASE/'all-final-verified.json').exists() or (BASE/'summary.json').exists():
        raise ValueError('Partial closeout refuses an existing global readout')
    status = dict(status='terminal-partial', scientific_retry=False, outcomes_read=False,
                  global_final_barrier=False, failure=read(BASE/'controller-failure.json'),
                  final={label:verification(BASE/'final'/label) for label in labels()})
    status['comparisons_completed'] = sum(r['status']=='replayed-and-reproduced' for r in status['final'].values())
    status['comparisons_planned'] = len(labels())
    status['original_final_hands_verified'] = sum(r.get('hands',0) for r in status['final'].values())
    models = {}
    for seed in SEEDS:
        audited = read(BASE/str(seed)/'5b.audit.json')
        if audited['status'] != 'verified':
            raise ValueError('Missing training audit')
        first = read(BASE/str(seed)/'stage1-verified.json')['checkpoint']
        pins = [first, *audited['files'].values()]
        models[str(seed)] = {Path(p['path']).name:p for p in pins}
    return status, models


def render(status, models):
    ASSETS.mkdir(exist_ok=True)
    plan = read(BASE/'canonical-plan.json')
    resources = summarize(BASE/'guards')
    output = ROOT/'docs/reports/hu100-v053-three-seeds.md'
    stamp = datetime.fromtimestamp(read(BASE/'guards/STOP.json')['at'], timezone.utc).isoformat()
    parts = ['# HU100 v0.5.3: terminal partial three-seed campaign\n',
        f'**Stopped {stamp}. Both new training seeds and every-row audits completed; evaluation stopped at preventive disk admission after {status["comparisons_completed"]}/17 comparisons. No qualification or poker outcome readout.** The owner decides merge and release.\n',
        '[PR286](https://github.com/dberweger2017/CFR-NLHE-6max/pull/286) repeats the unchanged [#260 → #272 recipe](../hu100-v053-three-seeds-protocol.md), sequentially on M4. No retry, failed-hand rerun, count reduction, budget extension, release, tag, default change or Slumbot.\n',
        '## Training and full audits\n']
    lineages=[];density=[];missing=[]
    for seed in SEEDS:
        first=read(BASE/str(seed)/'stage1-verified.json');final=read(BASE/str(seed)/'training-verified.json')
        a=first['header'];b=final['header'];audited=read(BASE/str(seed)/'5b.audit.json')
        lineages.append([seed,a['native_state']['completed_nodes'],a['iteration'],first['keys']['entries'],b['native_state']['completed_nodes'],b['iteration'],audited['all_nodes_verified']])
        for name,row in [('capacity terminal',first['telemetry'][-1]),('frozen 5B',final['telemetry'][-1])]:
            d=row['diagnostics'];n=d['entries']
            density.append([seed,name,f"{d['traverser_visits']/n:.6f}",d['visits_histogram']['0'],f"{100*d['visits_histogram']['0']/n:.4f}%",d['zero_average_mass_keys']])
        frozen=final['telemetry'][-1]['frozen_table']
        for i,street in enumerate(frozen['streets']):
            missing.append([seed,street,frozen['traverser_decisions_by_street'][i],frozen['missing_traverser_by_street'][i],frozen['opponent_decisions_by_street'][i],frozen['missing_opponent_by_street'][i]])
        f=b['frozen_table']
        parts.append(f"Seed **{seed}**: discarded updates **{f['discarded_updates']:,}**; discarded traverser visits **{f['discarded_traverser_visits']:,}**. All current/average rows pass `src.diagnostics.cfr_average.audit`, uniform zero-mass exports. [Audit](hu100-v053-three-seeds-artifacts/{seed}-5b.audit.json), [lineage](hu100-v053-three-seeds-artifacts/{seed}-training-verified.json).\n")
    parts += [table(['Seed','Stage-1 nodes','Iterations','Entries','Final nodes','Iterations','Rows audited'],lineages),
        'Stage 1 copies the fixed **97,290,497** cap, 500M milestone and 1B target, ending at the first complete iteration beyond capacity. Stage 2 hash-pins that seed’s terminal, freezes at its actual entry count and targets 5B. Sorted key lists match exactly by streaming count/SHA256. Filenames name targets, not actual work. HU100 10,000/10,000, 50/100, no ante/rake/raise cap; linear CFR, opponent-sampled average, one root per seat, ordered history, menu v2, pinned #163 K50. Native build SHA256 `4f6abcbb954c3c744448c2a3542ab3164e0e1a015cb05ecebeda077525991cd4` matches #272.\n',
        table(['Seed','Save','Visits/key','Zero visits','Zero visits %','Zero average mass keys'],density),
        table(['Seed','Street','Traverser decisions','Missing traverser','Opponent decisions','Missing opponent'],missing),
        'Frozen misses describe counterfactual training roles, not played-hand coverage. The full receipts retain street shares. Stored-key density does not establish convergence.\n',
        '## Evaluation retained without readout\n',
        f"All 17 cost-only pilots completed. Full **16,384 panel /131,072 direct blocks** were declared before final hands; [canonical plan](hu100-v053-three-seeds-artifacts/canonical-plan.json) and [exact PR-description readback](hu100-v053-three-seeds-artifacts/plan-posted.json). Whole prospective cost **{plan['counts']['projected_whole_seconds']/3600:.3f} hours**: elapsed {plan['counts']['elapsed_seconds']:.3f}s + 1.25 × ({plan['counts']['fixed_seconds']:.3f}s fixed + {plan['counts']['scaled_block_seconds']:.3f}s per-block + {plan['counts']['additional_preparation_seconds']:.3f}s future cache setup) + 1800s closeout. Only per-block work scales. The original deadline remains October 15, 20:34:12 UTC; unused budget is not permission to retry.\n"]
    old=read(ROOT/'docs/reports/hu100-v2-frozen-5b-artifacts/summary.json')
    parts.append(table(['Opponent','2026100601 (#272), paired BB/100 [95%]','2026101501','2026101502'],
        [[op,interval(old['primary'][op]['contrast']),'withheld (barrier incomplete)','not run'] for op in PANEL_OPPONENTS]))
    parts.append('The new primary panel uses frozen v2 translation off versus v0.5.2 translation on (512/128). #272’s historical rule met its declared limits; neither new seed has a reportable rule outcome. This is a historical comparison, not an improvement-release gate.\n')
    parts.append(table(['Fresh direct seed','Progress','Poker result / reached support'],
        [[seed,status['final'][f'direct-{seed}']['status'],'withheld / no global readout'] for seed in (2026100601,*SEEDS)]))
    parts.append(f"**{status['original_final_hands_verified']:,} original final hands** in nine comparisons independently replayed and exactly reproduced. All seven seed-2026101501 panels and direct matches for 2026100601/2026101501 completed; seed-2026101502’s eight final comparisons never started. Selected-key coverage verification receipts exist for all nine, including both-side support for the two direct matches. Native and effective coverage counts remain sealed with the raw traces. The required all-17 barrier is absent; no means, intervals, new historical-rule outcomes or coverage readout is produced. Cost-only pilots remain disjoint and unpooled.\n")
    parts += ['## Failure, resources and storage\n',
        'The next seed’s cache admission refused **25,183,547,392 free bytes − 8,076,238,511 additional bytes**, which would leave **17,107,308,881 bytes**, below the **17,179,869,184-byte (16 GiB)** floor. This was a preventive refusal before cache construction or that seed’s final hands. The controller exited 1; the outer guard retained the failure and stopped. No resource breach triggered it. [Terminal receipts](hu100-v053-three-seeds-artifacts/partial-status.json).\n']
    r=resources['all_operations']
    parts.append(f"Both shared locks and detached tmux on M4; all **{r['samples']:,}** five-second readings retained. Sampled peak family RSS **{r['peak_family_rss_bytes']/2**30:.3f} GiB**, minimum free memory **{r['minimum_free_percent']}%**, minimum free disk **{r['minimum_disk_free_bytes']/2**30:.3f} GiB**, maximum original-baseline swap growth **{r['maximum_swap_growth_bytes']/1e9:.3f} GB**, pressure levels **{r['pressure_levels']}**. Original guards/baseline are unchanged for archival preservation. [Resources](hu100-v053-three-seeds-artifacts/resource-summary.json).\n")
    parts += [*['- '+row['failure']+'\n' for row in read(BASE/'setup-failures.json')],
        '- Closeout process enumeration initially encountered macOS AccessDenied on PID 1; corrected to permission-tolerant metadata enumeration. No process was signalled.\n',
        '\nBoth new model sets remain in their unsynced campaign root. Only this campaign’s inactive regenerable release cache is eligible for removal, with manifest/input/rebuild provenance; models, unique evidence and synced files stay. Separate average-only ZIPs and raw trace/shared/lifecycle archives are sealed where headroom permits, with manifests and full local readback. Full checkpoint/current bundles remain pending if they cannot fit above 16 GiB. Exact archive/member hashes, Drive locators, upload status and restore commands are in [RESULTS_INDEX](../../RESULTS_INDEX.md); no archive or upload completion is implied by this partial report.\n',
        'The campaign’s 30-minute schedule is disabled on this terminal stop. The owner decides merge and release.\n']
    output.write_text('\n'.join(parts))
    exports={'partial-status.json':status,'model-index.json':models,'resource-summary.json':resources,
        'controller-failure.json':read(BASE/'controller-failure.json'),'guard-stop.json':read(BASE/'guards/STOP.json'),
        'whole-campaign-receipt.json':read(BASE/'guards/operations/whole-campaign/receipt.json'),
        'budget.json':read(BASE/'budget.json'),'evaluation-preparation.json':read(BASE/'evaluation-preparation.json')}
    for name,value in exports.items():
        (ASSETS/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def archive():
    owner=read(BASE/'partial-archive-owner.json')
    if owner['pr']['number']!=286 or not 0<=time()-owner['at']<600:
        raise ValueError('Fresh current owning-PR metadata required')
    if time()>=read(BASE/'budget.json')['deadline']:
        raise ValueError('Original deadline expired')
    status,models=inventory()
    put(BASE/'partial-status.json',status)
    cache=BASE/'policy-cache'
    if cache.exists():
        manifests={str(p.relative_to(cache)):read(p) for p in cache.rglob('manifest.json')}
        put(BASE/'partial-cache-removal.json',dict(at=time(),path=str(cache),manifests=manifests,
            scientific_workers_exited=True,all_consumers_terminal=True,canonical_models_retained=True,
            input_specs=read(BASE/'evaluation-inputs.json'),
            rebuild='AveragePolicy(pinned average, sha256, expected_schema, card_buckets, sorted_rows=True, cache=BASE/policy-cache)'))
        shutil.rmtree(cache)
    render(status,models)
    source=BASE/'partial-scientific-source.tar'
    with source.open('xb') as stream:
        subprocess.run(['git','archive','ed090caecde29569a5c7cd44e49f6dc434145e9c','scripts','native','src',
            'requirements.txt','requirements-dev.txt','requirements-play.txt','docs/hu100-v053-three-seeds-protocol.md'],stdout=stream,check=True)
    excluded_models={str(BASE/str(seed)/name) for seed in SEEDS for name in (*MODEL_NAMES,'5000000000.average.gz')}
    files=[]
    for p in sorted(BASE.rglob('*')):
        if not p.is_file() or p.is_symlink() or str(p) in excluded_models:continue
        rel=p.relative_to(BASE)
        if rel.parts[0] in ('inputs','policy-cache'):continue
        if rel.parts[:3]==('guards','operations','partial-preservation') or rel.name=='guard-held.json':continue
        files.append((p,'research/'+rel.as_posix()))
    for p in sorted((ROOT/'planning').rglob('*')):
        if p.is_file() and not p.is_symlink() and p.stat().st_size<1024**2:
            files.append((p,'planning/'+p.relative_to(ROOT/'planning').as_posix()))
    files.append((BINARY,'runtime/hu20-trainer'))
    files.append((ROOT/'docs/reports/hu100-v053-three-seeds.md','report/hu100-v053-three-seeds.md'))
    for p in sorted(ASSETS.glob('*.json')):files.append((p,'report/artifacts/'+p.name))
    average_bytes=sum(models[str(seed)]['5000000000.average.gz']['bytes'] for seed in SEEDS)
    require_disk(sum(p.stat().st_size for p,_ in files)+average_bytes+512*1024**2)
    metadata=dict(pr=286,status='terminal-partial',source='ed090caecde29569a5c7cd44e49f6dc434145e9c',
        no_outcome_readout=True,models=models,originals_retained=True,
        excludes_live_guard='guards/operations/partial-preservation',
        model_archive_status='Averages separately sealed; stage1/final/current bundles pending storage')
    for seed in SEEDS:
        average=BASE/str(seed)/'5000000000.average.gz'
        member=f'averages/{seed}/5000000000.average.gz'
        receipt=seal_models([(average,member)],
            FOLDER/f'hu100-v053-seed-{seed}-average.zip',metadata,
            {member:models[str(seed)]['5000000000.average.gz']})
        put(BASE/str(seed)/'average-archive-receipt.json',receipt)
    receipt=seal(files,FOLDER/'hu100-v053-terminal-partial-evidence.zip',metadata)
    put(BASE/'partial-archive-receipt.json',receipt)
    pending=sum(models[str(seed)][name]['bytes'] for seed in SEEDS for name in MODEL_NAMES)
    free=shutil.disk_usage(BASE).free
    put(BASE/'pending-model-archives.json',dict(at=time(),required_model_bytes=pending,free_bytes=free,
        floor_bytes=16*1024**3,additional_space_needed_bytes=max(0,pending+512*1024**2+16*1024**3-free),
        models={str(seed):{n:models[str(seed)][n] for n in MODEL_NAMES} for seed in SEEDS},
        originals_retained=True,scientific_campaign_terminal=True))


def archive_models():
    owner=read(BASE/'partial-archive-owner.json')
    if owner['pr']['number']!=286 or not 0<=time()-owner['at']<600:
        raise ValueError('Fresh current owning-PR metadata required')
    if time()>=read(BASE/'budget.json')['deadline']:
        raise ValueError('Original deadline expired')
    _,models=inventory()
    require_disk(sum(models[str(seed)][name]['bytes'] for seed in SEEDS for name in MODEL_NAMES)+512*1024**2)
    for seed in SEEDS:
        files=[(BASE/str(seed)/name,f'research/{seed}/{name}') for name in MODEL_NAMES]
        expected={member:models[str(seed)][path.name] for path,member in files}
        receipt=seal_models(files,FOLDER/f'hu100-v053-seed-{seed}-models.zip',
            dict(pr=286,status='terminal-partial',seed=seed,originals_retained=True,
                raw_evidence_archive='hu100-v053-terminal-partial-evidence.zip',
                average_archive=f'hu100-v053-seed-{seed}-average.zip'),expected)
        put(BASE/str(seed)/'model-archive-receipt.json',receipt)
    put(BASE/'model-archives-complete.json',dict(at=time(),originals_retained=True,
        pending_storage_resolved=True,within_original_deadline=time()<read(BASE/'budget.json')['deadline']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('archive','models','report','lifecycle','final-lifecycle'))
    args=parser.parse_args()
    if args.stage=='report':
        render(*inventory());return
    owner=read(BASE/'partial-archive-owner.json')
    if owner['pr']['number']!=286 or not 0<=time()-owner['at']<600:
        raise ValueError('Fresh current owning-PR metadata required')
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as one, \
         (Path.home()/'Local/.hu100-m4-research.lock').open('a+') as two:
        for stream in (one,two):fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='archive':archive()
        elif args.stage=='models':archive_models()
        else:
            if args.stage=='final-lifecycle' and not (BASE/'model-archives-complete.json').exists():
                raise ValueError('Full model archives must complete before final lifecycle')
            files=[]
            operations=('partial-preservation','partial-models') if args.stage=='lifecycle' else ('partial-models','partial-lifecycle')
            for name in operations:
                if (BASE/'guards/operations'/name).exists() and not (BASE/'guards/operations'/name/'receipt.json').exists():
                    raise ValueError('Cannot archive a live guard stream')
                files += [(p,'research/'+p.relative_to(BASE).as_posix())
                    for p in sorted((BASE/'guards/operations'/name).rglob('*')) if p.is_file()]
            files += [(p,'research/'+p.relative_to(BASE).as_posix()) for p in (
                BASE/'partial-archive-receipt.json',BASE/'pending-model-archives.json',
                *[BASE/str(s)/'average-archive-receipt.json' for s in SEEDS])]
            files += [(p,'research/'+p.relative_to(BASE).as_posix()) for p in (
                BASE/'model-archives-complete.json', *[BASE/str(s)/'model-archive-receipt.json' for s in SEEDS]) if p.exists()]
            if args.stage=='final-lifecycle':
                with ZipFile(FOLDER/'hu100-v053-terminal-partial-evidence.zip') as archive:
                    archived=json.loads(archive.read('ARCHIVE-MANIFEST.json'))['members']
                checked=0
                for label,row in read(BASE/'partial-status.json')['final'].items():
                    if row['status']!='replayed-and-reproduced':continue
                    for kind in ('play','reproduction'):
                        for name,pin in read(BASE/'final'/label/kind/'complete.json')['files'].items():
                            member=f'research/final/{label}/{kind}/{name}'
                            if any(archived[member][key]!=pin[key] for key in ('bytes','sha256')):
                                raise ValueError('Archived trace differs from verified completion pin')
                            checked+=1
                put(BASE/'partial-archived-trace-pins.json',dict(status='verified',trace_members=checked,
                    archive_source_hashes_match_existing_play_and_reproduction_pins=True,
                    additional_trace_hash_pass=False,outcomes_read=False,global_barrier_complete=False))
                for name in ('partial-final-lifecycle.json','partial-lifecycle-archive-receipt.json','partial-cache-removal.json'):
                    files.append((BASE/name,'research/'+name))
                files.append((BASE/'partial-archived-trace-pins.json','research/partial-archived-trace-pins.json'))
                files += [(p,'planning/'+p.name) for p in (ROOT/'planning').glob('*')
                          if p.is_file() and p.stat().st_size<1024**2]
                report=ROOT/'docs/reports/hu100-v053-three-seeds.md'
                files.append((report,'report/'+report.name))
                files += [(p,'report/artifacts/'+p.name) for p in ASSETS.glob('*.json')]
            require_disk(sum(p.stat().st_size for p,_ in files)+1024**2)
            final=args.stage=='final-lifecycle'
            put(BASE/('terminal-closeout-archive-receipt.json' if final else 'partial-lifecycle-archive-receipt.json'),seal(files,
                FOLDER/('hu100-v053-terminal-closeout.zip' if final else 'hu100-v053-terminal-partial-lifecycle.zip'),
                dict(pr=286,status='terminal-partial',excludes_live_guard='partial-final-lifecycle' if final else 'partial-lifecycle',originals_retained=True)))


if __name__=='__main__':main()
