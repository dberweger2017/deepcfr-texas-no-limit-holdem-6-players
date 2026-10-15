"""Three fixed-seed HU100 readouts behind one replay/reproduction barrier.

M4 workers never need GitHub credentials: the coordinator posts the prospective
plan and returns a hash-bound PR readback receipt before any final worker starts.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import sleep, time

from scripts import evaluate_hu100_direct as direct
from scripts.archive_hu100_4b_seed_ladder import seal
from scripts.evaluate_hu100_v2_frozen_5b import projected_cost, recommendation
from scripts.run_hu100_k50_v2 import restore, pin, header
from scripts.run_hu200_frozen_evaluation import series, estimate
from scripts.train_hu100_v053_seeds import BASE, ROOT, BINARY, TABLES, SEEDS, command, read, revision, require_disk
from src.blueprint.average import AveragePolicy
from src.blueprint.policy_cache import folder as cache_folder
from src.policies.files import file_hash

PR=286
put=direct.put
CACHE=BASE/'policy-cache'


def labels():
    return ['direct-2026100601', *[label for seed in SEEDS for label in
        [*[f'panel-{seed}-{opponent}' for opponent in direct.PANEL_OPPONENTS],f'direct-{seed}']]]


def select_counts(costs, elapsed, preparation_reserve=0):
    """Quote each run's fixed setup once, scaling only its measured block work."""
    if set(costs) != set(labels()):
        raise ValueError('Every seed and opponent needs a cost-only pilot')
    options=[]
    for panel, direct_blocks in ((16384,131072),(8192,65536)):
        parts=[projected_cost(cost,panel if label.startswith('panel-') else direct_blocks)
               for label,cost in costs.items()]
        fixed=sum(p['fixed_seconds'] for p in parts)
        work=sum(p['scaled_block_seconds'] for p in parts)
        whole=elapsed+1.25*(fixed+work+preparation_reserve)+1800
        option=dict(panel=panel,direct=direct_blocks,elapsed_seconds=elapsed,
            fixed_seconds=fixed,scaled_block_seconds=work,additional_preparation_seconds=preparation_reserve,
            planning_margin=1.25,closeout_seconds=1800,projected_whole_seconds=whole,fits=whole<=14*3600)
        options.append(option)
        if option['fits']:
            return option, options
    raise ValueError('Neither pre-outcome count option fits the original fourteen hours')


def pair_worker(folder, specs, blocks, root, opponent, pilot):
    registry,original=direct.execute(specs,blocks,root,'hu100-v053-qualification',folder/'play',revision(),
        stack_bb=100,opponent=opponent,cache=CACHE,cache_read_only=True)
    del registry; gc.collect()
    # cost_only seals means for pilots and final workers alike. No summary is
    # computed until the global barrier proves all seventeen final contrasts.
    checked=direct.audit(folder/'play',folder/'replay.json',cost_only=True)
    registry,repeat=direct.execute(specs,blocks,root,'hu100-v053-qualification',folder/'reproduction',revision(),
        reproduce=folder/'play',stack_bb=100,opponent=opponent,cache=CACHE,cache_read_only=True)
    del registry; gc.collect()
    if not repeat['reproduced_all_hands_and_decisions']:
        raise ValueError('Exact reproduction incomplete; no rerun')
    fixed=original['load_or_validation_seconds']+repeat['load_or_validation_seconds']+checked['fixed_setup_seconds']
    block_work=original['seconds']+repeat['seconds']+checked['seconds']-fixed
    put(folder/'costs.json',dict(primary=original,repeat=repeat,audit_seconds=checked['seconds'],
        audit_fixed_seconds=checked['fixed_setup_seconds'],audit_block_work_seconds=checked['block_work_seconds'],
        fixed_seconds=fixed,block_work_seconds=block_work,blocks=blocks,pilot=pilot,
        paired_sd_bb_per_100=checked['paired_sd_bb_per_100']))


def run_pair(label,specs,blocks,root,opponent=None,pilot=False):
    folder=BASE/('pilot' if pilot else 'final')/label
    spec_path=folder.parent/(label+'-specs.json'); put(spec_path,specs)
    args=[sys.executable,'-m','scripts.evaluate_hu100_v053_seeds','pair','--folder',folder,
          '--specs',spec_path,'--blocks',blocks,'--root',root]
    if opponent: args += ['--opponent',opponent]
    if pilot: args += ['--pilot']
    command(('pilot-' if pilot else 'final-')+label,args)
    return read(folder/'costs.json')


def prepare():
    tick=time()
    owners=read(BASE/'evaluation-owners.json')
    if (not 0<=time()-owners['at']<600 or owners['owners']['272']['state']!='MERGED'):
        raise ValueError('Fresh merged-owner check before restoring PR272 input required')
    original=read(ROOT/'docs/reports/hu100-v2-frozen-5b-artifacts/treatment.json')
    average=BASE/'inputs/2026100601.average.gz'
    archive=read(ROOT/'docs/reports/hu100-v2-frozen-5b-artifacts/model-index.json')['archives']['average']
    # A normal read may hydrate this existing cloud placeholder; never force
    # offload or delete it. The indexed whole and manifest hashes are checked.
    require_disk(2*original['bytes']+512*1024**2)
    restored=restore(archive,'averages/5000000000.average.gz',average)
    if pin(average)['sha256']!=original['sha256']:
        raise ValueError('Restored original candidate differs')
    old=dict(original,path=str(average),name='v2-2026100601',card_buckets=str(TABLES),translation=None)
    release=dict(read(ROOT/'docs/reports/hu100-k50-v2-artifacts/inputs.json')['models']['v052'],
        translation=dict(max_states=512,max_events=128))
    if pin(Path(release['path'])) != {k:release[k] for k in ('path','bytes','sha256')}:
        raise ValueError('Pinned released v0.5.2 differs')
    bundle=Path(release['path']).parent
    verified=subprocess.check_output([sys.executable,str(bundle/'verify_hu100_bundle.py'),str(bundle),
        '--expect-source','3a3bc4218bab96637d6a27a69b73383ee8bb196c','--require-publication'],text=True)
    models={2026100601:old,**{seed:read(BASE/str(seed)/'treatment.json') for seed in SEEDS}}
    for seed in SEEDS:
        audit=read(BASE/str(seed)/'5b.audit.json')
        if (audit['status']!='verified' or audit['all_nodes_verified']!=models[seed]['entries']
                or audit['average_sha256']!=models[seed]['sha256']):
            raise ValueError('New model lacks its every-row audit')
    pairs={label:[models[int(label.split('-')[1])],release] for label in labels()}
    put(BASE/'evaluation-inputs.json',dict(models=models,release=release,pairs=pairs,
        restored_original=restored,release_publication_verification=verified,owners=owners,source=revision()))
    put(BASE/'evaluation-input-preparation.json',dict(whole_seconds=time()-tick,
        input_verification_and_restoration_seconds=time()-tick,no_hands=True))


def build_cache(spec,phase):
    target=cache_folder(CACHE,spec['sha256'],spec['abstraction'])
    if target.exists():return None
    # One candidate plus the release cache is sufficient at any moment. Cache
    # rebuilds are fixed preparation, explicitly quoted before final hands.
    require_disk(spec['entries']*(35+8*(6 if spec.get('card_buckets') else 4))+1024**2)
    started=time()
    model=AveragePolicy(spec['path'],spec['sha256'],expected_schema=spec['abstraction'],
        card_buckets=Path(spec['card_buckets']) if spec.get('card_buckets') else None,
        sorted_rows=True,cache=CACHE)
    receipt=dict(phase=phase,seconds=time()-started,description=model.description,
        source_sha256=spec['sha256'],no_hands=True,path=str(target))
    del model;gc.collect()
    put(BASE/f"cache-{phase}-{spec['name']}.json",receipt)
    return receipt


def remove_candidate_cache(spec,phase):
    target=cache_folder(CACHE,spec['sha256'],spec['abstraction'])
    if not target.resolve().is_relative_to(CACHE.resolve()):
        raise ValueError('Cache cleanup outside this campaign')
    put(BASE/f"cache-removed-{phase}-{spec['name']}.json",dict(path=str(target),at=time(),
        source_sha256=spec['sha256'],manifest=read(target/'manifest.json'),
        all_phase_consumers_finished=True,canonical_average_retained=True,
        rebuild='AveragePolicy with this pinned average, schema, K50 tables and sorted_rows=True'))
    shutil.rmtree(target)


def evaluate():
    roots=read(BASE/'roots.json'); reservation=read(BASE/'root-reservation.json')
    if not reservation['fresh'] or sorted(reservation['roots'])!=sorted(n for r in roots.values() for n in r.values()):
        raise ValueError('Root reservation differs')
    pairs=read(BASE/'evaluation-inputs.json')['pairs']; costs={}; cache_costs=[]
    for label in labels():
        for spec in pairs[label]:
            built=build_cache(spec,'pilot')
            if built:cache_costs.append(built)
        opponent=label.split('-',2)[2] if label.startswith('panel-') else None
        costs[label]=run_pair(label,pairs[label],64 if opponent else 1024,roots[label]['pilot'],opponent,True)
        if not opponent:remove_candidate_cache(pairs[label][0],'pilot')
    put(BASE/'pilot-costs.json',costs)
    candidate_hashes={pair[0]['sha256'] for pair in pairs.values()}
    future_cache_seconds=sum(c['seconds'] for c in cache_costs if c['source_sha256'] in candidate_hashes)
    put(BASE/'evaluation-preparation.json',dict(input_preparation=read(BASE/'evaluation-input-preparation.json'),
        pilot_cache_builds=cache_costs,quoted_future_cache_build_seconds=future_cache_seconds,
        cache_rebuilds_are_fixed_one_off_work=True,one_off_not_scaled_with_blocks=True))
    counts,options=select_counts(costs,time()-read(BASE/'budget.json')['started'],future_cache_seconds)
    compact_costs={label:{k:cost[k] for k in ('blocks','fixed_seconds','block_work_seconds',
        'audit_fixed_seconds','audit_block_work_seconds','paired_sd_bb_per_100')} for label,cost in costs.items()}
    inputs=read(BASE/'evaluation-inputs.json')
    plan=dict(pr=PR,source=revision(),counts=counts,projection_options=options,roots=roots,
        pilots=compact_costs,pilot_costs_sha256=file_hash(BASE/'pilot-costs.json'),
        one_off_preparation=read(BASE/'evaluation-preparation.json'),
        models=inputs['models'],release=inputs['release'],
        evaluation_inputs_sha256=file_hash(BASE/'evaluation-inputs.json'),opponents=direct.PANEL_OPPONENTS,
        historical_rule='No primary upper<-20 and pot_pressure lower>-20; comparison with PR272 only',
        release_rule='No improvement-release pass/fail gate; owner judges the overall package',
        direct='Fresh v2-off versus released v0.5.2 translation512/128-on for all three seeds; descriptive, never gated',
        intervals='Nominal paired Student-t95 conditional on each fixed policy pair',
        no_failed_hand_reruns=True,no_outcome_extension=True,no_pilot_pooling=True,
        every_final_hand_independently_replayed_and_exactly_reproduced_before_readout=True,
        first_build=read(BASE/'first-build.json'),deadline=read(BASE/'budget.json')['deadline'],
        root_reservation_sha256=file_hash(BASE/'root-reservation.json'),frozen_at=time(),
        cost_scaling='Only per-block play, raw trace hashes, replay and reproduction scale; loading/table/model admission stays fixed',
        resource_guards=read(BASE/'guards/operations/whole-campaign/intent.json')['guard_rules'])
    put(BASE/'canonical-plan.json',plan)
    # The coordinator edits PR286's description with this exact canonical JSON,
    # reads it back, then deposits a receipt. Waiting never resets the clock.
    while not (BASE/'plan-posted.json').exists():
        if time()>=read(BASE/'budget.json')['deadline']-1800:
            raise ValueError('No prospective PR readback before science deadline')
        sleep(5)
    posted=read(BASE/'plan-posted.json')
    if not posted['readback_verified'] or posted['sha256']!=file_hash(BASE/'canonical-plan.json'):
        raise ValueError('Canonical plan not posted/read back')
    # Account for coordinator latency without changing the one pre-outcome count
    # choice. If the frozen size no longer fits, stop instead of reducing again.
    future=(counts['fixed_seconds']+counts['scaled_block_seconds']+counts['additional_preparation_seconds'])*1.25+1800
    if time()+future>read(BASE/'budget.json')['deadline']:
        raise ValueError('Posted count choice no longer fits; no second reduction')
    for label in labels():
        for spec in pairs[label]:build_cache(spec,'final')
        opponent=label.split('-',2)[2] if label.startswith('panel-') else None
        run_pair(label,pairs[label],counts['panel'] if opponent else counts['direct'],roots[label]['final'],opponent)
        if not opponent:remove_candidate_cache(pairs[label][0],'final')
    put(BASE/'all-final-verified.json',dict(status='verified',at=time(),labels=labels(),source=revision()))
    readout(plan)


def verify_final_files():
    if read(BASE/'all-final-verified.json')['status']!='verified':
        raise ValueError('Global final verification barrier incomplete')
    coverage={}
    for label in labels():
        folder=BASE/'final'/label
        if (read(folder/'replay.json')['status']!='verified'
                or not read(folder/'reproduction/complete.json')['reproduced_all_hands_and_decisions']):
            raise ValueError('A final replay or reproduction is incomplete')
        for stage in ('play','reproduction'):
            for name,expected in read(folder/stage/'complete.json')['files'].items():
                p=folder/stage/name
                if p.stat().st_size!=expected['bytes'] or file_hash(p)!=expected['sha256']:
                    raise ValueError('Verified final bytes changed')
        coverage[label]=read(folder/'replay.json')['coverage']
    return coverage


def readout(plan):
    coverage=verify_final_files()
    summary=dict(panels={},direct={},coverage=coverage,status='verified',
        canonical_plan_sha256=file_hash(BASE/'canonical-plan.json'),all_final_replayed_and_reproduced=True)
    for seed in (2026100601,*SEEDS):
        folder=BASE/'final'/f'direct-{seed}'
        summary['direct'][seed]=estimate(series(folder,plan['counts']['direct'])['candidate'])
        if seed in SEEDS:
            panel={}
            for opponent in direct.PANEL_OPPONENTS:
                arms=series(BASE/'final'/f'panel-{seed}-{opponent}',plan['counts']['panel'],panel=True)
                panel[opponent]=dict(arms={a:estimate(v) for a,v in arms.items()},
                    contrast=estimate([a-b for a,b in zip(arms['candidate'],arms['baseline'],strict=True)]))
            outcome,flags=recommendation(panel)
            summary['panels'][seed]=dict(opponents=panel,historical_pr272_rule_outcome=outcome,
                historical_upper_below_minus20_flags=flags,release_pass_fail_gate=False)
    put(BASE/'summary.json',summary)
    # Caches are this campaign's own regenerable outputs. Every final dependency
    # has finished before removal; canonical averages and all evidence survive.
    manifests={str(p.relative_to(CACHE)):read(p) for p in CACHE.rglob('manifest.json')}
    put(BASE/'cache-removal.json',dict(at=time(),path=str(CACHE),manifests=manifests,
        all_consumers_complete=True,rebuild='AveragePolicy with indexed averages, schemas, K50 tables and sorted_rows=True',
        canonical_models_and_raw_evidence_retained=True))
    shutil.rmtree(CACHE)


def archive_seed(seed):
    dest=Path.home()/f'Local/Research-Cloud/PR-{PR}-hu100-v053-three-seeds'
    dest.mkdir(parents=True,exist_ok=True)
    files=[(p,'research/'+p.relative_to(BASE).as_posix()) for p in sorted((BASE/str(seed)).rglob('*'))
        if p.is_file() and p.name!='5000000000.average.gz']
    for kind in ('pilot','final'):
        for label in labels():
            if label.split('-')[1]!=str(seed):continue
            files += [(p,'research/'+p.relative_to(BASE).as_posix()) for p in sorted((BASE/kind/label).rglob('*')) if p.is_file()]
    average=BASE/str(seed)/'5000000000.average.gz'
    additional=sum(p.stat().st_size for p,_ in files)+average.stat().st_size+512*1024**2
    require_disk(additional)
    metadata=dict(pr=PR,source=revision(),seed=seed,summary_sha256=file_hash(BASE/'summary.json'),
        average_export=dict(member=f'averages/{seed}/5000000000.average.gz',
            archive=f'hu100-v053-seed-{seed}-average.zip',**pin(average)),
        shared_dependencies='shared-evidence ZIP, separate seed average ZIP, PR272 original average, PR163 K50 tables and pinned v0.5.2 release',
        originals_retained=True)
    put(BASE/str(seed)/'archive-receipt.json',seal(files,dest/f'hu100-v053-seed-{seed}-evidence.zip',metadata))
    put(BASE/str(seed)/'average-archive-receipt.json',seal([(average,f'averages/{seed}/5000000000.average.gz')],
        dest/f'hu100-v053-seed-{seed}-average.zip',metadata))


def run():
    for stage in ('prepare','evaluate'):
        command('evaluation-'+stage,[sys.executable,'-m','scripts.evaluate_hu100_v053_seeds',stage])
    # Archive admission is a coordinator handshake with fresh current owning-PR
    # metadata. The user's explicit evidence request authorizes this open PR's
    # archive, while every other open root remains untouched.
    put(BASE/'archive-ready.json',dict(at=time(),source=revision()))
    while not (BASE/'archive-owner.json').exists():
        if time()>=read(BASE/'budget.json')['deadline']-900:
            raise ValueError('Archive ownership confirmation missing before closeout')
        sleep(5)
    owner=read(BASE/'archive-owner.json')
    if owner['pr']['number']!=PR or not 0<=time()-owner['at']<600:
        raise ValueError('Fresh current owning-PR metadata required for archive')
    for seed in SEEDS:
        # Archive is part of the original budget; leave lifecycle evidence for
        # the final shared bundle after the outer guard has closed.
        command(f'archive-{seed}',[sys.executable,'-m','scripts.evaluate_hu100_v053_seeds','archive','--seed',seed])
    put(BASE/'seed-archives-complete.json',dict(at=time(),seeds=SEEDS))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('run','prepare','evaluate','pair','archive'))
    p.add_argument('--folder',type=Path);p.add_argument('--specs',type=Path)
    p.add_argument('--blocks',type=int);p.add_argument('--root',type=int)
    p.add_argument('--opponent');p.add_argument('--pilot',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS)
    args=p.parse_args()
    if args.stage=='pair':
        pair_worker(args.folder,read(args.specs),args.blocks,args.root,args.opponent,args.pilot)
    elif args.stage=='archive':archive_seed(args.seed)
    else:globals()[args.stage]()


if __name__=='__main__':main()
