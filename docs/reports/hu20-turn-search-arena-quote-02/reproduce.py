"""Reproduce the outcome-blind quote using retained pilot bytes and catalog reads.

Run from the repository root with PYTHONPATH=. and --pilot pointing to evidence/timing.
The historical timestamp reproduces the quote; approval still needs fresh admission.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from math import ceil
from pathlib import Path

from scripts.freeze_hu20_turn_search import arena_plan
from scripts.quote_hu20_search_arena import quote
from src.arena.schedule import digest
from src.blueprint.hu20_turn_solver import file_hash, sampled_profile, PROFILE_RETENTION_RULE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).parent
    parent=here.parent/'hu20-turn-search-artifacts'
    original=json.loads((parent/'calibration-03-result.json').read_text())
    selected=next(c for c in original['full_curve'] if c['config']['iterations']==50
                  and c['config']['opponent_likelihood_floor']==0)
    assert selected['roots']==288 and selected['complete']
    assert selected['relaxed_root_ratio_violations']==selected['fallbacks']==0
    assert selected['cold_p95_seconds']<=30
    calibration={'status':'qualified','qualification':'owner-approved relative gate amendment',
        'owner_decision_url':'https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/148#issuecomment-5993910394',
        'original_result_sha256':digest(original),'original_strict_gate_passed':False,
        'original_absolute_relaxed_gate_passed':False,'selected':selected}
    plan=arena_plan(json.loads(Path('configs/diagnostics/hu20-turn-search-part-a.json').read_text()),
                    json.loads((parent/'part-a-result.json').read_text()),calibration)
    retention_policy={'profile_rule':PROFILE_RETENTION_RULE,
        'selector':'int(SHA256(final request.json bytes),16) < floor(2**256/100)',
        'sample_fraction_of_hash_space':.01,
        'retain_all':['hands','requests','receipts','responses','logs','manifests','failures'],
        'profile_hashes':'durably recorded with exact body size in every solve receipt before deletion',
        'scope':'all solves on newly created production pods; parity bodies follow the same rule after comparison; existing Mac/pilot/calibration evidence stays unchanged',
        'retrieval':'lossless gzip archives; stream member SHA/size verification without full extraction',
        'deletion':'only unsampled profile.jsonl bodies on owned production pods, after process exit and matrix parsing',
        'sample_count':'one percent of hash space, not a forced one-percent count; no outcome-driven additions or resampling'}
    plan['evidence_retention']=retention_policy
    blocks=defaultdict(list)
    inputs={}
    counts=Counter()
    statuses=Counter()
    for worker in sorted(args.pilot.glob('worker-[0-9]')):
        summary=json.loads((worker/'summary.json').read_text())
        assert summary['status']=='complete' and summary['hands']==312
        assert summary['plan']['selected_search_config']==selected['config']
        manifest=json.loads((worker/'manifest.json').read_text())
        for path in [worker/'summary.json',worker/'manifest.json',*sorted(worker.glob('*.hands.jsonl.gz'))]:
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            inputs[str(path.relative_to(args.pilot))]={'sha256':sha,'bytes':path.stat().st_size}
            if path.name.endswith('.hands.jsonl.gz'):
                assert manifest[path.name]=={'sha256':sha,'bytes':path.stat().st_size}
                with gzip.open(path,'rt') as stream:
                    for line in stream:
                        row=json.loads(line)
                        # Payoffs and action outcomes never enter sizing.
                        blocks[row['panel'],row['block']].append(
                            (row['seconds'],row['seed'],row['arm'],row['rotation']))
                        counts.update(row.get('search_counts',{}))
                        statuses.update(r['status'] for r in row.get('search_records',[]))
    panels={}
    for panel in plan['panels']:
        samples=[values for (name,_),values in blocks.items() if name==panel['name']]
        assert len(samples)==8
        assert all(len(v)==12 and len({(s,a,r) for _,s,a,r in v})==12 for v in samples)
        seconds=sorted(sum(x[0] for x in v) for v in samples)
        panels[panel['name']]={'paired_blocks':8,'seconds_per_joint_block_samples':seconds,
            'seconds_per_joint_block_p95':seconds[int(.95*len(seconds))],
            'seconds_per_joint_block_mean':sum(seconds)/len(seconds)}
    profile_inventory=[]
    for request in sorted(args.pilot.rglob('request.json')):
        profile=request.parent/'profile.jsonl'
        request_hash=file_hash(request)
        item={'request_path':str(request.relative_to(args.pilot)),
              'request_sha256':request_hash,'sample_selected':sampled_profile(request_hash),
              'profile_generated':profile.exists(),
              'profile_bytes':profile.stat().st_size if profile.exists() else 0,
              'profile_sha256':file_hash(profile) if profile.exists() else None}
        if profile.exists():
            manifest=json.loads((request.parent/'manifest.json').read_text())
            assert manifest['profile.jsonl']=={'bytes':item['profile_bytes'],'sha256':item['profile_sha256']}
        profile_inventory.append(item)
    total_bytes=sum(p.stat().st_size for p in args.pilot.rglob('*') if p.is_file())
    profile_bytes=sum(i['profile_bytes'] for i in profile_inventory)
    selected_bytes=sum(i['profile_bytes'] for i in profile_inventory if i['sample_selected'])
    scale=plan['expected_hands']/1248
    storage={'pilot_total_bytes':total_bytes,'pilot_profile_bytes':profile_bytes,
        'pilot_nonprofile_bytes':total_bytes-profile_bytes,
        'profile_share':profile_bytes/total_bytes,
        'pilot_fixed_sample_profile_bytes':selected_bytes,
        'pilot_fixed_sample_retained_bytes':total_bytes-profile_bytes+selected_bytes,
        'arena_hand_count_scale':scale,
        'arena_retained_raw_bytes_at_one_percent':(total_bytes-profile_bytes+.01*profile_bytes)*scale,
        'arena_retained_raw_bytes_fixed_pilot_sample':(total_bytes-profile_bytes+selected_bytes)*scale,
        'compressed_measurement':json.loads((here/'compression-measurement.json').read_text()),
        'archive_guard_bytes_total':4*10**9,'3090_pod_disk_gb':60,'cpu5_pod_disk_gb':80,'pod_disk_free_floor_gb':10,
        'required_storage_gb_per_worker':10,
        'disk_sizing':'raw retained forecast x2 plus 4 GB inputs/build scratch, transient profiles, archive output and 10 GB free floor; 20 GB container + 40 GB volume',
        'retrieval_destination':'compressed archives only; <=1 GB transfer chunks, streaming SHA verification; no complete raw expansion on either Mac'}
    assert storage['compressed_measurement']['retained_raw_bytes']==total_bytes-profile_bytes+selected_bytes
    storage['arena_compressed_projection_bytes']=storage['compressed_measurement']['gzip_tar_bytes']*scale
    reserves={'setup_build':1800,'actual_pod_parity':900,'replay_verification':3600,
              'retrieval_hash_verification':900,'shutdown':600,'retention_finalization':300}
    timing={'configuration_sha256':digest(selected['config']),
        'includes_preparation':True,'includes_parsing':True,'includes_lbr_speculative_solves':True,
        'includes_base_and_search_arms':True,'panels':panels,
        'reserves_seconds_per_worker':reserves,'required_storage_gb_per_worker':10,
        'rss_limit_bytes_per_worker':12*1024**3,'inputs':inputs,
        'search_counts':dict(counts),'record_statuses':dict(statuses),
        'pilot_concurrency':{'workers_per_pod':4,'threads_per_worker':6,'cpu_quota':23.8},
        'per_worker_latency_multiplier':1.0,
        'scaling_basis':'No unmeasured latency speedup; repartition all blocks over the reduced per-pod concurrency.',
        'timing_retained_bytes':total_bytes,'evidence_retention':retention_policy,
        'storage_forecast':storage,'profile_inventory_sha256':digest(profile_inventory)}
    reads={k:json.loads((here/('mcp-'+k+'-offer.json')).read_text()) for k in ('3090','cpu5')}
    now=max(r['retrieved_at'] for r in reads.values())+1
    quotes={}
    for key,workers,per_pod in [('3090',9,3),('cpu5',8,4)]:
        raw=reads[key]
        assert not raw['response']['result'].get('isError')
        data=json.loads(raw['response']['result']['content'][0]['text'])
        disk_gb=60 if key=='3090' else 80
        offer={'retrieved_at':raw['retrieved_at'],'source_url':'https://mcp.getrunpod.io/',
            'provider':'RunPod','architecture':'x86_64','gpu':key=='3090','compute_workload':'cpu',
            'availability':data['availability'],'catalog_id':data['id'],'catalog_request':raw['arguments'],
            'catalog_read_sha256':digest(raw),'minimum_cpu_per_pod':23.8 if key=='3090' else 32,
            'reserved_cpu_per_pod':4,'minimum_ram_bytes_per_pod':(60 if key=='3090' else 64)*10**9,
            'compute_hourly_usd':data['price']['community'] if key=='3090' else 32*data['price']['securePerVcpu'],
            'container_disk_hourly_usd':disk_gb*.10/720,'storage_gb':disk_gb,
            'storage_price_source':'https://www.runpod.io/pricing',
            'storage_price_basis':f'{disk_gb} GB total pod-local disk at $0.10/GB/month, conservatively divided by 720 hours',
            'retained_storage_reserve_usd':1.0,
            'resource_values':'Required actual-host admission minima, not promises made by the catalog.'}
        q=quote(plan,calibration,timing,offer,workers=workers,workers_per_pod=per_pod,price_only=True,now=now)
        mean_seconds=sum(ceil(p['blocks']/workers)*p['timing']['seconds_per_joint_block_mean']
                         for p in q['panel_forecasts'])
        mean_with_headroom_seconds=ceil(1.5*(mean_seconds+sum(reserves.values())))
        mean_with_headroom_cost=ceil((q['pods']*mean_with_headroom_seconds/3600
            *(offer['compute_hourly_usd']+offer['container_disk_hourly_usd'])+1)*100)/100
        q.update(mean_block_campaign_cost_with_1_5_headroom_usd=round(mean_with_headroom_cost+.5,2),
                 quote_timestamp_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),
                 prior_pilot_budget_reserve_usd=.50,
                 campaign_maximum_cost_usd=round(q['maximum_cost_usd']+.50,2),
                 campaign_expected_cost_usd=round(q['expected_cost_usd']+.50,2),
                 approved_ceiling_usd=25,
                 within_approved_ceiling=q['maximum_cost_usd']+.50<=25)
        quotes[key]=q
    result={'revision':2,'supersedes_commit':'a050002','status':'quote-awaiting-owner-approval','owner_approved':False,
        'arena_plan':plan,'amended_calibration':calibration,'timing':timing,'quotes':quotes,
        'stop_rule':{'first_checkpoint_search_decisions':500,'maximum_fallback_rate':.05,
            'first_checkpoint_halt_at_fallbacks':26,'scope':'globally and independently on each pod',
            'subsequent_checkpoints':'every additional 100 live search decisions; cumulative and trailing 500',
            'denominator':'target live turn/river search decisions, excluding preflop/flop blueprint queries and speculative LBR probes',
            'numerator':'any live search decision using base fallback, including cached/follow-on timeout fallbacks',
            'action':'stop dispatch, cancel owned workers, retain partials, retrieve and verify evidence, terminate only newly owned pods; no restart or retune without owner approval'},
        'spend_stop':{'campaign_ceiling_usd':25,'production_closeout_reserve_usd':4.0,
            'stop_dispatch_at_campaign_upper_charge_usd':21.0,
            'charge_basis':'wall-clock accrued compute plus provisioned disk at admitted rates, pilot reserve and any failure/restart costs',
            'additional_limit':'each pod deadline includes the full quoted worker_seconds; closeout starts before its priced reserves are consumed'}}
    (out/'profile-inventory.json').write_text(json.dumps(profile_inventory,indent=2,sort_keys=True)+'\n')
    (out/'quote.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    for k,q in quotes.items():
        print(k,{field:q[field] for field in ('pods','workers','workers_per_pod','maximum_worker_hours','maximum_cost_usd','campaign_maximum_cost_usd','campaign_expected_cost_usd','within_approved_ceiling','stock_available')})


if __name__=='__main__':main()
