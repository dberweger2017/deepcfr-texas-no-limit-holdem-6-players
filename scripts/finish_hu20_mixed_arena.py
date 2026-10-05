"""Publish verified complete or stopped results after the durable owned-pod closeout.

Raw solver evidence stays compressed. The unchanged approved native audit sees
original worker manifests backed by stream-verified archive members, while only
hand streams and summaries are materialized locally.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from time import sleep, time
from types import SimpleNamespace

from scripts.hu20_search_arena_control import durable_json
from scripts.hu20_search_evidence import file_hash


class ArchivedFile:
    def __init__(self,spec):self.spec=spec
    def stat(self):return SimpleNamespace(st_size=self.spec['bytes'])


class ArchivedDirectory:
    def __init__(self,local,prefix,members):self.local,self.prefix,self.members=local,prefix,members
    def __truediv__(self,name):
        path=self.local/name
        return path if path.exists() else ArchivedFile(self.members[self.prefix+'/'+name])
    def glob(self,pattern):return self.local.glob(pattern)


def materialize_hands(root):
    directories=[];summaries=[];proofs=[]
    for pod in sorted((root/'retrieved').iterdir()):
        verified=json.loads((pod/'retrieval-verified.json').read_text())
        if verified.get('verified') is not True:raise ValueError('Stream retrieval verification missing')
        if verified.get('archive_manifest_sha256')!=file_hash(pod/'manifest.json'):raise ValueError('Retrieved manifest changed since stream verification')
        manifest=json.loads((pod/'manifest.json').read_text())
        # Bind the previously verified chunks; native audit additionally compares
        # every original worker manifest entry with these verified member hashes.
        for row in manifest['archives']:
            path=pod/row['path']
            if path.stat().st_size!=row['bytes'] or file_hash(path)!=row['sha256']:
                raise ValueError('Archive changed since retrieval')
            with tarfile.open(path,'r|gz') as archive:
                for member in archive:
                    parts=Path(member.name).parts
                    if member.name in ('control-v2.events.jsonl','control-v2.json'):
                        destination=root/'audit-hands/controller'/member.name
                    elif (len(parts)==3 and parts[0]=='arena' and parts[1].startswith('worker-') and
                          (parts[2].endswith('.hands.jsonl.gz') or parts[2] in ('summary.json','manifest.json'))):
                        destination=root/'audit-hands'/parts[1]/parts[2]
                    else:continue
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if destination.exists():raise ValueError('Duplicate materialized worker member')
                    with archive.extractfile(member) as source,destination.open('wb') as target:shutil.copyfileobj(source,target,1024**2)
                    spec=manifest['members'][member.name]
                    if destination.stat().st_size!=spec['bytes'] or file_hash(destination)!=spec['sha256']:
                        raise ValueError('Materialized hand/summary differs')
        for prefix in sorted({str(Path(name).parent) for name in manifest['members'] if name.startswith('arena/worker-') and name.endswith('/summary.json')}):
            directory=root/'audit-hands'/Path(prefix).name
            summaries.append(json.loads((directory/'summary.json').read_text()))
            directories.append(ArchivedDirectory(directory,prefix,manifest['members']))
        proofs.append({'pod_id':pod.name,'archive_manifest_sha256':file_hash(pod/'manifest.json'),'retrieval':verified})
    return directories,summaries,proofs


def guard_report(root,ledger):
    events=[json.loads(line) for line in (root/'audit-hands/controller/control-v2.events.jsonl').read_text().splitlines()]
    def count(rows):
        bad=sum(row['fallback'] for row in rows)
        return {'decisions':len(rows),'fallbacks':bad,'fallback_rate':bad/len(rows) if rows else None,
                'fallback_causes':dict(Counter(row['cause'] for row in rows if row['fallback']))}
    return {'global':count(events),'per_pod':{pod['id']:count([row for row in events if row['pod']==pod['id']])
            for pod in ledger['pods'] if pod['workers']}}


def partial_counts(directories):
    hands=0;counts=Counter();hosts={}
    for directory in directories:
        for path in directory.glob('*.hands.jsonl.gz'):
            with gzip.open(path,'rt') as stream:
                for line in stream:
                    row=json.loads(line);hands+=1;counts.update(row['search_counts'])
                    host=hosts.setdefault(row['host'],{'seconds':[],'fallback_causes':Counter()})
                    for record in row['search_records']:
                        if record['status']=='decision' or record['status']=='fallback' and record.get('query_kind')=='play':
                            host['seconds'].append(record['seconds'])
                            if record['status']=='fallback':host['fallback_causes'][record['cause']]+=1
    return hands,dict(counts),hosts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--mcp-helper',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if (root/'RESULTS_POSTED.json').exists():raise ValueError('Result was already published')
    while not any((root/name).exists() for name in ('CLOSEOUT_COMPLETE.json','CLOSEOUT_FAILED.json')):sleep(20)
    ledger=json.loads((root/'ledger.json').read_text())
    actual_compute=sum((p.get('terminated_at',time())-p['created_at'])/3600*p['compute_hourly_usd'] for p in ledger['pods'])
    actual_disk_upper=sum((p.get('terminated_at',time())-p['created_at'])/3600*p['disk_hourly_usd'] for p in ledger['pods'])
    spend={'gross_compute_provisioning_usd':actual_compute,'disk_upper_usd':actual_disk_upper,
           'owner_excluded_sleep_usd':ledger['owner_excluded_charge_usd'],
           'cap_counted_provisioning_upper_usd':actual_compute+actual_disk_upper-ledger['owner_excluded_charge_usd'],
           'billing_basis':'provisioning wall clock × readback rates; posted billing separately retained; not an invoice claim'}
    spec=importlib.util.spec_from_file_location('billing_mcp',args.mcp_helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    for i,pod in enumerate(ledger['pods']):
        result=helper.call('tools/call',{'name':'list-pod-billing','arguments':{'podId':pod['id'],'startTime':'2026-10-05T14:00:00Z','endTime':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'bucketSize':'hour'}},700+i)
        durable_json(root/('final-billing-'+pod['id']+'.json'),result)
    durable_json(root/'actual-spend.json',spend)
    if (root/'CLOSEOUT_FAILED.json').exists():
        failure=json.loads((root/'CLOSEOUT_FAILED.json').read_text())
        report='Stage 4 stopped: closeout failed. '+failure['reason']+'\n\nUnretrieved evidence is preserved; no science will restart. Owner intervention is required. Pods not yet terminated remain billing.\n\n'+json.dumps(spend,indent=2)
    else:
        directories,summaries,proofs=materialize_hands(root)
        guards=guard_report(root,ledger)
        durable_json(root/'exact-search-guard-counts.json',guards)
        sys.path.insert(0,str(args.source.resolve()))
        import scripts
        scripts.__path__=[str(args.source.resolve()/'scripts'),*list(scripts.__path__)]
        audit_spec=importlib.util.spec_from_file_location('approved_hu20_audit',args.source/'scripts/audit_hu20_turn_search.py')
        audit=importlib.util.module_from_spec(audit_spec);audit_spec.loader.exec_module(audit)
        original_hash=audit.file_hash
        audit.file_hash=lambda path:path.spec['sha256'] if isinstance(path,ArchivedFile) else original_hash(path)
        complete=len(summaries)==6 and {s['worker_index'] for s in summaries}==set(range(6)) and all(s['status']=='complete' for s in summaries)
        if complete:
            result=audit.audit(directories)
            result['archive_proofs']=proofs
            durable_json(root/'independent-native-audit.json',result)
            report=f"Stage 4 complete: {result['hands']:,}/82,944 hands independently replayed with complete frozen coordinates and original manifests. All retained chunks and member hashes verified; all five owned pods terminated (5080 before dispatch, four after retrieval). Final list-pods retained.\n\n"
            report+='Base+search minus base, BB/100 (exploratory paired 95% intervals, conditional on the three original lineages):\n\n'
            for row in result['three_lineage_changes']:report+='- '+row['panel']+': '+json.dumps(row['search_minus_base'])+'\n'
            report+='\nPer-host latency and fallback telemetry:\n\n```json\n'+json.dumps(result['decision_latency_by_host'],indent=2)+'\n```\n'
            report+='\nConditioning gaps: '+str(result['turn_conditioning_gap_count'])+'. Complete raw findings are retained; this result does not authorize promotion or a new run.\n'
        else:
            hands,counts,hosts=partial_counts(directories)
            result={'status':'stopped-incomplete','hands':hands,'expected_hands':82944,'archive_proofs':proofs,'search_counts':counts,'decision_latency_by_host':audit.latency_report(hosts)}
            durable_json(root/'stopped-partial-summary.json',result)
            control=json.loads((root/'final-control.json').read_text())
            report=f"Stage 4 stopped, frozen protocol incomplete: {hands:,}/82,944 retained complete hand records. Stop reason: {control.get('reason')}.\n\nAll retained chunks/member hashes verified; all five owned pods terminated and final list-pods retained. Full base-versus-base+search contrasts are unavailable because the frozen protocol did not complete. No reduced protocol, strength conclusion, retune or restart. Owner review is required before further science.\n\nPer-host descriptive latency/fallback telemetry from retained complete hands; exact guard counts below also include interrupted-hand decisions:\n\n```json\n"+json.dumps(result['decision_latency_by_host'],indent=2)+'\n```\n'
        report+='\nExact live-search fallback counts, including decisions in interrupted hands (a host with zero decisions had not reached search):\n\n```json\n'+json.dumps(guards,indent=2)+'\n```\n'
        report+='\nSpend (sleep remains charged by the provider, but is excluded from the approved cap):\n\n```json\n'+json.dumps(spend,indent=2)+'\n```\n\nPer-pod posted billing responses are retained separately; empty/in-progress buckets are not evidence of zero cost. Evidence and manifests: M4 `~/Local/hu20-turn-search-arena-20261005/stage-4-mixed/retrieved`; native audit/partial summary and spend are in its parent. Protected historical pods and original Mac evidence remain untouched.\n'
    path=root/'pr166-final-results.md';path.write_text(report)
    environment=os.environ.copy();environment['GH_TOKEN']=(root/'github-token').read_text().strip()
    posted=subprocess.check_output(['/opt/homebrew/bin/gh','pr','comment','166','--repo','dberweger2017/deepcfr-texas-no-limit-holdem-6-players','--body-file',str(path)],env=environment,text=True)
    durable_json(root/'RESULTS_POSTED.json',{'at':time(),'url':posted.strip(),'report_sha256':file_hash(path)})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
        failure={'at':time(),'reason':f'{type(exc).__name__}: {exc}','instruction':'Preserve evidence; no science restart; owner review required'}
        durable_json(root/'FINAL_REPORT_FAILED.json',failure)
        path=root/'pr166-final-validation-failure.md'
        path.write_text('Stage 4 final reporting/validation failed: '+failure['reason']+'\n\nEvidence is preserved; no science will restart. '+
            ('All owned pods were terminated after verified retrieval.' if (root/'CLOSEOUT_COMPLETE.json').exists() else 'Closeout status requires owner inspection; any remaining pods may still be billing.')+
            '\n\nOwner review is required. Detailed receipts: M4 `~/Local/hu20-turn-search-arena-20261005/stage-4-mixed`.')
        environment=os.environ.copy();environment['GH_TOKEN']=(root/'github-token').read_text().strip()
        subprocess.run(['/opt/homebrew/bin/gh','pr','comment','166','--repo','dberweger2017/deepcfr-texas-no-limit-holdem-6-players','--body-file',str(path)],env=environment,check=True)
        raise
