import json, time, shutil, tarfile, hashlib, sys
from pathlib import Path
base=Path('/Users/dberweger/Local/runpod-mature-cpu-six-20260930')
sys.path.insert(0,str(base))
from scripts.mature_cpu_rental_guard import api
root=base/'results/runpod-mature-cpu-six-20260930-attempt-2'
first=base/'results/runpod-mature-cpu-six-20260930'
console=base/'results/runpod-mature-cpu-six-20260930-console-cpu3'
out=base/'results/mature-cpu-publication'
out.mkdir(exist_ok=False)
def save(name,value):
 (out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
# Billing is restricted to this pilot's six actual allocations.
pods=[]
for attempt in [first,root]:
 for p in json.loads((attempt/'pods.json').read_text()):
  if 'id' in p: pods.append((attempt,p))
billing=[]
for attempt,p in pods:
 try:
  ledger=api(Path('/tmp/doctor-research-mature-cpu-key.toml'),'/v2/billing/pods?podId='+p['id']+'&startTime=2026-09-30T00%3A00%3A00Z&endTime=2026-10-01T00%3A00%3A00Z&bucketSize=day')
 except Exception as e: ledger={'unavailable':str(e)}
 created=p.get('created_at',p.get('attempted'))
 if isinstance(created,str):
  import datetime
  created=datetime.datetime.fromisoformat(created.replace('Z','+00:00')).timestamp()
 ended=json.loads((attempt/'operator-finished.json').read_text())['finished']
 billing.append({'attempt':attempt.name,'pod_id':p['id'],'cpu_id':p['cpu_id'],'rate_usd_per_hour':p['total_rate'],'latest_possible_charge_seconds':ended-created,'compute_upper_estimate_usd':(ended-created)*p['total_rate']/3600,'ledger':ledger})
save('billing.json',{'queried':time.time(),'pods':billing,'lifetime_compute_upper_estimate_usd':sum(x['compute_upper_estimate_usd'] for x in billing)})
remaining=api(Path('/tmp/doctor-research-mature-cpu-key.toml'),'/v2/pods')['pods']
owned=[{'id':p['id'],'name':p['name']} for p in remaining if p['name'].startswith('doctor-research-mature-')]
assert not owned
save('absence.json',{'checked':time.time(),'owned_pods':owned,'owned_billable_storage':'30 GB disposable container disk only; destroyed with each pod; no persistent/network volumes requested'})
# Preserve only fully available small metadata from the truncated archive.
partial=root/'cpu5c/results.tar'
prefix=root/'cpu5c/preserved-prefix'
prefix.mkdir(exist_ok=False)
files=[]
try:
 with tarfile.open(partial,mode='r|') as stream:
  for member in stream:
   if member.isfile() and member.size < 1000000:
    content=stream.extractfile(member).read()
    if len(content)!=member.size: raise tarfile.ReadError('Truncated member')
    path=prefix/member.name
    if not path.resolve().is_relative_to(prefix): raise ValueError('Unsafe member')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(content)
    files.append({'name':member.name,'bytes':member.size,'sha256':hashlib.sha256(content).hexdigest()})
except (tarfile.ReadError,OSError) as error:
 truncation=str(error)
else: truncation='No tar parser truncation found; transport checksum still unavailable'
save('partial-retrieval.json',{'archive_bytes':partial.stat().st_size,'transport_hash_available':False,'fully_preserved_small_members':files,'truncation':truncation,'complete_state_parity':'NOT VERIFIED','remote_exit':0,'cause':'Controller terminated rentals after metadata-verifier failure while cpu5c transfer was still in progress; retained original incomplete archive; no workload rerun'})
metrics={}
for flavor in ['cpu5g','cpu5m']:
 folder=root/flavor
 work=folder/'results/work'
 direct=json.loads((work/'direct/result.json').read_text())
 resumed=json.loads((work/'resumed/result.json').read_text())
 worker=json.loads((work/'worker.json').read_text())
 resources=[json.loads(line) for line in (work/'resources.jsonl').read_text().splitlines()]
 verify=folder/('platform-verification-repaired.json' if flavor=='cpu5m' else 'platform-verification.json')
 assert json.loads(verify.read_text())['passed']
 price=next(p['total_rate'] for _,p in pods if p['id']==next(p['id'] for p in json.loads((root/'pods.json').read_text()) if p['cpu_id']==flavor))
 metrics[flavor]={'direct':direct,'resumed':resumed,'worker':worker,'resource_summary':{'samples':len(resources),'peak_owned_rss_bytes':max(x['owned_rss_bytes'] for x in resources),'peak_cgroup_bytes':max(x['cgroup_memory_peak_bytes'] for x in resources),'maximum_swap_growth_bytes':max(x['swap_growth_bytes'] for x in resources),'minimum_free_disk_bytes':min(x['free_disk_bytes'] for x in resources)},'direct_process_wall_seconds':worker['attempts'][0]['finished']-worker['attempts'][0]['started'],'resumed_process_wall_seconds':worker['attempts'][1]['finished']-worker['attempts'][1]['started'],'training_usd_per_million_nodes':price*1000000/(3600*direct['nodes_per_second']),'all_worker_compute_usd_per_unique_million_nodes':price*(worker['finished']-worker['started'])/3600/(direct['added_nodes']/1000000),'one_worker_400M_training_hours':400000000/direct['nodes_per_second']/3600,'three_lineage_training_compute_usd':price*1200000000/direct['nodes_per_second']/3600}
 target=out/flavor;target.mkdir()
 for name in ['worker.json','resume-verification.json','lscpu.txt','cpu-topology.txt','cpu.cfs_quota_us.txt','cpu.cfs_period_us.txt']:
  shutil.copyfile(work/name,target/name)
 shutil.copyfile(verify,target/'platform-verification.json')
 shutil.copyfile(folder/'transport-verification.json',target/'transport-verification.json')
 shutil.copyfile(work/'direct/result.json',target/'direct.json')
 shutil.copyfile(work/'resumed/result.json',target/'resumed.json')
 shutil.copyfile(folder/'results/work-manifest.json',target/'work-manifest.json')
save('measurements.json',metrics)
for origin,label in [(first,'startup-attempt-1'),(root,'training-attempt-2'),(console,'console-not-launched')]:
 target=out/label;target.mkdir()
 for name in ['lease.json','plan.json','pods.json','operator-finished.json','watchdog.json']:
  if (origin/name).exists():shutil.copyfile(origin/name,target/name)
# Original failed verification is retained independently of the repaired sidecar.
shutil.copyfile(root/'cpu5m/platform-verification.json',out/'original-metadata-verifier-failure.json')
shutil.copyfile(root/'publication-verification/campaign.json',out/'publication-verification.json')
shutil.copyfile(base/'results/prelaunch/reporting-repair-tests.log',out/'focused-tests.log')
shutil.copyfile(Path(__file__),out/'publication-script.py')
save('publication-summary.json',{'status':'partial-six-class-comparison','verified_classes':['cpu5g','cpu5m'],'unverified_completed_workload':['cpu5c'],'allocation_rejected':['cpu3c','cpu3g','cpu3m'],'meaningful_training_state_divergence':False,'original_cutoff':1790809835.502777,'completed':time.time(),'no_new_training':True})
print(json.dumps({'metrics':{k:{n:v for n,v in x.items() if n not in ['direct','resumed','worker']} for k,x in metrics.items()},'billing_upper_estimate':sum(x['compute_upper_estimate_usd'] for x in billing),'ledger_counts':[len(x['ledger'].get('records',[])) for x in billing],'prefix_members':len(files),'truncation':truncation}))
