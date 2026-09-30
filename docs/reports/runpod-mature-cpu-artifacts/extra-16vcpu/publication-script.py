import json,hashlib,time,shutil,sys,datetime
from pathlib import Path
base=Path('/Users/dberweger/Local/runpod-mature-cpu-six-20260930')
sys.path.insert(0,str(base))
from scripts.mature_cpu_rental_guard import api
root=base/'results/runpod-mature-cpu-16vcpu-20260930'
out=base/'results/mature-cpu-16vcpu-publication'
out.mkdir(exist_ok=False)
def save(name,value):(out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
operator=json.loads((root/'operator-finished.json').read_text())
assert operator['status']=='complete' and not operator['remaining_owned_ids']
assert json.loads((root/'watchdog.json').read_text())['no_owned_pods']
rows=json.loads((root/'pods.json').read_text());assert len(rows)==1
pod=rows[0];assert pod['status']=='verified'
work=root/'cpu5c/results/work'
verify=json.loads((root/'cpu5c/platform-verification.json').read_text());assert verify['passed']
worker=json.loads((work/'worker.json').read_text())
direct=json.loads((work/'direct/result.json').read_text());resumed=json.loads((work/'resumed/result.json').read_text())
resources=[json.loads(x) for x in (work/'resources.jsonl').read_text().splitlines()]
price=pod['total_rate'];created=datetime.datetime.fromisoformat(pod['created_at'].replace('Z','+00:00')).timestamp()
seconds=operator['finished']-created
ledger=api(Path('/tmp/doctor-research-mature-cpu-key.toml'),'/v2/billing/pods?podId='+pod['id']+'&startTime=2026-09-30T00%3A00%3A00Z&endTime=2026-10-01T00%3A00%3A00Z&bucketSize=day')
owned=[{'id':p['id'],'name':p['name']} for p in api(Path('/tmp/doctor-research-mature-cpu-key.toml'),'/v2/pods')['pods'] if p['name'].startswith('doctor-research-mature-')]
assert not owned
save('billing.json',{'pod_id':pod['id'],'compute_upper_estimate_usd':seconds*price/3600,'latest_possible_charge_seconds':seconds,'rate_usd_per_hour':price,'initial_compute_upper_estimate_usd':.06886513233568933,'combined_compute_upper_estimate_usd':.06886513233568933+seconds*price/3600,'ledger':ledger,'queried':time.time(),'invoice_settled':False,'posted_records_available':bool(ledger.get('records')),'owned_pods_remaining':owned})
metrics={'direct':direct,'resumed':resumed,'worker':worker,'extra_control':True,'single_trial':True,'resource_summary':{'samples':len(resources),'peak_owned_rss_bytes':max(x['owned_rss_bytes'] for x in resources),'peak_cgroup_bytes':max(x['cgroup_memory_peak_bytes'] for x in resources),'maximum_swap_growth_bytes':max(x['swap_growth_bytes'] for x in resources),'minimum_free_disk_bytes':min(x['free_disk_bytes'] for x in resources)},'direct_process_wall_seconds':worker['attempts'][0]['finished']-worker['attempts'][0]['started'],'resumed_process_wall_seconds':worker['attempts'][1]['finished']-worker['attempts'][1]['started'],'training_usd_per_million_nodes':price*1000000/(3600*direct['nodes_per_second']),'all_worker_compute_usd_per_unique_million_nodes':price*(worker['finished']-worker['started'])/3600/(direct['added_nodes']/1000000),'one_worker_400M_training_hours':400000000/direct['nodes_per_second']/3600,'three_lineage_training_compute_usd':price*1200000000/direct['nodes_per_second']/3600}
save('measurements.json',metrics)
for name in ['worker.json','resume-verification.json','lscpu.txt','cpu-topology.txt','cpu.cfs_quota_us.txt','cpu.cfs_period_us.txt']:shutil.copyfile(work/name,out/name)
for name in ['lease.json','pods.json','plan.json','operator-finished.json','watchdog.json']:shutil.copyfile(root/name,out/name)
for name in ['platform-verification.json','transport-verification.json']:shutil.copyfile(root/'cpu5c'/name,out/name)
shutil.copyfile(root/'cpu5c/results/work-manifest.json',out/'work-manifest.json')
shutil.copyfile(base/'results/prelaunch/single-16vcpu-focused-tests.log',out/'focused-tests.log')
shutil.copyfile(Path(__file__),out/'publication-script.py')
print(json.dumps({'nodes_per_second':direct['nodes_per_second'],'wall':metrics['direct_process_wall_seconds'],'training_usd_per_million_nodes':metrics['training_usd_per_million_nodes'],'resource_summary':metrics['resource_summary'],'combined_compute_upper_estimate_usd':.06886513233568933+seconds*price/3600,'ledger_records':len(ledger.get('records',[]))}))
# This phase logs outside these roots; the closed coordinator/worker logs are sealed.
files=[p for r in [root,out] for p in sorted(r.rglob('*')) if p.is_file() and p.name not in ['ssh-key','ssh-key.pub','known-hosts']]
files.append(Path('/Users/dberweger/Local/runpod-mature-cpu-16vcpu-20260930-coordinator.log'))
records={}
for p in files:
 if time.time()>=1790809835.502777 or shutil.disk_usage(base).free<8*2**30:raise RuntimeError('Guard during final seal')
 before=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 records[str(p.relative_to(base)) if p.is_relative_to(base) else str(p)]={'bytes':after.st_size,'sha256':h.hexdigest()}
p=base/'results/mature-cpu-16vcpu-final-manifest.json'
p.write_text(json.dumps({'root':str(base),'files':records,'sealed':time.time(),'original_cutoff':1790809835.502777,'excluded':'Ephemeral SSH keys and known-hosts; provider key outside artifact roots'},sort_keys=True,indent=2)+'\n')
(base/'results/mature-cpu-16vcpu-seal.json').write_text(json.dumps({'files':len(records),'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sealed':time.time(),'status':'complete'})+'\n')
