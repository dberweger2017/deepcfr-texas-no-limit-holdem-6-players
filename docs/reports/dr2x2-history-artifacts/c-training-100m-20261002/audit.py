from pathlib import Path
from collections import Counter
from dataclasses import asdict
import json,gzip,hashlib,tarfile,subprocess,time,resource,math
from scripts.hu20_platform_pilot import canonical
from scripts.dr2x2_control import verify_archive
from scripts.dr2x2_worker import export_fingerprint
from src.blueprint.artifact import load_training,save_training,export_policy
r=Path('/Users/dberweger/Local/dr2x2-c-campaign-20261001');out=r/'final-audit-attempt-2';out.mkdir(exist_ok=False)
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source=='e45b297cdab45a2f4eb70f0325415794883c2163'
assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip()
plan=json.loads((r/'plan.json').read_text());plan_sha=hashlib.sha256(canonical(plan)).hexdigest();assert plan_sha=='d6c01b4685dff805de93ac566a007778e065e9493b075656a7f279e93a24a36c'
def h(p):
 s=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):s.update(b)
 return s.hexdigest()
def write(p,d):p.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
start=time.time();results=[]
try:
 for pod in json.loads((r/'pods.json').read_text()):
  seed=pod['seed'];folder=out/str(seed);folder.mkdir();archive=r/'jobs/C'/str(seed)/'final.tar';assert pod['status']=='retrieved-complete' and pod.get('terminated') and pod['outer_exit']=='0'
  assert h(archive)==pod['archive_sha256'];archive_check=verify_archive(archive)
  with tarfile.open(archive,'r:') as t:
   def read(name):return json.load(t.extractfile('results/worker/'+name))
   result=read('training/result.json');assert result['status']=='complete' and not result.get('failure') and not result.get('discarded_nodes')
   assert result['plan_sha256']==plan_sha and result['parent']==dict(cell='C',seed=seed,completed_nodes=0,checkpoint_sha256='from-zero')
   parity=read('preflight-parity.json');assert parity['passed'] and all(parity['checks'].values())
   a=read('final-reload-a/result.json');b=read('final-reload-b/result.json');assert a==b and a['status']=='complete'
   assert a['iteration']==result['iteration'] and a['entries']==result['entries'] and a['completed_nodes']==result['completed_nodes']
   chain=hashlib.sha256(canonical(result['parent'])).hexdigest();counters=Counter();total=0;iterations=0;max_step=0
   with gzip.open(t.extractfile('training/iterations.jsonl.gz'.join(['results/worker/',''])),'rt') as stream:
    for line in stream:
     row=json.loads(line);iterations+=1;assert row['iteration']==iterations
     total+=row['nodes'];max_step=max(max_step,row['nodes']);assert total==row['lifetime_completed_nodes']
     assert row['nodes']==row['terminals']+sum(row['attempted_work']['nodes_by_street'].values())
     assert row['raw_traverser_visits']==sum(row['traverser_visits_by_street'].values())
     assert row['new_entries']==sum(row['new_entries_by_street'].values())
     stable={k:v for k,v in row.items() if k not in ('elapsed_seconds','replay_seconds','worker_rss_sum_bytes','work_chain_sha256')}
     chain=hashlib.sha256(bytes.fromhex(chain)+canonical(stable)).hexdigest();assert chain==row['work_chain_sha256']
     for field,value in stable.items():
      if type(value) in (int,float) and field not in ('iteration','entries','lifetime_completed_nodes'):counters[field]+=value
      elif isinstance(value,dict):
       for street,amount in value.items():
        if type(amount) in (int,float):counters[field+':'+street]+=amount
   assert iterations==result['iteration'] and total==result['completed_nodes'] and counters==Counter(result['cumulative_work']) and chain==result['work_chain_sha256']
   assert total>=100000000 and total-100000000<max_step and counters['new_entries']==result['entries']
   with t.extractfile('results/worker/training/saved.jsonl') as stream:saves=[json.loads(x) for x in stream]
   assert [s['requested_total_nodes'] for s in saves]==plan['recovery_totals']
   for saved in saves:
    ack=read('training/ack/'+saved['id']+'.json');assert ack['id']==saved['id'] and ack['files']==saved['files']
   final=saves[-1];assert final['completed_nodes']==total and final['work_chain_sha256']==chain and final['checkpoint_sha256']==a['checkpoint_sha256'] and final['files'][1]['sha256']==a['policy_sha256']
   artifact=r/'artifacts/C'/str(seed)/'attempt-1';cp=artifact/final['files'][0]['name'];policy=artifact/final['files'][1]['name']
   assert h(cp)==final['checkpoint_sha256'] and h(policy)==final['policy_sha256']
   trainer=load_training(cp);assert asdict(trainer.config)==final['config'] and trainer.iteration==iterations and len(trainer.nodes)==result['entries']
   assert sum(n.visits for n in trainer.nodes.values())==counters['raw_traverser_visits']
   assert all(math.isfinite(x) for n in trainer.nodes.values() for x in [*n.regrets,*n.average])
   assert save_training(trainer,folder/'reload.json.gz')==h(cp)
   export_policy(trainer,folder/'current.json.gz');assert export_fingerprint(folder/'current.json.gz')['os_normalized_sha256']==export_fingerprint(policy)['os_normalized_sha256']
   del trainer
  write(folder/'checks.json',dict(status='passed',seed=seed,source=source,plan_sha256=plan_sha,archive_verification=archive_check,all_work_chain_rows=iterations,whole_iteration_overshoot=total-100000000,all_recovery_acks=len(saves),final_reload_export=True,pod_two_fresh_next_processes_equal=True))
  results.append(dict(seed=seed,nodes=total,iteration=iterations,entries=result['entries'],visits=counters['raw_traverser_visits'],work_chain_sha256=chain,training_seconds=result['training_seconds'],step_nodes_per_second=total/result['training_seconds'],phase_seconds=pod['worker_final']['finished']-pod['worker_final']['started'],peak_rss_bytes=pod['worker_final']['peak_owned_rss_bytes'],swap_growth_bytes=pod['worker_final']['swap_growth_bytes'],archive_path=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=h(archive),verified_manifest_files=archive_check['verified_files'],recovery_acks=len(saves),all_checks_passed=True,final_checkpoint_path=str(cp),final_checkpoint_sha256=h(cp),final_current_path=str(policy),final_current_sha256=h(policy),resources=pod['worker_final']))
  write(out/'progress.json',dict(completed=[x['seed'] for x in results],heartbeat=time.time()))
  print('verified',seed,flush=True)
 write(out/'summary.json',dict(status='passed',source=source,plan_sha256=plan_sha,workers=results,wall_seconds=time.time()-start,audit_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,strength_outcomes_inspected=False,upper_cost=json.loads((r/'final-cost-estimate.json').read_text())['upper_cost_usd'],remaining_ids=json.loads((r/'operator-finished.json').read_text())['remaining_ids']))
except Exception as e:
 write(out/'failure.json',dict(error=type(e).__name__,message=str(e),completed=[x['seed'] for x in results]));raise
