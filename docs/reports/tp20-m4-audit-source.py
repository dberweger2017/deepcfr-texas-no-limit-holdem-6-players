import gzip, hashlib, json, math, statistics, subprocess, time
from pathlib import Path
from scipy.stats import t
from src.arena.schedule import canonical, digest

root=Path('/Users/dberweger/Local/tp20-pr113/results/tp20-m4-20260928')
plan=json.loads(Path('configs/blueprint/tp20-m4.json').read_text())
report=json.loads((root/'final-report.json').read_text())
campaign=json.loads((root/'campaign.json').read_text())
manifest=json.loads((root/'artifact-manifest.json').read_text())
errors=[]
def require(check,message):
 if not check: errors.append(message)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
require(digest(plan)=='efc2e794c76db36568530e9cd5239ecf4bb80d17e1a2bc4c5a648eb6ee312690','Plan changed')
require(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='204e351def48a630a795ffd1d67d7df03871afc1','Source changed')
require(campaign['status']=='complete' and report['status']=='complete','Incomplete report/campaign')
require(time.time()<campaign['deadline_unix_seconds'],'Post-run audit exceeded original deadline')
require(digest(manifest['files'])==manifest['files_sha256'],'Inventory digest')
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='artifact-manifest.json'}
require(actual==set(manifest['files']),'Unlisted or missing inventory files')
for name,expected in manifest['files'].items():require(sha(root/name)==expected,'Hash '+name)
require(all(a['status']=='complete' and a.get('returncode')==0 for a in campaign['attempts']),'Failed or omitted attempt')
require(len({(a['stage'],a['name']) for a in campaign['attempts']})==len(campaign['attempts'])==132,'Duplicate/missing attempts')
require(all(a['finished_unix_seconds']<=a['deadline_unix_seconds'] for a in campaign['attempts']),'Attempt deadline')
require(campaign['updated_unix_seconds']<campaign['deadline_unix_seconds'],'Campaign deadline')
pre=root/'preflight-training'/'2026092901'/'checkpoint-0.json.gz'
replay=root/'training'/'2026092901'/'checkpoint-1.json.gz'
require(sha(pre)==sha(replay),'5M preflight/main checkpoint differs')
require(sha(root/'preflight-training'/'2026092901'/'policy-0.json.gz')==sha(root/'training'/'2026092901'/'policy-1.json.gz'),'5M inference replay differs')
resources=report['resources']
require(all(x['rss_bytes']<10.5*1024**3 and x['free_disk_bytes']>=8*1024**3 for x in resources),'Resource sample bounds')
require(all('used = 761.38M' in x['swap'] for x in resources),'System swap changed')
require(all(a['sampled_peak_rss_bytes']<10.5*1024**3 for a in campaign['attempts']),'Supervised RSS cap')
for seed in plan['training_seeds']:
 run=root/'training'/str(seed)
 result=json.loads((run/'result.json').read_text())
 manifest_seed=json.loads((run/'manifest.json').read_text())
 require(manifest_seed['seed']==seed and manifest_seed['initialization']=='zero regrets; no parent checkpoint','From-zero lineage')
 require(result['discarded_nodes']==0 and 20000000<=result['completed_nodes']<=20250000,'Work accounting')
 with gzip.open(run/'checkpoint-3.json.gz','rt') as f:header=json.loads(next(f))
 require(header['iteration']==result['iterations'] and header['config']['seed']==seed and header['identity']['players']==3 and header['table']['stacks']==[2000]*3,'Checkpoint identity')
# Independently recompute block means and contrasts from hand rows.
by_phase={}; counts={}; all_deals={}
for phase in ('development','confirmation','secondary','crossplay'):
 rows_by_run={}; counts[phase]=0; deals=set()
 for folder in sorted((root/phase).iterdir()):
  if not folder.is_dir():continue
  manifest_run=json.loads((folder/'manifest.json').read_text())
  result=json.loads((folder/'result.json').read_text())
  require(result['status']=='complete','Incomplete '+str(folder))
  sched=json.loads((folder/'schedule.json').read_text())
  require(digest(sched)==manifest_run['schedule_sha256'],'Schedule digest '+str(folder))
  indexed={}; observed={}
  for line in (folder/'hands.jsonl').open():
   row=json.loads(line); key=(row['block'],row['rotation'])
   require(key not in indexed,'Duplicate hand '+str(folder))
   require(row['status']=='completed' and len(row['net_chips'])==3 and sum(row['net_chips'])==0,'Illegal/incomplete/conservation '+str(folder))
   require(row['candidate_chips']==row['net_chips'][row['rotation']],'Hero chip mapping '+str(folder))
   block=sched['blocks'][row['block']]
   require(row['deal_seed']==block['deal_seeds'][0] and row['button']==block['button'] and row['action_seeds']==block['action_seeds'] and row['opponents']==block['opponents'],'Hand schedule mismatch '+str(folder))
   require(len(set(row['action_seeds']))==3,'Opponent stream collision')
   indexed[key]=row['candidate_chips']; observed[key]=(row['deal_seed'],row['button'],tuple(row['action_seeds']),tuple(row['opponents']))
   deals.add(row['deal_seed']);counts[phase]+=1
  n=manifest_run['blocks']
  require(len(indexed)==3*n,'Missing rotations '+str(folder))
  rows_by_run[folder.name]={'means':[statistics.mean(indexed[(b,r)] for r in range(3)) for b in range(n)],'schedule':observed}
 by_phase[phase]=rows_by_run;all_deals[phase]=deals
require(counts=={'development':59904,'confirmation':294912,'secondary':18432,'crossplay':9216},'Frozen hand counts')
for p,deals in all_deals.items():
 for q,others in all_deals.items():
  if p<q:require(not deals&others,'Overlapping phase deals '+p+' '+q)
for phase,runset in by_phase.items():
 grouped={}
 for name,v in runset.items():
  # Schedules paired within lineup; arm is last segment except E milestone suffix.
  manifest_run=json.loads((root/phase/name/'manifest.json').read_text())
  lineup=manifest_run['lineup']
  if lineup in grouped:require(v['schedule']==grouped[lineup],'Cross-arm pairing '+phase+' '+lineup)
  else:grouped[lineup]=v['schedule']
def stratified(groups):
 center=statistics.mean(statistics.mean(g) for g in groups)
 terms=[statistics.variance(g)/len(g)/len(groups)**2 for g in groups]
 v=sum(terms);df=v*v/sum(x*x/(len(g)-1) for x,g in zip(terms,groups))
 radius=float(t.ppf(.975,df))*math.sqrt(v)
 return {'bb_per_100':center,'interval':[center-radius,center+radius],'blocks':sum(map(len,groups))}
groups=[]; absolutes=[]
for lineup in plan['primary_lineups']:
 runs=by_phase['confirmation'];u=runs[lineup+'-uniform']['means'];trained=[runs[lineup+'-C'+str(s)]['means'] for s in plan['training_seeds']]
 groups.append([statistics.mean(x[b]-u[b] for x in trained) for b in range(len(u))])
 absolutes.append([statistics.mean(x[b] for x in trained) for b in range(len(u))])
primary=stratified(groups); saved=report['suites']['confirmation']['effects']['final/aggregate-minus-uniform']
require(abs(primary['bb_per_100']-saved['bb_per_100'])<1e-12 and max(abs(a-b) for a,b in zip(primary['interval'],saved['interval']))<1e-12,'Independent primary estimate mismatch')
peak=max([v['result']['peak_process_rss_bytes'] for v in report['training'].values()]+[v['result']['peak_process_rss_bytes'] for p in report['suites'].values() for v in p['runs'].values()]+[v[k]['result']['peak_process_rss_bytes'] for v in report['crossplay'].values() for k in ('early','final')])
out={'schema':'tp20-post-run-audit-v1','status':'complete' if not errors else 'failed','errors':errors,'audited_unix_seconds':time.time(),'source_revision':'204e351def48a630a795ffd1d67d7df03871afc1','plan_sha256':digest(plan),'global_inventory_file_sha256':sha(root/'artifact-manifest.json'),'verified_files':len(actual),'completed_attempts':len(campaign['attempts']),'evaluation_hands':counts,'total_evaluation_hands':sum(counts.values()),'independent_primary':primary,'descriptive_five_scripted_effect':stratified(groups[:5]),'descriptive_absolute_six_lineup_panel':stratified(absolutes),'descriptive_absolute_five_scripted':stratified(absolutes[:5]),'peak_process_rss_gib':peak/1024**3,'system_swap_change_mib':0,'minimum_recorded_free_disk_gib':min(x['free_disk_bytes'] for x in resources)/1024**3,'campaign_elapsed_hours':(campaign['updated_unix_seconds']-campaign['started_unix_seconds'])/3600,'preflight_5m_checkpoint_equals_main':sha(pre)==sha(replay)}
Path('/Users/dberweger/Local/tp20-pr113/results/tp20-completed-audit-20260928.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(bool(errors))
