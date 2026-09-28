from pathlib import Path
from collections import defaultdict,Counter
import json,gzip,hashlib,time
p=Path('/Users/dberweger/Local/hu20-native-reopening-ab/results/hu20-native-reopening-m4-20260928')
stats=defaultdict(lambda:{'hands':0,'target':{'decisions':0,'seconds':0,'maximum_seconds':0},'opponent':{'decisions':0,'seconds':0,'maximum_seconds':0},'membership':Counter()})
rows=0
with gzip.open(p/'evaluation/hands.jsonl.gz','rt') as stream:
 for line in stream:
  r=json.loads(line);rows+=1
  if r['milestone']!=3 or r['arm'] not in ('A','B'):continue
  x=stats[(r['arm'],r['attacker'])];x['hands']+=1
  for a in r['actions']:
   who='target' if a['logical_player']==0 else 'opponent';d=x[who];d['decisions']+=1;d['seconds']+=a['seconds'];d['maximum_seconds']=max(d['maximum_seconds'],a['seconds'])
   for flag in ('on_original_cap2_menu','on_target_menu','preceding_original_off_menu','preceding_target_off_menu'):
    x['membership'][who+':'+flag+':'+str(a[flag])]+=1
   if who=='target':
    x['membership'][a['street']+':'+('trained' if a['target_trained'] else 'fallback')]+=1
    if a['preceding_original_off_menu']:x['membership']['after_original_off_menu:'+('trained' if a['target_trained'] else 'fallback')]+=1
    if a['preceding_target_off_menu']:x['membership']['after_target_off_menu:'+('trained' if a['target_trained'] else 'fallback')]+=1
for x in stats.values():
 for who in ('target','opponent'):
  d=x[who];d['mean_seconds']=d['seconds']/d['decisions'] if d['decisions'] else None
work=[]
for f in sorted(p.glob('training/*/iterations.jsonl')):
 durations=[];new=repeat=dec=upd=0;nodes=Counter();updates=Counter()
 with f.open() as stream:
  for line in stream:
   r=json.loads(line);durations.append(r['complete_outer_seconds']);new+=r['new_entries_after_cap'];repeat+=r['revisited_entries_after_cap'];w=r['attempted_work'];nodes.update(w['nodes_by_street']);updates.update(w['updates_by_street_raise_count'])
   dec+=sum(v for k,v in w['decisions_by_street_raise_count'].items() if int(k.rsplit(':',1)[1])>2);upd+=sum(v for k,v in w['updates_by_street_raise_count'].items() if int(k.rsplit(':',1)[1])>2)
 durations.sort();n=len(durations)
 overhead=f.parent/'checkpoint-overhead.jsonl';checkpoint=sum(json.loads(s)['seconds'] for s in overhead.read_text().splitlines()) if overhead.exists() else 0
 work.append({'run':f.parent.name,'completed_iterations':n,'complete_outer_seconds':{'sum':sum(durations),'mean':sum(durations)/n,'p95':durations[int((n-1)*.95)],'p99':durations[int((n-1)*.99)],'max':durations[-1]},'new_keys_at_least_two_raises':new,'revisited_keys_at_least_two_raises':repeat,'decisions_more_than_two_raises':dec,'updates_more_than_two_raises':upd,'nodes_by_street':dict(nodes),'updates_by_street_raise_count':dict(updates),'periodic_checkpoint_seconds':checkpoint})
r={'scope':'postprocessing frozen retained logs; no new evaluation','finished':time.time(),'raw_hands':rows,'final_suite_telemetry':[{'arm':k[0],'attacker':k[1],**v} for k,v in sorted(stats.items())],'training_work':work}
f=p.with_name(p.name+'-diagnostics.json');f.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print('Wrote',f,'with',rows,'raw hand rows and',len(work),'training runs')
