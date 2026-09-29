"""Independent paired LBR aggregation from retained concrete hand rows."""
import argparse,gzip,json,math,hashlib
from collections import defaultdict
from statistics import mean,stdev
from scipy.stats import t
parser=argparse.ArgumentParser();parser.add_argument('--hands',required=True);source=parser.parse_args().hands
blocks=defaultdict(lambda:defaultdict(list));roles=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
with gzip.open(source,'rt') as f:
 for line in f:
  if '"rules": ["lbr"]' not in line:continue
  r=json.loads(line)
  blocks[r['policy']][r['block']].append(r['target_chips'])
  role='button_small_blind' if r['rotation']==r['button'] else 'big_blind'
  roles[r['policy']][role][r['block']].append(r['target_chips'])
blocks={p:{b:mean(v) for b,v in bs.items()} for p,bs in blocks.items()}
def ci(a):
 n=len(a);mu=mean(a);half=t.ppf(.975,n-1)*stdev(a)/math.sqrt(n)
 return dict(blocks=n,bb100=mu,ci95=[mu-half,mu+half])
seeds=[2026092801,2026092802,2026092803];out={}
for work in [2,5,10,20]:
 profiles=[blocks[f'2p-{s}-{work}M'] for s in seeds]
 vals=[mean([p[b] for p in profiles]) for b in range(512)]
 out[f'absolute-{work}M']=ci(vals)
 out[f'relative-{work}M']=ci([vals[b]-blocks['uniform'][b] for b in range(512)])
for high,low in [(20,2),(20,10)]:
 out[f'contrast-{high}M-{low}M']=ci([mean([blocks[f'2p-{s}-{high}M'][b]-blocks[f'2p-{s}-{low}M'][b] for s in seeds]) for b in range(512)])
for role in ['button_small_blind','big_blind']:
 vals=[mean([mean(roles[f'2p-{s}-20M'][role][b]) for s in seeds]) for b in range(512)]
 out[f'target-role-{role}']=ci(vals)
 out[f'attacker-opposite-target-role-{role}']=ci([-v for v in vals])
with open(source,'rb') as f:out['source_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(out,indent=2))
