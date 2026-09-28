"""Second arithmetic path from raw rows, independent of the production report grouping."""

import argparse
from collections import defaultdict
import gzip
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t


def check(hands,report,seeds):
    data=defaultdict(dict)
    with gzip.open(hands,'rt') as rows:
        for line in rows:
            r=json.loads(line)
            if r.get('milestone')!=3 or r.get('arm') not in ('A','B') or r['attacker'] not in ('Pressure-native','LBR-original-cap2'):continue
            if r['status']!='complete':raise ValueError('Primary failed hand')
            k=(r['attacker'],r['block'],r['rotation'],r['training_seed'],r['arm'])
            if k in data:raise ValueError('Duplicate primary observation')
            data[k]=r['target_chips']
    out={}
    for attack in ('Pressure-native','LBR-original-cap2'):
        blocks=sorted({k[1] for k in data if k[0]==attack})
        byblock=[]
        for b in blocks:
            difference=0
            for s in seeds:
                for rotation in (0,1):
                    difference+=data[(attack,b,rotation,s,'B')]-data[(attack,b,rotation,s,'A')]
            byblock.append(difference/(len(seeds)*2))
        mu=mean(byblock);half=t.ppf(.9875,len(byblock)-1)*stdev(byblock)/sqrt(len(byblock))
        out[attack]={'blocks':len(byblock),'bb100':mu,'interval':[mu-half,mu+half]}
        expected=report['primary'][attack]['B_minus_A']
        if expected['blocks']!=len(byblock) or abs(expected['bb100']-mu)>1e-8 or any(abs(a-b)>1e-8 for a,b in zip(expected['interval'],out[attack]['interval'])):
            raise ValueError('Independent block arithmetic disagreement')
    with hands.open('rb') as f:out['hands_sha256']=__import__('hashlib').file_digest(f,'sha256').hexdigest()
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    plan=json.loads((a.root/'frozen-plan.json').read_text());report=json.loads((a.root/'audit/results.json').read_text())
    if report['status']!='complete':return 0
    out=check(a.root/'evaluation/hands.jsonl.gz',report,plan['training_seeds'])
    (a.root/'audit/independent-summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out));return 0

if __name__=='__main__':raise SystemExit(main())
