"""Stream an independent every-row audit of N-player inference probabilities."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def audit(checkpoint,policy):
    cp_sha,policy_sha=sha(checkpoint),sha(policy)
    streets=[{'entries':0,'visits':0,'zero_visit_entries':0} for _ in range(4)]
    counts={'rows':0,'average_rows':0,'current_rows':0,'zero_mass_rows':0}
    with gzip.open(checkpoint,'rt') as a,gzip.open(policy,'rt') as b:
        source=json.loads(next(a));header=json.loads(next(b))
        assert source['format']=='nplayer-cfr-v1' and header['format']=='nplayer-policy-v1'
        assert header['source_checkpoint_sha256']==cp_sha
        assert source['game']==header['identity'] and source['players']==header['players']
        assert source['entries']==header['entries'] and source['last_average_street']==header['last_average_street']
        mode=header['strategy'];assert mode in ('current','average','scoped-average')
        assert header['continuation']==('current-after-average-scope' if mode=='scoped-average' else 'none')
        assert mode!='average' or header['last_average_street']==3
        for line in a:
            src=json.loads(line);row=json.loads(next(b))
            assert src[:4]==row[:4]
            stats=streets[src[2]];stats['entries']+=1;stats['visits']+=src[4];stats['zero_visit_entries']+=src[4]==0
            use_average=mode!='current' and src[2]<=source['last_average_street']
            weights=src[6] if use_average else [max(0.,x) for x in src[5]]
            assert weights and all(math.isfinite(x) and x>=0 for x in weights)
            total=math.fsum(weights)
            expected=[w/total for w in weights] if total else [1/len(weights)]*len(weights)
            assert len(row[4])==len(expected)
            assert all(math.isfinite(x) and 0<=x<=1 for x in row[4])
            assert max(abs(x-y) for x,y in zip(row[4],expected))<=1e-14,(counts['rows'],src[0])
            counts['rows']+=1;counts['average_rows' if use_average else 'current_rows']+=1;counts['zero_mass_rows']+=not total
        assert not b.readline()
    assert counts['rows']==source['entries']
    for stats in streets:
        stats['visits_per_key']=stats['visits']/stats['entries'] if stats['entries'] else None
    return {'by_street':streets,'status':'passed','checkpoint_sha256':cp_sha,'policy_sha256':policy_sha,'strategy':mode,'last_average_street':source['last_average_street'],**counts}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('policy',type=Path);a=p.parse_args();print(json.dumps(audit(a.checkpoint,a.policy),indent=2))
