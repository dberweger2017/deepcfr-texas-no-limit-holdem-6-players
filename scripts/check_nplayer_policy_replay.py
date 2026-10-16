"""Recompute every policy-play transition using the pinned independent pokers engine."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.check_nplayer_parity import engine_identity,record


def check(path):
    pin=engine_identity();hands=decisions=0
    with path.open() as f:
        header=json.loads(next(f));assert header['format']=='nplayer-privileged-policy-replay-v1'
        n=header['identity']['players'];assert header['distinct_policy_instances']==n
        arena=header.get('harness')=='nplayer-duplicate-v1'
        if arena:
            from scripts.analyze_nplayer_arena import PERMUTATIONS
            assert n==3 and header['hands']==6*header['blocks']
        duplicate_deck=None
        for line in f:
            native=json.loads(line)
            oracle=record(native['hand'],n,preset=native)
            for field in ('actions','states','history','settlement'):
                assert native[field]==oracle[field],(native['hand'],field)
            assert native['actors']==[s['actor'] for s in oracle['states'][:-1]]
            if arena:
                assert native['hand']==hands
                block,permutation=divmod(hands,6)
                assert native['block']==block and native['permutation']==permutation
                assert tuple(native['identities'])==PERMUTATIONS[permutation]
                assert native['button']==block%3
                assert native['rewards']==oracle['states'][-1]['rewards']
                if permutation==0:duplicate_deck=native['deck']
                assert native['deck']==duplicate_deck
            hands+=1;decisions+=len(native['actions'])
    assert hands==header['hands']
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'status':'passed','engine':pin,'hands':hands,'decisions':decisions,'players':n,'trace_sha256':digest}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace',type=Path);a=p.parse_args();print(json.dumps(check(a.trace),indent=2))
