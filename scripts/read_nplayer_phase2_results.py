"""Only read settled results after the complete fixed matrix passes its gates."""
import argparse
import json
from pathlib import Path
from scripts.analyze_nplayer_arena import verified_readout, sha
from scripts.nplayer_phase2_matrix import matrix


def read_results(manifest_path, evaluation, guards):
    manifest=json.loads(manifest_path.read_text())
    if manifest['cases']!=matrix(): raise ValueError('changed matrix')
    # Admission is checked across the entire matrix before reading any payoffs.
    for case in matrix():
        root=evaluation/case['name']
        result=json.loads((root/'result.json').read_text())
        guard=json.loads((guards/case['name']/'result.json').read_text())
        admission=json.loads((root/'admission.json').read_text())
        if (result['status']!='passed' or guard['status']!='passed' or result['case']!=case
            or admission['manifest_sha256']!=sha(manifest_path)
            or result['hands']!=case['blocks']*6):
            raise ValueError(f'unsettled contrast: {case["name"]}')
    rows=[]
    for case in matrix():
        root=evaluation/case['name']
        readout=verified_readout(root/'play-0.jsonl',root/'replay.json',root/'play-1.jsonl')
        if readout['root']!=case['root'] or readout['blocks']!=case['blocks']:
            raise ValueError('readout schedule mismatch')
        result=json.loads((root/'result.json').read_text())
        coverage=result['operations'][0]['report']
        rows.append({'case':case,**readout,'coverage':{
            k:coverage[k] for k in ('decisions_by_identity_street','misses_by_identity_street')}})
    primary=[r for r in rows if r['case']['primary']]
    passed=all(r['singleton']['lower95_bb100']>0 for r in primary)
    return {'status':'settled','learning_rule_passed':passed,'primary_count':len(primary),
            'final_hands':sum(r['hands'] for r in rows),'manifest_sha256':sha(manifest_path),
            'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--evaluation',required=True,type=Path)
    p.add_argument('--guards',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    result=read_results(a.manifest,a.evaluation,a.guards)
    with a.out.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'learning_rule_passed':result['learning_rule_passed'],'final_hands':result['final_hands']}))
