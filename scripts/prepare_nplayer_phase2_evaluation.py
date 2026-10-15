"""Pin audited endpoints and quote the frozen evaluation before final hands."""
import argparse
import json
import math
from pathlib import Path

from scripts.analyze_nplayer_arena import sha
from scripts.nplayer_phase2_matrix import matrix
from scripts.run_nplayer_blueprint_seed import SEEDS


def prepare(training, binary, costs_path, out):
    costs=json.loads(costs_path.read_text())
    if costs['status']!='passed': raise ValueError('harness qualification did not pass')
    models={}
    for seed in SEEDS:
        folder=training/str(seed)
        result=json.loads((folder/'result.json').read_text())
        guard=json.loads((training.parent/'guards'/f'seed-{seed}'/'result.json').read_text())
        if result['status']!='passed' or guard['status']!='passed': raise ValueError('training/guard failure')
        if [m['target'] for m in result['milestones']] != [200_000_000,500_000_000,1_000_000_000]:
            raise ValueError('missing scheduled endpoint')
        for milestone in result['milestones']:
            label=f'{milestone["target"]//1_000_000}m'
            path=folder/f'{label}-average.gz'
            audit=milestone['audit']
            digest=sha(path)
            if (audit['status']!='passed' or audit['strategy']!='average' or audit['last_average_street']!=3
                or audit['rows']!=milestone['entries'] or digest!=audit['policy_sha256']
                or digest!=milestone['average_sha256']):
                raise ValueError('export/audit identity mismatch')
            models[f'{seed}/{label}']={'member':str(path.relative_to(training)),
                'sha256':digest,'bytes':path.stat().st_size,'entries':milestone['entries'],
                'strategy':'average','last_average_street':3,
                'checkpoint_sha256':milestone['checkpoint_sha256']}
    cases=matrix()
    # Repetition reloads from disk: these one-off costs are charged twice per case.
    load_rate=max(run['load_seconds'] for c in costs['cases'] for run in c['runs'])/8_246_876
    play_rate=max(run['play_seconds']/run['hands'] for c in costs['cases'] for run in c['runs'])
    replay_rate=max(c['replay_seconds']/(c['blocks']*6) for c in costs['cases'])
    total_load=total_play=total_replay=0
    per_case=[]
    for case in cases:
        ids={case['singleton'],case['pair']}
        entries=sum(models[i]['entries'] for i in ids if not i.startswith('bot='))
        hands=case['blocks']*6
        load=2*entries*load_rate;play=2*hands*play_rate;replay=hands*replay_rate
        total_load+=load;total_play+=play;total_replay+=replay
        per_case.append({'name':case['name'],'one_off_load_seconds':load,
                         'two_pass_play_seconds':play,'engine_replay_seconds':replay})
    measured_scaled=total_load+total_play+total_replay
    budget=math.ceil((2*measured_scaled+600)/60)*60
    manifest={'format':'nplayer-phase2-evaluation-v1','arena_binary_sha256':sha(binary),
        'models':models,'cases':cases,'costs':{'qualification_sha256':sha(costs_path),
        'load_seconds_per_row':load_rate,'play_seconds_per_hand':play_rate,'replay_seconds_per_hand':replay_rate,
        'one_off_loading_seconds':total_load,'two_pass_play_seconds':total_play,
        'engine_replay_seconds':total_replay,'scaled_total_seconds':measured_scaled,
        'budget_seconds':budget,'budget_rule':'twice scaled work plus 600 seconds; rounded up to minute',
        'per_case':per_case},'final_hands':sum(c['blocks']*6 for c in cases),
        'decision':'all three primary singleton lower95_bb100 > 0; other comparisons descriptive'}
    with out.open('x') as f: json.dump(manifest,f,indent=2);f.write('\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--costs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.training,a.binary,a.costs,a.out),indent=2))
