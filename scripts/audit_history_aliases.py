"""Native tree/history audit only; no solver, training or production changes."""
import argparse
import json
from pathlib import Path
from src.diagnostics.history_alias import enumerate_aliases, exposure, self_play_exposure
from scripts.prepare_flop_check import load_policy

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('tree','stored','self-play'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tree',type=Path)
    parser.add_argument('--plan',type=Path,default=Path('configs/diagnostics/hu20-exact-flop-check-inputs.json'))
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--hands',type=Path,default=Path('docs/reports/hu20-card-v2-artifacts/production'))
    parser.add_argument('--policy-index',type=int,default=0)
    parser.add_argument('--deals',type=int,default=3000)
    args=parser.parse_args()
    if args.stage=='tree':enumerate_aliases(args.out)
    else:
        plan=json.loads(args.plan.read_text())
        if args.stage=='stored':exposure(args.tree,args.hands,plan,args.out)
        else:
            source=load_policy(plan['policies'][args.policy_index],args.inputs)
            self_play_exposure(args.tree,source,args.out,deals=args.deals,
                               seed=202610020102+args.policy_index*10)
if __name__=='__main__':main()
