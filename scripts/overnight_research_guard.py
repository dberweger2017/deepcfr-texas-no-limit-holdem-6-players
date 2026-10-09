"""M4 one-use operation guard with whole-family limits and a fixed swap cap."""
import argparse
import json
from pathlib import Path
import sys
from scripts import run_native_hu100_growth_1b as resources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--seconds',type=float,help='Measured operation budget, if frozen')
    p.add_argument('--stop-file',type=Path)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.command[0]=='--':a.command=a.command[1:]
    resources.OUT=a.out
    resources.FAMILY_SOFT=7*resources.GIB
    resources.FAMILY_HARD=9*resources.GIB
    resources.DISK_FLOOR=16*resources.GIB
    resources.SWAP_GROWTH=3_000_000_000
    # The authorized cap is total system swap, not growth from a large baseline.
    original=resources.limits
    def limits(sample,baseline,rss,*,swap_limit=3_000_000_000):
        if sample['swap_bytes']>3_000_000_000:return '3 GB total system swap cap'
        return original(sample,baseline,rss,swap_limit=swap_limit)
    resources.limits=limits
    resources.identity=lambda: __import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip()
    receipt=resources.operation(a.name,a.command,stop_file=a.stop_file,
        deadline=__import__('time').time()+a.seconds if a.seconds else None)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
