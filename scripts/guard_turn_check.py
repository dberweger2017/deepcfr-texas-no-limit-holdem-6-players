"""Run one turn preparation/validation worker under the M4 RSS/swap guard."""
import argparse
from pathlib import Path
import sys
from scripts.preflight_flop_check import prepare_guarded
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.flop_check_runtime import machine_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);before=machine_snapshot();atomic_json(a.out/'machine-before.json',before)
    budget=min(6*1024**3,int(before['reclaimable_bytes']*.8/1024**3)*1024**3)
    if budget<2*1024**3:raise MemoryError('Insufficient measured turn headroom')
    atomic_json(a.out/'admission.json',{'budget_bytes':budget,'swap_baseline_bytes':before['swap_used_bytes']})
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    prepare_guarded(command,a.out,budget,before['swap_used_bytes'])
if __name__=='__main__':main()
