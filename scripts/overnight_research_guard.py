"""M4 one-use operation guard with whole-family limits and a fixed swap cap."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
from time import sleep

import psutil
from scripts import run_native_hu100_growth_1b as resources
from scripts.research_process_family import owned_processes


def remember_descendants(known):
    owned_processes(known)


def stop_owned(child, known):
    """Keep identities across reparenting, including descendants in new sessions."""
    remember_descendants(known)
    live = []
    for pid, created in known.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == created:
                live.append(process)
                process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, surviving = psutil.wait_procs(live, timeout=5)
    for process in surviving:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, surviving = psutil.wait_procs(surviving, timeout=5)
    if child is not None:
        child.wait(timeout=5)
    if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in surviving):
        raise RuntimeError('Owned descendant survived cleanup')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--seconds',type=float,help='Measured operation budget, if frozen')
    p.add_argument('--stop-file',type=Path)
    p.add_argument('--accept-capacity-stop',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.command[0]=='--':a.command=a.command[1:]
    if a.accept_capacity_stop and (len(a.command)<2 or a.command[1]!='train'):
        raise ValueError('Capacity stop is accepted only for native training')
    resources.OUT=a.out
    resources.FAMILY_SOFT=7*resources.GIB
    resources.FAMILY_HARD=9*resources.GIB
    resources.DISK_FLOOR=16*resources.GIB
    resources.SWAP_GROWTH=3_000_000_000
    original_limits, original_host = resources.limits, resources.host
    known = {}
    def host():
        return original_host()
    def limits(sample,baseline,rss,*,swap_limit=3_000_000_000):
        if sample['swap_bytes']>3_000_000_000:return '3 GB total system swap cap'
        return original_limits(sample,baseline,rss,swap_limit=swap_limit)
    resources.host=host
    resources.limits=limits
    resources.terminate_child=lambda child:stop_owned(child,known)
    # Host ownership is shared with the existing HU100 campaigns. Each operation
    # keeps its original fixed baseline, and no failed science is readmitted.
    with (Path.home()/'Local/.hu100-m4-research.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resources.identity()
        try:
            receipt=resources.operation(a.name,a.command,stop_file=a.stop_file,accepted=(0,3) if a.accept_capacity_stop else (0,),
                deadline=__import__('time').time()+a.seconds if a.seconds else None,
                family_processes=lambda:owned_processes(known))
            remember_descendants(known)
            survivors=[]
            for pid, created in known.items():
                try:
                    process=psutil.Process(pid)
                    if process.create_time()==created and process.status()!=psutil.STATUS_ZOMBIE:
                        survivors.append(pid)
                except psutil.NoSuchProcess:
                    pass
            if survivors:
                failure=a.out/'campaign-failure.json'
                if not failure.exists():
                    resources.write(failure,{'operation':a.name,'failure':'Owned descendants outlived operation','pids':survivors})
                raise RuntimeError('Owned descendants outlived successful operation: '+str(survivors))
            print(json.dumps(receipt),flush=True)
        finally:
            # Also runs if operation() raises after its direct child already
            # exited; inherited cleanup alone cannot cover reparented workers.
            stop_owned(None,known)

if __name__=='__main__':main()
