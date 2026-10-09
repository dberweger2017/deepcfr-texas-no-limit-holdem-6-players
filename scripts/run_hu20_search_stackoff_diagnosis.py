"""Sequential, low-priority M1 stages for recorded-hand diagnosis only."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import time
from uuid import uuid4

from scripts.hu20_scaling_supervise import run


def jobs(stage: str, root: Path, archive: Path, binary: Path) -> list[dict]:
    inputs=root/'inputs'; analysis=root/'analysis'; offline=root/'offline'
    def command(module: str, *args: object) -> list[str]:
        return [sys.executable,'-m','scripts.'+module,*map(str,args)]
    if stage=='restore':
        return [{'name':'restore','command':command('diagnose_hu20_search_stackoff','restore','--archive',archive,'--out',inputs)}]
    if stage=='analyze':
        return [{'name':c,'command':command('diagnose_hu20_search_stackoff',c,'--inputs',inputs,'--out',analysis)}
                for c in ('analyze','gain-scope')]
    common=('--inputs',inputs,'--analysis',analysis,'--out',offline)
    if stage=='retrieve':
        return [{'name':'retrieve','command':command('resolve_hu20_search_stackoff','retrieve','--archive',archive,*common)}]
    if stage=='resolve':
        return [{'name':'resolve','command':command('resolve_hu20_search_stackoff','resolve',*common,'--binary',binary)},
                {'name':'responses','command':command('resolve_hu20_search_stackoff','responses',*common)}]
    if stage=='tests':
        return [{'name':'focused-tests','command':[sys.executable,'-m','pytest','-q','tests/test_hu20_search_stackoff_diagnosis.py']}]
    raise ValueError('Unknown diagnosis stage: '+stage)


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('restore','analyze','retrieve','resolve','tests'))
    p.add_argument('--root',type=Path,default=Path('results/stackoff-diagnosis'))
    p.add_argument('--archive',type=Path,default=Path.home()/'Local/Research-Cloud/PR-166-HU20-turn-search-arena/attempt-2-full-closeout-M1-20261006.zip')
    p.add_argument('--binary',type=Path,default=Path.home()/'Local/hu20-fixed50-search-tool/harness/target/release/hu20-exact-flop-tool')
    p.add_argument('--owner-power-waiver',action='store_true',help='Use only when the owner waives the AC stop')
    a=p.parse_args(); root=a.root.resolve()
    if platform.node()!='dberweger-m1': raise RuntimeError('This diagnosis is M1-only')
    if 'CloudStorage' in str(root) or 'Research-Cloud' in str(root): raise ValueError('Use a non-synced working root')
    root.mkdir(parents=True,exist_ok=True)
    ignored=subprocess.run(['git','check-ignore',str(root)],capture_output=True,check=False)
    if ignored.returncode: raise ValueError('Working root must be Git-ignored')
    os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
    with (root/'phase.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        seconds={'restore':2400,'analyze':600,'retrieve':900,'resolve':1200,'tests':300}[a.stage]
        result=run(jobs(a.stage,root,a.archive.expanduser(),a.binary.expanduser()),
                   root/f'guard-{a.stage}-{uuid4().hex[:8]}',time()+seconds,
                   require_ac=not a.owner_power_waiver,rss_gib=4,disk_gib=10,swap_gib=.5,system_memory_guard=True)
    print(json.dumps({'status':result['status'],'attempts':result['attempts'],'failure':result['failure']}))
    return int(result['status']!='complete')


if __name__=='__main__': raise SystemExit(main())
