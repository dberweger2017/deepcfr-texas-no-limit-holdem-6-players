"""Seal shared qualification evidence after the training/evaluation guard closes."""
import fcntl
import json
from pathlib import Path
import subprocess
from time import time

from scripts.archive_hu100_4b_seed_ladder import seal
from scripts.evaluate_hu100_direct import put
from scripts.report_hu100_v053_seeds import render
from scripts.train_hu100_v053_seeds import BASE,ROOT,BINARY,SEEDS,read,revision,require_disk
from src.policies.files import file_hash


def main():
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as one, \
         (Path.home()/'Local/.hu100-m4-research.lock').open('a+') as two:
        for stream in (one,two):fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (BASE/'guards/operations/whole-campaign/receipt.json').exists():
            raise ValueError('Whole-campaign guard must close before shared evidence seals')
        owner=read(BASE/'shared-archive-owner.json')
        if owner['pr']['number']!=286 or not 0<=time()-owner['at']<600:
            raise ValueError('Fresh current owning-PR status required')
        render()
        source=BASE/'source.tar'
        with source.open('xb') as target:
            subprocess.run(['git','archive','HEAD','scripts','native','requirements.txt',
                'requirements-dev.txt','requirements-play.txt','docs/hu100-v053-three-seeds-protocol.md'],
                stdout=target,check=True)
        files=[]
        for p in sorted(BASE.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            rel=p.relative_to(BASE)
            if rel.parts[0] in (*map(str,SEEDS),'inputs','policy-cache'):continue
            if rel.parts[:3]==('guards','operations','shared-archive'):continue
            if rel.name=='guard-held.json':continue
            if rel.parts[0] in ('pilot','final') and len(rel.parts)>1 and '2026100601' not in rel.parts[1]:continue
            files.append((p,'research/'+rel.as_posix()))
        for p in sorted((ROOT/'planning').glob('*')):
            if p.is_file() and p.stat().st_size<1024**2:
                files.append((p,'planning/'+p.name))
        files.append((BINARY,'runtime/hu20-trainer'))
        report=ROOT/'docs/reports/hu100-v053-three-seeds.md'
        files.append((report,'report/'+report.name))
        for p in sorted((ROOT/'docs/reports/hu100-v053-three-seeds-artifacts').glob('*.json')):
            files.append((p,'report/artifacts/'+p.name))
        dest=Path.home()/'Local/Research-Cloud/PR-286-hu100-v053-three-seeds/hu100-v053-shared-evidence.zip'
        require_disk(sum(p.stat().st_size for p,_ in files)+512*1024**2)
        receipt=seal(files,dest,dict(pr=286,source=revision(),native_build=read(BASE/'build-receipt.json'),
            original_candidate_restore=read(BASE/'evaluation-inputs.json')['restored_original'],
            excludes_current_archive_guard=True,originals_retained=True,
            seed_archive_receipts={str(seed):read(BASE/str(seed)/'archive-receipt.json') for seed in SEEDS}))
        put(BASE/'shared-archive-receipt.json',receipt)
        put(BASE/'archive-closeout-complete.json',dict(at=time(),within_original_deadline=time()<read(BASE/'budget.json')['deadline']))


if __name__=='__main__':main()
