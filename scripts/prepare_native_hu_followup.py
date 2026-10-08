"""One owner-authorized retained-file verification, with no native retraining."""
import hashlib
import json
from pathlib import Path
import sys
from time import time

from scripts.prepare_native_hu_campaign import file_hash
from scripts.prepare_native_hu_execution import DEADLINE, bind_followup, qualification, write_plan

MANIFEST_SHA='b5e506be66fb68700be6332155bc726835e2dc42e19fa4b746165477327aa3d5'
TRAINING_SOURCE='64318398fea423a9c43c0db8635a3724e05bd55f'


def prepare_verification(out,binary,qualification_path,original_root,approval,swap_baseline):
    out,binary,original_root=(p.resolve() for p in (out,binary,original_root))
    if DEADLINE-time()<1200: raise ValueError('Verification and closeout cannot fit remaining deadline')
    source,q=qualification(qualification_path,binary)
    m=json.loads((original_root/'archive-member-index.json').read_text())
    if hashlib.sha256(json.dumps(m,sort_keys=True,indent=2).encode()).hexdigest()!=MANIFEST_SHA:
        raise ValueError('Original archive manifest provenance differs')
    inputs={}
    def retained(name):
        p=original_root/name
        matches=[x for x in m['members'] if x['path']=='research/'+name]
        if len(matches)!=1 or matches[0]['bytes']!=p.stat().st_size or matches[0]['sha256']!=file_hash(p):
            raise ValueError('Retained archive member changed: '+name)
        inputs[str(p)]={'bytes':p.stat().st_size,'sha256':file_hash(p)}
        return p
    tq=retained('qualification.json'); original_q=json.loads(tq.read_text())
    if original_q['source']!=TRAINING_SOURCE or original_q['binary_sha256']!=file_hash(binary):
        raise ValueError('Original executed training source/binary differs')
    reference,resumed=retained('reference-02/training/HU20-2026100601-1000000000.json.gz'),retained('recovery-02/training/HU20-2026100601-1000000000.json.gz')
    args={'reference':reference,'resumed':resumed,'parent':retained('inputs/historical-500M.json.gz'),
          'reference-telemetry':retained('reference-02/checkpoints.jsonl'),
          'resumed-telemetry':retained('recovery-02/checkpoints.jsonl'),
          'reference-current':retained('reference-02/current.json.gz'),
          'reference-average':retained('reference-02/average.jsonl.gz'),
          'resumed-current':retained('recovery-02/current.json.gz'),
          'resumed-average':retained('recovery-02/average.jsonl.gz')}
    for n in ('reference-02/plan.json','recovery-02/plan.json'): retained(n)
    command=[sys.executable,'-m','scripts.compare_native_hu_recovery']
    for k,v in args.items(): command+=['--'+k,str(v)]
    command+=['--training-qualification',str(tq),'--isolated-audits','--out',str(out/'equivalence.json')]
    plan={'stage':'verification','source':source,'executed_training_source':TRAINING_SOURCE,
          'binary':str(binary),'binary_sha256':file_hash(binary),'qualification_path':str(qualification_path.resolve()),
          'qualification_sha256':file_hash(qualification_path),'qualification':q,'retained_inputs':inputs,
          'original_manifest_sha256':MANIFEST_SHA,'command':command,'verification_attempt_limit':1}
    bind_followup(plan,approval)
    return write_plan(out,plan,[('verification',[{'name':'compare','command':command}],900,[])],
                      swap_baseline,min(DEADLINE-300,time()+900))
