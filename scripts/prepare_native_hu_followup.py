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


def pilot_save_reserve(equivalence):
    """Bound the fresh pilot's saves using retained native serialization measurements."""
    ref=next(Path(n).parent for n in equivalence['files'] if n.endswith('/reference-02/plan.json'))
    root=ref.parent
    manifest=json.loads((root/'archive-member-index.json').read_text())
    if hashlib.sha256(json.dumps(manifest,sort_keys=True,indent=2).encode()).hexdigest()!=MANIFEST_SHA:
        raise ValueError('Original serialization measurement manifest differs')
    paths=(ref/'checkpoints.jsonl',ref/'training-guard/resources.jsonl')
    proof={}
    for p in paths:
        spec=next(m for m in manifest['members'] if m['path']=='research/'+str(p.relative_to(root)))
        if p.stat().st_size!=spec['bytes'] or file_hash(p)!=spec['sha256']:
            raise ValueError('Original serialization measurements changed')
        proof[str(p)]={'bytes':spec['bytes'],'sha256':spec['sha256']}
    rows=[json.loads(x) for x in paths[0].read_text().splitlines()]
    samples=[json.loads(x) for x in paths[1].read_text().splitlines()]
    largest=max(rows,key=lambda r:r['diagnostics']['entries']); ceiling=10000000
    growth=ceiling/largest['diagnostics']['entries']
    extra=max(0,max(s['aggregate_job_rss_bytes'] for s in samples)-largest['process_rss_bytes_after_save'])
    reserve=max(32*ceiling+64*1024**2,2*extra*growth)
    if not 0 < reserve < 8*1024**3: raise ValueError('No measured pilot serialization headroom')
    seconds=max(r['write_seconds'] for r in rows)*growth*2
    if not 0 < seconds < 900: raise ValueError('Pilot save time cannot fit fixed phase cap')
    return {'max_entries':ceiling,'serialization_rss_bytes':reserve,
            'save_reserve_seconds':seconds,'files':proof}


def original_baseline(plans, baseline):
    if len(plans)!=2 or not baseline or any(json.loads(p.read_text()).get("campaign_swap_baseline")!=baseline for p in plans):
        raise ValueError("Retained reference/recovery campaign swap baseline differs")


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
    plans=[retained(n) for n in ('reference-02/plan.json','recovery-02/plan.json')]
    original_baseline(plans,swap_baseline)
    phase_deadline=DEADLINE-300
    command=[sys.executable,'-m','scripts.compare_native_hu_recovery']
    for k,v in args.items(): command+=['--'+k,str(v)]
    command+=['--training-qualification',str(tq),'--isolated-audits','--audit-approval',str(approval.resolve()),
              '--audit-swap-baseline',swap_baseline,'--audit-deadline',str(phase_deadline),'--out',str(out/'equivalence.json')]
    plan={'stage':'verification','source':source,'executed_training_source':TRAINING_SOURCE,
          'binary':str(binary),'binary_sha256':file_hash(binary),'qualification_path':str(qualification_path.resolve()),
          'qualification_sha256':file_hash(qualification_path),'qualification':q,'retained_inputs':inputs,
          'stage_fit_seconds':900,'original_manifest_sha256':MANIFEST_SHA,'command':command,'verification_attempt_limit':1}
    bind_followup(plan,approval)
    return write_plan(out,plan,[('verification',[{'name':'compare','command':command}],900,[])],
                      swap_baseline,phase_deadline)
