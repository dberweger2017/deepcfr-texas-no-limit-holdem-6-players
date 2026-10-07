"""Prepare authorized PR197 recovery/growth commands without launching them.

Original #196 stages remain available. This separate entry point binds the
owner's 10B capacity scope to qualification/equivalence/pilot receipts instead
of weakening the original proposed-1B quote/approval gate.
"""
import argparse
from datetime import datetime, timezone
from gzip import open as gzip_open
import json
from math import isfinite
from pathlib import Path
import shlex
import subprocess
import sys

from scripts.compare_native_hu_recovery import PARENT_SHA, receipt
from scripts.prepare_native_hu_campaign import SEED, digest, file_hash
from src.blueprint.average import checked_header
from src.blueprint.abstraction import HU100_SCHEMA

DEADLINE = datetime(2026, 10, 8, 8, tzinfo=timezone.utc).timestamp()
THREAD_URI = 't3://thread/495ca3f8-32db-4e73-98ad-29d01fb9e282'


def qualification(path, binary):
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Use committed clean tracked source')
    q = json.loads(path.read_text())
    if (q.get('status') != 'verified' or q.get('source') != source
        or q.get('binary_sha256') != file_hash(binary) or not q.get('independent_review')
        or not q.get('checks') or any(c.get('status') != 'passed' for c in q['checks'])):
        raise ValueError('Source/binary qualification and resolved independent review required')
    return source, q


def write_plan(out, plan, phases, swap_baseline, deadline):
    if out.exists(): raise FileExistsError('Use a fresh attempt root; never duplicate a launch')
    out.mkdir(parents=True); (out/'training').mkdir()
    plan.update(status='prepared-only', prepared_at=datetime.now(timezone.utc).isoformat(),
                owner_instruction=THREAD_URI, campaign_swap_baseline=swap_baseline,
                hard_deadline=deadline, limits={'rss_gib':5.5, 'swap_gib':.5, 'disk_gib':15.5, 'require_ac':True})
    plan['plan_sha256'] = digest(plan)
    (out/'plan.json').write_text(json.dumps(plan, sort_keys=True, indent=2)+'\n')
    lines = ['set -eu', '# PR188 MERGED and M4 worker-side closeout/idle admission required before execution.',
             'export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1']
    for name, jobs, seconds, extra in phases:
        jobs_path = out/f'{name}-jobs.json'
        jobs_path.write_text(json.dumps(jobs, indent=2)+'\n')
        guard = [sys.executable, '-m', 'scripts.hu20_scaling_supervise', '--jobs', str(jobs_path),
                 '--out', str(out/f'{name}-guard'), '--rss-gib','5.5', '--disk-gib','15.5',
                 '--swap-gib','.5', '--require-ac', '--swap-baseline', swap_baseline] + extra
        phase_deadline = shlex.join([sys.executable, '-c', f'import time; print(min({deadline!r},time.time()+{seconds!r}))'])
        lines += [shlex.join(guard)+f' --deadline "$({phase_deadline})"']
    (out/'commands.txt').write_text('\n'.join(lines)+'\n')
    return plan


def prepare_recovery(out, binary, qualification_path, reference_root, parent, swap_baseline, deadline):
    out, binary, reference_root, parent = (p.resolve() for p in (out,binary,reference_root,parent))
    source, q = qualification(qualification_path, binary)
    if file_hash(parent) != PARENT_SHA: raise ValueError('Retained 500M input hash differs')
    rp = json.loads((reference_root/'plan.json').read_text())
    guard = json.loads((reference_root/'export-audit-guard/campaign.json').read_text())
    audit = json.loads((reference_root/'audit.json').read_text())
    reference = Path(rp['export_audit_jobs'][0]['command'][2])
    records = [json.loads(line) for line in (reference_root/'checkpoints.jsonl').read_text().splitlines()]
    matches = [r for r in records if r['checkpoint_sha256'] == PARENT_SHA]
    if (rp['source'] != source or rp['binary_sha256'] != file_hash(binary) or rp['stage'] != 'hu20'
        or rp.get('campaign_swap_baseline') != swap_baseline or guard['status'] != 'complete'
        or audit['status'] != 'verified' or audit.get('hu20_reference_checked') is not True
        or audit['audit']['checkpoint_sha256'] != file_hash(reference) or len(matches) != 1):
        raise ValueError('Completed pinned reference audit/parent reproduction required')
    pr = receipt(reference_root/'checkpoints.jsonl', Path(matches[0]['path']))
    if pr['requested_nodes'] != 500000000: raise ValueError('Wrong legacy parent milestone')
    result = out/'training/HU20-2026100601-1000000000.json.gz'
    current, average = out/'current.json.gz', out/'average.jsonl.gz'
    command = [str(binary),'train','--stack-bb','20','--nodes','1000000000',
        '--average-rule','opponent-sampled','--resume',str(parent),'--resume-sha256',PARENT_SHA,
        '--completed-nodes',str(pr['completed_nodes']),'--max-entries','1000000000',
        '--max-seconds','900','--out',str(result),'--telemetry',str(out/'checkpoints.jsonl')]
    exports = [{'name':'export','command':[str(binary),'export',str(result),'--current',str(current),
                                          '--average',str(average),'--zero-mass','uniform']}]
    compare = [sys.executable,'-m','scripts.compare_native_hu_recovery',
        '--reference',str(reference),'--resumed',str(result),'--parent',str(parent),
        '--reference-telemetry',str(reference_root/'checkpoints.jsonl'),
        '--resumed-telemetry',str(out/'checkpoints.jsonl'),
        '--reference-current',str(reference_root/'current.json.gz'),
        '--reference-average',str(reference_root/'average.jsonl.gz'),
        '--resumed-current',str(current),'--resumed-average',str(average),'--out',str(out/'equivalence.json')]
    # A single fixed session cap covers both startup-inclusive phases.
    session_deadline = min(deadline, datetime.now(timezone.utc).timestamp()+1800)
    return write_plan(out, {'stage':'recovery','source':source,'binary_sha256':file_hash(binary),
        'qualification':q,'seed':SEED,'parent_sha256':PARENT_SHA,'parent_path':str(parent),
        'parent_completed_nodes':pr['completed_nodes'],'reference_plan_sha256':rp['plan_sha256'],
        'command':command}, [('training',[{'name':'train','command':command}],900,[]),
                            ('export-audit',exports+[{'name':'compare','command':compare}],900,[])],
        swap_baseline, session_deadline)


def prepare_growth(out, binary, qualification_path, equivalence_path, pilot_root, swap_baseline, deadline, save_reserve):
    out, binary, pilot_root = (p.resolve() for p in (out,binary,pilot_root))
    source, q = qualification(qualification_path, binary)
    equivalence = json.loads(equivalence_path.read_text())
    if equivalence.get('status') != 'verified' or equivalence.get('seed') != SEED:
        raise ValueError('Full HU20 recovery equivalence required')
    # Rehash the full comparison inputs once at preparation, not on every agent wake.
    for name, spec in equivalence['files'].items():
        p = Path(name)
        if p.stat().st_size != spec['bytes'] or file_hash(p) != spec['sha256']:
            raise ValueError('Equivalence input changed')
    pp = json.loads((pilot_root/'plan.json').read_text())
    a = json.loads((pilot_root/'audit.json').read_text())
    if (pp['source'] != source or pp['binary_sha256'] != file_hash(binary) or pp['stage'] != 'pilot'
        or pp.get('campaign_swap_baseline') != swap_baseline or a['status'] != 'verified'
        or a['stack_bb'] != 100 or a.get('native_state', {}).get('completed_nodes',0) < 10000000
        or a['native_state']['coverage_start'] != [0,0,0]
        or any(n <= 0 for n in a['native_state']['traverser_visits_by_street'])):
        raise ValueError('Complete source-bound audited HU100 pilot/coverage required')
    parent = Path(pp['export_audit_jobs'][0]['command'][2])
    if file_hash(parent) != a['audit']['checkpoint_sha256']: raise ValueError('Pilot parent changed')
    with gzip_open(parent,'rt') as f: h = json.loads(f.readline())
    checked_header(h, {'seed':SEED,'iteration':a['iteration']}, expected_schema=HU100_SCHEMA)
    if h.get('average_rule') != 'opponent-sampled' or 'training_options' in h or h['native_state'] != a['native_state']:
        raise ValueError('Pilot recipe/state differs')
    remaining = deadline-datetime.now(timezone.utc).timestamp()
    if not isfinite(save_reserve) or save_reserve <= 0 or remaining <= save_reserve:
        raise ValueError('No measured serialization/time headroom remains')
    command = [str(binary),'train','--stack-bb','100','--average-rule','opponent-sampled',
        '--resume',str(parent),'--resume-sha256',file_hash(parent),'--nodes','10000000000',
        '--milestones',','.join(str(n*1000000000) for n in range(1,10)),
        '--max-entries','1000000000','--max-seconds',str(remaining-save_reserve),
        '--out',str(out/'training/HU100-2026100601-{nodes}.json.gz'),
        '--stop-file',str(out/'controlled-stop.json'),'--telemetry',str(out/'checkpoints.jsonl')]
    extra = ['--stop-file',str(out/'controlled-stop.json'),'--soft-rss-gib','4.0',
             '--save-reserve-seconds',str(save_reserve)]
    return write_plan(out, {'stage':'growth','source':source,'binary_sha256':file_hash(binary),
        'qualification':q,'seed':SEED,'target_total_nodes':10000000000,
        'equivalence_path':str(equivalence_path.resolve()),'equivalence_sha256':file_hash(equivalence_path),
        'pilot_audit_sha256':file_hash(pilot_root/'audit.json'),'parent_path':str(parent),
        'parent_sha256':file_hash(parent),'command':command,
        'serialization_headroom_gib':1.5,'audit_reserve_note':'Operator deadline must already exclude measured final export/audit/archive allowance'},
        [('training',[{'name':'train','command':command}],remaining,extra)],swap_baseline,deadline)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('recovery','growth'),required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True); p.add_argument('--swap-baseline',required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--reference-root',type=Path); p.add_argument('--parent',type=Path)
    p.add_argument('--equivalence',type=Path); p.add_argument('--pilot-root',type=Path)
    p.add_argument('--save-reserve-seconds',type=float); a = p.parse_args()
    if not isfinite(a.deadline) or not datetime.now(timezone.utc).timestamp() < a.deadline <= DEADLINE:
        raise ValueError('Deadline must fit the owner October 8 10:00 Madrid ceiling')
    if a.stage == 'recovery':
        if a.reference_root is None or a.parent is None: p.error('Recovery needs reference root and retained parent')
        plan = prepare_recovery(a.out,a.binary,a.qualification,a.reference_root,a.parent,a.swap_baseline,a.deadline)
    else:
        if a.equivalence is None or a.pilot_root is None or a.save_reserve_seconds is None:
            p.error('Growth needs equivalence, audited pilot and measured save reserve')
        plan = prepare_growth(a.out,a.binary,a.qualification,a.equivalence,a.pilot_root,a.swap_baseline,a.deadline,a.save_reserve_seconds)
    print(json.dumps({'status':plan['status'],'plan_sha256':plan['plan_sha256']}))


if __name__ == '__main__': main()
