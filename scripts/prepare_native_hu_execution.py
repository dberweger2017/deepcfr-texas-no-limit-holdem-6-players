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
from scripts.prepare_native_hu_campaign import SEED, digest, file_hash, prepare
from src.blueprint.average import checked_header
from src.blueprint.abstraction import HU100_SCHEMA
from scripts.native_hu_followup_limits import envelope

DEADLINE = datetime(2026, 10, 8, 10, tzinfo=timezone.utc).timestamp()
THREAD_URI = 't3://thread/495ca3f8-32db-4e73-98ad-29d01fb9e282'
PILOT_NODES = (100000,1000000,5000000,10000000)


def bind_followup(plan, approval):
    if approval is not None:
        envelope(approval)
        plan.update(followup_approval_path=str(approval.resolve()),followup_approval_sha256=file_hash(approval))
    return envelope(approval)


def pilot_prerequisites(root):
    """Enumerate the complete admission evidence; callers cannot omit a member."""
    root=root.resolve()
    paths={root/'plan.json',root/'checkpoints.jsonl',root/'training-jobs.json',root/'export-audit-jobs.json'}
    for phase in ('training','export-audit'):
        paths.update((root/f'{phase}-guard/campaign.json',root/f'{phase}-guard/resources.jsonl'))
    for nodes in PILOT_NODES:
        audit_path=root/f'audit-{nodes}.json'
        paths.update((audit_path,root/'training'/f'HU100-{SEED}-{nodes}.json.gz',
                      root/f'current-{nodes}.json.gz',root/f'average-{nodes}.jsonl.gz'))
        # Include any additional input declared by the independent audit too.
        paths.update(Path(name).resolve() for name in json.loads(audit_path.read_text())['files'])
    return {str(p):{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(paths)}


def verify_pilot_prerequisites(root, manifest):
    if not manifest or pilot_prerequisites(root) != manifest:
        raise ValueError('Complete pilot prerequisite manifest changed or has missing members')


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
    limits=envelope(plan.get('followup_approval_path'))
    if not isfinite(deadline) or not datetime.now(timezone.utc).timestamp() < deadline <= DEADLINE:
        raise ValueError('Deadline must fit the owner October 8 12:00 Madrid ceiling')
    if out.exists(): raise FileExistsError('Use a fresh attempt root; never duplicate a launch')
    out.mkdir(parents=True); (out/'training').mkdir()
    plan.update(status='prepared-only', prepared_at=datetime.now(timezone.utc).isoformat(),
                owner_instruction=THREAD_URI, campaign_swap_baseline=swap_baseline,
                hard_deadline=deadline, limits={'rss_gib':limits['rss_gib'], 'swap_gib':.5, 'disk_gib':15.5, 'require_ac':True})
    plan['phase_jobs'] = {name: {'sha256':digest(jobs),'seconds':seconds,'extra_guard_arguments':extra} for name,jobs,seconds,extra in phases}
    plan['plan_sha256'] = digest(plan)
    (out/'plan.json').write_text(json.dumps(plan, sort_keys=True, indent=2)+'\n')
    lines = ['set -eu', '# PR188 MERGED and M4 worker-side closeout/idle admission required before execution.',
             shlex.join([sys.executable,'-m','scripts.verify_native_hu_launch','--plan',str(out/'plan.json')]),
             'export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1']
    for name, jobs, seconds, extra in phases:
        jobs_path = out/f'{name}-jobs.json'
        jobs_path.write_text(json.dumps(jobs, indent=2)+'\n')
        guard = [sys.executable, '-m', 'scripts.hu20_scaling_supervise', '--jobs', str(jobs_path),
                 '--out', str(out/f'{name}-guard'), '--rss-gib',str(limits['rss_gib']), '--disk-gib','15.5',
                 '--swap-gib','.5', '--require-ac', '--swap-baseline', swap_baseline] + extra
        if limits['system_memory_guard']: guard += ['--system-memory-guard']
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
        'binary':str(binary),'qualification_path':str(qualification_path.resolve()),'qualification_sha256':file_hash(qualification_path),
        'qualification':q,'seed':SEED,'parent_sha256':PARENT_SHA,'parent_path':str(parent),
        'parent_completed_nodes':pr['completed_nodes'],'reference_plan_sha256':rp['plan_sha256'],
        'command':command}, [('training',[{'name':'train','command':command}],900,[]),
                            ('export-audit',exports+[{'name':'compare','command':compare}],900,[])],
        swap_baseline, session_deadline)


def checked_equivalence(path, source, binary, swap_baseline):
    e = json.loads(path.read_text())
    if (e.get('status') != 'verified' or e.get('seed') != SEED or e.get('source') != source
        or e.get('binary_sha256') != file_hash(binary) or e.get('campaign_swap_baseline') != swap_baseline):
        raise ValueError('Full source-bound HU20 equivalence with campaign baseline required')
    for name, spec in e['files'].items():
        p=Path(name)
        if p.stat().st_size != spec['bytes'] or file_hash(p) != spec['sha256']:
            raise ValueError('Equivalence input changed')
    return e


def prepare_reference(out,binary,qualification_path,swap_baseline,deadline):
    out,binary=out.resolve(),binary.resolve()
    source,q=qualification(qualification_path,binary)
    plan=prepare('hu20',out,binary,swap_baseline=swap_baseline,deadline=deadline)
    plan.update(qualification=q,qualification_path=str(qualification_path.resolve()),
        qualification_sha256=file_hash(qualification_path),hard_deadline=deadline)
    plan['phase_jobs']={name:{'sha256':digest(json.loads((out/f'{name}-jobs.json').read_text())),
        'seconds':900,'extra_guard_arguments':[]} for name in ('training','export-audit')}
    plan.pop('plan_sha256'); plan['plan_sha256']=digest(plan)
    (out/'plan.json').write_text(json.dumps(plan,sort_keys=True,indent=2)+'\n')
    commands=(out/'commands.txt').read_text().replace('set -eu\n','set -eu\n'+
        shlex.join([sys.executable,'-m','scripts.verify_native_hu_launch','--plan',str(out/'plan.json')])+'\n',1)
    (out/'commands.txt').write_text(commands)
    return plan


def prepare_pilot(out,binary,qualification_path,equivalence_path,swap_baseline,deadline,approval=None):
    out,binary = out.resolve(),binary.resolve()
    source,q = qualification(qualification_path,binary)
    checked_equivalence(equivalence_path,source,binary,swap_baseline)
    plan = prepare('pilot',out,binary,swap_baseline=swap_baseline,deadline=deadline)
    limits=bind_followup(plan,approval)
    if approval is not None: plan['limits']['rss_gib']=limits['rss_gib']
    plan.update(qualification=q,qualification_path=str(qualification_path.resolve()),
        qualification_sha256=file_hash(qualification_path),equivalence_path=str(equivalence_path.resolve()),
        equivalence_sha256=file_hash(equivalence_path),hard_deadline=deadline)
    # Audit every early save. A failed or incomplete training phase still blocks
    # all these jobs through the foreground script's fail-fast shell gate.
    jobs = []
    for nodes in PILOT_NODES:
        checkpoint = out/'training'/f'HU100-{SEED}-{nodes}.json.gz'
        current,average = out/f'current-{nodes}.json.gz',out/f'average-{nodes}.jsonl.gz'
        jobs += [{'name':f'export-{nodes}','command':[str(binary),'export',str(checkpoint),
                    '--current',str(current),'--average',str(average),'--zero-mass','uniform']},
                 {'name':f'audit-{nodes}','command':[sys.executable,'-m','scripts.audit_native_hu_checkpoint',
                    '--checkpoint',str(checkpoint),'--current',str(current),'--average',str(average),
                    '--stack-bb','100','--target-nodes',str(nodes),'--out',str(out/f'audit-{nodes}.json')]}]
    plan['export_audit_jobs']=jobs
    training_jobs=json.loads((out/'training-jobs.json').read_text())
    plan['phase_jobs']={'training':{'sha256':digest(training_jobs),'seconds':900,'extra_guard_arguments':[]},
                        'export-audit':{'sha256':digest(jobs),'seconds':900,'extra_guard_arguments':[]}}
    plan.pop('plan_sha256'); plan['plan_sha256']=digest(plan)
    (out/'plan.json').write_text(json.dumps(plan,sort_keys=True,indent=2)+'\n')
    (out/'export-audit-jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    commands=(out/'commands.txt').read_text()
    if approval is not None:
        commands=commands.replace('--rss-gib 5.5','--rss-gib 10').replace('--require-ac','--require-ac --system-memory-guard')
    commands=commands.replace('set -eu\n','set -eu\n'+shlex.join([sys.executable,'-m','scripts.verify_native_hu_launch','--plan',str(out/'plan.json')])+'\n',1)
    (out/'commands.txt').write_text(commands)
    return plan


def prepare_growth(out, binary, qualification_path, equivalence_path, pilot_root, swap_baseline, deadline, capacity_path,approval=None):
    out, binary, pilot_root = (p.resolve() for p in (out,binary,pilot_root))
    source, q = qualification(qualification_path, binary)
    equivalence=checked_equivalence(equivalence_path,source,binary,swap_baseline)
    pp = json.loads((pilot_root/'plan.json').read_text())
    limits=envelope(approval)
    if approval is not None and pp.get('followup_approval_sha256')!=file_hash(approval):
        raise ValueError('Pilot must bind the same owner follow-up envelope')
    a = json.loads((pilot_root/'audit-10000000.json').read_text())
    if (pp['source'] != source or pp['binary_sha256'] != file_hash(binary) or pp['stage'] != 'pilot'
        or pp.get('campaign_swap_baseline') != swap_baseline or pp.get('equivalence_sha256') != file_hash(equivalence_path)
        or a['status'] != 'verified'
        or a['stack_bb'] != 100 or a.get('native_state', {}).get('completed_nodes',0) < 10000000
        or a['native_state']['coverage_start'] != [0,0,0]
        or any(n <= 0 for n in a['native_state']['traverser_visits_by_street'])):
        raise ValueError('Complete source-bound audited HU100 pilot/coverage required')
    for phase in ('training','export-audit'):
        if json.loads((pilot_root/f'{phase}-guard/campaign.json').read_text())['status'] != 'complete':
            raise ValueError('Both pilot guards must finish successfully')
    for phase in ('training','export-audit'):
        g=json.loads((pilot_root/f'{phase}-guard/campaign.json').read_text())
        samples=[json.loads(line) for line in (pilot_root/f'{phase}-guard/resources.jsonl').read_text().splitlines()]
        if approval is not None and (g['limits'].get('rss_gib')!=10 or not g['limits'].get('system_memory_guard')
            or any(not s.get('system_memory') or s['system_memory']['pressure_level']!=1
                   or s['system_memory']['free_percent']<15 for s in samples)):
            raise ValueError('Follow-up pilot needs continuous system-pressure evidence')
        if (g.get('swap_baseline') != swap_baseline or not samples
            or any(s['aggregate_job_rss_bytes'] >= limits['rss_gib']*1024**3 or s['swap_growth_bytes'] > .5*1024**3
                   or s['free_disk_bytes'] < 15.5*1024**3 or 'AC Power' not in (s.get('power') or '') for s in samples)):
            raise ValueError('Pilot resource telemetry/baseline is incomplete or breaches limits')
    for nodes in PILOT_NODES:
        cp=pilot_root/'training'/f'HU100-{SEED}-{nodes}.json.gz'
        rr=receipt(pilot_root/'checkpoints.jsonl',cp)
        aa=json.loads((pilot_root/f'audit-{nodes}.json').read_text())
        if (rr['requested_nodes'] != nodes or rr['binary_sha256'] != file_hash(binary)
            or aa['status'] != 'verified' or aa['audit']['checkpoint_sha256'] != file_hash(cp)):
            raise ValueError('Every early pilot save needs bound telemetry and full audit')
        for name,spec in aa['files'].items():
            if Path(name).stat().st_size != spec['bytes'] or file_hash(Path(name)) != spec['sha256']:
                raise ValueError('Pilot checkpoint/export changed since audit')
    parent = pilot_root/'training'/f'HU100-{SEED}-10000000.json.gz'
    if file_hash(parent) != a['audit']['checkpoint_sha256']: raise ValueError('Pilot parent changed')
    with gzip_open(parent,'rt') as f: h = json.loads(f.readline())
    checked_header(h, {'seed':SEED,'iteration':a['iteration']}, expected_schema=HU100_SCHEMA)
    if h.get('average_rule') != 'opponent-sampled' or 'training_options' in h or h['native_state'] != a['native_state']:
        raise ValueError('Pilot recipe/state differs')
    remaining = deadline-datetime.now(timezone.utc).timestamp()
    capacity=json.loads(capacity_path.read_text())
    if approval is not None and capacity.get('followup_approval_sha256')!=file_hash(approval):
        raise ValueError('Capacity must bind owner follow-up envelope')
    minimum_existing=sum(spec['bytes'] for spec in equivalence['files'].values())+sum(
        p.stat().st_size for p in pilot_root.rglob('*') if p.is_file())
    if capacity.get('retained_non_growth_bytes',0) < minimum_existing:
        raise ValueError('Archive duplicate must account for retained reference/recovery/pilot artifacts')
    validate_capacity(capacity,pilot_root,deadline)
    save_reserve=capacity['save_reserve_seconds']
    if not isfinite(save_reserve) or save_reserve <= 0 or remaining <= save_reserve:
        raise ValueError('No measured serialization/time headroom remains')
    command = [str(binary),'train','--stack-bb','100','--average-rule','opponent-sampled',
        '--resume',str(parent),'--resume-sha256',file_hash(parent),'--nodes','10000000000',
        '--milestones',','.join(str(n*1000000000) for n in range(1,10)),
        '--max-entries',str(capacity['forecast_entry_ceiling']),'--max-seconds',str(remaining-save_reserve),
        '--out',str(out/'training/HU100-2026100601-{nodes}.json.gz'),
        '--stop-file',str(out/'controlled-stop.json'),'--telemetry',str(out/'checkpoints.jsonl')]
    soft_rss=4.0 if approval is None else limits['rss_gib']-capacity['serialization_rss_bytes']/1024**3-.5
    extra = ['--stop-file',str(out/'controlled-stop.json'),'--soft-rss-gib',str(soft_rss),
             '--save-reserve-seconds',str(save_reserve)]
    plan={'stage':'growth','source':source,'binary_sha256':file_hash(binary),
        'binary':str(binary),'qualification_path':str(qualification_path.resolve()),'qualification_sha256':file_hash(qualification_path),
        'qualification':q,'seed':SEED,'target_total_nodes':10000000000,
        'equivalence_path':str(equivalence_path.resolve()),'equivalence_sha256':file_hash(equivalence_path),
        'pilot_root':str(pilot_root),'pilot_prerequisites':pilot_prerequisites(pilot_root),
        'pilot_audit_sha256':file_hash(pilot_root/'audit-10000000.json'),'parent_path':str(parent),
        'parent_sha256':file_hash(parent),'command':command,
        'serialization_headroom_gib':limits['rss_gib']-soft_rss,'capacity_plan':capacity,'capacity_path':str(capacity_path.resolve()),
        'capacity_sha256':file_hash(capacity_path)}
    bind_followup(plan,approval)
    return write_plan(out,plan,
        [('training',[{'name':'train','command':command}],remaining,extra)],swap_baseline,deadline)


def validate_capacity(c,pilot_root,training_deadline):
    import shutil
    limits=envelope(c.get('followup_approval_path'))
    if c.get('followup_approval_path') and file_hash(Path(c['followup_approval_path']))!=c.get('followup_approval_sha256'):
        raise ValueError('Capacity owner approval changed')
    rows=[json.loads(line) for line in (pilot_root/'checkpoints.jsonl').read_text().splitlines()]
    if (c.get('pilot_telemetry_sha256') != file_hash(pilot_root/'checkpoints.jsonl')
        or c.get('pilot_audit_sha256') != file_hash(pilot_root/'audit-10000000.json')):
        raise ValueError('Capacity plan must bind pilot measurements')
    verify_pilot_prerequisites(pilot_root,c['measurement_files'])
    # Forecasts are uncertain; the raw arithmetic is retained and hard guards
    # remain authoritative. Positive values alone cannot satisfy reserve admission.
    n=c['forecast_entry_ceiling']; growth=max(1,n/max(r['diagnostics']['entries'] for r in rows))
    minimum_write=max(r['write_seconds'] for r in rows)*growth*2
    maximum_bytes_per_key=max(r['checkpoint_bytes']/max(1,r['diagnostics']['entries']) for r in rows)
    max_cp=maximum_bytes_per_key*n*2
    minimum_disk=max_cp*(10+1+10+2)  # ten retained growth saves, atomic temp, archive duplicate, export pair
    if (type(n) is not int or n <= max(r['diagnostics']['entries'] for r in rows)
        or any(not isfinite(c[k]) or c[k] <= 0 for k in
        ('save_reserve_seconds','export_audit_reserve_seconds','archive_reserve_seconds',
         'serialization_rss_bytes','export_audit_rss_bytes','disk_reserve_bytes'))
        or c['save_reserve_seconds'] < minimum_write
        or c['serialization_rss_bytes'] < 32*n+64*1024**2
        or c['serialization_rss_bytes'] >= limits['serialization_gib']*1024**3
        or c['disk_reserve_bytes'] < minimum_disk+c['retained_non_growth_bytes']
        or c['archive_reserve_seconds'] < minimum_write*11
        or shutil.disk_usage(pilot_root).free-c['disk_reserve_bytes'] < 15.5*1024**3
        or training_deadline+c['export_audit_reserve_seconds']+c['archive_reserve_seconds'] > DEADLINE):
        raise ValueError('Measured capacity/save/export/archive reserve fails admission')
    # Each export/audit command timing is measured by its actual supervisor receipt.
    attempts=json.loads((pilot_root/'export-audit-guard/campaign.json').read_text())['attempts']
    if c['export_audit_reserve_seconds'] < sum(a['finished']-a['started'] for a in attempts)*growth*2:
        raise ValueError('Measured export/audit reserve is too short')
    # Both tools retain table-sized data. Forecast their measured aggregate RSS
    # at the largest pilot table independently, including process overhead,
    # with 2x headroom. Also cover every earlier measured peak; small-table
    # process startup costs are not extrapolated as a per-key allocation.
    # The continuous external guard remains authoritative if the forecast errs.
    entries={r['requested_nodes']:r['diagnostics']['entries'] for r in rows}
    required={f'{phase}-{nodes}' for phase in ('export','audit') for nodes in PILOT_NODES}
    if (len(attempts)!=len(required) or {a['name'] for a in attempts}!=required
        or any(a.get('status')!='complete' or not isfinite(a.get('peak_aggregate_job_rss_bytes',0))
               or a.get('peak_aggregate_job_rss_bytes',0)<=0 for a in attempts)
        or any(entries.get(nodes,0)<=0 for nodes in PILOT_NODES)):
        raise ValueError('Measured RSS for every pilot export and audit required')
    largest=max(entries,key=entries.get)
    forecast=max(max(a['peak_aggregate_job_rss_bytes'] for a in attempts),
                 max(a['peak_aggregate_job_rss_bytes'] for a in attempts
                     if a['name'].endswith(f'-{largest}'))*n/entries[largest])*2
    if c['export_audit_rss_bytes'] < forecast or c['export_audit_rss_bytes'] >= limits['rss_gib']*1024**3:
        raise ValueError('Measured export/audit memory forecast exceeds admitted capacity')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('reference','recovery','pilot','growth'),required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True); p.add_argument('--swap-baseline',required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--reference-root',type=Path); p.add_argument('--parent',type=Path)
    p.add_argument('--equivalence',type=Path); p.add_argument('--pilot-root',type=Path)
    p.add_argument('--capacity-plan',type=Path); a = p.parse_args()
    if not isfinite(a.deadline) or not datetime.now(timezone.utc).timestamp() < a.deadline <= DEADLINE:
        raise ValueError('Deadline must fit the owner October 8 12:00 Madrid ceiling')
    if a.stage == 'reference':
        plan=prepare_reference(a.out,a.binary,a.qualification,a.swap_baseline,a.deadline)
    elif a.stage == 'recovery':
        if a.reference_root is None or a.parent is None: p.error('Recovery needs reference root and retained parent')
        plan = prepare_recovery(a.out,a.binary,a.qualification,a.reference_root,a.parent,a.swap_baseline,a.deadline)
    elif a.stage == 'pilot':
        if a.equivalence is None: p.error('Pilot needs verified HU20 equivalence')
        plan=prepare_pilot(a.out,a.binary,a.qualification,a.equivalence,a.swap_baseline,a.deadline)
    else:
        if a.equivalence is None or a.pilot_root is None or a.capacity_plan is None:
            p.error('Growth needs equivalence, audited pilot and measured save reserve')
        plan = prepare_growth(a.out,a.binary,a.qualification,a.equivalence,a.pilot_root,a.swap_baseline,a.deadline,a.capacity_plan)
    print(json.dumps({'status':plan['status'],'plan_sha256':plan['plan_sha256']}))


if __name__ == '__main__': main()
