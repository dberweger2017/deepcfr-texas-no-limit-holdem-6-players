"""Wait without heavy work; run only the frozen 1M parity reference after #128."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HOME = Path('/Users/dberweger/Local')
CLONE = HOME/'hu20-platform-parity-pr129'
AUDIT = HOME/'hu20-posterior-audit-v2/results/hu20-posterior-audit-v2-m4-20260930'
FINISHED = HOME/'posterior-audit-v2-wrapper-finished-20260930.json'
STATE = HOME/'hu20-platform-parity-pr129-reference-status.json'
COORDINATION = Path('/tmp/DR_RESEARCH_M4_COORDINATION.txt')
ARCHIVE = HOME/'hu20-linux-pilot-pr129.tar'
EXPECTED_ARCHIVE = 'ca096bc89593f1f9dee4106389fb698a5fedb2bc9352c89d769d60db7018cb39'
EXPECTED_SOURCE = 'e18f0079a14addc90938acca8c30795e8af09691'
PYTHON = str(HOME/'deepcfr-texas-no-limit-holdem-6-players/.venv/bin/python')
WAIT_DEADLINE = 1790814715.584733  # #128 hard cutoff plus 30 minutes; no indefinite queue.

def record(status, **fields):
    STATE.write_text(json.dumps({'status':status, 'pid':os.getpid(), 'time':time.time(), **fields},indent=2)+'\n')

record('waiting_for_audit_release', wait_deadline=WAIT_DEADLINE, source=EXPECTED_SOURCE)
try:
    while True:
        if time.time() >= WAIT_DEADLINE:
            raise TimeoutError('Audit did not release before parity queue expiry')
        if FINISHED.exists() and (AUDIT/'reporting-status.json').exists():
            audit = json.loads((AUDIT/'coordinator.json').read_text())
            if audit['status'] != 'running':
                owned = [audit['owner_pid']] + [p['pid'] for p in audit['phases']]
                alive = False
                for pid in owned:
                    try:
                        os.kill(pid,0)
                        alive = True
                    except ProcessLookupError:
                        pass
                if not alive:
                    break
        time.sleep(30)
    os.chdir(CLONE)
    sys.path.insert(0,str(CLONE))
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() != EXPECTED_SOURCE:
        raise ValueError('Frozen parity source changed')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Parity source is not clean')
    from scripts.run_exact_ranker_experiment import guard, swap
    started = time.time()
    clock = {'deadline':started+3600, 'swap_start_mib':swap()}
    root = CLONE/'results/m4-platform-pilot'
    root.mkdir(parents=True,exist_ok=False)
    guard(root,clock)
    record('running', source=EXPECTED_SOURCE, clock=clock, started=started)
    with COORDINATION.open('a') as f:
        f.write(f'\nDoctor Research PR #129 CLAIM after posterior-audit release: reference supervisor {os.getpid()}, fixed 1M direct/resume, source {EXPECTED_SOURCE}; one heavy child, at most one hour including checks, no playing evaluation.\n')
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != EXPECTED_ARCHIVE:
        raise ValueError('Linux transfer archive checksum mismatch')
    commands = [
        ('extract-linux', ['tar','-xf',str(ARCHIVE),'-C',str(root)]),
        ('focused-tests', [PYTHON,'-m','pytest','-q','tests/test_hu20_platform_pilot.py']),
        ('direct', [PYTHON,'-m','scripts.hu20_platform_pilot','run','--plan','configs/blueprint/runpod-hu20-parity.json','--out',str(root/'direct')]),
        ('resumed', [PYTHON,'-m','scripts.hu20_platform_pilot','run','--plan','configs/blueprint/runpod-hu20-parity.json','--out',str(root/'resumed'),'--resume',str(root/'direct')]),
        ('resume-comparison', [PYTHON,'-m','scripts.hu20_platform_pilot','compare','--left',str(root/'direct'),'--right',str(root/'resumed'),'--out',str(root/'resume-comparison.json')]),
        ('platform-comparison', [PYTHON,'-m','scripts.hu20_platform_pilot','compare','--left',str(root/'direct'),'--right',str(root/'results/platform-pilot/direct'),'--out',str(root/'platform-comparison.json')]),
    ]
    samples = []
    for phase, command in commands:
        guard(root,clock)
        with (root/f'{phase}.log').open('x') as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while child.poll() is None:
                    samples.append(guard(root,clock,child.pid))
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f'{phase} exited {child.returncode}; retained, no retry')
            except Exception:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    child.wait(timeout=10)
                raise
    (root/'resources.json').write_text(json.dumps(samples,indent=2)+'\n')
    expected = json.loads((root/'results/linux-inventory.json').read_text())
    for name, metadata in expected.items():
        p = root/'results'/name
        if p.stat().st_size != metadata['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != metadata['sha256']:
            raise ValueError(f'Linux archive member changed: {name}')
    direct = [json.loads(x) for x in (root/'direct/iterations.jsonl').read_text().splitlines()]
    linux = [json.loads(x) for x in (root/'results/platform-pilot/direct/iterations.jsonl').read_text().splitlines()]
    resumed = [json.loads(x) for x in (root/'resumed/iterations.jsonl').read_text().splitlines()]
    mid = json.loads((root/'direct/midpoint.json').read_text())
    m4_result = json.loads((root/'direct/result.json').read_text())
    linux_result = json.loads((root/'results/platform-pilot/direct/result.json').read_text())
    checks = {'linux_files_verified':len(expected), 'all_non_timing_iterations_equal':direct==linux,
              'resume_suffix_equal':resumed==direct[mid['iteration']:],
              'next_rng_streams_equal':m4_result['next_streams']==linux_result['next_streams']}
    (root/'independent-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not all(checks.values()):
        raise ValueError('Independent work/RNG/resume mismatch')
    inventory = {str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(root.rglob('*')) if p.is_file()}
    (root/'final-manifest.json').write_text(json.dumps({'source':EXPECTED_SOURCE,'clock':clock,'files':inventory},indent=2)+'\n')
    record('complete', source=EXPECTED_SOURCE, root=str(root), clock=clock, started=started)
except Exception as exc:
    record('failed', failure=f'{type(exc).__name__}: {exc}', source=EXPECTED_SOURCE)
    raise
finally:
    with COORDINATION.open('a') as f:
        f.write(f'\nDoctor Research PR #129 reference supervisor {os.getpid()} ended: {STATE}; no heavy child remains.\n')
