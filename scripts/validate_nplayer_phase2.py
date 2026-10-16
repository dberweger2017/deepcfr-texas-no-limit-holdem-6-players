"""Sequential native and real-K50 regressions inside the Phase 2 supervisor."""
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root=Path.cwd().parent
    if not os.environ.get('NPLAYER_STOP_FILE'): raise RuntimeError('VRM supervisor required')
    out=root/'validation';out.mkdir(exist_ok=False)
    native=Path('native/hu20-trainer/target/release/examples')
    if not native.exists(): native.symlink_to(root/'target/release/examples',target_is_directory=True)
    env=dict(os.environ,CARGO_TARGET_DIR=str(root/'target'),CARGO_BUILD_JOBS='2',
        PATH='/home/dberweger/.cargo/bin:'+os.environ['PATH'],
        MKL_CBWR='AVX2,STRICT',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        NPLAYER_ARENA_BINARY=str(root/'target/release/examples/nplayer_arena'),
        NPLAYER_K50_TABLES=str(root/'tables'),K50_REAL_TABLES=str(root/'tables'))
    tests=sorted(map(str,Path('tests').glob('test_native*.py')))+[
        'tests/test_k50_evaluation.py','tests/diagnostics/test_compact_policy.py',
        'tests/diagnostics/test_streaming_cfr_audit.py','tests/test_diagnose_hu20_checkpoints.py',
        'tests/test_nplayer_arena.py','tests/test_nplayer_phase2_runner.py',
        'tests/test_nplayer_phase2_evaluation.py','scripts/machine_qualification/test_vrm_watchdog.py']
    commands=[['cargo','test','--release','--locked','--manifest-path','native/hu20-trainer/Cargo.toml',
               '--all-targets','-j','2','--','--test-threads=4'],
              ['.venv/bin/python','-m','pytest','-q',*tests,'--basetemp='+str(out/'pytest')]]
    receipts=[]
    for i,command in enumerate(commands):
        if Path(os.environ['NPLAYER_STOP_FILE']).exists(): raise RuntimeError('stop requested')
        tick=time.monotonic()
        with (out/f'{i}.log').open('x') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        receipts.append({'command':command,'returncode':result.returncode,'seconds':time.monotonic()-tick})
        (out/'commands.json').write_text(json.dumps(receipts,indent=2)+'\n')
        if result.returncode: raise RuntimeError(f'validation {i} failed')
    (out/'result.json').write_text(json.dumps({'status':'passed','commands':receipts},indent=2)+'\n')


if __name__=='__main__':main()
