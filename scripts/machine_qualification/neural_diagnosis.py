"""Bounded diagnostic matrix; preserves every probe and strict pytest outcome."""
import json
import os
from pathlib import Path
import subprocess

root = Path(os.environ['QUALIFICATION_ROOT'])
out = root / os.environ.get('QUALIFICATION_DIAGNOSIS_LABEL', 'neural-diagnosis-results')
out.mkdir(exist_ok=False)
repo = root / 'repo-main'
python = str(repo / '.venv/bin/python')
cases = [('baseline2', {'QUALIFICATION_TORCH_THREADS': '2'}),
         ('single1', {'QUALIFICATION_TORCH_THREADS': '1'}),
         ('compatible2', {'QUALIFICATION_TORCH_THREADS': '2', 'MKL_CBWR': 'COMPATIBLE'}),
         ('aten-default2', {'QUALIFICATION_TORCH_THREADS': '2', 'ATEN_CPU_CAPABILITY': 'default'})]
if os.environ.get('QUALIFICATION_DIAGNOSIS_STRICT'):
    cases = [('baseline2-gru', {'QUALIFICATION_TORCH_THREADS': '2'}),
             ('strict-avx2', {'QUALIFICATION_TORCH_THREADS': '2', 'MKL_CBWR': 'AVX2,STRICT'})]
results = []
for name, options in cases:
    env = dict(os.environ, OMP_NUM_THREADS=options['QUALIFICATION_TORCH_THREADS'],
               MKL_NUM_THREADS=options['QUALIFICATION_TORCH_THREADS'],
               OPENBLAS_NUM_THREADS=options['QUALIFICATION_TORCH_THREADS'], **options)
    with (out / f'{name}-probe.json').open('w') as stream:
        probe = subprocess.run([python, str(root / 'neural_identity_probe.py')],
            cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
    with (out / f'{name}-pytest.log').open('w') as stream:
        test = subprocess.run([python, '-m', 'pytest', '-q', '-ra',
            'tests/test_multistreet_invariants.py'], cwd=repo, env=env,
            stdout=stream, stderr=subprocess.STDOUT)
    results.append({'name': name, 'environment': options,
                    'probe_exit': probe.returncode, 'strict_test_exit': test.returncode})
    (out / 'outcomes.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results[-1]), flush=True)
if any(r['probe_exit'] for r in results):
    raise SystemExit('Diagnostic probe failed; inspect before interpretation')
