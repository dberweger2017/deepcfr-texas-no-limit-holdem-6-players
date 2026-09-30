import gzip, hashlib, json, pathlib

root = pathlib.Path('/workspace/hu20-parity/results/platform-pilot')
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
direct = [json.loads(x) for x in (root/'direct/iterations.jsonl').read_text().splitlines()]
resumed = [json.loads(x) for x in (root/'resumed/iterations.jsonl').read_text().splitlines()]
mid = json.loads((root/'direct/midpoint.json').read_text())
suffix = direct[mid['iteration']:]
a = (pathlib.Path('/workspace/m4-historical')/'current.json.gz').read_bytes()
b = (root/'direct/current.json.gz').read_bytes()
differences = [i for i,(x,y) in enumerate(zip(a,b)) if x != y]
checks = {
    'midpoint_completed_nodes': mid['completed_nodes'],
    'midpoint_iteration': mid['iteration'],
    'resumed_iterations': len(resumed),
    'iteration_suffix_exact': resumed == suffix,
    'reload_byte_identical': digest(root/'direct/midpoint.json.gz') == digest(root/'resumed/reload.json.gz'),
    'historical_export_different_offsets': differences,
    'historical_export_only_gzip_os_byte_differs': len(a)==len(b) and differences == [9],
    'historical_export_gzip_os': {'m4':a[9], 'linux':b[9]},
    'historical_export_deflate_and_crc_identical': a[10:] == b[10:],
    'historical_export_payload_identical': gzip.decompress(a)==gzip.decompress(b),
}
(root/'independent-parity-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
if not checks['iteration_suffix_exact'] or not checks['reload_byte_identical'] or not checks['historical_export_payload_identical']:
    raise SystemExit('Meaningful pilot divergence')
print(json.dumps(checks))
base = pathlib.Path('/workspace/hu20-parity/results')
inventory = {str(p.relative_to(base)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(base.rglob('*')) if p.is_file()}
(base/'linux-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
