"""Run the once-only recipe and recovery gates under the external campaign guard."""
import gzip
import json
from pathlib import Path
import subprocess
import sys
from time import time

from scripts.hu20_k50_first_pass_assets import check, recipe_gate, recovery_gate, restore_tables, write, REFERENCE
from src.policies.files import file_hash

BASE = Path('results/hu20-k50-5b')
BINARY = Path('native/hu20-trainer/target/release/hu20-trainer')


def run(name, command, output, telemetry):
    if output.exists() or telemetry.exists():
        raise FileExistsError('Preserve prior native stage: ' + name)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(BASE/'gates'/f'{name}-intent.json', {'name': name, 'command': command, 'started': time(),
          'binary_sha256': file_hash(BINARY), 'source': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    subprocess.run(command, check=True)
    records = [json.loads(line) for line in telemetry.read_text().splitlines()]
    if not records or any(row['status'] != 'saved' or row['stop_requested'] for row in records):
        raise ValueError('Native stage did not reach its exact requested complete-iteration target')
    for row in records:
        check(Path(row['path']), row['checkpoint_sha256'], row['checkpoint_bytes'])
    return records


def main():
    if len(sys.argv) != 1:
        raise ValueError('This is a once-only gate worker, with no extra flags')
    if not json.loads((BASE/'reference-restoration.json').read_text())['status']=='verified':
        raise ValueError('Reference not verified')
    check(BASE/'inputs/v1-seed602-1b.json.gz', REFERENCE)
    with gzip.open(BASE/'inputs/v1-seed602-1b.json.gz','rt') as source:
        header=json.loads(source.readline())
    cap=header['config']['max_entries']
    fresh=BASE/'gates/v1-seed602-fresh-1b.json.gz'
    telemetry=BASE/'gates/v1-seed602-fresh-1b.telemetry.jsonl'
    run('recipe-v1', [str(BINARY), 'train', '--nodes','1000000000','--seed','2026100602',
        '--average-rule','opponent-sampled','--max-entries',str(cap),'--telemetry',str(telemetry),'--out',str(fresh)], fresh, telemetry)
    recipe_gate(BASE)
    restore_tables(BASE)
    tables=str(BASE/'tables')
    template=BASE/'gates/k50-fresh-{nodes}.json.gz'
    fresh=BASE/'gates/k50-fresh-4000000.json.gz'
    telemetry=BASE/'gates/k50-fresh.telemetry.jsonl'
    records=run('recovery-fresh', [str(BINARY),'train','--nodes','4000000','--milestones','2000000',
        '--seed','2026100601','--average-rule','opponent-sampled','--max-entries',str(cap),'--card-buckets',tables,
        '--telemetry',str(telemetry),'--out',str(template)],fresh,telemetry)
    midpoint,terminal=records
    if midpoint['requested_nodes']!=2000000 or terminal['requested_nodes']!=4000000:
        raise ValueError('Foreign short-run schedule')
    resumed=BASE/'gates/k50-resumed-4m.json.gz'
    telemetry=BASE/'gates/k50-resumed.telemetry.jsonl'
    result=run('recovery-resumed', [str(BINARY),'train','--nodes','4000000','--resume',midpoint['path'],
        '--resume-sha256',midpoint['checkpoint_sha256'],'--completed-nodes',str(midpoint['completed_nodes']),
        '--seed','2026100601','--average-rule','opponent-sampled','--max-entries',str(cap),'--card-buckets',tables,
        '--telemetry',str(telemetry),'--out',str(resumed)],resumed,telemetry)
    if result[0]['completed_nodes']!=terminal['completed_nodes'] or result[0]['iteration']!=terminal['iteration']:
        raise ValueError('Resumed stop semantics differ')
    recovery_gate(BASE)
    write(BASE/'gates-complete.json',{'status':'verified','finished':time(), 'recipe_gate':'recipe-gate.json',
          'recovery_gate':'recovery-gate.json','source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'binary_sha256':file_hash(BINARY)})


if __name__=='__main__':
    main()
