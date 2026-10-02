"""Prepare an explicitly owner-resumed campaign without changing frozen estimands."""
import argparse
import json
import math
from pathlib import Path
import shutil
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.flop_check_runtime import append,machine_snapshot
from src.diagnostics.saved_hu20 import file_hash
from scripts.run_turn_check import counter_record


def prepare(original,inventory,old_run,out,config_out,repo,amendment,*,memory_budget_gib=None,results_inventory=None):
    original=Path(original).resolve();inventory=Path(inventory).resolve();amendment=Path(amendment).resolve()
    old_run=Path(old_run).resolve();out=Path(out).resolve();repo=Path(repo).resolve()
    config=json.loads(original.read_text());old_inv=json.loads((repo/config['inventory_path']).read_text())
    if file_hash(repo/config['inventory_path'])!=config['inventory_sha256']:raise ValueError('Original inventory differs')
    new_inv=json.loads(Path(inventory).read_text())
    if not new_inv['all_inputs_verified']:raise ValueError('Unverified resumed inputs')
    for name in ('upstream_commit','external_tool_files_sha256','pokers_native_sha256'):
        if new_inv[name]!=old_inv[name]:raise ValueError('Qualified external/native solver changed: '+name)
    changes={p for p,w in old_inv['repository_source_sha256'].items()
             if new_inv['repository_source_sha256'].get(p)!=w}
    if changes-{'scripts/run_turn_check.py','scripts/report_turn_check.py','scripts/readmit_turn_check.py','src/diagnostics/turn_report.py'}:
        raise ValueError('Unqualified solver adapter changes: '+str(sorted(changes)))
    failure=json.loads((old_run/'failure.json').read_text());admission=json.loads((old_run/'admission.json').read_text())
    if admission['protocol_sha256']!=file_hash(original):raise ValueError('Original admission/config differs')
    prior=failure['elapsed_seconds']
    if not 0<=prior<config['main_seconds_ceiling']:raise ValueError('No remaining original compute allowance')
    evidence=Path(results_inventory).resolve() if results_inventory else repo/'docs/reports/hu20-exact-turn-check-artifacts/main-file-inventory.json'
    source_inventory=json.loads(evidence.read_text())['files']
    results=[]
    for path in sorted((old_run/'spots').glob('*/result.json')):
        relative=str(path.relative_to(old_run));wanted=source_inventory[relative]['sha256']
        if file_hash(path)!=wanted:raise ValueError('Original completed result differs')
        row=json.loads(path.read_text())
        attempt=Path(row['attempt_path'])
        if file_hash(attempt/'prepared/request.json')!=row['request_sha256']:raise ValueError('Original request differs')
        if file_hash(attempt/'solver/response.jsonl')!=row['runtime']['response_sha256']:raise ValueError('Original response differs')
        results.append((path,row))
    if len(results)!=failure['jobs_done']:raise ValueError('Completed result count differs')
    before=machine_snapshot();gib=1024**3
    if memory_budget_gib is not None:
        if not config['memory_budget_gib']<=memory_budget_gib<=10:
            raise ValueError('Resource readmission must remain within the owner 10-GiB limit')
        if before['swap_used_bytes']-config['swap_baseline_bytes']>gib:
            raise MemoryError('Resource readmission cannot reset a failed swap guard')
        config['memory_budget_gib']=memory_budget_gib
    if math.floor(before['reclaimable_bytes']*.8/gib)<config['memory_budget_gib']:raise MemoryError('Insufficient resumed admission headroom')
    out.mkdir(parents=True,exist_ok=False)
    preserved={}
    for path,row in results:
        destination=out/'spots'/path.parent.name;destination.parent.mkdir(exist_ok=True)
        shutil.copytree(path.parent,destination)
        relative=str((destination/'result.json').relative_to(out));preserved[relative]=file_hash(destination/'result.json')
        append(out/'progress.jsonl',counter_record(row))
    atomic_json(out/'machine-readmission.json',before)
    config.update(inventory_path=str(Path(inventory).relative_to(repo)),inventory_sha256=file_hash(inventory),
                  swap_baseline_bytes=config['swap_baseline_bytes'] if memory_budget_gib is not None else before['swap_used_bytes'],
                  resume_amendment={'path':str(Path(amendment).relative_to(repo)),'sha256':file_hash(amendment)},
                  resume={'from_run':str(old_run),'original_config_sha256':file_hash(original),
                          'old_admission_sha256':file_hash(old_run/'admission.json'),
                          'old_failure_sha256':file_hash(old_run/'failure.json'),
                          'previous_active_seconds':prior,'owner_pause_excluded':memory_budget_gib is None,
                          'preserved_results':preserved,'python_source_changes':sorted(changes),
                          'results_inventory_sha256':file_hash(evidence),
                          'machine_snapshot_sha256':file_hash(out/'machine-readmission.json')})
    atomic_json(config_out,config)
    return {'preserved_jobs':len(results),'remaining_jobs':config['jobs_total']-len(results),
            'previous_active_seconds':prior,'remaining_seconds':config['main_seconds_ceiling']-prior,
            'swap_baseline_bytes':config['swap_baseline_bytes'],'config_sha256':file_hash(config_out)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original','inventory','old-run','out','config-out','amendment'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--memory-budget-gib',type=float,help='Resource-only readmission within measured headroom and the owner 10-GiB ceiling')
    p.add_argument('--results-inventory',type=Path,help='Hash inventory of the stopped run being preserved')
    p.add_argument('--repo',type=Path,default=Path.cwd());a=p.parse_args()
    print(json.dumps(prepare(a.original,a.inventory,a.old_run,a.out,a.config_out,a.repo,a.amendment,
        memory_budget_gib=a.memory_budget_gib,results_inventory=a.results_inventory),sort_keys=True))
if __name__=='__main__':main()
