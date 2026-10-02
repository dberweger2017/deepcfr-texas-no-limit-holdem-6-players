"""Outcome-free D resource prefixes and independent checkpoint recovery audit."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.preflight_hu20_history import canonical, file_hash, write
from src.blueprint.artifact import load_training, save_training, export_policy


def audit_worker(root, name, mode):
    folder = root/name
    result = json.loads((folder/'result.json').read_text())
    checkpoint = folder/'checkpoint-2000000.json.gz'
    assert file_hash(checkpoint) == result['milestones'][-1]['checkpoint_sha256']
    trainer = load_training(checkpoint)
    assert asdict(trainer.config) == result['config']
    assert trainer.iteration == result['iterations']
    assert len(trainer.nodes) == result['milestones'][-1]['entries']
    rows = [json.loads(line) for line in (folder/'iterations.jsonl').open()]
    assert sum(node.visits for node in trainer.nodes.values()) == sum(row['raw_traverser_visits'] for row in rows)
    out = root/'verification'/name/mode
    out.mkdir(parents=True, exist_ok=False)
    if mode == 'reload':
        assert save_training(trainer, out/'checkpoint.gz') == file_hash(checkpoint)
        assert export_policy(trainer, out/'current.gz') == result['export_sha256']
        checks = dict(status='passed', exact_reload_and_export=True)
    else:
        step = asdict(trainer.step())
        step.pop('elapsed_seconds'); step.pop('updated_keys')
        checks = dict(status='passed', work=step,
                      checkpoint_sha256=save_training(trainer, out/'checkpoint.gz'),
                      export_sha256=export_policy(trainer, out/'current.gz'))
    write(out/'checks.json', checks)


def run(plan_path, root):
    from hashlib import sha256
    plan = json.loads(plan_path.read_text())
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Clean frozen executable required')
    if file_hash(Path('src/blueprint/cards_v2.py')) != plan['descriptor_sha256']:
        raise ValueError('PR143 descriptor hash mismatch')
    if plan['execution_cells'] != ['compressed'] or plan['milestones'][-1] != 2000000:
        raise ValueError('Only the frozen D 2M resource prefix is admitted')
    root.mkdir(parents=True, exist_ok=False)
    write(root/'frozen-plan.json', plan)
    start = time.time()
    state = dict(status='running', pid=os.getpid(), source=source,
                 plan_sha256=sha256(canonical(plan)).hexdigest(), started=start,
                 deadline=start+7200, completed=[], child_pid=None, paid_spend_usd=0)
    write(root/'supervisor.json', state)
    def child(label, args):
        if time.time() >= state['deadline']:
            raise RuntimeError('Resource prefix total engineering time guard')
        with (root/(label+'.log')).open('w') as log:
            p = subprocess.Popen([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT)
            state.update(phase=label, child_pid=p.pid, heartbeat=time.time())
            write(root/'supervisor.json', state)
            try:
                while p.poll() is None:
                    if time.time() >= state['deadline']:
                        p.terminate()
                        raise RuntimeError('Resource prefix total engineering time guard')
                    time.sleep(10)
                    state['heartbeat'] = time.time(); write(root/'supervisor.json', state)
                state['completed'].append(dict(phase=label, exit_code=p.returncode))
                if p.returncode:
                    raise RuntimeError(f'{label} exited {p.returncode}; retained, no retry')
            finally:
                if p.poll() is None:
                    p.terminate()
                    try: p.wait(timeout=10)
                    except subprocess.TimeoutExpired: p.kill(); p.wait()
                state['child_pid'] = None; write(root/'supervisor.json', state)
    try:
        child('corpus', ['-m', 'scripts.preflight_hu20_history', 'corpus', '--plan', str(plan_path), '--out', str(root/'corpus')])
        checks = []
        for seed in plan['seeds']:
            name = f'compressed-{seed}'
            child(name, ['-m', 'scripts.preflight_hu20_history', 'worker', '--plan', str(plan_path), '--out', str(root/name), '--probe', str(root/'corpus'), '--seed', str(seed), '--cell', 'compressed'])
            result = json.loads((root/name/'result.json').read_text())
            assert result['status'] == 'complete' and result['failure'] is None
            assert result['source'] == source and result['plan_sha256'] == state['plan_sha256']
            rows = [json.loads(line) for line in (root/name/'iterations.jsonl').open()]
            total = visits = new_entries = 0
            for i, row in enumerate(rows, 1):
                assert row['iteration'] == i
                total += row['nodes']; visits += row['raw_traverser_visits']; new_entries += row['new_entries']
                assert row['completed_nodes'] == total
                assert row['nodes'] == row['terminals'] + sum(row['attempted_work']['nodes_by_street'].values())
                assert row['raw_traverser_visits'] == sum(row['traverser_visits_by_street'].values())
            assert total == result['completed_nodes'] and len(rows) == result['iterations']
            assert 2000000 <= total < 2250000
            assert new_entries == result['milestones'][-1]['entries']
            for milestone in result['milestones']:
                p = root/name/f"checkpoint-{milestone['requested_nodes']}.json.gz"
                assert file_hash(p) == milestone['checkpoint_sha256'] and p.stat().st_size == milestone['checkpoint_bytes']
                assert rows[milestone['iteration']-1]['completed_nodes'] == milestone['completed_nodes']
            assert file_hash(root/name/'current.json.gz') == result['export_sha256']
            for mode in ('reload', 'next-a', 'next-b'):
                child(name+'-'+mode, ['-m', 'scripts.preflight_dr2x2_resources', '--out', str(root), '--audit-worker', name, '--mode', mode])
            a = json.loads((root/'verification'/name/'next-a/checks.json').read_text())
            b = json.loads((root/'verification'/name/'next-b/checks.json').read_text())
            assert a == b
            checks.append(dict(seed=seed, status='passed', nodes=total, entries=new_entries, visits=visits,
                               all_milestone_hashes=True, exact_reload_export=True, fresh_process_next_step=True))
        write(root/'verification.json', dict(status='passed', source=source, workers=checks,
                                             strength_outcomes_inspected=False))
        state.update(status='complete', finished=time.time(), wall_seconds=time.time()-start)
    except Exception as exc:
        state.update(status='failed', failure=f'{type(exc).__name__}: {exc}', finished=time.time())
    write(root/'supervisor.json', state)
    files = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=file_hash(p))
             for p in sorted(root.rglob('*')) if p.is_file()}
    write(root/'final-manifest.json', dict(source=source, plan_sha256=state['plan_sha256'], files=files))
    return state['status'] != 'complete'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit-worker')
    parser.add_argument('--mode', choices=('reload', 'next-a', 'next-b'))
    args = parser.parse_args()
    if args.audit_worker:
        audit_worker(args.out, args.audit_worker, args.mode)
        return 0
    return run(args.plan, args.out)


if __name__ == '__main__':
    raise SystemExit(main())
