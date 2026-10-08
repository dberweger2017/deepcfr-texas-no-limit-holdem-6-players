"""Full HU20 legacy recovery equivalence; metadata exclusions are explicit and bounded.

Run only under the campaign resource supervisor. Never infer legacy completed
nodes from a filename: reference telemetry must bind the retained parent hash.
"""
import argparse
from gzip import open as gzip_open
from itertools import zip_longest
import json
from pathlib import Path
import struct
import subprocess
import sys
import shutil
from time import time
from scripts.native_hu_followup_limits import envelope, memory_snapshot, unsafe_memory
from scripts.train_hu20 import system
from scripts.run_tp20_campaign import swap_bytes

from scripts.audit_native_hu_checkpoint import inspect
from scripts.prepare_native_hu_campaign import REFERENCE, SEED, file_hash
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.blueprint.average import checked_header, checked_row

PARENT_SHA = 'a5318cd586c0b170a68334e4236111faddabaf7f686c071958757db888afab47'


def exact(a, b):
    """Compare IEEE-754 bits, including signed zero, rather than tolerant probabilities."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return struct.pack('!d', a) == struct.pack('!d', b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def read_header(path):
    with gzip_open(path, 'rt') as source:
        h = json.loads(source.readline())
    checked_header(h, {'seed': SEED, 'iteration': h['iteration']}, expected_schema=HU20_UNCAPPED_SCHEMA)
    if (h.get('average_rule') != 'opponent-sampled' or 'training_options' in h
        or h['config']['roots_per_seat'] != 1 or h['table']['button'] != 0):
        raise ValueError('Recovery comparison requires the frozen native linear recipe')
    return h


def receipt(path, checkpoint):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    matches = [r for r in rows if Path(r['path']).resolve() == checkpoint.resolve()]
    if len(matches) != 1:
        raise ValueError('Exactly one successful checkpoint receipt is required')
    r = matches[0]
    if (r['status'] != 'saved' or r['checkpoint_sha256'] != file_hash(checkpoint)
        or r['checkpoint_bytes'] != checkpoint.stat().st_size
        or r['completed_nodes'] < r['requested_nodes']):
        raise ValueError('Checkpoint receipt identity/completeness differs')
    return r


def compare(reference, resumed, parent, reference_telemetry, resumed_telemetry,
            reference_current, reference_average, resumed_current, resumed_average,
            training_qualification=None, isolated_audits=False, audit_guard=None):
    reference_plan_path, resumed_plan_path = reference_telemetry.parent/'plan.json', resumed_telemetry.parent/'plan.json'
    plans = [json.loads(p.read_text()) for p in (reference_plan_path,resumed_plan_path)]
    baseline = plans[0].get('campaign_swap_baseline')
    if not baseline or plans[1].get('campaign_swap_baseline') != baseline:
        raise ValueError('Reference/recovery must share one campaign swap baseline')
    if audit_guard is not None and audit_guard['swap_baseline']!=baseline:
        raise ValueError('Audit guard must preserve retained campaign swap baseline')
    parent_hash = file_hash(parent)
    if parent_hash != PARENT_SHA:
        raise ValueError('Retained historical 500M parent differs')
    # Locate the new 500M save by its independently retained member hash, then
    # validate its bytes/receipt before trusting its lifetime node counter.
    records = [json.loads(line) for line in reference_telemetry.read_text().splitlines()]
    parent_records = [r for r in records if r['checkpoint_sha256'] == parent_hash]
    if len(parent_records) != 1:
        raise ValueError('Reference 500M save does not reproduce the retained parent')
    parent_receipt = receipt(reference_telemetry, Path(parent_records[0]['path']))
    if parent_receipt['requested_nodes'] != 500000000 or parent_receipt['iteration'] != 1095942:
        raise ValueError('Wrong retained parent endpoint')
    r, s = receipt(reference_telemetry, reference), receipt(resumed_telemetry, resumed)
    if (not r.get('binary_sha256') or r['binary_sha256'] != s.get('binary_sha256')
        or r['binary_sha256'] != parent_receipt.get('binary_sha256')):
        raise ValueError('Reference/resume executed binary differs')
    source = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    training_source = source
    if training_qualification is not None:
        q = json.loads(training_qualification.read_text())
        if (q.get('status') != 'verified' or not q.get('independent_review')
            or not q.get('checks') or any(c.get('status') != 'passed' for c in q['checks'])
            or q.get('binary_sha256') != r['binary_sha256'] or not q.get('source')):
            raise ValueError('Retained training qualification differs')
        training_source = q['source']
        if any(p.get('qualification_sha256') != file_hash(training_qualification) for p in plans):
            raise ValueError('Original plans do not bind retained training qualification')
    if any(p.get('source') != training_source or p.get('binary_sha256') != r['binary_sha256'] for p in plans):
        raise ValueError('Executed source/binary differs from reference/recovery plans')
    if (r['requested_nodes'] != 1000000000 or s['requested_nodes'] != 1000000000
        or r['completed_nodes'] != s['completed_nodes'] or r['iteration'] != s['iteration']):
        raise ValueError('Recovery final actual nodes/iterations differ')
    h, recovered, ph = read_header(reference), read_header(resumed), read_header(parent)
    if 'native_state' in h or 'native_state' in ph:
        raise ValueError('Reference and retained parent must preserve legacy headers')
    state = recovered.get('native_state')
    scientific = {k: v for k, v in recovered.items() if k != 'native_state'}
    if not exact(h, scientific):
        raise ValueError('Recovery changed scientific header/configuration')
    pd, rd = parent_receipt['diagnostics'], r['diagnostics']
    expected_state = {'version': 1, 'completed_nodes': r['completed_nodes'],
        'coverage_start': [ph['iteration'], parent_receipt['completed_nodes'], pd['traverser_visits']],
        'decisions_by_street': [x-y for x, y in zip(rd['decisions_by_street'], pd['decisions_by_street'], strict=True)],
        'traverser_visits_by_street': [x-y for x, y in zip(rd['traverser_visits_by_street'], pd['traverser_visits_by_street'], strict=True)]}
    if (not exact(state, expected_state) or any(v < 0 for v in expected_state['decisions_by_street'])
        or any(v < 0 for v in expected_state['traverser_visits_by_street'])
        or s['diagnostics']['coverage_start'] != expected_state['coverage_start']
        or s['diagnostics']['decisions_by_street'] != expected_state['decisions_by_street']
        or s['diagnostics']['traverser_visits_by_street'] != expected_state['traverser_visits_by_street']):
        raise ValueError('Recovery-only coverage metadata differs from reference deltas')
    entries = visits = 0
    previous = None
    with gzip_open(reference, 'rt') as left, gzip_open(resumed, 'rt') as right:
        next(left); next(right)
        for a, b in zip_longest(left, right):
            if a is None or b is None:
                raise ValueError('Recovery table length differs')
            x, y = json.loads(a), json.loads(b)
            checked_row(x, h['iteration'], False); checked_row(y, h['iteration'], False)
            if previous is not None and x[0] <= previous:
                raise ValueError('Reference keys must be unique and sorted')
            if not exact(x, y):
                raise ValueError(f'Recovery complete training row differs: {x[0]}')
            previous = x[0]; entries += 1; visits += x[4]
    if (entries != rd['entries'] or entries != s['diagnostics']['entries']
        or visits != rd['traverser_visits'] or visits != s['diagnostics']['traverser_visits']
        or visits != state['coverage_start'][2] + sum(state['traverser_visits_by_street'])):
        raise ValueError('Recovery telemetry/table counts differ')
    if file_hash(reference_current) != file_hash(resumed_current):
        raise ValueError('Current inference bytes differ')
    # These audits independently recompute every output row from training state.
    # A fresh process releases each full audit's table/allocator before the next
    # audit. The parent retains only its compact receipt; checks are unchanged.
    auditor = isolated_inspect if isolated_audits else inspect
    extra={'audit_guard':audit_guard} if isolated_audits else {}
    original_audit = auditor(reference, reference_current, reference_average, 20, **extra)
    recovered_audit = auditor(resumed, resumed_current, resumed_average, 20, hu20_reference=False, **extra)
    average_rows = 0
    with gzip_open(reference_average, 'rt') as left, gzip_open(resumed_average, 'rt') as right:
        lm, rm = json.loads(next(left)), json.loads(next(right))
        exclusions = {'checkpoint_header', 'source_checkpoint_sha256'}
        if (not exact({k:v for k,v in lm.items() if k not in exclusions},
                      {k:v for k,v in rm.items() if k not in exclusions})
            or not exact(lm['checkpoint_header'], h) or not exact(rm['checkpoint_header'], recovered)
            or lm['source_checkpoint_sha256'] != file_hash(reference)
            or rm['source_checkpoint_sha256'] != file_hash(resumed)):
            raise ValueError('Unexpected average metadata difference')
        for a, b in zip_longest(left, right):
            if a is None or b is None or not exact(json.loads(a), json.loads(b)):
                raise ValueError('Average probabilities/mass/visits differ')
            average_rows += 1
    if average_rows != entries:
        raise ValueError('Average export does not cover the complete training table')
    return {'status': 'verified', 'scope': 'full HU20 legacy recovery state and current/average policy equivalence',
        'source': source, 'campaign_swap_baseline': baseline,
        'executed_training_source': training_source, 'verifier_source': source,
        'isolated_audits': isolated_audits,
        'binary_sha256': r['binary_sha256'],
        'seed': SEED, 'completed_nodes': r['completed_nodes'], 'iteration': h['iteration'],
        'entries': entries, 'training_float_equality': 'IEEE-754 bits, including signed zero',
        'current_equality': 'identical file bytes by SHA256; both independently audited',
        'average_equality': 'all probabilities/mass/visits exact; validated metadata differences only',
        'metadata_differences': {'checkpoint': ['native_state'],
            'average': ['source_checkpoint_sha256', 'checkpoint_header.native_state'],
            'validated_recovery_state': state},
        'reference_average_sha256': REFERENCE, 'reference_average_bytes': reference_average.stat().st_size,
        'reference_audit': original_audit, 'resumed_audit': recovered_audit,
        'files': {str(p.resolve()): {'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in
            (reference, resumed, parent, reference_telemetry, resumed_telemetry,
             reference_current, reference_average, resumed_current, resumed_average,
             reference_plan_path,resumed_plan_path,
             *((training_qualification,) if training_qualification is not None else ()))}}


def admit_audit(guard, checkpoint):
    if guard is None: raise ValueError('Isolated follow-up audit requires fresh guarded admission')
    limits=envelope(guard['approval'])
    if limits['rss_gib']!=10: raise ValueError('Owner follow-up audit approval required')
    memory=memory_snapshot()
    power=system(['pmset','-g','batt'])
    swap=system(['sysctl','vm.swapusage'])
    free=shutil.disk_usage(checkpoint.parent).free
    if (time()>=guard['deadline'] or unsafe_memory(memory,admission=True,rss_gib=10)
        or power is None or 'AC Power' not in power or swap is None
        or swap_bytes(swap)-swap_bytes(guard['swap_baseline'])>.5*1024**3 or free<15.5*1024**3):
        raise ValueError('Fresh isolated audit resource/headroom admission refused')
    return {'at':time(),'system_memory':memory,'power':power,'swap':swap,'free_disk_bytes':free}


def isolated_inspect(checkpoint, current, average, bb, *, hu20_reference=True, audit_guard=None):
    code = ('import json,sys; from pathlib import Path; '
            'from scripts.audit_native_hu_checkpoint import inspect; '
            'print(json.dumps(inspect(*(Path(p) for p in sys.argv[1:4]), '
            'int(sys.argv[4]), hu20_reference=sys.argv[5]=="true")))')
    admission=admit_audit(audit_guard,checkpoint)
    result=json.loads(subprocess.check_output([sys.executable, '-c', code,
        str(checkpoint),str(current),str(average),str(bb),str(hu20_reference).lower()],text=True))
    result['fresh_admission']=admission
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fields = ('reference', 'resumed', 'parent', 'reference_telemetry', 'resumed_telemetry',
              'reference_current', 'reference_average', 'resumed_current', 'resumed_average')
    for field in fields:
        parser.add_argument('--'+field.replace('_','-'), type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--training-qualification',type=Path)
    parser.add_argument('--isolated-audits',action='store_true')
    parser.add_argument('--audit-approval',type=Path)
    parser.add_argument('--audit-swap-baseline')
    parser.add_argument('--audit-deadline',type=float)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve prior audit evidence')
    guard=None
    if args.isolated_audits:
        if not all((args.audit_approval,args.audit_swap_baseline,args.audit_deadline)): parser.error('Isolated audits require approval, baseline and deadline')
        guard={'approval':args.audit_approval,'swap_baseline':args.audit_swap_baseline,'deadline':args.audit_deadline}
    result = compare(**{k: getattr(args, k) for k in fields},
                     training_qualification=args.training_qualification,isolated_audits=args.isolated_audits,audit_guard=guard)
    with args.out.open('x') as target:
        json.dump(result, target, sort_keys=True, indent=2, allow_nan=False); target.write('\n')


if __name__ == '__main__': main()
