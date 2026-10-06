"""The R/O subset retains real hand replay and refuses modified observations."""
import gzip
import json

import pytest

from scripts.audit_hu20_cfr_plus_arena import audit
from scripts.evaluate_hu20_cfr_average import Uniform, play
from scripts.evaluate_hu20_v041_arena import report
from src.arena.schedule import digest
from src.blueprint.abstraction import choices


class UniformSource:
    def distribution(self, view):
        menu = choices(view, raise_cap=None, free_fold=False)
        return menu, [1 / len(menu)] * len(menu), False


def test_ro_audit_checks_every_native_action_and_rejects_changed_cards(tmp_path):
    source = 'test-source'
    models = [{'name': f'{arm}-{seed}', 'arm': arm, 'seed': seed, 'strategy': strategy}
              for arm, strategy in [('R', 'current'), ('O', 'average')] for seed in (1, 2, 3)]
    panels = [{'name': name, 'contract': 'menu', 'rule': 'uniform', 'blocks': 32}
              for name in ('lbr', 'native-pressure', 'uniform')]
    plan = {'models': models, 'panels': panels, 'root': 19, 'expected_hands': 1152}
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    for spec in models:
        rows = [play(UniformSource(), spec, panel, plan['root'], block, rotation,
                     rival=Uniform(block + rotation))
                for panel in panels for block in range(32) for rotation in (0, 1)]
        with gzip.open(tmp_path / f"{spec['name']}.hands.jsonl.gz", 'wt') as stream:
            for row in rows:
                row['arm'] = spec['arm']
                stream.write(json.dumps(row) + '\n')
        (tmp_path / f"{spec['name']}.result.json").write_text(json.dumps({
            'model': spec['name'], 'arm': spec['arm'], 'status': 'complete', 'failure': None,
            'hands': len(rows), 'plan_sha256': digest(plan), 'source': source}))
    report(plan, tmp_path)
    audit_path = tmp_path / 'audit.json'
    audit(plan_path, tmp_path, audit_path, digest(plan), source)
    verified = json.loads(audit_path.read_text())
    assert verified['hands_replayed'] == 1152
    assert verified['decisions_checked'] > 1152
    assert set(verified['contrasts_bb_per_100']) == {'O-R'}
    assert set(verified['absolute_bb_per_100']) == {'R', 'O'}
    path = tmp_path / 'O-1.hands.jsonl.gz'
    with gzip.open(path, 'rt') as stream:
        lines = [json.loads(line) for line in stream]
    lines[0]['actions'][0]['observation']['hole_cards'] = ['As', 'As']
    with gzip.open(path, 'wt') as stream:
        for row in lines:
            stream.write(json.dumps(row) + '\n')
    with pytest.raises(AssertionError):
        audit(plan_path, tmp_path, audit_path, digest(plan), source)
