"""D combines the frozen v2 cards with unchanged public history compression."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import gzip
import json
import subprocess
import sys

import pytest
from src.blueprint.abstraction import (HU20_CARD_V2_SCHEMA, HU20_COMPRESSED_CARD_V2_SCHEMA,
    HU20_COMPRESSED_SCHEMA, choices, information_key, compressed_history)
from src.blueprint.artifact import save_training, load_training, export_policy
from src.blueprint.cards_v2 import VERSION
from src.blueprint.solver import BlueprintTrainer, PilotConfig, HU20_UNCAPPED_GAME
from src.game.hand import Table
from src.game.types import Street, ActionKind
from tests.test_blueprint_hu20 import coupled


def test_exact_descriptor_and_preflop_b_partition():
    assert sha256(Path('src/blueprint/cards_v2.py').read_bytes()).hexdigest() == '190d530ce66d65a031334d95600ce0f005d81324a7353170dcebf1d64a6ccd92'
    for button in (0, 1):
        for hidden in (('Qc', 'Qd'), ('Jc', 'Jd')):
            view = coupled(button, hidden).observe(button)
            menu = choices(view, raise_cap=None, free_fold=False)
            assert information_key(view, menu, schema=HU20_COMPRESSED_CARD_V2_SCHEMA) == information_key(view, menu, schema=HU20_CARD_V2_SCHEMA)


def test_d_hidden_world_rotation_current_history_and_menu():
    hands = [coupled(button, hidden) for button in (0, 1) for hidden in (('Qc', 'Qd'), ('Jc', 'Jd'))]
    for _ in range(8):
        views = [h.observe(h.actor) for h in hands]
        menus = [choices(v, raise_cap=None, free_fold=False) for v in views]
        keys = [information_key(v, m, schema=HU20_COMPRESSED_CARD_V2_SCHEMA) for v, m in zip(views, menus)]
        assert keys[0] == keys[2] and keys[1] == keys[3]
        if views[0].seat == views[0].button:
            assert len(set(keys)) == 1
        for v, m in zip(views, menus):
            assert compressed_history(v) == compressed_history(replace(v, hole_cards=('As', 'Ah')))
            for choice in m:
                v.legal_actions.validate(choice.action)
            if v.street != Street.PREFLOP:
                assert information_key(v, m, schema=HU20_COMPRESSED_CARD_V2_SCHEMA) != information_key(v, m, schema=HU20_COMPRESSED_SCHEMA)
        actions = [next(c.action for c in m if c.action.kind in (ActionKind.CALL, ActionKind.CHECK)) for m in menus]
        hands = [h.apply(a) for h, a in zip(hands, actions)]


def test_d_fresh_process_state_export_and_cross_cell_rejection(tmp_path):
    cfg = PilotConfig(seed=2026093001, raise_cap=None, max_nodes=250000,
                      abstraction=HU20_COMPRESSED_CARD_V2_SCHEMA, game=HU20_UNCAPPED_GAME)
    t = BlueprintTrainer(Table(('a', 'b'), (2000, 2000)), cfg)
    for _ in range(8): t.step()
    mid = tmp_path/'mid.gz'; save_training(t, mid)
    for _ in range(4): t.step()
    expected = tmp_path/'expected.gz'; save_training(t, expected)
    fresh = tmp_path/'fresh.gz'
    subprocess.run([sys.executable, '-c', 'import sys; from pathlib import Path; from src.blueprint.artifact import load_training,save_training; t=load_training(Path(sys.argv[1])); [t.step() for _ in range(4)]; save_training(t,Path(sys.argv[2]))', str(mid), str(fresh)], check=True)
    assert fresh.read_bytes() == expected.read_bytes()
    assert export_policy(t, tmp_path/'a.gz') == export_policy(load_training(fresh), tmp_path/'b.gz')
    header, *rows = gzip.decompress(expected.read_bytes()).splitlines()
    doc = json.loads(header)
    assert doc['identity']['card_descriptor'] == VERSION
    assert doc['identity']['history_descriptor'] == 'hu20-earlier-streets-public-summary-v1'
    doc['identity']['card_descriptor'] = 'legacy-postflop-descriptor-v1'
    expected.write_bytes(gzip.compress(json.dumps(doc).encode()+b'\n'+b'\n'.join(rows)+b'\n', mtime=0))
    with pytest.raises(ValueError, match='schema'): load_training(expected)
