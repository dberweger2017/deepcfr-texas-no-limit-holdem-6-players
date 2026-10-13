import gzip
import json
from pathlib import Path
from time import time
import pytest
from scripts import evaluate_hu20_cfr_plus_direct as direct
from scripts.audit_hu20_k50_first_pass_direct import replay
from src.blueprint.abstraction import choices, information_key, HU20_UNCAPPED_SCHEMA


class Uniform:
    visits={}
    def distribution(self,view):
        menu=choices(view,raise_cap=None,free_fold=False)
        return menu,tuple(1/len(menu) for _ in menu),False
    def information_key(self,view,menu):
        return information_key(view,menu,schema=HU20_UNCAPPED_SCHEMA)


def played(tmp_path,monkeypatch):
    monkeypatch.setattr(direct,'load',lambda spec,root:Uniform())
    spec=lambda name:dict(name=name,strategy='average',seed=11,abstraction=HU20_UNCAPPED_SCHEMA)
    plan=dict(root=42,blocks=33,started_at=time(),max_seconds=300,
              pairs=[dict(candidate=spec('candidate'),reference=spec('reference'))])
    result=direct.run(plan,tmp_path,tmp_path/'out',1)
    assert result['status']=='complete'
    return plan,tmp_path/'out'


def test_independent_replay_checks_both_schema_keys_and_coverage(tmp_path,monkeypatch):
    plan,out=played(tmp_path,monkeypatch)
    cells,receipt=replay(plan,out,tmp_path)
    assert len(cells)==66 and receipt['hands_replayed']==66
    assert receipt['policy_owned_key_checks'] and receipt['decisions_checked']>66
    assert receipt['coverage']['candidate']['unique_stored_reached_keys']==0
    assert sum(v['missing_reached_keys'] for v in receipt['coverage']['candidate']['streets'].values())>0


def test_independent_replay_rejects_raw_accounting_corruption(tmp_path,monkeypatch):
    plan,out=played(tmp_path,monkeypatch)
    path=out/'direct-lineage-1.hands.jsonl.gz'
    with gzip.open(path,'rt') as source:rows=list(map(json.loads,source))
    rows[0]['target_chips']+=1
    with gzip.open(path,'wt') as target:
        for row in rows:target.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='accounting'):
        replay(plan,out,tmp_path)


def test_independent_replay_rejects_foreign_rival_key(tmp_path,monkeypatch):
    plan,out=played(tmp_path,monkeypatch)
    path=out/'direct-lineage-1.hands.jsonl.gz'
    with gzip.open(path,'rt') as source:rows=list(map(json.loads,source))
    decision=next(a for row in rows for a in row['actions'] if a['logical_player'])
    decision['reference_decision']['key']='0'*32
    with gzip.open(path,'wt') as target:
        for row in rows:target.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='schema key'):
        replay(plan,out,tmp_path)

def test_independent_replay_rejects_rival_decision_action_disagreement(tmp_path,monkeypatch):
    plan,out=played(tmp_path,monkeypatch)
    path=out/'direct-lineage-1.hands.jsonl.gz'
    with gzip.open(path,'rt') as source:rows=list(map(json.loads,source))
    decision=next(a for row in rows for a in row['actions'] if a['logical_player'])
    decision['reference_decision']['kind']='foreign-action'
    with gzip.open(path,'wt') as target:
        for row in rows:target.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='decision/action'):
        replay(plan,out,tmp_path)


@pytest.mark.parametrize("candidate_schema,reference_schema", [
    ("hu20-native-reopening-ordered-history-equity-k50-menu-v2", HU20_UNCAPPED_SCHEMA),
    (HU20_UNCAPPED_SCHEMA, "hu20-native-reopening-ordered-history-equity-k50-menu-v2"),
    ("hu20-native-reopening-ordered-history-equity-k50-menu-v2",
     "hu20-native-reopening-ordered-history-equity-k50-v1"),
])
def test_postflop_equity_v2_replay_owns_pinned_tables(tmp_path, monkeypatch, candidate_schema, reference_schema):
    from src.arena.schedule import stream_seed
    from src.blueprint import equity_buckets
    from src.blueprint.abstraction import EQUITY_SCHEMAS, schema_choices
    from src.game.hand import Hand, Table
    from src.game.types import Action, ActionKind
    from tests.native_equity_tables import write_tables
    # Enumerate the two passive played hands' actual private holdings and boards.
    observations = []
    for rotation in (0, 1):
        hand = Hand.start(Table(("seat0", "seat1"), (2000, 2000)), hand_id="tables",
                          seed=stream_seed(42, "test", "deal", 2, 0))
        while not hand.finished:
            view = hand.observe(hand.actor)
            observations.append(view)
            hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL))
    (tmp_path / "tables").mkdir()
    tables = write_tables(tmp_path / "tables", [(v.hole_cards, v.board) for v in observations if v.board])
    cards = equity_buckets.EquityCards(tables, pinned=False)
    monkeypatch.setattr(equity_buckets, "EQUITY_K50_SHA256", cards.sha256)
    monkeypatch.setattr(equity_buckets, "_registered", None)
    class Passive:
        visits = {}
        def __init__(self, schema): self.abstraction = schema
        def distribution(self, view):
            menu = schema_choices(view, self.abstraction)
            probabilities = tuple(float(c.name in ("check", "call")) for c in menu)
            return menu, probabilities, False
        def information_key(self, view, menu):
            return information_key(view, menu, schema=self.abstraction,
                                   equity_cards=cards if self.abstraction in EQUITY_SCHEMAS else None)
    monkeypatch.setattr(direct, "load", lambda spec, root: Passive(spec["abstraction"]))
    def spec(name, schema):
        return dict(name=name, strategy="average", seed=11, abstraction=schema,
                    **({"card_buckets": "tables"} if schema in EQUITY_SCHEMAS else {}))
    plan = dict(root=42, blocks=1, started_at=time(), max_seconds=30,
                pairs=[dict(candidate=spec("candidate", candidate_schema), reference=spec("reference", reference_schema))])
    result = direct.run(plan, tmp_path, tmp_path / "out", 1)
    assert result["status"] == "complete", result["failure"]
    cells, receipt = replay(plan, tmp_path / "out", tmp_path)
    assert len(cells) == receipt["hands_replayed"] == 2
    assert receipt["coverage"]["candidate"]["streets"]["flop"]["missing_reached_keys"] > 0
    assert receipt["coverage"]["reference"]["streets"]["river"]["missing_reached_keys"] > 0
    assert equity_buckets._registered is None
