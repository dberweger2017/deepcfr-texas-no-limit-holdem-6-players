"""Generated-policy paired execution, native replay and shared action streams."""
from scripts.evaluate_hu20_river import play, root_hand
from src.blueprint.hu20_river import HU20RiverConfig, RiverProfileCache
from tests.test_hu20_river import Uniform
from src.game.types import ActionKind, Street


def test_generated_policy_complete_paired_hands_and_earlier_street_identity():
    class CallEarlier(Uniform):
        def distribution(self, view):
            menu, p, trained = super().distribution(view)
            if view.street != Street.RIVER:
                p = tuple(float(c.action.kind in (ActionKind.CHECK, ActionKind.CALL)) for c in menu)
            return menu, p, trained
    source = CallEarlier(); spec = {'name':'fixture','seed':1}
    config = HU20RiverConfig(sweeps=1);cache = RiverProfileCache(source)
    panel = {'name':'passive','rule':'passive','contract':'menu'}
    pairs = []
    for rotation in (0,1):
        direct = play(source,spec,panel,202610010401,0,rotation,'current',config,cache,lambda:None)
        search = play(source,spec,panel,202610010401,0,rotation,'average',config,cache,lambda:None)
        pairs.append((direct,search))
        assert direct['native_replay_verified'] and search['native_replay_verified']
        assert direct['deal_seed'] == search['deal_seed']
        def prefix(row):return [a for a in row['actions'] if a['street'] != 'river']
        assert prefix(direct) == prefix(search)
        assert sum(search['net_chips_by_seat']) == 0
    assert cache.stats['solves'] > 0


def test_declared_timing_roots_have_exact_stacks_and_pots():
    for seed,raise_to,pot in ((202610010301,None,200),(202610010302,400,1000),(202610010303,1200,2600)):
        hand = root_hand(seed,raise_to)
        assert hand.observe(0).pot == pot


def test_closed_fixture_report_replays_and_rejects_tampered_payoff(tmp_path):
    import gzip,json
    import pytest
    from scripts.evaluate_hu20_cfr_average import summarize
    from scripts.report_hu20_river import report, audit, exploratory_interval
    from src.arena.schedule import digest
    from src.diagnostics.saved_hu20 import file_hash
    source=Uniform();config=HU20RiverConfig(sweeps=1);cache=RiverProfileCache(source)
    spec={'name':'fixture','seed':1};panel={'name':'uniform','rule':'uniform','contract':'menu','blocks':2}
    plan={'models':[spec],'panels':[panel],'root':202610010601,'river_config':{'sweeps':1},'expected_hands':8}
    directory=tmp_path/'raw';directory.mkdir();rows=[]
    with gzip.open(directory/'fixture.hands.jsonl.gz','wt') as f:
        for strategy in ('current','average'):
            for block in range(2):
                for rotation in (0,1):
                    row=play(source,spec,panel,plan['root'],block,rotation,strategy,config,cache,lambda:None)
                    rows.append(row);f.write(json.dumps(row)+'\n')
    summary={'status':'complete','hands':8,'plan':plan,'plan_sha256':digest(plan),'comparison':summarize(rows)}
    (directory/'summary.json').write_text(json.dumps(summary))
    def manifest():
        (directory/'manifest.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in directory.iterdir() if p.name!='manifest.json'}))
    manifest();result=report(directory,tmp_path/'report')
    assert result['hands']==8 and result['identical_earlier_street_pairs']==4
    value=exploratory_interval([0,10,20,30]);assert value['exploratory_t95_low']<15<value['exploratory_t95_high']
    rows[0]['target_chips']+=1
    with gzip.open(directory/'fixture.hands.jsonl.gz','wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    manifest()
    with pytest.raises(ValueError,match='payoff'):audit(directory)
