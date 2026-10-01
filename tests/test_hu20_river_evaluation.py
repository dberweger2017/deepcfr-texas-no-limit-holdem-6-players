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
