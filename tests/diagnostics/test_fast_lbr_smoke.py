"""Exercise the actual smoke runner with a small generated uniform policy."""
from copy import deepcopy
from scripts.smoke_hu20_fast_lbr import compare, run_hand
from src.blueprint.abstraction import choices
from src.diagnostics.robustness import LBRConfig


class Uniform:
    def distribution(self,view):
        menu=choices(view,raise_cap=None,free_fold=False)
        return menu,tuple(1/len(menu) for _ in menu),False


def test_native_and_fast_complete_hand_and_rng():
    source=Uniform()
    rows=[run_hand(source,0,0,fast,root=91,config=LBRConfig(1,5)) for fast in (False,True)]
    assert compare(*rows)=={'identical':True,'different_fields':[],'all_batches_complete':True}
    assert rows[0]['native_replay_verified'] and sum(rows[0]['net_chips'])==0
    assert rows[0]['lbr_telemetry'] and rows[1]['shared_cache']['misses']>0
    changed=deepcopy(rows[1]);changed['actions'][0]['raise_to']=4321
    assert compare(rows[0],changed)['different_fields']==['actions']
    changed=deepcopy(rows[1]);changed['lbr_rng_state']=(3,(),None)
    assert compare(rows[0],changed)['different_fields']==['lbr_rng_state']
    changed=deepcopy(rows[1]);changed['lbr_telemetry'][0]['completed']=False
    assert not compare(rows[0],changed)['all_batches_complete']

    changed=deepcopy(rows[1])
    decision=next(a for a in changed['actions'] if 'lbr' in a)
    decision['lbr']['values_chips'][0]+=1
    assert compare(rows[0],changed)['different_fields']==['actions']
    changed=deepcopy(rows[1]);changed['net_chips']=[123,-123]
    assert compare(rows[0],changed)['different_fields']==['net_chips']
