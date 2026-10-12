from random import Random
from types import SimpleNamespace
from scripts.evaluate_hu20_cfr_plus_direct import PolicyRival
from src.game.types import Action, ActionKind


def test_rival_evidence_preserves_action_rng_and_uses_own_keys():
    menu=[SimpleNamespace(name='fold', action=Action(ActionKind.FOLD)),
          SimpleNamespace(name='check',action=Action(ActionKind.CHECK))]
    class Source:
        visits={'own-schema-key':17}
        def distribution(self, view):
            assert view=='own-view'
            return menu,[.3,.7],True
        def information_key(self, view, own_menu):
            assert view=='own-view' and own_menu is menu
            return 'own-schema-key'
    rival=PolicyRival(Source(),31); old=Random(31)
    for _ in range(30):
        expected=old.choices(menu,weights=[.3,.7],k=1)[0].action
        actual=rival.choose_action('own-view')
        assert actual==expected
        row=rival.decisions[-1]
        assert row['key']=='own-schema-key' and row['training_visits']==17
        assert row['probabilities']==[.3,.7]
        assert row['kind']==actual.kind.value and row['raise_to']==actual.raise_to
