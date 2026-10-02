"""Turn-only fallback and public native replay controls."""
from src.diagnostics.flop_check import compile_tree, fixture_root
from src.game.types import Street
from src.game.observation import replay
from src.blueprint.abstraction import choices


def test_turn_cap_two_retains_native_lookup_and_legal_jams():
    request,histories=compile_tree(fixture_root('3-bet',street=Street.TURN),raise_cap=2)
    assert request['initial_street']=='turn' and len(request['board'])==4
    removed=0
    for node,history in zip(request['nodes'],histories,strict=True):
        if node['terminal']:continue
        view=replay(history,request['seat_map'][node['player']],())
        native=choices(view,raise_cap=None,free_fold=False)
        assert node['native_names']==[c.name for c in native]
        for c in native:
            jam=c.action.raise_to is not None and c.action.raise_to-view.players[view.seat].street_bet==view.players[view.seat].stack
            if jam:assert c.name in node['names']
        removed+=len(node['native_names'])-len(node['names'])
    assert removed>0


def test_turn_root_identity_survives_real_native_replay_and_rejects_tampering():
    from src.diagnostics.turn_check import root_record,replay_root
    import pytest
    for kind in ('limped','min-raised','pot-raised','3-bet'):
        for button in (0,1):
            record=root_record(fixture_root(kind,street=Street.TURN,button=button))
            assert root_record(replay_root(record))==record
            with pytest.raises(ValueError,match='replay mismatch'):
                replay_root(dict(record,spot='tampered'))
