from src.arena.river_final_schedule import block_at, ROOTS, COUNTS
from src.arena.schedule import Plan, Scenario, build_schedule


def test_random_access_coordinates_match_independent_existing_schedule():
    for panel in ROOTS:
        root=4127
        expected=build_schedule(Plan((Scenario('river-final-'+panel,(20000,20000)),),
            candidate='search',baseline='blend',opponents=(panel if panel!='direct' else 'blend',),
            blocks=13,root_seed=root,split='test'))
        assert tuple(block_at(panel,root,i) for i in range(13))==expected
    assert COUNTS['direct']==131072 and sum(COUNTS.values())==145408
