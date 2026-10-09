from copy import deepcopy
from hashlib import blake2b
import json

from scripts.run_hu20_equity_bench import projected
from src.blueprint.abstraction import HU20_EQUITY_SCHEMA, HU20_UNCAPPED_SCHEMA
from src.diagnostics.flop_check import factored_key, line_key


def test_trained_projection_keeps_v1_and_uses_versioned_scalar_payload():
    template=[HU20_UNCAPPED_SCHEMA,2,1,'turn',None,[[False,False],[False,False]],[],['check','jam']]
    node={'terminal':False,'street':'turn','line':[],'template':template}
    table=line_key([])
    original={'codes':[[0,1],[0,255]],'labels':{'50':[[3,4],[3,65535]]},
        'pool_keys':{'v1':{table:{str(c):factored_key(template,[0,c,0,0,0]) for c in (0,1)}}},
        'node_tables':{table:table}}
    # Let the existing adapter recompute v1 from its actual descriptor encoder.
    from src.diagnostics.board_pooling_features import add_pool_keys
    original['pool_keys']['v1']=add_pool_keys({'nodes':[node]},original,original)['v1']
    snapshot=deepcopy(original)
    output=projected({'nodes':[node]},original,[[7,8],[7,65535]])
    assert original==snapshot
    assert output['pool_keys']['v1']==snapshot['pool_keys']['v1']
    payload=deepcopy(template);payload[0]=HU20_EQUITY_SCHEMA;payload[4]=7
    expected=blake2b(json.dumps(payload,separators=(',',':')).encode(),digest_size=16).hexdigest()
    assert output['pool_keys']['eq50-fit0'][table]['7']==expected
    assert expected!=factored_key(template,['equity-bucket',7])
    assert '65535' not in output['pool_keys']['eq50-fit0'][table]
    assert set(output['pool_keys'])=={'v1','eq50-fit0'}
