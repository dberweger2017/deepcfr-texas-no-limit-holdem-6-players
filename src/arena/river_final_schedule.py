"""Constant-memory final coordinates, exactly the repository schedule law."""

from random import Random
from src.arena.schedule import Block, stream_seed
from scripts.evaluate_hu100_direct import PANEL_OPPONENTS

ROOTS={name:20261017317900+i for i,name in enumerate(('direct',*PANEL_OPPONENTS))}
COUNTS={name:131072 if name=='direct' else 2048 for name in ROOTS}


def block_at(panel, root, index):
    if panel not in ROOTS or index<0:raise ValueError('Unknown final block')
    scenario='river-final-'+panel
    def seed(stream,*extra):return stream_seed(root,'test',stream,scenario,index,*extra)
    opponent=panel if panel!='direct' else 'blend'
    selection=Random(seed('opponent'))
    return Block(scenario,index,index%2,(seed('deal',0),),
        tuple(seed('action',p) for p in (0,1)),seed('opponent'),(selection.choice((opponent,)),))
