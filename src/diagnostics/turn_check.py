"""Compact, card-exact turn inputs for the separately installed solver."""
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from src.blueprint.search import DECK
from src.diagnostics.flop_check import atomic_json,compile_tree,descriptor,descriptor_code,export_policy_tables,line_key
from src.diagnostics.flop_check_analysis import uniform_river_equities,emd_clusters,equity_quantiles
from src.game.observation import BoardDealt,replay
from src.game.types import Street
from scripts.prepare_flop_check import load_policy


def features(board,*,bins=20,seed=202610020204):
    board=tuple(board)
    if len(board)!=4 or len(set(board))!=4:raise ValueError('Need four distinct turn cards')
    cards=[c for c in DECK if c not in board];holdings=list(combinations(cards,2))
    lookup={tuple(sorted(h)):i for i,h in enumerate(holdings)}
    boards=[board]+[board+(c,) for c in cards]
    codes=np.full((49,len(holdings)),255,dtype=np.uint8)
    river=np.full((48,len(holdings)),np.nan,dtype=np.float32)
    histogram=np.zeros((len(holdings),bins),dtype=np.uint16)
    for row,b in enumerate(boards):
        for i,h in enumerate(holdings):
            if not set(h).intersection(b):codes[row,i]=descriptor_code(descriptor(h,b))
        if row:
            hands,equity=uniform_river_equities(b);mapped=np.asarray([lookup[tuple(sorted(h))] for h in hands])
            river[row-1,mapped]=equity
            band=np.minimum((equity*bins).astype(int),bins-1)
            np.add.at(histogram,(mapped,band),1)
    probability=histogram/histogram.sum(axis=1)[:,None]
    labels={};occupied={};valid=np.isfinite(river)
    for k in (50,200):
        turn,_=emd_clusters(probability,k,seed=seed+k)
        rivers=np.full(river.shape,65535,dtype=np.uint16)
        rivers[valid]=equity_quantiles(river[valid],k)
        labels[str(k)]=np.vstack([turn[None,:],rivers]).tolist()
        occupied[str(k)]={'turn':len(set(turn.tolist())),'river':len(set(rivers[valid].tolist()))}
    return {'board':list(board),'boards':[list(b) for b in boards],'holdings':[list(h) for h in holdings],
            'codes':codes.tolist(),'labels':labels,'root_equity':np.nanmean(river,axis=0).tolist(),
            'histogram_bins':bins,'seed':seed,'occupied_buckets':occupied}


def export(root,spec,inputs,out,*,raise_cap=None,source=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    request,_=compile_tree(root,raise_cap=raise_cap)
    if request['initial_street']!='turn':raise ValueError('Turn export needs turn root')
    if raise_cap is not None:raise ValueError('Capped export requires a completed removed-reach audit')
    data=features(request['board']);source=load_policy(spec,inputs) if source is None else source
    if source.description['weights_sha256']!=spec['sha256']:
        raise ValueError('Export source differs from the pinned policy')
    codes=np.asarray(data['codes']);by_street={'turn':set(map(int,np.unique(codes[:1])))-{255},
                                             'river':set(map(int,np.unique(codes[1:])))-{255}}
    tables=export_policy_tables(request,source,by_street)
    data['tables']=tables['tables'];data['node_tables']=tables['node_tables'];data['source']=source.description
    path=out/'compact.json';atomic_json(path,data)
    atomic_json(out/'native-tree.json',request)
    return path


def root_record(history):
    view=replay(history,0,())
    if view.street!=Street.TURN or view.finished:raise ValueError('Need live turn root')
    from dataclasses import asdict
    from hashlib import sha256
    from src.game.observation import HandStarted
    events=[asdict(replace(e,hand_id='')) if isinstance(e,HandStarted) else asdict(e) for e in history]
    identity=sha256(json.dumps(events,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'spot':identity,'board':list(view.board),'pot':view.pot,'button':view.button,
            'effective_stack':view.players[0].stack,'events':events}


def replay_root(record):
    from src.game.hand import Hand,Table
    from src.game.types import Action,ActionKind
    from src.arena.endgame_quality import _world
    hand=Hand.start(Table(('a','b'),(2000,2000),button=record['button']),
                    hand_id='turn-selection',seed=0)
    for event in record['events']:
        if 'action' in event:
            if hand.actor!=event['seat']:raise ValueError('Turn root actor mismatch')
            hand=hand.apply(Action(ActionKind(event['action']['kind']),event['action']['raise_to']))
    board=tuple(record['board'])
    history=tuple(replace(e,cards=board[:3] if e.street==Street.FLOP else (board[3],))
                  if isinstance(e,BoardDealt) else e for e in hand.events)
    root=_world(history,board,{}).events
    if root_record(root)['spot']!=record['spot']:raise ValueError('Turn public root replay mismatch')
    return root
