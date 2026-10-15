"""Count policy keys actually selected after independently replayed translation.

The original direct harness's lookup/visit fields describe the exact native key,
including when translation subsequently finds another key. Preserve those fields
and report effective selected-key support alongside them, without changing play.
"""
from copy import deepcopy
import gc
import gzip
import json
from time import perf_counter

from src.blueprint import policy_cache


def selected_coverage(folder,specs,cache,replayed):
    if replayed['status']!='verified':
        raise ValueError('Independent replay required before selected-key coverage')
    coverage=deepcopy(replayed['coverage'])
    for cell in coverage.values():
        cell['native_lookup']=cell.pop('lookup')
        cell['native_visit_bands']=cell.pop('visit_bands')
        cell.update(lookup={},visit_bands={},translation_modes={},selected_decisions=0)
    fixed=work=0.0
    for spec in specs:
        tick=perf_counter()
        loaded=policy_cache.load(cache,spec['sha256'],spec['abstraction'])
        if loaded is None:raise ValueError('Selected-key audit requires verified source cache')
        entries,zero_mass,visits=loaded
        fixed+=perf_counter()-tick
        tick=perf_counter()
        with gzip.open(folder/'decisions.jsonl.gz','rt') as stream:
            for row in map(json.loads,stream):
                if row['policy']!=spec['name']:continue
                info=row['translation'];selected=info['selected_key']
                if selected is None:
                    lookup='missing-key';n=0
                else:
                    saved=entries.get(selected)
                    if saved is None or list(saved[1])!=row['probabilities']:
                        raise ValueError('Selected key/probabilities differ from audited source cache')
                    lookup='zero-mass' if selected in zero_mass else 'positive-mass-known-key'
                    n=int(visits[selected])
                if info['mode']=='translated' and lookup!='positive-mass-known-key':
                    raise ValueError('Translated witness must select a positive-mass stored key')
                band='0' if n==0 else '1-9' if n<10 else '10-99' if n<100 else '100+'
                cell=coverage[row['policy']+'/'+row['street']]
                cell['selected_decisions']+=1
                for field,value in (('lookup',lookup),('visit_bands',band),('translation_modes',info['mode'])):
                    cell[field][value]=cell[field].get(value,0)+1
        work+=perf_counter()-tick
        del entries,zero_mass,visits,loaded;gc.collect()
    if any(c['selected_decisions']!=c['decisions'] for c in coverage.values()):
        raise ValueError('Selected-key coverage coordinates incomplete')
    return dict(status='verified',coverage=coverage,fixed_setup_seconds=fixed,block_work_seconds=work,
        seconds=fixed+work,method='Actual selected key after translation; exact native-key coverage retained separately',
        independent_replay_required=True)
