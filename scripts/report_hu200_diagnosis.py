"""Exploratory tables from sealed HU200 evidence; never reruns play or selects a model."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev
from math import sqrt, isclose
from scipy.stats import t

from scripts.evaluate_hu200_diagnosis import rows
from src.arena.schedule import digest
from src.policies.files import file_hash


def quantiles(values):
    v=sorted(values)
    return dict(n=len(v),mean=mean(v),median=median(v),minimum=v[0],maximum=v[-1]) if v else dict(n=0)


def produce(root,out):
    out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((root/'frozen-plan.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    if summary['status']!='verified' or summary['plan_sha256']!=digest(plan):
        raise ValueError('Unverified/foreign primary summary')
    recount=Counter();blocks=defaultdict(dict);coverage=defaultdict(Counter)
    groups=defaultdict(Counter);sizes=defaultdict(lambda:defaultdict(list));handgroups=defaultdict(Counter)
    for m in plan['models']:
        cost=json.loads((root/'final'/str(m['target'])/'costs.json').read_text())
        if file_hash(root/'final'/str(m['target'])/'hands.jsonl.gz')!=cost['hands_sha256']:
            raise ValueError('Raw worker hash differs')
        seen=set()
        for r in rows(root/'final'/str(m['target'])/'hands.jsonl.gz'):
            coordinate=(r['opponent'],r['block'],r['seat'])
            if coordinate in seen or r['sha256']!=digest({k:v for k,v in r.items() if k!='sha256'}):
                raise ValueError('Changed/duplicate hand during independent recount')
            seen.add(coordinate)
            recount['hands']+=1;recount['actions']+=len(r['actions']);recount['decisions']+=len(r['decisions'])
            blocks[(m['target'],r['opponent'],r['block'])][r['seat']]=r['net_chips']
            target=m['target'];op=r['opponent'];net=r['net_chips']/100
            for d in r['decisions']:
                cc=coverage[(target,op,d['street'])];cc['decisions']+=1;cc[d['lookup']]+=1
                cc['visits_'+d['visit_band']]+=1;cc['support_'+d['support']]+=1;cc['action_'+d['action']['kind']]+=1
                # All outcomes are included. Repeated decisions repeat the final payoff.
                state=('unsupported' if d['support']=='unsupported-abstract-history-or-menu' else
                    'unresolved' if d['support']=='unresolved' else
                    'supported-absent' if d['lookup']=='missing-key' else
                    'stored-zero-visits' if d['visits']==0 else
                    'stored-zero-mass-visited' if d['lookup']=='zero-mass' else 'positive-mass-visited')
                for dimension,value in [('lookup',d['lookup']),('support',d['support']),('visits',d['visit_band']),('state',state)]:
                    c=groups[(target,op,d['street'],dimension,value)]
                    c['decisions']+=1;c['final_net_bb_sum']+=net;c['win_hand_decisions']+=net>0;c['loss_hand_decisions']+=net<0;c['tie_hand_decisions']+=net==0
                if d['raise_ratio'] is not None:
                    s=sizes[(target,op,d['street'])];s['raise_ratio'].append(d['raise_ratio']);s['raise_to_bb'].append(d['action']['raise_to']/100)
                if r['stackoff'] and r['stackoff']['decision']==d['decision']:
                    s=sizes[(target,op,'stackoff')];s['score'].append(d['score']);s['committed_before_bb'].append(d['committed_bb'])
            h=handgroups[(target,op)]
            h['hands']+=1
            for label,test in [('large_pot_loss',r['max_pot_bb']>=64 and net<0),('stackoff_loss',r['stackoff'] is not None and net<0)]:
                if test:h[label+'_hands']+=1;h[label+'_net_bb']+=net
    if dict(recount)!=summary['counts']:
        raise ValueError('Independent hand/action/decision counts differ')
    for cell in json.loads((root/'coverage-behavior.json').read_text()):
        cc=coverage[(cell['target'],cell['opponent'],cell['street'])]
        if any(cell.get(k,0)!=v for k,v in cc.items()):
            raise ValueError('Independent coverage/action recount differs')
    means=defaultdict(list)
    for (target,op,block),rotations in sorted(blocks.items()):
        if set(rotations)!={0,1}:raise ValueError('Incomplete independent block')
        means[(target,op)].append((rotations[0]+rotations[1])/2)
    for panel in summary['panels']:
        values=means[(panel['target'],panel['opponent'])]
        if len(values)!=plan['blocks'] or not isclose(mean(values),panel['absolute']['bb_per_100'],abs_tol=1e-10):
            raise ValueError('Independent absolute rate differs')
    for op,gain in summary['gains'].items():
        values=[a-b for a,b in zip(means[(100000000,op)],means[(20000000,op)],strict=True)]
        center=mean(values);margin=float(t.ppf(.995,len(values)-1))*stdev(values)/sqrt(len(values))
        if not isclose(center,gain['bb_per_100'],abs_tol=1e-10) or any(not isclose(a,b,abs_tol=1e-10) for a,b in zip([center-margin,center+margin],gain['interval'],strict=True)):
            raise ValueError('Independent primary estimate/Bonferroni interval differs')
    (out/'independent-recount.json').write_text(json.dumps(dict(status='verified',counts=dict(recount),all_five_paired_primary_means_and_adjusted_intervals=True,all_ten_absolute_means=True,all_coverage_visit_support_action_counts=True),indent=2)+'\n')
    with (out/'decision-strata.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['target','opponent','street','dimension','value','decisions','win_hand_decisions','loss_hand_decisions','tie_hand_decisions','mean_final_net_bb_per_decision'])
        writer.writeheader()
        for (target,op,street,dim,value),c in sorted(groups.items()):
            writer.writerow(dict(target=target,opponent=op,street=street,dimension=dim,value=value,
                decisions=c['decisions'],win_hand_decisions=c['win_hand_decisions'],loss_hand_decisions=c['loss_hand_decisions'],tie_hand_decisions=c['tie_hand_decisions'],mean_final_net_bb_per_decision=c['final_net_bb_sum']/c['decisions']))
    size_rows=[dict(target=target,opponent=op,street=street,**{k:quantiles(v) for k,v in values.items()}) for (target,op,street),values in sorted(sizes.items())]
    (out/'bet-sizes.json').write_text(json.dumps(size_rows,indent=2)+'\n')
    # Rank was frozen prospectively; merge multiple illustration tags for the same hand.
    examples={}
    for r in json.loads((root/'representatives.json').read_text()):
        h=r['hand'];key=(r['target'],h['opponent'],h['block'],h['seat'])
        if key not in examples:examples[key]=dict(target=r['target'],rank=r['rank'],tags=[],hand=h)
        examples[key]['tags'].append(r['tag'])
    (out/'representative-hands.json').write_text(json.dumps(list(examples.values()),indent=1)+'\n')
    (out/'tail-losses.json').write_text(json.dumps([dict(target=t,opponent=o,**dict(c),
        **{label+'_contribution_bb_per_100':100*c[label+'_net_bb']/c['hands'] for label in ('large_pot_loss','stackoff_loss')}) for (t,o),c in sorted(handgroups.items())],indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(primary_summary_sha256=file_hash(root/'summary.json'),
        frozen_plan_sha256=file_hash(root/'frozen-plan.json'),raw_inputs={str(m['target']):file_hash(root/'final'/str(m['target'])/'hands.jsonl.gz') for m in plan['models']},
        scope='Post-closeout exploratory tabulation of all sealed outcomes; no new evaluation or inference',
        conditional_payoffs_are_not_causal=True,representative_rule='lowest predeclared coordinate SHA per category/sign; exact coordinates deduplicated'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();produce(a.root,a.out)


if __name__=='__main__':main()
