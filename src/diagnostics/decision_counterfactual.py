"""Observation-only assumed-range values with independent selection/evaluation."""
from math import sqrt
from random import Random
from statistics import mean, stdev

from scipy.stats import t

from src.arena.schedule import digest, stream_seed
from src.blueprint.search import _sample_world
from src.diagnostics.strong_rollout import AssumedRange, StrongRollout, action_menu


def summarize_values(values,probabilities,selection):
    if selection<2 or len(values)!=2*selection or any(len(r)!=len(probabilities) for r in values):
        raise ValueError('Need equal independent selection/evaluation halves')
    if abs(sum(probabilities)-1)>1e-9 or min(probabilities)<0:raise ValueError('Invalid root mixture')
    selection_means=[mean(r[i] for r in values[:selection]) for i in range(len(probabilities))]
    chosen=max(range(len(probabilities)),key=lambda i:selection_means[i])
    evaluation=values[selection:]
    mixture=[sum(p*v for p,v in zip(probabilities,r)) for r in evaluation]
    differences=[r[chosen]-m for r,m in zip(evaluation,mixture)]
    gap=mean(differences);half=float(t.ppf(.975,selection-1))*stdev(differences)/sqrt(selection)
    return {'selected_index':chosen,'selection_action_mean_bb':selection_means,
        'evaluation_action_mean_bb':[mean(r[i] for r in evaluation) for i in range(len(probabilities))],
        'evaluation_policy_mean_bb':mean(mixture),'gap_bb':gap,'gap_ci95_bb':[gap-half,gap+half],
        'selection_worlds':selection,'evaluation_worlds':selection,
        'interval_scope':'conditional on this observation, fixed assumed range and continuation model; exploratory'}


def continuation(hand,hero,source,config,mode,seed,*,trace=False):
    target_random=Random(stream_seed(seed,'test','action','target'))
    opponent=StrongRollout(stream_seed(seed,'test','opponent','continuation'),config,mode)
    lookups={'trained':0,'fallback':0};actions=[]
    for _ in range(1000):
        if hand.finished:
            return (hand.observe(hero).players[hero].stack-2000)/100,lookups,actions
        view=hand.observe(hand.actor)
        if hand.actor==hero:
            menu,probabilities,trained=source.distribution(view)
            lookups['trained' if trained else 'fallback']+=1
            action=target_random.choices(menu,weights=probabilities,k=1)[0].action
        else:action=opponent.choose_action(view)
        view.legal_actions.validate(action)
        if trace:actions.append({'seat':hand.actor,'street':view.street.value,'board':view.board,
                                 'own_cards':view.hole_cards,'kind':action.kind.value,'raise_to':action.raise_to})
        hand=hand.apply(action)
    raise RuntimeError('Counterfactual continuation exceeded action bound')


def score_decision(view,source,config,mode,seed,*,selection_worlds=16):
    if not 2<=selection_worlds<=64:raise ValueError('Invalid bounded world budget')
    menu,probabilities,trained=source.distribution(view)
    alternatives=list(menu);mixture=list(probabilities)
    if mode=='native':
        for c in action_menu(view,mode):
            if not any(c.action==a.action for a in alternatives):alternatives.append(c);mixture.append(0)
    for c in alternatives:view.legal_actions.validate(c.action)
    belief=AssumedRange(stream_seed(seed,'test','opponent','range'),config)
    rows=belief.update(view);values=[];world_records=[];lookups={'trained':0,'fallback':0}
    for index in range(2*selection_worlds):
        world_seed=stream_seed(seed,'test','deal','world',index)
        world=_sample_world(view,{1-view.seat:rows},Random(world_seed))
        continuation_seed=stream_seed(seed,'test','action','continuation',index)
        returns=[];traces=[]
        for c in alternatives:
            value,exposure,trace=continuation(world.apply(c.action),view.seat,source,config,mode,continuation_seed,trace=index==0)
            returns.append(value)
            for k,v in exposure.items():lookups[k]+=v
            if index==0:traces.append({'alternative':c.name,'actions':trace})
        values.append(returns)
        world_records.append({'world_index':index,'world_seed':world_seed,'continuation_seed':continuation_seed,
                              'returns_bb':returns,**({'traces':traces} if index==0 else {})})
    result=summarize_values(values,mixture,selection_worlds)
    return {**result,'version':'assumed-range-decision-gap-v1','mode':mode,'root_trained':trained,
        'legal_alternatives':[{'name':c.name,'kind':c.action.kind.value,'raise_to':c.action.raise_to,'root_probability':p}
                              for c,p in zip(alternatives,mixture)],
        'range':belief.summary(),'range_holdings':rows,'continuation_lookups':lookups,
        'assumptions':f"{config['particles']}-particle heuristic public-history range; sampled fresh worlds; saved target vs freshly initialized frozen v1; no cross-hand adaptation",
        'observation_sha256':digest({'own_cards':view.hole_cards,'board':view.board,'events':[repr(e) for e in view.history]}),
        'world_records':world_records,
        'warning':'A conditional model-based estimate, not actual-opponent EV or a proven error; one world is a realized counterfactual.'}
