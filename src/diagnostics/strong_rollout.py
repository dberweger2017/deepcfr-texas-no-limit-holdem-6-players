"""Transparent observation-only HU20 heuristic; strength remains unvalidated."""
from collections import Counter
from dataclasses import replace
from itertools import combinations
from math import exp
from random import Random

from src.blueprint.abstraction import Choice, RANKS, _preflop, choices
from src.blueprint.search import DECK
from src.diagnostics.exact_ranker import exact_seven_card
from src.game.observation import ActionTaken, BoardDealt, Observation, replay
from src.game.types import Action, ActionKind

VERSION = 'strong-rollout-hu20-v1'


def expand_range(groups):
    result=set()
    for group in groups:
        plus=group.endswith('+');token=group.rstrip('+')
        high,low=RANKS.index(token[0]),RANKS.index(token[1])
        if len(token)==2:
            if high!=low:raise ValueError('Pair notation required')
            result.update(r+r+'p' for r in RANKS[low:] if plus or r==token[0])
        else:
            if token[2] not in ('s','o') or high<=low:raise ValueError('Invalid range notation')
            result.update(token[0]+RANKS[r]+token[2] for r in range(low,high) if plus or r==low)
    return frozenset(result)


def action_menu(view,mode):
    if mode=='restricted':return choices(view,raise_cap=None,free_fold=False)
    if mode!='native':raise ValueError('Unknown sizing mode')
    menu=[c for c in choices(view,raise_cap=None,free_fold=False) if c.action.kind!=ActionKind.RAISE]
    legal=view.legal_actions
    if ActionKind.RAISE in legal.kinds:
        own=view.players[view.seat];matched=own.street_bet+legal.call_amount;pot=view.pot+legal.call_amount
        targets=[('min',legal.min_raise_to)]
        if not view.board:
            prior=[e for e in view.history if isinstance(e,ActionTaken) and e.action.kind==ActionKind.RAISE]
            targets.append(('open-2.5bb' if not prior else 'reraise-3x',250 if not prior else 3*max(p.street_bet for p in view.players)))
        else:
            targets += [('one-third',matched+(pot+1)//3),('two-thirds',matched+(2*pot+1)//3)]
        if legal.max_raise_to-matched<=2*pot or legal.min_raise_to==legal.max_raise_to:
            targets.append(('jam',legal.max_raise_to))
        used=set()
        for name,target in targets:
            target=max(legal.min_raise_to,min(legal.max_raise_to,target))
            if target not in used:menu.append(Choice(name,Action(ActionKind.RAISE,target)));used.add(target)
    for c in menu:view.legal_actions.validate(c.action)
    return tuple(menu)


def features(view,cards=None):
    cards=tuple(cards or view.hole_cards);board=view.board
    suits=Counter(c[1] for c in board);joint=Counter(c[1] for c in cards+board)
    ranks={RANKS.index(c[0]) for c in cards+board}
    windows=[{12,0,1,2,3}]+[set(range(i,i+5)) for i in range(9)]
    flush_draw=bool(len(board)<5 and max(joint.values())==4)
    straight_draw=bool(len(board)<5 and any(len(w-ranks)==1 for w in windows))
    nut_blocker=any(c[0]=='A' and suits[c[1]]>=2 for c in cards)
    value=exact_seven_card(cards+board) if board else None
    board_only=bool(len(board)==5 and value==exact_seven_card(board))
    own_ranks={RANKS.index(c[0])+2 for c in cards}
    top=max((RANKS.index(c[0])+2 for c in board),default=0)
    if not board:tier='preflop'
    elif board_only:tier='board_only'
    elif value[0]>=2:tier='strong'
    elif value[0]==1 and value[1] in own_ranks:
        tier='top_pair' if value[1]>=top else 'lower_pair'
    elif value[0]==1:tier='board_pair'
    else:tier='weak'
    return {'made_value':value,'tier':tier,'flush_draw':flush_draw,'straight_draw':straight_draw,
            'nut_suit_blocker':nut_blocker,'board_only':board_only,
            'board_paired':len({c[0] for c in board})<len(board),'max_board_suit':max(suits.values(),default=0)}


def preflop_plan(view,ranges,mode):
    menu=action_menu(view,mode);by={c.name:c for c in menu};hand=_preflop(view.hole_cards)
    raises=[e for e in view.history if isinstance(e,ActionTaken) and e.action.kind==ActionKind.RAISE]
    passive=by.get('check',by.get('call'));fold=by.get('fold',passive)
    aggression=next((by[n] for n in ('pot','reraise-3x','open-2.5bb','min') if n in by),None)
    biggest=max((c for c in menu if c.action.kind==ActionKind.RAISE),key=lambda c:c.action.raise_to,default=None)
    if view.legal_actions.call_amount>=800:
        rule='stackoff';preferred=passive if hand in ranges['stackoff'] else fold;prob=.95
    elif len(raises)>=2:
        if hand in ranges['stackoff']:rule='stackoff';preferred=biggest or passive;prob=.90
        else:rule='call_reraise';preferred=passive if hand in ranges[rule] else fold;prob=.95
    elif raises:
        if hand in ranges['value_reraise']:rule='value_reraise';preferred=aggression or passive;prob=.85
        elif hand in ranges['blocker_reraise']:rule='blocker_reraise';preferred=aggression or passive;prob=.25
        else:rule='bb_call_open';preferred=passive if hand in ranges[rule] else fold;prob=.95
    elif view.seat==view.button:
        rule='sb_open';preferred=(aggression or passive) if hand in ranges[rule] else fold;prob=.90
    else:
        rule='value_reraise';preferred=aggression if hand in ranges[rule] and aggression else passive;prob=.85
    # A bluff re-raise's alternative is a range-qualified call, not a mandatory fold.
    alternative=passive if passive and (view.legal_actions.call_amount==0 or hand in ranges['bb_call_open']) else fold
    if preferred.action.kind in (ActionKind.FOLD,ActionKind.CALL,ActionKind.CHECK):
        alternative=fold if preferred.action.kind==ActionKind.CALL else preferred
    weights=[(prob if c==preferred else 0)+(1-prob if c==alternative else 0) for c in menu]
    return menu,weights,rule


def continuing_probability(view,pair,owed,pot,ranges):
    if not view.board:
        group='stackoff' if owed>=800 else 'call_reraise' if owed>=300 else 'bb_call_open'
        return .96 if _preflop(pair) in ranges[group] else .06
    f=features(view,pair);price=owed/max(1,pot+owed)
    base={'strong':.98,'top_pair':.85,'lower_pair':.55,'board_pair':.30,'board_only':.18,'weak':.12}[f['tier']]
    if f['flush_draw'] or f['straight_draw']:base=max(base,.62 if price<.32 else .35)
    # Larger prices increasingly filter weak holdings; nut blockers are card removal,
    # not permission to peek at an actual opponent holding.
    return max(.03,min(.99,base-1.1*max(0,price-.2)))


class AssumedRange:
    def __init__(self,seed,config):
        self.random=Random(seed);self.config=config
        self.ranges={k:expand_range(v) for k,v in config['ranges'].items()}
        self.pairs=None;self.weights=None;self.processed=0;self.hand_id=None;self.resets=0

    def _initial(self,view):
        available=[c for c in DECK if c not in view.hole_cards+view.board]
        pairs=list(combinations(available,2))
        self.pairs=self.random.sample(pairs,min(self.config['particles'],len(pairs)))
        self.weights=[1/len(self.pairs)]*len(self.pairs);self.processed=0

    def update(self,view):
        if self.hand_id is not None and self.hand_id!=view.hand_id:raise ValueError('Range memory is hand-local')
        self.hand_id=view.hand_id
        if self.pairs is None:self._initial(view)
        if len(view.history)<self.processed:raise ValueError('Public history moved backwards')
        for index in range(self.processed,len(view.history)):
            event=view.history[index]
            if isinstance(event,BoardDealt):
                for i,pair in enumerate(self.pairs):
                    if set(pair).intersection(event.cards):self.weights[i]=0
            elif isinstance(event,ActionTaken) and event.seat!=view.seat:
                pair=next((p for p,w in zip(self.pairs,self.weights) if w>0),None)
                if pair is None:break
                base=replay(view.history[:index],event.seat,pair)
                for i,pair in enumerate(self.pairs):
                    if not self.weights[i]:continue
                    hypothetical=replace(base,hole_cards=pair)
                    if not hypothetical.board:
                        menu,probabilities,_=preflop_plan(hypothetical,self.ranges,'native')
                        likelihood=sum(p for c,p in zip(menu,probabilities) if c.action.kind==event.action.kind)
                    else:
                        continued=continuing_probability(hypothetical,pair,hypothetical.legal_actions.call_amount,hypothetical.pot,self.ranges)
                        f=features(hypothetical)
                        aggression=.55 if f['tier']=='strong' else .22 if f['tier']=='top_pair' or f['flush_draw'] or f['straight_draw'] else .06
                        if event.action.kind==ActionKind.RAISE:
                            ratio=event.paid/max(1,hypothetical.pot)
                            likelihood=aggression*(1 if ratio<=1 else .75 if f['tier']=='strong' else .25)
                        elif event.action.kind==ActionKind.FOLD:likelihood=1-continued
                        elif event.action.kind==ActionKind.CHECK:likelihood=1-aggression
                        else:likelihood=continued*(1-aggression)
                    self.weights[i]*=max(self.config['likelihood_floor'],likelihood)
            total=sum(self.weights)
            if total<=0:
                self.resets+=1;self._initial(view);return self.update(view)
            self.weights=[w/total for w in self.weights]
        self.processed=len(view.history)
        return tuple((pair,w) for pair,w in zip(self.pairs,self.weights) if w>0)

    def summary(self):
        return {'assumption':'heuristic-public-likelihood-v1','particles':sum(w>0 for w in self.weights),
                'effective_sample_size':1/sum(w*w for w in self.weights),'resets':self.resets}


class StrongRollout:
    version=VERSION
    def __init__(self,seed,config,mode='restricted'):
        if config['version']!=VERSION or not 1<=config['particles']<=256 or not 1<=config['equity_worlds']<=128:
            raise ValueError('Invalid bounded opponent configuration')
        self.config=config;self.mode=mode
        self.random=Random(seed);self.chance=Random(seed^0xE017)
        self.belief=AssumedRange(seed^0xA55A,config);self.telemetry=[]
        self.ranges=self.belief.ranges

    def choose_action(self,view):
        if not isinstance(view,Observation):raise TypeError('Opponent accepts only an Observation')
        if view.capacity!=2 or any(p.starting_stack!=2000 for p in view.players):raise ValueError('Opponent requires HU20 reset stacks')
        rows=self.belief.update(view);menu=action_menu(view,self.mode)
        own,rival=view.players[view.seat],view.players[1-view.seat]
        common={'street':view.street.value,'position':'button' if view.seat==view.button else 'big_blind',
                'pot':view.pot,'call_amount':view.legal_actions.call_amount,
                'pot_odds':view.legal_actions.call_amount/max(1,view.pot+view.legal_actions.call_amount),
                'effective_stack':min(own.stack,rival.stack),'spr':min(own.stack,rival.stack)/max(1,view.pot),
                'features':features(view), 'range':self.belief.summary()}
        if not view.board:
            menu,weights,rule=preflop_plan(view,self.ranges,self.mode)
            scores=[{'name':c.name,'probability':p} for c,p in zip(menu,weights)]
            common.update(rule=rule)
        else:
            pairs,probabilities=zip(*rows);worlds=[]
            for _ in range(self.config['equity_worlds']):
                pair=self.chance.choices(pairs,weights=probabilities,k=1)[0]
                available=[c for c in DECK if c not in view.hole_cards+view.board+pair]
                board=view.board+tuple(self.chance.sample(available,5-len(view.board)))
                a,b=exact_seven_card(view.hole_cards+board),exact_seven_card(pair+board)
                worlds.append((pair,1 if a>b else .5 if a==b else 0))
            eq=sum(e for _,e in worlds)/len(worlds);scores=[]
            for c in menu:
                action=c.action;fe=0;continuing_eq=eq;allowed=True
                if action.kind==ActionKind.FOLD:score=-own.contributed
                else:
                    paid=action.raise_to-own.street_bet if action.kind==ActionKind.RAISE else view.legal_actions.call_amount if action.kind==ActionKind.CALL else 0
                    owed=min(rival.stack,max(0,own.street_bet+paid-rival.street_bet))
                    matched=min(own.contributed+paid,rival.contributed+owed)
                    if action.kind==ActionKind.RAISE:
                        # Use the rival's public perspective; its prospective call is
                        # conditioned on the candidate raise and its own hypothetical pair.
                        rival_view=replace(view,seat=1-view.seat)
                        continuation=[continuing_probability(rival_view,pair,owed,view.pot+paid,self.ranges) for pair,_ in worlds]
                        mass=sum(continuation);fe=1-mass/len(worlds)
                        continuing_eq=sum(p*e for p,(_,e) in zip(continuation,worlds))/mass
                        score=fe*rival.contributed+(1-fe)*(2*continuing_eq-1)*matched
                        f=common['features']
                        allowed=(continuing_eq>=.58 or (f['flush_draw'] or f['straight_draw']) and eq>=.40
                                 or f['nut_suit_blocker'] and fe>=.35 or fe>=.55)
                    else:score=(2*eq-1)*matched
                scores.append({'name':c.name,'raise_to':action.raise_to,'score_chips':score,
                               'estimated_fold':fe,'continuing_equity':continuing_eq,'eligible':allowed})
            best=max(s['score_chips'] for s in scores if s['eligible'])
            weights=[exp((s['score_chips']-best)/self.config['mix_temperature_chips'])
                     if s['eligible'] and s['score_chips']>=best-self.config['near_best_chips'] else 0 for s in scores]
            common.update(equity=eq,worlds=len(worlds),rule='range-checkdown-raise-v1')
            for s,w in zip(scores,weights):s['probability']=w/sum(weights)
        chosen=self.random.choices(menu,weights=weights,k=1)[0]
        view.legal_actions.validate(chosen.action)
        self.telemetry.append({**common,'scores':scores,'chosen':chosen.name})
        return chosen.action
