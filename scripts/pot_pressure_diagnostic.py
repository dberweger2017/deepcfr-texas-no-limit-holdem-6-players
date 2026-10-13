"""Diagnostic arms on PR236's scripted schedule/streams and public-hand runner."""
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from math import isclose
from pathlib import Path
from time import perf_counter

from scripts.evaluate_native_hu100_baseline import action_seed
from scripts.evaluate_hu100_direct import put
from src.arena.catalog import Checkpoint
from src.arena.heuristics import STYLES, Style, StylePolicy
from src.arena.registry import PolicyRegistry
from src.arena.runner import public_events, run_schedule
from src.arena.schedule import Plan, Scenario, build_schedule, canonical, digest
from src.blueprint.abstraction import HU100_SCHEMA, EQUITY_SCHEMAS, schema_choices, information_key
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.equity_buckets import EquityCards
from src.game.hand import Hand, Table
from src.game.observation import Observation
from src.game.types import Action, ActionKind
from src.policies.files import file_hash

ARMS = {'baseline':('v052',3,None,True), 'size1x':('v052',2,None,True),
        'passive':('v052',3,'call',True), 'folding':('v052',3,'fold',True),
        'k50':('k50',3,None,False), 'v1':('v051',3,None,False)}


def make_plan(spec, blocks, root):
    model = Checkpoint(**{k:spec[k] for k in ('name','path','sha256','format','abstraction','card_buckets') if k in spec})
    return Plan((Scenario('pot_pressure',(10000,10000)),),candidate=model.name,
                baseline='native_hu100_uniform',opponents=('pot_pressure',),blocks=blocks,
                root_seed=root,split='test',max_decisions=1000,models=(model,))


def diagnostic_action(view: Observation, fallback: str) -> Action:
    if ActionKind.CHECK in view.legal_actions.kinds:
        return Action(ActionKind.CHECK)
    return Action(ActionKind.CALL if fallback=='call' else ActionKind.FOLD)


class Probe:
    def __init__(self, player, model, context, emit, fallback):
        self.player,self.model,self.context,self.emit,self.fallback = player,model,context,emit,fallback
    def choose_action(self, view: Observation) -> Action:
        if not isinstance(view, Observation): raise TypeError("Diagnostic policies require a legal player observation")
        action = self.player.choose_action(view)
        row = {**self.context,'hand_id':view.hand_id,'seat':view.seat,'street':view.street.value,'action':asdict(action)}
        if self.model is not None:
            menu, probabilities, known, info = self.player.last_decision
            info = {k:v for k,v in info.items() if k!='lookup_seconds'}
            missing = info['mode']=='uniform' and info['reason']=='missing-key'
            effective = list(probabilities)
            if missing and self.fallback:
                # Consume the released sampler's draw first, preserving its stream.
                action = diagnostic_action(view,self.fallback)
                effective = [float(item.action==action) for item in menu]
            row.update(action=asdict(action),key=self.model.information_key(view,menu),
                       known=known,translation=info,menu=[asdict(c) for c in menu],
                       probabilities=list(probabilities),effective_probabilities=effective,
                       diagnostic_override=bool(missing and self.fallback))
        self.emit(row)
        return action


def execute(spec, arm, blocks, root, out, source, registry=None, reproduce=None):
    import subprocess
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=source or subprocess.check_output(
            ['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Clean frozen source required')
    out.mkdir(parents=True,exist_ok=False)
    started = perf_counter(); plan = make_plan(spec,blocks,root)
    tick = perf_counter()
    if registry is None: registry = PolicyRegistry(plan,sorted_average_rows=True)
    registry.plan = plan
    model = registry.models[spec['name']]
    if model.source_path.stat().st_size!=spec['bytes'] or file_hash(model.source_path)!=spec['sha256']:
        raise ValueError('Model pin differs')
    if any(model.description[k]!=spec[k] for k in ('entries','iteration','source_checkpoint_sha256')):
        raise ValueError('Training identity differs')
    _, numerator, fallback, translate = ARMS[arm]
    model.configure_translation(TranslationOptions(max_states=512,max_events=128) if translate else None)
    model.record_translation = True
    load = perf_counter()-tick
    put(out/'inputs.json',{'source':source,'arm':arm,'model':spec,'plan':asdict(plan),'plan_sha256':digest(asdict(plan)),
                          'style':asdict(Style(0.40,0.60,0.95,0.20,numerator)), 'fallback':fallback,'translation':translate})
    contexts = {}; hands = decisions = 0
    old_h = gzip.open(reproduce/'hands.jsonl.gz','rt') if reproduce else None
    old_d = gzip.open(reproduce/'decisions.jsonl.gz','rt') if reproduce else None
    tick = perf_counter()
    try:
        with gzip.open(out/'hands.jsonl.gz','wt') as hs, gzip.open(out/'decisions.jsonl.gz','wt') as ds:
            def seed_factory(block,rotation,unused_arm,role):
                seed = action_seed(root,block,rotation,'candidate',role)
                if seed in contexts: raise ValueError('Private stream collision')
                contexts[seed] = {'block':block.index,'rotation':rotation,'role':role,'seed':seed}
                return seed
            def trace(row):
                nonlocal decisions
                if old_d and canonical(json.loads(next(old_d)))!=canonical(row):
                    raise ValueError('Decision reproduction differs')
                ds.write(canonical(row)+'\n'); decisions+=1
            def factory(name,seed):
                context = contexts.pop(seed)
                if name==spec['name']:
                    return Probe(model.policy(seed),model,context,trace,fallback)
                if name!='pot_pressure': raise ValueError('Unknown policy')
                return Probe(StylePolicy(Style(0.40,0.60,0.95,0.20,numerator),seed),None,context,trace,None)
            def emit(row,timing):
                nonlocal hands
                hs.write(canonical(row)+'\n'); hs.flush()
                if row['status']!='completed': raise ValueError('Failed final hand; never rerun: '+str(row['error']))
                if old_h and canonical(json.loads(next(old_h)))!=canonical(row): raise ValueError('Hand reproduction differs')
                hands+=1
            if not run_schedule(plan,build_schedule(plan),emit,factory=factory,seed_factory=seed_factory,arms=('candidate',)):
                raise ValueError('Scripted runner stopped')
        if reproduce and (next(old_h,None) is not None or next(old_d,None) is not None):
            raise ValueError('Extra reproduction traces')
    finally:
        if old_h: old_h.close()
        if old_d: old_d.close()
    put(out/'complete.json',{'status':'complete','hands':hands,'decisions':decisions,'load_seconds':load,
        'play_seconds':perf_counter()-tick,'seconds':perf_counter()-started,'reproduced_all_hands_and_decisions':bool(reproduce),
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.glob('*.jsonl.gz')}})
    return registry,json.loads((out/'complete.json').read_text())


def audit(out, output, *, cost_only=False):
    """Independent trace replay: no call to the player, Probe or production readout."""
    tick = perf_counter(); inputs = json.loads((out/'inputs.json').read_text()); done = json.loads((out/'complete.json').read_text())
    plan = Plan.from_dict(inputs['plan']); spec = inputs['model']; arm = inputs['arm']
    if digest(inputs['plan'])!=inputs['plan_sha256']: raise ValueError('Plan changed')
    for name,pin in done['files'].items():
        if (out/name).stat().st_size!=pin['bytes'] or file_hash(out/name)!=pin['sha256']: raise ValueError('Trace changed')
    cards = EquityCards(Path(spec['card_buckets'])) if spec['abstraction'] in EQUITY_SCHEMAS else None
    schedule = {b.index:b for b in build_schedule(plan)}
    expected = {(b,r) for b in range(plan.blocks) for r in (0,1)}; seen = set(); actions = 0
    values = {}; classes = {}; positive_exact_hands = {}; action_mix = defaultdict(Counter); committed = Counter(); terminal_net = Counter(); lost_by_street = Counter(); net_losses_by_terminal = Counter()
    with gzip.open(out/'hands.jsonl.gz','rt') as hands, gzip.open(out/'decisions.jsonl.gz','rt') as traces:
        for line in hands:
            row = json.loads(line); coord = row['block'],row['rotation']
            if coord not in expected or coord in seen or row['status']!='completed' or row['arm']!='candidate': raise ValueError('Incomplete/foreign hand')
            if row['outcome_sha256']!=digest({k:v for k,v in row.items() if k!='outcome_sha256'}): raise ValueError('Outcome changed')
            seen.add(coord); b = schedule[row['block']]; r = row['rotation']
            ids = tuple(f'player-{(seat-r)%2}' for seat in (0,1))
            hand = Hand.start(Table(ids,(10000,10000),b.button,50,100,'0.01'),hand_id=f'table-0/{b.index}/{r}/0',seed=b.deal_seeds[0])
            rival = StylePolicy(Style(**inputs['style']),action_seed(plan.root_seed,b,r,'candidate',1))
            missing = translated = False; positive_exact_only = True; investments = Counter(); final_street = 'preflop'
            investments['preflop'] = 10000-hand.observe(r).players[r].stack
            for event in row['events']:
                if event['event']!='ActionTaken': continue
                if hand.finished or hand.actor!=event['seat']: raise ValueError('Actor differs')
                view = hand.observe(hand.actor); d = json.loads(next(traces)); role = int(view.player_id!='player-0')
                action = Action(ActionKind(event['action']['kind']),event['action']['raise_to']); view.legal_actions.validate(action)
                if (d['block'],d['rotation'],d['role'],d['seed'],d['hand_id'],d['seat'],d['street'],d['action'])!=(
                    b.index,r,role,action_seed(plan.root_seed,b,r,'candidate',role),view.hand_id,view.seat,view.street.value,asdict(action)):
                    raise ValueError('Private stream/action trace differs')
                final_street = view.street.value
                if role==0:
                    menu = schema_choices(view,spec["abstraction"])
                    key = information_key(view,menu,schema=spec['abstraction'],equity_cards=cards)
                    info = d['translation']; probs = d['effective_probabilities']
                    unresolved = info['mode']=='uniform' and info['reason']=='missing-key'
                    positive_exact_only &= info['mode']=='exact' and info['reason']=='positive-mass'
                    if info['reason']=='zero-mass': action_mix['known_zero_mass'][action.kind.value]+=1
                    if info['mode']=='exact': action_mix['positive_mass_exact'][action.kind.value]+=1
                    if d['key']!=key or d['menu']!=json.loads(canonical([asdict(c) for c in menu])):
                        raise ValueError('Schema/menu differs')
                    if len(probs)!=len(menu) or not isclose(sum(probs),1,abs_tol=1e-8) or not any(c.action==action and p>0 for c,p in zip(menu,probs)):
                        raise ValueError('Unsupported action probability')
                    if info['mode']=='translated':
                        selected = information_key(view,menu,schema=HU100_SCHEMA,
                            history_label_overrides={int(i):label for i,label in info['overrides']})
                        if selected!=info['selected_key'] or not inputs['translation'] or d['known']:
                            raise ValueError('Translated witness differs')
                        translated = True; action_mix['translated'][action.kind.value]+=1
                    if unresolved:
                        missing = True; action_mix['missing'][action.kind.value]+=1
                        if d['probabilities']!=[1/len(menu)]*len(menu): raise ValueError('Missing baseline differs')
                    if d['diagnostic_override']!=bool(unresolved and inputs['fallback']): raise ValueError('Override classification differs')
                    if d['diagnostic_override']:
                        expected_action = Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else
                            ActionKind.CALL if inputs['fallback']=='call' else ActionKind.FOLD)
                        if action!=expected_action or probs!=[float(c.action==expected_action) for c in menu]: raise ValueError('Fallback intervention differs')
                    elif probs!=d['probabilities']: raise ValueError('Stored decision altered')
                    cost = view.legal_actions.call_amount if action.kind==ActionKind.CALL else (
                        action.raise_to-view.players[view.seat].street_bet if action.kind==ActionKind.RAISE else 0)
                    investments[view.street.value]+=cost
                elif rival.choose_action(view)!=action: raise ValueError('Scripted opponent replay differs')
                hand=hand.apply(action); actions+=1
            net = [p.stack-10000 for p in hand.observe(0).players]
            if not hand.finished or canonical(public_events(hand.events))!=canonical(row['events']) or net!=row['net_chips'] or sum(net)!=0 or row['candidate_chips']!=net[r] or row['participants']!=list(ids):
                raise ValueError('Replay/settlement differs')
            kind = 'missing' if missing else 'translated_only' if translated else 'exact_or_zero_mass'
            values[coord]=net[r]; classes[coord]=kind; positive_exact_hands[coord]=positive_exact_only
            committed.update(investments); terminal_net[final_street]+=net[r]
            if net[r]<0:
                net_losses_by_terminal[final_street]-=net[r]
                for street,chips in investments.items(): lost_by_street[street]+=chips
        if next(traces,None) is not None or seen!=expected or len(seen)!=done['hands'] or actions!=done['decisions']: raise ValueError('Incomplete traces')
    result = {'status':'verified','hands_replayed':len(seen),'actions_replayed':actions,'seconds':perf_counter()-tick,
              'all_settlements_replayed':True,'all_opponent_actions_independently_reproduced':True}
    if not cost_only:
        result.update(values={f'{b}/{r}':v for (b,r),v in values.items()},hand_classes={f'{b}/{r}':v for (b,r),v in classes.items()},
             hand_positive_exact_only={f'{b}/{r}':v for (b,r),v in positive_exact_hands.items()},
             action_mix={k:dict(v) for k,v in action_mix.items()},chips_committed_by_street=dict(committed),
             chips_committed_in_losing_hands_by_street=dict(lost_by_street),net_chips_by_terminal_street=dict(terminal_net),gross_chips_lost_by_terminal_street=dict(net_losses_by_terminal))
    put(output,result); return result
