"""Enumerate native HU20 history collisions without running or changing a solver."""

from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import sqlite3
from time import monotonic

from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, _history, choices, information_key
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken
from src.game.types import Action, ActionKind


def signature(view, menu):
    """Exactly the production payload apart from its card descriptor."""
    payload = [HU20_UNCAPPED_SCHEMA, 2, (view.seat-view.button) % 2,
               view.street.value,
               [(view.players[(view.button+i) % 2].folded,
                 view.players[(view.button+i) % 2].all_in) for i in (0, 1)],
               _history(view), [c.name for c in menu]]
    text = json.dumps(payload, separators=(",", ":"))
    return sha256(text.encode()).hexdigest(), text


def action_record(view, item):
    return {"street": view.street.value, "position": (view.seat-view.button) % 2,
            "name": item.name, "kind": item.action.kind.value,
            "raise_to": item.action.raise_to, "pot_before": view.pot,
            "paid": (item.action.raise_to-view.players[view.seat].street_bet
                     if item.action.kind == ActionKind.RAISE else
                     view.legal_actions.call_amount if item.action.kind == ActionKind.CALL else 0)}


def token_identity(payload):
    # A band collision alone need not be a policy alias: the legal menu is
    # also in information_key and can split the two public contexts.
    value = json.loads(payload) if isinstance(payload, str) else payload
    return sha256(json.dumps(value[:-1], separators=(",", ":")).encode()).hexdigest()


def history_token_counts(db):
    counts = Counter()
    for payload, n in db.execute("SELECT payload,n FROM groups"):
        counts[token_identity(payload)] += n
    return counts


def alias_origin(db, sig, cache):
    if sig not in cache:
        contexts = [json.loads(c) for (c,) in db.execute(
            "SELECT context FROM nodes WHERE sig=?", (sig,))]
        a, b = contexts[:2]
        index = next(i for i, (x, y) in enumerate(zip(a["line"], b["line"], strict=True)) if x != y)
        cache[sig] = {"action_index": index, "street": a["line"][index]["street"],
                      "alternatives": [a["line"][index], b["line"][index]]}
    return cache[sig]


def enumerate_aliases(out, *, max_decisions=5_000_000, seconds=1200, seed=202610020101):
    """Complete fixed-card tree; cards never change betting legality.

    Every live node uses the real information_key. Within each actor/street the
    fixed deal gives identical cards on every path, so a differing real key in
    a shared card-free signature is a validation failure, not an assumed alias.
    The SQLite index and exhaustive JSONL cases stay as retained run artifacts.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(out / "tree.sqlite")
    db.executescript("CREATE TABLE groups(sig TEXT PRIMARY KEY,payload TEXT,real_key TEXT,n INTEGER);"
                     "CREATE TABLE nodes(sig TEXT,street TEXT,context TEXT);"
                     "CREATE INDEX nodes_sig ON nodes(sig);")
    counts = Counter(); started = monotonic(); cache = {}; terminal = 0
    root = Hand.start(Table(("a", "b"), (2000, 2000)), hand_id="history-alias-tree", seed=seed)

    def visit(hand, path):
        nonlocal terminal
        if hand.finished:
            terminal += 1; return
        if sum(counts.values()) >= max_decisions or monotonic()-started > seconds:
            raise RuntimeError("Native alias enumeration guard; partial tree is not complete")
        view = hand.observe(hand.actor); menu = choices(view, raise_cap=None, free_fold=False)
        sig, payload = signature(view, menu)
        real = information_key(view, menu, schema=HU20_UNCAPPED_SCHEMA)
        if sig in cache:
            if cache[sig] != real:
                raise ValueError("Real information_key differs within card-free signature")
            db.execute("UPDATE groups SET n=n+1 WHERE sig=?", (sig,))
        else:
            cache[sig] = real
            db.execute("INSERT INTO groups VALUES(?,?,?,1)", (sig,payload,real))
        context = {"line":path, "pot":view.pot,
                   "stacks_by_position":[view.players[(view.button+i) % 2].stack for i in (0,1)],
                   "menu":[{"name":c.name,"raise_to":c.action.raise_to} for c in menu]}
        db.execute("INSERT INTO nodes VALUES(?,?,?)", (sig,view.street.value,json.dumps(context,separators=(",", ":"))))
        counts[view.street.value] += 1
        if sum(counts.values()) % 10000 == 0:
            db.commit(); atomic_json(out/"progress.json", {"decisions":sum(counts.values()),
                       "elapsed_seconds":monotonic()-started,"complete":False})
        for item in menu:
            visit(hand.apply(item.action), path+[action_record(view,item)])

    try:
        visit(root, [])
        db.commit()
    except BaseException as exc:
        db.commit(); atomic_json(out/"failure.json", {"error":str(exc),"counts":dict(counts),
                   "complete":False,"elapsed_seconds":monotonic()-started})
        db.close(); raise
    aliased = Counter(); groups = Counter(); examples = []
    temporary = out/"aliases.jsonl.gz.tmp"
    with gzip.open(temporary,"wt") as target:
        for sig, payload, real, n in db.execute("SELECT sig,payload,real_key,n FROM groups WHERE n>1 ORDER BY sig"):
            contexts = [json.loads(c) for (c,) in db.execute("SELECT context FROM nodes WHERE sig=?",(sig,))]
            street = json.loads(payload)[3]; aliased[street] += n; groups[street] += 1
            row = {"signature":sig,"payload_without_cards":json.loads(payload),
                   "verified_real_information_key":real,"contexts":contexts,"distinct_lines":n}
            target.write(json.dumps(row,separators=(",", ":"))+"\n")
            if len(examples)<12 or any(c["line"][-1].get("name") == "pot" and c["pot"]==400 for c in contexts):
                if len(examples)<24: examples.append(row)
    temporary.replace(out/"aliases.jsonl.gz")
    result = {"complete":True,"decision_nodes_by_street":dict(counts),"terminal_nodes":terminal,
              "unique_signatures":len(cache),"aliased_nodes_by_street":dict(aliased),
              "alias_groups_by_street":dict(groups),"every_node_real_key_verified":True,
              "key_mismatches":0,"seed":seed,"elapsed_seconds":monotonic()-started,
              "cases_sha256":file_hash(out/"aliases.jsonl.gz"),"cases_path":str(out/"aliases.jsonl.gz"),
              "tree_sha256":file_hash(out/"tree.sqlite"),"examples":examples,
              "scope":"all native-reopening live HU20 20BB decision paths; one fixed deal; no cap"}
    atomic_json(out/"result.json",result); db.close(); return result


def exposure(db_path, hands, plan, out):
    """Exact stored replay; do not infer unseen off-menu alternate histories."""
    db = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro",uri=True)
    tokens = history_token_counts(db)
    counts = Counter(); strata = defaultdict(Counter); decisions=[]; sources=[]; origins={}
    for relative, expected in sorted(plan["stored_hands"].items()):
        path=Path(hands)/relative
        if file_hash(path)!=expected: raise ValueError("Stored hand hash mismatch")
        sources.append({"path":str(path),"sha256":expected})
        with gzip.open(path,"rt") as stream:
            for number,text in enumerate(stream,1):
                row=json.loads(text)
                if row["version"]!="v1" or row["panel"]!="lbr": continue
                hand=Hand.start(Table(("a","b"),(2000,2000),button=row["button"]),
                                hand_id="alias-exposure",seed=row["deal_seed"])
                first_bet = None
                for recorded in row["actions"]:
                    view=hand.observe(hand.actor)
                    if recorded["seat"]!=view.seat or list(view.hole_cards)!=recorded["observation"]["hole_cards"]:
                        raise ValueError("Stored action/deal replay mismatch")
                    is_first = (recorded["logical_player"] == 1 and view.street.value == "flop"
                        and recorded["kind"] == "raise" and not any(
                            isinstance(e,ActionTaken) and e.street.value == "flop"
                            and e.action.kind == ActionKind.RAISE for e in hand.events))
                    if is_first:
                        paid = recorded["raise_to"]-view.players[view.seat].street_bet
                        first_bet = {"pot_sized":paid == view.pot}
                        counts['lbr_flop_first_bets'] += 1
                        counts['lbr_flop_first_bets_pot_sized'] += first_bet['pot_sized']
                    if recorded["logical_player"]==0:
                        menu=choices(view,raise_cap=None,free_fold=False); sig,payload=signature(view,menu)
                        found=db.execute("SELECT n FROM groups WHERE sig=?",(sig,)).fetchone()
                        aliased=found is not None and found[0]>1
                        token_aliased = tokens[token_identity(payload)] > 1
                        origin = alias_origin(db,sig,origins) if aliased else None
                        if first_bet is not None:
                            counts['lbr_first_bet_full_alias'] += aliased
                            counts['lbr_pot_first_bet_full_alias'] += aliased and first_bet['pot_sized']
                            first_bet = None
                        group=view.street.value+('/facing-bet' if view.legal_actions.call_amount>0 else '/free')
                        counts['decisions']+=1; counts['aliased']+=aliased
                        counts['history_token_aliased'] += token_aliased
                        strata[group]['decisions']+=1; strata[group]['aliased']+=aliased
                        strata[group]['history_token_aliased'] += token_aliased
                        strata[group]['outside_native_tree']+=found is None
                        if origin:
                            strata[group]['first_alias_'+origin['street']] += 1
                        if view.legal_actions.call_amount>0:
                            counts['facing_bet']+=1; counts['facing_bet_aliased']+=aliased
                            counts['facing_bet_history_token_aliased'] += token_aliased
                            if view.street.value=='flop':
                                counts['set_a']+=1; counts['set_a_aliased']+=aliased
                                counts['set_a_history_token_aliased'] += token_aliased
                        decisions.append({"source_sha256":expected,"source_line":number,
                            "action_index":recorded['index'],"street":view.street.value,
                            "facing_bet":view.legal_actions.call_amount>0,"signature":sig,
                            "aliased":aliased,"native_contexts":found[0] if found else None,
                            "history_token_aliased":token_aliased,
                            "alias_origin":origin,
                            "real_key":information_key(view,menu,schema=HU20_UNCAPPED_SCHEMA)})
                    hand=hand.apply(Action(ActionKind(recorded['kind']),recorded['raise_to']))
    if counts['set_a']!=273: raise ValueError("Pinned Set A decisions differ")
    result={"counts":dict(counts),"strata":{k:dict(v) for k,v in sorted(strata.items())},
            "decisions":decisions,"sources":sources,"hashes_verified":True,
            "outside_tree_interpretation":"unavailable, not evidence of no alias"}
    atomic_json(out,result);db.close();return result


def self_play_exposure(db_path, source, out, *, deals=3000, seed=202610020102):
    """Fresh realized context mixture; exports lack per-concrete-size visit counts."""
    db=sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro",uri=True)
    tokens = history_token_counts(db)
    deal_rng=Random(seed);action_rng=Random(seed+1);counts=Counter();groups={};origins={}
    for index in range(deals):
        hand=Hand.start(Table(('a','b'),(2000,2000),button=index%2),hand_id='alias-self-play',seed=deal_rng.getrandbits(64))
        path=[]
        while not hand.finished:
            view=hand.observe(hand.actor);menu,p,trained=source.distribution(view)
            sig,payload=signature(view,menu);found=db.execute('SELECT n FROM groups WHERE sig=?',(sig,)).fetchone()
            counts['decisions']+=1
            counts['history_token_aliased_decisions'] += tokens[token_identity(payload)] > 1
            if found and found[0]>1:
                counts['aliased_decisions']+=1
                key=information_key(view,menu,schema=HU20_UNCAPPED_SCHEMA)
                group=groups.setdefault(key,{"key":key,"signature":sig,"street":view.street.value,
                    "trained_key_present":trained,"native_distinct_lines":found[0],
                    "alias_origin":alias_origin(db,sig,origins),"observed_contexts":{}})
                identity=json.dumps(path,separators=(',',':'))
                cell=group['observed_contexts'].setdefault(identity,{"line":path,"decisions":0,
                    "first_aliased_action":path[group['alias_origin']['action_index']],
                    "last_raise":next((a for a in reversed(path) if a['kind']=='raise'),None),
                    "probabilities":list(p)})
                cell['decisions']+=1
            item=action_rng.choices(menu,weights=p,k=1)[0]
            path=path+[action_record(view,item)];hand=hand.apply(item.action)
    for group in groups.values():group['observed_contexts']=list(group['observed_contexts'].values())
    result={"source":source.description,"deals":deals,"deal_seed":seed,"action_seed":seed+1,
            "counts":dict(counts),"groups":list(groups.values()),"training_per_size_counts":None,
            "interpretation":"fresh self-play occupancy, not historical training-update attribution"}
    atomic_json(out,result);db.close();return result
