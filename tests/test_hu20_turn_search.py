"""Public-state search, exact insertion, likelihood floors and failure contracts."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pytest

from src.arena.endgame_quality import _world
from src.blueprint.abstraction import Choice, choices
from src.blueprint.hu20_turn_search import HU20TurnSearchPolicy, TurnSearchConfig
from src.blueprint.hu20_turn_solver import ExternalTurnSolver, PolicyMatrix, SolveFailure, parse_profiles
from src.blueprint.hu20_turn_tree import betting_line, compile_tree, line_key, round_root
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street


class Uniform:
    description = {"fixture": "uniform"}

    def distribution(self, view):
        menu = choices(view, raise_cap=None, free_fold=False)
        return menu, (1 / len(menu),) * len(menu), False


class FakeSolver:
    expected_sha256 = "fixture"

    def __init__(self):
        self.requests = []

    def solve(self, request, deadline):
        self.requests.append(request)
        result = {}
        locks = {line_key(n["line"]): n for n in request["locks"]}
        for node in request["nodes"]:
            if node["terminal"] or node["street"] != request["initial_street"]:
                continue
            holdings = tuple(tuple(sorted(r["hand"])) for r in request["ranges"][node["player"]]
                             if r["weight"] > 0)
            p = np.full((len(holdings), len(node["actions"])), 1 / len(node["actions"]))
            if line_key(node["line"]) in locks:
                lock = locks[line_key(node["line"])]
                lookup = {tuple(sorted(h)): i for i, h in enumerate(lock["holdings"])}
                values = np.asarray(lock["strategy"]).reshape(len(node["actions"]), -1).T
                p = np.asarray([values[lookup[h]] for h in holdings])
            p.setflags(write=False)
            menu = tuple(Choice(name, Action(ActionKind(a["kind"]), a["raise_to"]))
                         for name, a in zip(node["names"], node["native_actions"], strict=True))
            result[line_key(node["line"])] = PolicyMatrix(menu, holdings, p)
        return result


def fixture(street=Street.TURN, *, deep=False):
    hand = Hand.start(Table(("a", "b"), (2000, 2000), button=0), hand_id="search", seed=31)
    if not deep:
        hand = hand.apply(Action(ActionKind.RAISE, 1000))
    while hand.observe(hand.actor).street != street:
        view = hand.observe(hand.actor)
        hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL))
    return hand


def test_holding_queries_share_one_public_solution_and_hidden_worlds():
    hand = fixture(); view = hand.observe(hand.actor); solver = FakeSolver()
    policy = HU20TurnSearchPolicy(Uniform(), solver)
    first = policy.distribution(view)
    available = [c for c in ("2c", "2d", "3c", "3d", "4c", "4d", "5c", "5d") if c not in view.board + view.hole_cards]
    other = _world(view.history, view.board, {view.seat: tuple(available[:2])})
    second = policy.distribution(other.observe(view.seat))
    assert first == second and len(solver.requests) == 1
    changed = _world(view.history, view.board, {view.seat: view.hole_cards,
                                              1 - view.seat: tuple(available[2:4])})
    assert policy.distribution(changed.observe(view.seat)) == first
    assert "hole_cards" not in json.dumps(solver.requests)


def test_on_tree_bet_is_retained_and_exact_off_menu_relocks_hero():
    hand = fixture(); solver = FakeSolver(); policy = HU20TurnSearchPolicy(Uniform(), solver)
    actor = hand.actor; view = hand.observe(actor); policy.distribution(view)
    first = solver.requests[0]
    hand = hand.apply(Action(ActionKind.CHECK))
    native = min((c.action for c in choices(hand.observe(hand.actor), raise_cap=None, free_fold=False)
                  if c.action.kind == ActionKind.RAISE), key=lambda a: a.raise_to)
    on_tree = hand.apply(native)
    policy.distribution(on_tree.observe(actor))
    assert len(solver.requests) == 1
    off_tree = hand.apply(Action(ActionKind.RAISE, 333))
    policy.distribution(off_tree.observe(actor))
    assert len(solver.requests) == 2
    req = solver.requests[-1]
    assert any(a.get("amount") == 333 for n in req["nodes"] if not n["terminal"] for a in n["actions"])
    assert len(req["locks"]) == 1
    lock = req["locks"][0]
    assert lock["actions"] == first["nodes"][0]["actions"]
    p = np.asarray(lock["strategy"]).reshape(len(lock["actions"]), -1)
    assert np.allclose(p, 1 / len(lock["actions"]))


def test_opponent_floor_restores_action_support_but_not_cards_or_own_support():
    class ZeroCall(Uniform):
        def distribution(self, view):
            menu, _, trained = super().distribution(view)
            weights = [float(c.action.kind != ActionKind.CALL) for c in menu]
            return menu, tuple(w / sum(weights) for w in weights), trained
    root = fixture(deep=True).events
    with pytest.raises(SolveFailure, match="no support") as caught:
        HU20TurnSearchPolicy(ZeroCall(), FakeSolver(), TurnSearchConfig(opponent_likelihood_floor=0))._ranges(root, 1, monotonic()+30)
    assert caught.value.cause == "zero_support_opponent"
    policy = HU20TurnSearchPolicy(ZeroCall(), FakeSolver())
    ranges, counts = policy._ranges(root, 1, monotonic()+30)
    assert counts["floored_opponent_factors"] > 0
    board = fixture(deep=True).observe(1).board
    assert all(not set(h).intersection(board) for rows in ranges.values() for h, _ in rows)
    with pytest.raises(SolveFailure) as caught:
        policy._ranges(root, 0, monotonic()+30)
    assert caught.value.cause == "zero_support_own"


def test_river_ranges_use_turn_solution_instead_of_blueprint():
    class BetOnly(Uniform):
        def distribution(self, view):
            menu, p, trained = super().distribution(view)
            if view.street == Street.TURN:
                p = tuple(float(c.action.kind == ActionKind.RAISE) for c in menu)
                p = tuple(v / sum(p) for v in p)
            return menu, p, trained
    hand = fixture(); bot = hand.actor
    policy = HU20TurnSearchPolicy(BetOnly(), FakeSolver(), TurnSearchConfig(opponent_likelihood_floor=0))
    policy.distribution(hand.observe(bot))
    hand = hand.apply(Action(ActionKind.CHECK)).apply(Action(ActionKind.CHECK))
    ranges, counts = policy._ranges(round_root(hand.events), bot, monotonic()+30)
    assert counts["turn_solution_factors"] > 0
    assert all(sum(w for _, w in ranges[s]) == pytest.approx(1) for s in (0, 1))
    policy.distribution(hand.observe(hand.actor))
    assert policy.solver.requests[-1]["initial_street"] == "river"


def test_sampled_river_conditioning_independently_matches_used_matrices():
    from random import Random
    from scripts.validate_hu20_search_rivers import sampled_river,check_conditioning
    hand=fixture();bot=hand.actor
    policy=HU20TurnSearchPolicy(Uniform(),FakeSolver())
    river,counts=sampled_river(policy,hand.events,bot,Random(77),lambda:None)
    root=round_root(river.events)
    solution=policy._resolve(root,bot,monotonic()+30)
    assert counts["attempts"]>=1
    assert check_conditioning(policy,root,bot,solution.ranges)<=1e-10
    assert all(not set(h).intersection(river.observe(bot).board)
               for rows in solution.ranges.values() for h,w in rows)


def test_exact_insertions_follow_pinned_solver_action_order():
    hand=fixture();later=hand.apply(Action(ActionKind.CHECK)).apply(Action(ActionKind.RAISE,333))
    request=compile_tree(hand.events,later.events)
    node=next(n for n in request["nodes"] if n["line"]==[{"kind":"Check"}])
    assert node["actions"]==[{"kind":"Check"},{"kind":"Bet","amount":100},
                             {"kind":"Bet","amount":333},{"kind":"AllIn","amount":1000}]


def test_live_fallback_locks_survive_probe_eviction_and_new_hand_resets():
    class Recovering(FakeSolver):
        def solve(self,request,deadline):
            if not self.requests:
                self.requests.append(request)
                raise SolveFailure("timeout","First live solve failed")
            profiles=super().solve(request,deadline)
            locks={line_key(n["line"]) for n in request["locks"]}
            for key,matrix in list(profiles.items()):
                if key in locks:continue
                column=next((i for i,c in enumerate(matrix.menu) if c.action.kind==ActionKind.RAISE),0)
                p=np.zeros_like(matrix.probabilities);p[:,column]=1;p.setflags(write=False)
                profiles[key]=PolicyMatrix(matrix.menu,matrix.holdings,p)
            return profiles
    hand=fixture();bot=hand.actor;source=Uniform();solver=Recovering()
    policy=HU20TurnSearchPolicy(source,solver,TurnSearchConfig(cache_entries=1))
    assert policy.distribution(hand.observe(bot),query_kind="play")==source.distribution(hand.observe(bot))
    played=dict(policy.played)
    hypothetical=hand.apply(Action(ActionKind.CHECK)).apply(Action(ActionKind.RAISE,444))
    policy.distribution(hypothetical.observe(bot))
    assert policy.played==played
    actual=hand.apply(Action(ActionKind.CHECK)).apply(Action(ActionKind.RAISE,333))
    policy.distribution(actual.observe(bot),query_kind="play")
    lock=next(n for n in solver.requests[-1]["locks"] if n["line"]==[])
    assert np.allclose(np.asarray(lock["strategy"]),1/len(lock["actions"]))
    assert policy.distribution(hand.observe(bot))==source.distribution(hand.observe(bot))
    fresh=Hand.start(Table(("a","b"),(2000,2000),button=0),hand_id="fresh",seed=33)
    policy.distribution(fresh.observe(fresh.actor),query_kind="play")
    assert not policy.played and policy.live_hand_id=="fresh"
    assert not policy.live_solutions and not policy.live_turn_models and not policy.last_turn_source


def test_live_turn_profiles_survive_lbr_eviction_without_river_resolve():
    class NoSecondTurn(FakeSolver):
        def solve(self,request,deadline):
            if self.requests and request["initial_street"]=="turn":
                raise SolveFailure("timeout","Evicted turn must not be re-solved")
            return super().solve(request,deadline)
    hand=fixture();bot=hand.actor;solver=NoSecondTurn()
    policy=HU20TurnSearchPolicy(Uniform(),solver,TurnSearchConfig(cache_entries=1))
    policy.distribution(hand.observe(bot),query_kind="play")
    retained=policy.live_solutions[bot,hand.events]
    live=dict(policy.live_solutions)
    probe=hand.apply(Action(ActionKind.CHECK)).apply(Action(ActionKind.RAISE,333))
    policy.distribution(probe.observe(bot))
    assert policy.live_solutions==live and policy.last_turn_source[bot][1] is retained
    policy.cache.clear();policy.range_cache.clear()
    river=hand.apply(Action(ActionKind.CHECK)).apply(Action(ActionKind.CHECK))
    policy.distribution(river.observe(bot),query_kind="play")
    assert [r["initial_street"] for r in solver.requests]==["turn","river"]
    assert not any(k.startswith("range:turn_conditioning_fallback:") for k in policy.stats)
    assert policy.records[-2]["range_coverage"]["turn_solution_factors"]>0


def test_failed_first_live_turn_keeps_declared_base_opponent_model():
    class FailTurn(FakeSolver):
        def solve(self,request,deadline):
            if request["initial_street"]=="turn":
                self.requests.append(request)
                raise SolveFailure("timeout","Declared base turn fallback")
            return super().solve(request,deadline)
    hand=fixture().apply(Action(ActionKind.CHECK));bot=hand.actor
    policy=HU20TurnSearchPolicy(Uniform(),FailTurn())
    policy.distribution(hand.observe(bot),query_kind="play")
    turn_attempts=len(policy.solver.requests)
    river=hand.apply(Action(ActionKind.CHECK));policy.cache.clear();policy.range_cache.clear()
    solution=policy._resolve(round_root(river.events),bot,monotonic()+30)
    assert solution.coverage["played_base_turn_actions"]==1
    assert [r["initial_street"] for r in policy.solver.requests]==["turn"]*turn_attempts+["river"]
    assert not any(k.startswith("range:turn_conditioning_fallback:") for k in policy.stats)


@pytest.mark.parametrize("cause", ["solver_unavailable", "timeout", "memory_refusal", "invalid_response"])
def test_failed_search_is_a_legal_base_fallback_and_is_cached(cause):
    class Failed(FakeSolver):
        def solve(self, request, deadline):
            self.requests.append(request)
            raise SolveFailure(cause, "retained failure")
    hand = fixture(); view = hand.observe(hand.actor); source = Uniform()
    solver = Failed(); policy = HU20TurnSearchPolicy(source, solver)
    assert policy.distribution(view, query_kind="play") == source.distribution(view)
    assert policy.distribution(view) == source.distribution(view)
    assert len(solver.requests) == 1
    assert policy.stats["play:fallback:" + cause] == 1


def test_new_hand_and_probe_do_not_mutate_prior_context():
    hand = fixture(); policy = HU20TurnSearchPolicy(Uniform(), FakeSolver())
    view = hand.observe(hand.actor); initial = policy.distribution(view)
    original = list(policy.solver.requests)
    hypothetical = hand.apply(Action(ActionKind.CHECK))
    hypothetical = hypothetical.apply(Action(ActionKind.RAISE, 333))
    policy.distribution(hypothetical.observe(view.seat))
    assert policy.distribution(view) == initial
    assert policy.solver.requests[0] == original[0]
    assert policy.distribution(replace(view, hand_id="another")) == policy.distribution(view)


def test_profile_parser_rejects_tampering_and_missing_rows(tmp_path):
    hand = fixture(Street.RIVER)
    request = compile_tree(hand.events, hand.events)
    request["ranges"] = [[{"hand": ["2c", "2d"], "weight": 1}], [{"hand": ["3c", "3d"], "weight": 1}]]
    node = request["nodes"][0]
    row = {"line": [], "board": request["board"], "player": node["player"],
           "actions": node["actions"], "holdings": [["4c", "4d"]], "strategy": [1]}
    path = tmp_path / "profiles.jsonl"; path.write_text(json.dumps(row)+"\n")
    with pytest.raises(SolveFailure, match="holdings"):
        parse_profiles(request, path)
    path.write_text("")
    with pytest.raises(SolveFailure, match="Incomplete"):
        parse_profiles(request, path)


def test_profile_parser_rejects_normalized_strategy_that_violates_hero_lock(tmp_path):
    hand=fixture(Street.RIVER);request=compile_tree(hand.events,hand.events)
    cards=[c for c in ("2c","2d","3c","3d","4c","4d","5c","5d") if c not in request["board"]]
    request["ranges"]=[[{"hand":cards[:2],"weight":1}],[{"hand":cards[2:4],"weight":1}]]
    request["locks"]=[];profiles=FakeSolver().solve(request,monotonic()+30)
    nodes={line_key(n["line"]):n for n in request["nodes"] if not n["terminal"]}
    rows=[{"line":nodes[k]["line"],"board":request["board"],"player":nodes[k]["player"],
        "actions":nodes[k]["actions"],"holdings":m.holdings,"strategy":m.probabilities.T.ravel().tolist()}
        for k,m in profiles.items()]
    path=tmp_path/"profiles.jsonl";path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    lock=dict(rows[0],strategy=[1]+[0]*(len(rows[0]["actions"])-1))
    request["locks"]=[lock]
    with pytest.raises(SolveFailure,match="prior hero lock"):parse_profiles(request,path)


def test_process_timeout_kills_child_and_retains_receipt(tmp_path):
    executable = tmp_path / "slow"
    executable.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n")
    executable.chmod(0o755)
    solver = ExternalTurnSolver(executable, tmp_path / "evidence")
    with pytest.raises(SolveFailure) as caught:
        solver.solve({"spot": "fixture", "threads": 1}, monotonic()+.15)
    assert caught.value.cause == "timeout"
    assert solver.records[-1]["cause"] == "timeout"
    receipt = Path(solver.records[-1]["path"])
    assert (receipt / "manifest.json").is_file() and (receipt / "stderr.log").is_file()


@pytest.mark.parametrize("kw", [{"decision_seconds":31}, {"iterations":0},
    {"opponent_likelihood_floor":.05}, {"menu":"unknown"}, {"compress":1}])
def test_configuration_rejects_undeclared_settings(kw):
    with pytest.raises(ValueError): TurnSearchConfig(**kw)


def test_native_memory_refusal_uses_family_cap_and_keeps_requested_identity(tmp_path):
    executable=tmp_path/'refuse'
    executable.write_text('#!/usr/bin/env python3\nimport json,sys\n'
        'r=json.load(open(sys.argv[1]))\n'
        'assert r["memory_budget_bytes"]==64*1024**2\n'
        'assert r["requested_memory_budget_bytes"]==5*1024**3\n'
        'open(sys.argv[2],"w").write(json.dumps({"event":"completion","status":"oversize","allocated":False})+"\\n")\n')
    executable.chmod(0o755)
    solver=ExternalTurnSolver(executable,tmp_path/'evidence',allocation_budget=lambda requested:64*1024**2)
    request={'spot':'fixture','threads':1,'memory_budget_bytes':5*1024**3}
    with pytest.raises(SolveFailure) as caught:solver.solve(request,monotonic()+5)
    assert caught.value.cause=='memory_refusal'
    assert request['memory_budget_bytes']==5*1024**3 and 'requested_memory_budget_bytes' not in request
    receipt=solver.records[-1]
    assert receipt['memory_admission']=={'configured_bytes':5*1024**3,'admitted_bytes':64*1024**2}
    assert json.loads((Path(receipt['path'])/'manifest.json').read_text())['request.json']['bytes']>0


def test_empty_family_headroom_refuses_without_launching_native_process(tmp_path,monkeypatch):
    executable=tmp_path/'unused';executable.write_text('not executed')
    def forbidden(*args,**kwargs):raise AssertionError('A native process must not start')
    monkeypatch.setattr('src.blueprint.hu20_turn_solver.subprocess.Popen',forbidden)
    solver=ExternalTurnSolver(executable,tmp_path/'evidence',allocation_budget=lambda requested:0)
    with pytest.raises(SolveFailure) as caught:
        solver.solve({'spot':'fixture','threads':1,'memory_budget_bytes':5*1024**3},monotonic()+5)
    assert caught.value.cause=='memory_refusal'
    assert solver.records[-1]['memory_admission']['admitted_bytes']==0


def test_research_deadline_is_explicit_and_bounded():
    assert TurnSearchConfig(decision_seconds=120).decision_seconds==120
    assert TurnSearchConfig().decision_seconds==30
    with pytest.raises(ValueError):TurnSearchConfig(decision_seconds=121)


def test_unlocked_turn_requests_and_full_matrices_are_identical_for_both_seats_at_epsilon_zero():
    from scripts.forecast_hu20_search_final import requests_shareable
    hand=fixture();solvers=[FakeSolver(),FakeSolver()]
    policies=[HU20TurnSearchPolicy(Uniform(),s,TurnSearchConfig(opponent_likelihood_floor=0)) for s in solvers]
    results=[p._resolve(hand.events,bot,monotonic()+30) for bot,p in enumerate(policies)]
    assert requests_shareable(results[0].request,results[1].request)
    assert results[0].request==results[1].request and not results[0].request['locks']
    for key in results[0].profiles:
        a,b=results[0].profiles[key],results[1].profiles[key]
        assert a.menu==b.menu and a.holdings==b.holdings
        np.testing.assert_array_equal(a.probabilities,b.probabilities)
    altered=dict(results[1].request,locks=[{'line':[]}])
    assert not requests_shareable(results[0].request,altered)


def test_opponent_only_floor_can_make_unlocked_requests_seat_dependent():
    from scripts.forecast_hu20_search_final import requests_shareable
    class RareAction(Uniform):
        def distribution(self,view):
            menu,_,trained=super().distribution(view)
            if all(c.action.kind!=ActionKind.CALL for c in menu):
                return menu,(1/len(menu),)*len(menu),trained
            rare=.001 if any(c.startswith('2') for c in view.hole_cards) else .02
            other=sum(c.action.kind!=ActionKind.CALL for c in menu)
            p=[rare if c.action.kind==ActionKind.CALL else (1-rare)/other for c in menu]
            return menu,tuple(p),trained
    root=fixture(deep=True).events
    results=[HU20TurnSearchPolicy(RareAction(),FakeSolver(),TurnSearchConfig(opponent_likelihood_floor=.01))._resolve(
        root,bot,monotonic()+30) for bot in (0,1)]
    assert not requests_shareable(results[0].request,results[1].request)
