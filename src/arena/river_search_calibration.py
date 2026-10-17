"""Checked duplicate hands; evaluator state never enters a policy observation."""

from dataclasses import asdict
from math import fsum, isfinite
import json
from pathlib import Path
from time import monotonic

from src.arena.policies import make_policy
from src.arena.runner import public_events
from src.arena.schedule import canonical, digest
from src.blueprint.hu200_river_search import HU200RiverSearchPlayer, HU200RiverSearchPolicy
from src.blueprint.hu20_turn_solver import ExternalTurnSolver
from src.blueprint.river_evidence import COMPACT_RETENTION_RULE, file_pin
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street


class ForbiddenSolver:
    def __init__(self, identity):
        self.expected_sha256 = identity
        self.records = []

    def solve(self, *args, **kwargs):
        raise AssertionError("Ledger reproduction invoked a native solver")


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(canonical(value) + "\n")


def view_key(view):
    return digest(dict(history=public_events(view.history), seat=view.seat,
                       hole_cards=view.hole_cards, board=view.board))


def replay_actions(table, hand_id, deal, decisions):
    hand = Hand.start(table, hand_id=hand_id, seed=deal)
    for row in decisions:
        if row.get("sha256") != digest({k:v for k,v in row.items() if k != "sha256"}):
            raise ValueError("Decision checksum differs")
        if hand.finished or hand.actor != row["seat"]:
            raise ValueError("Independent replay actor differs")
        view = hand.observe(hand.actor)
        if view_key(view) != row["observation_sha256"]:
            raise ValueError("Independent replay observation differs")
        action = Action(ActionKind(row["action"]["kind"]), row["action"]["raise_to"])
        view.legal_actions.validate(action)
        hand = hand.apply(action)
    if not hand.finished:
        raise ValueError("Independent replay is incomplete")
    return hand


def outcome(hand, target):
    view = hand.observe(0)
    chips = [p.stack - p.starting_stack for p in view.players]
    if not hand.finished or sum(chips):
        raise ValueError("Incomplete or nonconserving hand")
    return dict(net_chips=chips, target_chips=chips[target],
                public_events_sha256=digest(public_events(hand.events)))


def make_players(base, binary, identity, config, directory, seeds, opponent,
                 searched, replay, check):
    if searched:
        solver = (ForbiddenSolver(identity) if replay else ExternalTurnSolver(
            binary, directory / "native", expected_sha256=identity,
            compact_profiles=True, profile_retention=COMPACT_RETENTION_RULE,
            resource_check=check))
        policy = HU200RiverSearchPolicy(base, solver, config,
            ledger=directory / "ledger", replay_ledger=replay)
        original = policy._check
        def guarded(deadline):
            check()
            original(deadline)
        policy._check = guarded
        hero = HU200RiverSearchPlayer(policy, seeds[0])
    else:
        policy = solver = None
        hero = base.policy(seeds[0])
    rival = make_policy(opponent, seeds[1]) if opponent else base.policy(seeds[1])
    return (hero, rival), policy, solver


def play_hand(base, binary, identity, config, directory, *, deal, seeds, button,
              rotation, hand_id, opponent=None, searched=True, check=lambda: None,
              player_builder=make_players):
    """Play once, independently replay, then reproduce with fresh player state."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    ids = tuple(f"player-{(seat - rotation) % 2}" for seat in (0, 1))
    table = Table(ids, (20000, 20000), button, 50, 100, "0.01")
    started = monotonic()
    decisions = []
    search_seconds = 0.
    search_rows = []
    def one(reproduce=False):
        nonlocal search_seconds
        players, policy, solver = player_builder(base, binary, identity, config,
            directory, seeds, opponent, searched, reproduce, check)
        hand = Hand.start(table, hand_id=hand_id, seed=deal)
        observed = []
        for index in range(1000):
            check()
            if hand.finished:
                break
            view = hand.observe(hand.actor)
            role = int(view.player_id != "player-0")
            player = players[role]
            native_before = len(solver.records) if solver is not None else 0
            tick = monotonic()
            action = player.choose_action(view)
            view.legal_actions.validate(action)
            row = dict(index=index, seat=view.seat, role=role, street=view.street.value,
                observation_sha256=view_key(view), action=asdict(action))
            if role == 0 and policy is not None and view.street == Street.RIVER:
                telemetry = dict(player.last_decision)
                for key in ("search_action", "blueprint_action"):
                    telemetry[key] = asdict(telemetry[key])
                if telemetry["fallback"]:
                    menu, probabilities, _ = base.distribution(view)
                else:
                    matrix = policy.played[view.seat, view.history][0]
                    menu, probabilities = matrix.menu, matrix.row(view.hole_cards)
                row.update(ledger_key=telemetry["public_history"], fallback=telemetry["fallback"],
                    fallback_cause=telemetry["fallback_cause"],
                    changed_action=telemetry["changed_action"],
                    model_ev_chips=telemetry["model_ev_chips"],
                    model_ev_delta_chips=telemetry.get("model_ev_delta_chips"))
                if not reproduce:
                    telemetry["native_records"] = solver.records[native_before:]
                    telemetry["sha256"] = digest(telemetry)
                    put(directory / "telemetry" / f"{index}.json", telemetry)
                    elapsed = monotonic() - tick
                    timing = dict(complete_seconds=elapsed)
                    put(directory / "telemetry" / f"{index}-time.json", dict(timing,sha256=digest(timing)))
                    search_seconds += elapsed
                    search_rows.append(dict(complete_seconds=elapsed, **telemetry))
            elif opponent is None or role == 0:
                cached = getattr(player, "last_decision", None)
                if isinstance(cached, tuple) and len(cached) == 4:
                    menu, probabilities = cached[:2]
                else:
                    menu, probabilities, _ = base.distribution(view)
            else:
                menu = probabilities = None
            if menu is not None:
                if (len(menu) != len(probabilities) or not all(isfinite(p) and p >= 0 for p in probabilities)
                        or abs(fsum(probabilities)-1) > 1e-8):
                    raise ValueError("Invalid played probability distribution")
                row.update(menu=[asdict(c) for c in menu], probabilities=list(map(float, probabilities)))
                if not any(c.action == action and p > 0 for c, p in zip(menu, probabilities, strict=True)):
                    raise ValueError("Played action has no positive policy support")
            row["sha256"] = digest(row)
            if reproduce:
                if index >= len(decisions) or canonical(row) != canonical(decisions[index]):
                    raise ValueError("Fresh-player reproduced decision differs")
            else:
                decisions.append(row)
                # A crash retains every completed decision and ledger; terminal
                # outcomes do not feed policy input or retention selection.
                put(directory / "decisions" / f"{index}.json", row)
            observed.append(row)
            hand = hand.apply(action)
        if not hand.finished or (reproduce and len(observed) != len(decisions)):
            raise ValueError("Hand or reproduction decision inventory incomplete")
        if reproduce and solver is not None and solver.records:
            raise ValueError("Reproduction performed native work")
        return hand
    try:
        hand = one()
        primary = outcome(hand, rotation)
        primary_seconds = monotonic() - started
        put(directory / "hand.json", dict(deal_seed=deal, action_seeds=seeds,
            hand_id=hand_id, button=button, rotation=rotation, opponent=opponent,
            searched=searched, decisions=decisions, public_events=public_events(hand.events),
            **primary))
        checked = replay_actions(table, hand_id, deal, decisions)
        if outcome(checked, rotation) != primary:
            raise ValueError("Independent event/settlement replay differs")
        repeated = one(True)
        if outcome(repeated, rotation) != primary:
            raise ValueError("Fresh-player outcome differs")
        pins = {p.relative_to(directory).as_posix(): file_pin(p)
                for p in sorted(directory.rglob('*')) if p.is_file()}
        put(directory / "manifest.json", pins)
        put(directory / "verified.json", dict(independent_replay=True,
            fresh_player_reproduction=True, native_replay_solves=0,
            hand_sha256=pins['hand.json']['sha256']))
        return dict(target_chips=primary["target_chips"], primary_seconds=primary_seconds,
            total_seconds=monotonic() - started, search_seconds=search_seconds,
            search_rows=search_rows, decisions=len(decisions), verified=True)
    except BaseException as exc:
        put(directory / "incomplete.json", dict(error=repr(exc), decisions=decisions,
            hand_id=hand_id, deal_seed=deal, action_seeds=seeds))
        raise


def block_value(chips):
    """100-chip big blind, two seat rotations; convert chips to BB/100."""
    if len(chips) != 2:
        raise ValueError("A duplicate score needs both rotations")
    return sum(chips) / 2 / 100 * 100
