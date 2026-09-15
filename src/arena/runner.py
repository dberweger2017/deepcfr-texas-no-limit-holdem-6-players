"""Run paired rotations without giving policies the schedule or hidden simulator."""

from collections.abc import Callable
from dataclasses import asdict
from time import perf_counter

from src.arena.policies import make_policy
from src.arena.schedule import Block, Plan, digest
from src.game.hand import Hand, Table
from src.game.session import Session


class InvalidAction(ValueError):
    pass


def public_events(events) -> list[dict]:
    return [{"event": type(event).__name__, **asdict(event)} for event in events]


def _decision(view, policy, timings):
    started = perf_counter()
    try:
        action = policy.choose_action(view)
    finally:
        timings.append(
            {"player_id": view.player_id, "seconds": perf_counter() - started}
        )
    try:
        view.legal_actions.validate(action)
    except (ValueError, TypeError) as exc:
        raise InvalidAction(f"{view.player_id}: {exc}; returned {action!r}") from exc
    return action


def _fixed(scenario, block, rotation, ids, policies, max_decisions, trace, timings):
    table = Table(
        ids,
        scenario.stacks,
        block.button,
        scenario.small_blind,
        scenario.big_blind,
        scenario.chip_unit,
    )
    hand = Hand.start(
        table,
        hand_id=f"{scenario.name}/{block.index}/{rotation}/0",
        seed=block.deal_seeds[0],
    )
    trace["events"] = public_events(hand.events)
    for _ in range(max_decisions):
        if hand.finished:
            break
        view = hand.observe(hand.actor)
        action = _decision(view, policies[view.player_id], timings)
        hand = hand.apply(action)
        trace["events"] = public_events(hand.events)
    if not hand.finished:
        raise RuntimeError("Hand exceeded its decision limit")
    final = hand.observe(0)
    return [p.stack - p.starting_stack for p in final.players], list(ids)


def _session(scenario, block, rotation, ids):
    session = Session(
        f"{scenario.name}/{block.index}/{rotation}",
        capacity=len(ids),
        small_blind=scenario.small_blind,
        big_blind=scenario.big_blind,
        chip_unit=scenario.chip_unit,
        min_buy_in=min(scenario.stacks),
        max_buy_in=max(scenario.stacks),
    )
    for seat, identity in enumerate(ids):
        session.join(identity, seat, scenario.stacks[seat])
    return session


def _session_hand(
    session, scenario, block, hand_index, ids, policies, max_decisions, trace, timings
):
    reloads = {}
    for player in session.seats:
        if player.status == "busted":
            chips = scenario.stacks[player.seat]
            session.top_up(player.player_id, chips)
            reloads[player.player_id] = chips
    trace["reloads"] = reloads
    starting = {p.player_id: p.stack for p in session.seats}
    session.start_hand(
        seed=block.deal_seeds[hand_index],
        opening_button=block.button if hand_index == 0 else None,
    )
    participants = list(session.participants)
    trace["events"] = public_events(session.events[-1].hand_events)
    for _ in range(max_decisions):
        if session.actor is None:
            break
        view = session.observe(session.actor)
        action = _decision(view, policies[view.player_id], timings)
        session.apply(view.player_id, action)
        trace["events"] = public_events(session.events[-1].hand_events)
    if session.actor is not None:
        raise RuntimeError("Hand exceeded its decision limit")
    session.settle()
    final = {p.player_id: p.stack for p in session.seats}
    return [final[identity] - starting[identity] for identity in ids], participants


def run_schedule(
    plan: Plan,
    blocks: tuple[Block, ...],
    emit: Callable[[dict, dict], None],
    *,
    factory=make_policy,
) -> bool:
    """Emit completed or failed attempts; stop on the first failure. No fallback actions."""
    scenarios = {scenario.name: scenario for scenario in plan.scenarios}
    for block in blocks:
        scenario = scenarios[block.scenario]
        n = len(scenario.stacks)
        for rotation in range(n):
            # Player 0 is the evaluated identity in both arms. Policy names stay host-side.
            ids = tuple(f"player-{(seat - rotation) % n}" for seat in range(n))
            for arm, policy_name in (
                ("candidate", plan.candidate),
                ("baseline", plan.baseline),
            ):
                session = None
                policies = None
                for hand_index in range(scenario.hands_per_rotation):
                    trace = {"events": [], "reloads": {}}
                    latency = []
                    started = perf_counter()
                    key = {
                        "scenario": scenario.name,
                        "block": block.index,
                        "rotation": rotation,
                        "arm": arm,
                        "hand": hand_index,
                    }
                    row = {
                        **key,
                        "mode": scenario.mode,
                        "big_blind": scenario.big_blind,
                        "initial_stacks": list(scenario.stacks),
                        "opponents": list(block.opponents),
                        "status": "completed",
                        "candidate_chips": None,
                        "net_chips": None,
                        "participants": [],
                        "error": None,
                    }
                    try:
                        if policies is None:
                            names = (policy_name, *block.opponents)
                            policies = {
                                f"player-{i}": factory(name, block.action_seeds[i])
                                for i, name in enumerate(names)
                            }
                            if len({id(policy) for policy in policies.values()}) != n:
                                raise ValueError(
                                    "Policies cannot share instances across identities"
                                )
                        if scenario.mode == "fixed":
                            net, participants = _fixed(
                                scenario,
                                block,
                                rotation,
                                ids,
                                policies,
                                plan.max_decisions,
                                trace,
                                latency,
                            )
                        else:
                            if session is None:
                                session = _session(scenario, block, rotation, ids)
                            net, participants = _session_hand(
                                session,
                                scenario,
                                block,
                                hand_index,
                                ids,
                                policies,
                                plan.max_decisions,
                                trace,
                                latency,
                            )
                        if sum(net) != 0:
                            raise RuntimeError("Hand does not conserve chips")
                        row.update(
                            candidate_chips=net[rotation],
                            net_chips=net,
                            participants=participants,
                        )
                    except Exception as exc:  # noqa: BLE001 -- Policy failures must invalidate the saved run.
                        row.update(
                            status="invalid_action"
                            if isinstance(exc, InvalidAction)
                            else "failed",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    row.update(trace)
                    row["outcome_sha256"] = digest(row)
                    emit(
                        row,
                        {
                            **key,
                            "wall_seconds": perf_counter() - started,
                            "decisions": latency,
                        },
                    )
                    if row["status"] != "completed":
                        return False
    return True
