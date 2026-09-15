from dataclasses import FrozenInstanceError, replace

import pytest

from src.arena.policies import CheckCall
from src.arena.runner import run_schedule
from src.arena.schedule import Plan, Scenario, build_schedule
from src.game.types import Action, ActionKind


def collect(plan, **kwargs):
    rows, times = [], []
    ok = run_schedule(
        plan,
        build_schedule(plan),
        lambda row, timing: (rows.append(row), times.append(timing)),
        **kwargs,
    )
    return ok, rows, times


def test_paired_rotation_replays_identical_hands_for_identical_policies():
    plan = Plan(
        (Scenario("four", (2000,) * 4),),
        candidate="random",
        baseline="random",
        opponents=("random",),
        blocks=2,
    )
    ok, rows, _ = collect(plan)
    assert ok and len(rows) == 16
    for candidate, baseline in zip(rows[::2], rows[1::2]):
        assert candidate["net_chips"] == baseline["net_chips"]
        assert candidate["events"] == baseline["events"]
        assert (
            candidate["candidate_chips"]
            == candidate["net_chips"][candidate["rotation"]]
        )
        assert sum(candidate["net_chips"]) == 0
        start = candidate["events"][0]
        assert start["player_ids"][candidate["rotation"]] == "player-0"
    assert {r["rotation"] for r in rows} == {0, 1, 2, 3}
    assert rows == collect(plan)[1]


def test_seat_rotations_reuse_physical_deal_and_keep_policy_inputs_private():
    views = []
    instances = []

    class Observer(CheckCall):
        def choose_action(self, view):
            views.append(view)
            assert not hasattr(view, "_state")
            assert not hasattr(view, "seed")
            assert not hasattr(view, "opponents")
            assert (
                all(p.shown_cards == () for p in view.players)
                if len(view.board) == 0
                else True
            )
            with pytest.raises(FrozenInstanceError):
                view.hole_cards = ()
            return super().choose_action(view)

    def factory(name, seed):
        policy = Observer()
        instances.append(policy)
        return policy

    plan = Plan((Scenario("four", (2000,) * 4),), blocks=1)
    assert collect(plan, factory=factory)[0]
    assert len({id(p) for p in instances}) == 4 * 4 * 2
    for physical_seat in range(4):
        cards = {v.hole_cards for v in views if v.seat == physical_seat}
        assert len(cards) == 1
    assert all(v.previous_hands == () for v in views)


def test_session_carries_bankroll_and_history_but_resets_between_rotations():
    seen = []

    class Observer(CheckCall):
        def choose_action(self, view):
            seen.append(view)
            return super().choose_action(view)

    plan = Plan(
        (Scenario("session", (2000,) * 4, mode="session", hands_per_rotation=3),),
        candidate="check_call",
        baseline="check_call",
        blocks=2,
    )
    ok, rows, _ = collect(plan, factory=lambda name, seed: Observer())
    assert ok and len(rows) == 2 * 4 * 2 * 3
    assert max(len(v.previous_hands) for v in seen) == 2
    assert all(
        len(v.previous_hands) == int(v.hand_id.rsplit("/", 1)[1]) - 1 for v in seen
    )
    for first, second, third in zip(rows[::3], rows[1::3], rows[2::3]):
        assert second["events"][0]["stacks"] == first["events"][-1]["stacks"]
        assert third["events"][0]["stacks"] == second["events"][-1]["stacks"]
    assert rows == collect(plan, factory=lambda name, seed: Observer())[1]


def test_busted_session_reloads_are_not_counted_as_winnings():
    plan = Plan(
        (
            Scenario(
                "short", (100, 200, 300, 400), mode="session", hands_per_rotation=8
            ),
        ),
        candidate="random",
        baseline="random",
        opponents=("random",),
        blocks=2,
    )
    ok, rows, _ = collect(plan)
    assert ok and any(r["reloads"] for r in rows)
    assert all(sum(r["net_chips"]) == 0 for r in rows)
    assert any(len(r["participants"]) < 4 for r in rows)


@pytest.mark.parametrize(
    "broken, status", [("illegal", "invalid_action"), ("exception", "failed")]
)
def test_policy_failure_stops_the_run_and_keeps_the_attempt(broken, status):
    class Broken:
        def choose_action(self, view):
            if broken == "exception":
                raise RuntimeError("policy crashed")
            return Action(ActionKind.RAISE, 10**9)

    plan = Plan((Scenario("four", (2000,) * 4),), blocks=2)
    ok, rows, timings = collect(plan, factory=lambda name, seed: Broken())
    assert not ok and len(rows) == 1
    assert rows[0]["status"] == status
    assert rows[0]["candidate_chips"] is None
    assert rows[0]["events"][0]["event"] == "HandStarted"
    assert timings[0]["decisions"]


def test_decision_cap_and_shared_policy_instance_are_failures():
    plan = Plan((Scenario("four", (2000,) * 4),), blocks=1, max_decisions=1)
    assert collect(plan)[1][0]["status"] == "failed"
    shared = CheckCall()
    assert (
        collect(replace(plan, max_decisions=100), factory=lambda name, seed: shared)[1][
            0
        ]["status"]
        == "failed"
    )
