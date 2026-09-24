"""Evaluate a paused blueprint trainer without duplicating its large table."""

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from random import Random

from src.arena.policies import make_policy
from src.arena.report import summarize
from src.arena.runner import run_schedule
from src.arena.schedule import Plan, build_schedule, digest
from src.blueprint.abstraction import choices, information_key
from src.blueprint.solver import BlueprintTrainer, regret_match
from src.game.observation import Observation


class _TablePolicy:
    def __init__(
        self,
        trainer: BlueprintTrainer,
        seed: int,
        counts: Counter,
        *,
        uniform: bool,
        on_candidate_decision: Callable[[Observation, bool], None] | None = None,
    ):
        self.trainer = trainer
        self.random = Random(seed)
        self.counts = counts
        self.uniform = uniform
        self.on_candidate_decision = on_candidate_decision

    def choose_action(self, view):
        menu = choices(view, raise_cap=self.trainer.config.raise_cap)
        key = information_key(view, menu, schema=self.trainer.config.abstraction)
        node = self.trainer.nodes.get(key)
        if node is not None and node.names != tuple(item.name for item in menu):
            raise ValueError("Blueprint action labels differ from the observation")
        if not self.uniform:
            self.counts[(view.street.value, "trained" if node else "fallback")] += 1
            if self.on_candidate_decision is not None:
                self.on_candidate_decision(view, node is not None)
        weights = (
            regret_match(tuple(node.regrets))
            if node is not None and not self.uniform
            else (1 / len(menu),) * len(menu)
        )
        selected = self.random.choices(menu, weights=weights, k=1)[0]
        if not self.uniform:
            self.counts[
                (view.street.value, f"action_{selected.action.kind.value}")
            ] += 1
        return selected.action


def evaluate(
    trainer: BlueprintTrainer,
    plan: Plan,
    *,
    on_candidate_decision: Callable[[Observation, bool], None] | None = None,
) -> dict:
    """Return paired arena results and actual candidate decision coverage by street."""
    if plan.candidate != "blueprint_live" or plan.baseline != "blueprint_uniform":
        raise ValueError("Blueprint evaluation needs live and uniform policy arms")
    if plan.models or any(
        s.mode != "fixed" or len(s.stacks) != trainer.table.capacity
        for s in plan.scenarios
    ):
        raise ValueError(
            "Blueprint evaluation requires fixed matching tables and no artifacts"
        )
    counts: Counter = Counter()
    rows = []

    def factory(name, seed):
        if name == "blueprint_live":
            return _TablePolicy(
                trainer,
                seed,
                counts,
                uniform=False,
                on_candidate_decision=on_candidate_decision,
            )
        if name == "blueprint_uniform":
            return _TablePolicy(trainer, seed, counts, uniform=True)
        return make_policy(name, seed)

    def emit(row, _timing):
        # Retain the outcome needed for paired estimates, without hand histories in RAM.
        compact = {
            key: value
            for key, value in row.items()
            if key not in {"events", "reloads", "outcome_sha256"}
        }
        compact["outcome_sha256"] = digest(compact)
        rows.append(compact)

    valid = run_schedule(plan, build_schedule(plan), emit, factory=factory)
    report = summarize(plan, rows)
    if valid != (report["status"] == "valid"):
        raise RuntimeError("Arena completion and report validity disagree")
    coverage = {}
    for street in ("preflop", "flop", "turn", "river"):
        trained = counts[(street, "trained")]
        fallback = counts[(street, "fallback")]
        total = trained + fallback
        coverage[street] = {
            "trained": trained,
            "fallback": fallback,
            "decisions": total,
            "trained_fraction": trained / total if total else None,
            "actions": {
                action: counts[(street, f"action_{action}")]
                for action in ("fold", "check", "call", "raise")
            },
        }
    return {"report": report, "coverage": coverage, "plan": asdict(plan)}
