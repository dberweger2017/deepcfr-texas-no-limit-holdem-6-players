"""Finite river references; hidden worlds are integrated before policy queries."""

from dataclasses import dataclass
from hashlib import sha256
from math import fsum
from random import Random
from time import perf_counter
from types import MappingProxyType

import numpy as np
import torch

from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street
from src.holdem.actions import bet_candidates
from src.holdem.betting import BettingNetwork, betting_loss
from src.holdem.collection import CollectionLimitExceeded
from src.holdem.policy import FrozenProfile
from src.holdem.targets import CandidateTargets

DECK = tuple(r + s for s in "cdhs" for r in "23456789TJQKA")


def check_deadline(deadline):
    if perf_counter() >= deadline:
        raise CollectionLimitExceeded("River diagnostic exceeded its time budget")


class ReferenceProfile(FrozenProfile):
    """Fixed public policies with immutable, own-information baseline lookups."""

    def __init__(self, kind, baselines=None):
        if kind not in ("uniform", "increasing"):
            raise ValueError("Unknown reference policy")
        super().__init__([None] * 6)
        entries = tuple((baselines or {}).items())
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "baselines", MappingProxyType(dict(entries)))
        object.__setattr__(self, "_snapshot", (kind, entries))
        object.__setattr__(
            self, "fingerprint", sha256(repr((kind, entries)).encode()).hexdigest()
        )
        object.__setattr__(self, "_expected_fingerprint", self.fingerprint)

    def assert_unchanged(self):
        if (
            self.kind,
            tuple(self.baselines.items()),
        ) != self._snapshot or self.fingerprint != self._expected_fingerprint:
            raise RuntimeError("Reference policy changed")

    def distribution(self, candidates):
        n = len(candidates.actions)
        if self.kind == "uniform":
            return (1 / n,) * n
        return tuple((i + 1) / (n * (n + 1) / 2) for i in range(n))

    def action_values(self, candidates):
        return self.baselines[candidates.decision.source]


@dataclass(frozen=True)
class Context:
    name: str
    split: str
    worlds: tuple[Hand, ...]
    assignments: tuple[tuple[tuple[str, ...], ...], ...]


def contexts(plan):
    result = []
    for board_index, spec in enumerate(plan["boards"]):
        excluded = set(spec["board"]) | {c for h in spec["hands"] for c in h}
        available = [c for c in DECK if c not in excluded]
        random = Random(plan["root_seed"] + board_index)
        deals = [random.sample(available, 10) for _ in range(2)]
        for hand_index, hero_cards in enumerate(spec["hands"]):
            for facing in (False, True):
                name = f"board-{board_index}/hand-{hand_index}/{'facing' if facing else 'open'}"
                result.append(
                    river_context(
                        name, spec["split"], spec["board"], hero_cards, deals, facing
                    )
                )
    return tuple(result)


def river_context(name, split, board, hero_cards, deals, facing):
    """Build the same legal shallow river for an explicit equally weighted range."""
    hero = 2 if facing else 1
    worlds, assignments = [], []
    for deal in deals:
        hands = [None] * 6
        hands[hero] = tuple(hero_cards)
        for index, seat in enumerate(s for s in range(6) if s != hero):
            hands[seat] = tuple(deal[2 * index : 2 * index + 2])
        order = (1, 2, 3, 4, 5, 0)
        prefix = tuple(hands[s][r] for r in range(2) for s in order) + tuple(board)
        if len(prefix) != 17 or len(set(prefix)) != 17 or not set(prefix) <= set(DECK):
            raise ValueError("Incompatible reference cards")
        deck = prefix + tuple(c for c in DECK if c not in prefix)
        table = Table(
            tuple(f"player-{i}" for i in range(6)),
            (4,) * 6,
            small_blind=1,
            big_blind=2,
            chip_unit="1",
        )
        # The public identifier carries neither the holding nor a hidden-world index.
        node = Hand.from_deck(
            table,
            hand_id=name.split("/hand-")[0].replace("board-", "river-")
            + ("/facing" if facing else "/open"),
            deck=deck,
        )
        while node.observe(node.actor).street != Street.RIVER:
            view = node.observe(node.actor)
            kind = (
                ActionKind.CHECK
                if ActionKind.CHECK in view.legal_actions.kinds
                else ActionKind.CALL
            )
            node = node.apply(Action(kind))
        if facing:
            node = node.apply(Action(ActionKind.RAISE, 2))
        if node.actor != hero or any(p.folded for p in node.observe(hero).players):
            raise ValueError("Invalid multiway river prefix")
        if tuple(node.observe(hero).board) != tuple(board):
            raise ValueError("Deck does not reproduce the declared board")
        worlds.append(node)
        assignments.append(tuple(hands))
    if not worlds or any(w.observe(hero) != worlds[0].observe(hero) for w in worlds):
        raise ValueError("Hidden worlds must share one hero observation")
    return Context(name, split, tuple(worlds), tuple(assignments))


@dataclass(frozen=True)
class Reference:
    target: CandidateTargets
    baselines: dict
    nodes: int
    seconds: float


def enumerate_reference(context, profile, *, max_nodes, deadline):
    """Conditional values integrate the declared joint range and opponent reach."""
    started = perf_counter()
    hero = context.worlds[0].actor
    totals, candidates_by_view = {}, {}
    total_nodes = 0
    for root in context.worlds:
        nodes = 0

        def visit(node, reach):
            nonlocal nodes
            check_deadline(deadline)
            nodes += 1
            if nodes > max_nodes:
                raise CollectionLimitExceeded(
                    "Exact river tree exceeded its node budget"
                )
            if node.finished:
                stacks = node.events[-1].stacks
                if sum(stacks) != sum(node.table.stacks):
                    raise ValueError("Reference settlement is not zero sum")
                return (stacks[hero] - node.table.stacks[hero]) / node.table.big_blind
            view = node.observe(node.actor)
            candidates = bet_candidates(view)
            probabilities = profile.distribution(candidates)
            values = tuple(
                visit(node.apply(a), reach * (p if node.actor != hero else 1))
                for a, p in zip(candidates.actions, probabilities)
            )
            if node.actor == hero:
                if (
                    view in candidates_by_view
                    and candidates_by_view[view].actions != candidates.actions
                ):
                    raise ValueError("An information set has inconsistent actions")
                candidates_by_view[view] = candidates
                weight = reach / len(context.worlds)
                mass, sums = totals.setdefault(view, [0.0, np.zeros(len(values))])
                totals[view][0] = mass + weight
                sums += weight * np.asarray(values)
            return fsum(p * v for p, v in zip(probabilities, values))

        visit(root, 1.0)
        total_nodes += nodes
    baselines = {
        v: tuple(sums / mass) for v, (mass, sums) in totals.items() if mass > 0
    }
    root_view = context.worlds[0].observe(hero)
    candidates = candidates_by_view[root_view]
    values = baselines[root_view]
    probabilities = profile.distribution(candidates)
    center = fsum(p * v for p, v in zip(probabilities, values))
    target = CandidateTargets(
        candidates, probabilities, values, tuple(v - center for v in values)
    )
    return Reference(target, baselines, total_nodes, perf_counter() - started)


def combine_targets(targets):
    """The declared two-profile schedule has iteration weights one and two."""
    if (
        len(targets) != 2
        or targets[0].candidates != targets[1].candidates
        or targets[0].candidates.decision.source
        != targets[1].candidates.decision.source
    ):
        raise ValueError("Combine matching decisions from the two frozen profiles")
    values = tuple(
        (a + 2 * b) / 3 for a, b in zip(targets[0].values_bb, targets[1].values_bb)
    )
    regrets = tuple(
        (a + 2 * b) / 3 for a, b in zip(targets[0].regrets_bb, targets[1].regrets_bb)
    )
    # Stored policy is provenance only: averaged regrets need not equal Q minus one policy's V.
    return CandidateTargets(targets[0].candidates, targets[0].policy, values, regrets)


def probabilities(regrets):
    r = np.asarray(regrets, dtype=float)
    positive = np.maximum(r, 0)
    if positive.max() > 0:
        return positive / positive.sum()
    return np.eye(len(r))[r.argmax()]


def prediction_metrics(regrets, targets):
    differences = np.concatenate(
        [np.asarray(r) - t.regrets_bb for r, t in zip(regrets, targets)]
    )
    truth = np.concatenate([t.regrets_bb for t in targets])
    rmse, scale = (
        float(np.sqrt(np.mean(differences**2))),
        float(np.sqrt(np.mean(truth**2))),
    )
    tv, costs, decisions = [], [], []
    for r, target in zip(regrets, targets):
        p = probabilities(r)
        tv.append(float(np.abs(p - probabilities(target.regrets_bb)).sum() / 2))
        costs.append(float(max(target.values_bb) - p @ target.values_bb))
        decisions.append(
            {
                "context": target.candidates.decision.source.hand_id,
                "hole_cards": target.candidates.decision.source.hole_cards,
                "predicted_regrets_bb": list(r),
                "reference_regrets_bb": target.regrets_bb,
                "reference_values_bb": target.values_bb,
                "probabilities": p.tolist(),
                "policy_tv": tv[-1],
                "decision_cost_bb": costs[-1],
            }
        )
    return {
        "regret_rmse_bb": rmse,
        "target_rms_bb": scale,
        "relative_rmse": rmse / scale if scale else None,
        "mean_policy_tv": float(np.mean(tv)),
        "mean_decision_cost_bb": float(np.mean(costs)),
        "decisions": decisions,
    }


def fit_reference(train, validation, fit_targets, *, seed, plan, deadline):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = BettingNetwork(plan["width"])
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])
    batch = [t.candidates for t in fit_targets]

    def metrics():
        with torch.no_grad():
            return {
                name: prediction_metrics(
                    [
                        s.regrets.tolist()
                        for s in model([t.candidates for t in targets])
                    ],
                    targets,
                )
                for name, targets in (("train", train), ("validation", validation))
            }

    initial = metrics()
    started = perf_counter()
    for _ in range(plan["fit_steps"]):
        check_deadline(deadline)
        optimizer.zero_grad()
        loss = betting_loss(model(batch), fit_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), plan["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite diagnostic model")
    return model, {
        "initial": initial,
        "final": metrics(),
        "seconds": perf_counter() - started,
    }
