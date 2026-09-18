"""Observation-safe shallow multi-street reliable targets.

The river diagnostic has a deliberately complete-board contract.  This module
keeps that contract intact and builds a separate context for a visible flop or
turn.  Each context contains complete, independently sampled worlds; the
trainer only receives the common observation at the requested street.
"""

from dataclasses import dataclass
from hashlib import sha256
from itertools import permutations
import json
from math import fsum
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np

from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street
from src.holdem.actions import bet_candidates
from src.holdem.encoding import _canonical_cards
from src.holdem.representation_reference import range_support
from src.holdem.river_reference import DECK, ReferenceProfile, check_deadline
from src.holdem.targets import CandidateTargets


STREET_NAMES = {"flop": Street.FLOP, "turn": Street.TURN, "river": Street.RIVER}


@dataclass(frozen=True)
class MultiStreetContext:
    name: str
    split: str
    street: str
    visible_board: tuple[str, ...]
    worlds: tuple[Hand, ...]
    assignments: tuple[tuple[tuple[str, ...], ...], ...]
    world_seeds: tuple[int, ...]
    hero_seat: int
    facing: bool


def flop_key(cards):
    """Canonical key for a flop ancestor and its public reveal order."""

    cards = tuple(cards)
    if len(cards) < 3:
        raise ValueError("A board ancestor needs at least a flop")
    return _canonical_cards((cards[:3],))


def _ordered_deck(hero, board, deal, hero_seat):
    # The engine deals two rounds clockwise from the seat left of the button,
    # then burns/deals the board.  This is the order used by river_reference.
    hands = [None] * 6
    hands[hero_seat] = tuple(hero)
    order = (1, 2, 3, 4, 5, 0)
    for index, seat in enumerate(s for s in order if s != hero_seat):
        hands[seat] = tuple(deal[2 * index : 2 * index + 2])
    prefix = tuple(hands[s][r] for r in range(2) for s in order) + tuple(board)
    if len(prefix) != 17 or len(set(prefix)) != 17:
        raise ValueError("A sampled world contains duplicate cards")
    return prefix + tuple(c for c in DECK if c not in prefix), tuple(hands)


def compatible_visible_deals(support, board, holding):
    """Condition the declared range on cards visible at this street."""

    visible = tuple(board) + tuple(holding)
    if (
        len(holding) != 2
        or len(set(visible)) != len(visible)
        or not set(visible) <= set(DECK)
    ):
        raise ValueError("Expected distinct visible cards from the standard deck")
    result = tuple(deal for deal in support if not set(deal).intersection(visible))
    if not result:
        raise ValueError("Visible cards have zero probability under the declared range")
    return result


def _advance_to_hero(hand, hero, target, facing):
    """Run the public check/call prefix until hero owns the target decision."""

    while not hand.finished:
        view = hand.observe(hand.actor)
        if view.street == target and hand.actor == hero:
            return hand
        if ("preflop", "flop", "turn", "river").index(view.street.value) > (
            "preflop", "flop", "turn", "river"
        ).index(target.value):
            raise ValueError("Prefix passed the requested decision street")
        if view.street == target and facing:
            # Put one public raise immediately before hero's decision.  The
            # action is fixed by the public history and independent of worlds.
            action = Action(ActionKind.RAISE, view.legal_actions.min_raise_to)
            hand = hand.apply(action)
            facing = False
            continue
        kind = (
            ActionKind.CHECK
            if ActionKind.CHECK in view.legal_actions.kinds
            else ActionKind.CALL
        )
        hand = hand.apply(Action(kind))
    raise ValueError("Prefix ended before the requested hero decision")


def _world(hero, board, deal, name, target, facing, hero_seat):
    deck, hands = _ordered_deck(hero, board, deal, hero_seat)
    table = Table(
        tuple(f"player-{i}" for i in range(6)),
        (4,) * 6,
        small_blind=1,
        big_blind=2,
        chip_unit="1",
    )
    hand = Hand.from_deck(table, hand_id=name, deck=deck)
    hand = _advance_to_hero(hand, hero_seat, target, facing)
    if tuple(hand.observe(hero_seat).board) != tuple(board[: len(hand.observe(hero_seat).board)]):
        raise ValueError("Engine board reveal disagrees with the declared prefix")
    return hand, hands


def build_context(
    *,
    name,
    split,
    street,
    board,
    holding,
    support,
    samples=4,
    deals_per_sample=2,
    seed=0,
    facing=False,
):
    """Build complete worlds that share one hero-visible prefix.

    A future board card is sampled independently for each world.  Opponent
    holdings are conditioned on the visible board before the engine is
    created. Future cards are then drawn without replacement after all hole
    cards, so an undealt future card never changes an earlier-street prior.
    """

    if street not in STREET_NAMES:
        raise ValueError("street must be flop, turn, or river")
    target = STREET_NAMES[street]
    hero_seat = 2 if facing else 1
    board = tuple(board)
    expected = {"flop": 3, "turn": 4, "river": 5}[street]
    if len(board) != expected or len(set(board)) != expected:
        raise ValueError(f"{street} contexts need {expected} visible board cards")
    hero = tuple(holding)
    if len(hero) != 2 or len(set(hero)) != 2 or set(hero) & set(board):
        raise ValueError("Hero holding and visible board must be disjoint")
    rng = Random(seed)
    worlds, assignments, world_seeds = [], [], []
    used = 0
    try:
        deals = compatible_visible_deals(support, board, hero)
    except ValueError as error:
        raise ValueError("No compatible sampled hidden worlds") from error
    if len(deals) < deals_per_sample:
        raise ValueError("Range support has too few compatible worlds")
    for draw_index in range(samples * deals_per_sample):
        # Sampling with replacement preserves the declared multiplicity of
        # the joint range and gives independent world draws for the SE.
        deal = rng.choice(deals)
        used_cards = set(hero) | set(board) | set(deal)
        available = [c for c in DECK if c not in used_cards]
        future = tuple(rng.sample(available, 5 - expected))
        full_board = board + future
        hand, hands = _world(hero, full_board, deal, name, target, facing, hero_seat)
        worlds.append(hand)
        assignments.append(tuple(hands))
        world_seeds.append(draw_index)
        used += 1
    if not worlds:
        raise ValueError("No compatible sampled hidden worlds")
    first = worlds[0].observe(hero_seat)
    if any(world.observe(hero_seat) != first for world in worlds[1:]):
        raise ValueError("Sampled worlds do not share one public observation")
    return MultiStreetContext(
        name,
        split,
        street,
        board,
        tuple(worlds),
        tuple(assignments),
        tuple(world_seeds),
        hero_seat,
        facing,
    )


@dataclass(frozen=True)
class MultiStreetReference:
    target: CandidateTargets
    world_action_values_bb: tuple[tuple[float, ...], ...]
    action_standard_error_bb: tuple[float, ...]
    uncertainty_status: str
    nodes: int
    worlds: int
    seconds: float


def enumerate_reference(context, profile, *, max_nodes, deadline):
    """Enumerate each complete sampled world and report paired uncertainty."""

    started = perf_counter()
    hero = context.worlds[0].actor
    totals, candidates_by_view = {}, {}
    per_world = []
    total_nodes = 0
    for root in context.worlds:
        nodes = 0
        root_values = None

        def visit(node, reach):
            nonlocal nodes, root_values
            check_deadline(deadline)
            nodes += 1
            if nodes > max_nodes:
                raise RuntimeError("Multi-street reference exceeded its node budget")
            if node.finished:
                stacks = node.events[-1].stacks
                return (stacks[hero] - node.table.stacks[hero]) / node.table.big_blind
            view = node.observe(node.actor)
            candidates = bet_candidates(view)
            probs = profile.distribution(candidates)
            values = tuple(
                visit(node.apply(action), reach * (p if node.actor != hero else 1.0))
                for action, p in zip(candidates.actions, probs, strict=True)
            )
            if node is root and node.actor == hero:
                root_values = values
            if node.actor == hero:
                if view in candidates_by_view and candidates_by_view[view].actions != candidates.actions:
                    raise ValueError("One information set has inconsistent actions")
                candidates_by_view[view] = candidates
                weight = reach / len(context.worlds)
                mass, sums = totals.setdefault(view, [0.0, np.zeros(len(values))])
                totals[view][0] = mass + weight
                sums += weight * np.asarray(values)
            return fsum(p * value for p, value in zip(probs, values))

        visit(root, 1.0)
        if root_values is None:
            raise ValueError("Reference root was not a hero decision")
        per_world.append(tuple(root_values))
        total_nodes += nodes
    root_view = context.worlds[0].observe(hero)
    candidates = candidates_by_view[root_view]
    values = tuple(totals[root_view][1] / totals[root_view][0])
    probs = profile.distribution(candidates)
    center = fsum(p * v for p, v in zip(probs, values))
    target = CandidateTargets(candidates, probs, values, tuple(v - center for v in values))
    arr = np.asarray(per_world, dtype=float)
    # Root action values are paired because every action uses the same sampled
    # hidden world.  This is the uncertainty used for the cost comparison.
    se = (
        tuple(float(x) for x in arr.std(axis=0, ddof=1) / np.sqrt(len(arr)))
        if len(arr) > 1
        else (0.0,) * len(values)
    )
    return MultiStreetReference(
        target,
        tuple(tuple(float(v) for v in row) for row in per_world),
        se,
        "estimated" if len(arr) > 1 else "insufficient_worlds",
        total_nodes,
        len(context.worlds),
        perf_counter() - started,
    )


def _iter_specs(plan, context_filter=None):
    """Yield contexts while enforcing whole-flop-family splits."""

    support = range_support(plan["range_templates"])
    forbidden = set()
    for path in plan.get("forbidden_flop_plans", ()):
        source = json.loads(Path(path).read_text())
        boards = source.get("boards", ())
        # Nested board diagnostics declare a generator; its complete material-
        # ized set is needed to exclude every previously observed ancestor.
        if "board_counts" in source and "fresh_board_seed" in source:
            from src.holdem.card_diversity import expanded_plan

            boards = expanded_plan(source)["boards"]
        # Prior diagnostics use boards, contexts, or materialized campaign
        # families.  Treat every representation as an ancestor exclusion.
        forbidden.update(
            flop_key(tuple(row.get("cards", row.get("board", row.get("flop")))))
            for row in boards
        )
        forbidden.update(
            flop_key(tuple(row["board"]))
            for row in source.get("contexts", ())
            if "board" in row
        )
        forbidden.update(
            flop_key(tuple(row["flop"]))
            for row in source.get("families", ())
            if "flop" in row
        )
    groups = {}
    for index, row in enumerate(plan["contexts"]):
        if context_filter is not None and not context_filter(row, index):
            continue
        board = tuple(row["board"])
        key = flop_key(board)
        if key in forbidden:
            raise ValueError("A held-out or training flop repeats a prior observed family")
        previous = groups.setdefault(key, row["split"])
        if previous != row["split"]:
            raise ValueError("A flop ancestor must stay in one split")
        if "stream_namespace" in plan:
            seed_material = (
                f"{plan['stream_namespace']}|{plan['context_seed']}|{row['street']}|"
                f"{index}|{bool(row.get('facing', False))}"
            ).encode()
            context_seed = int.from_bytes(sha256(seed_material).digest(), "big")
        else:
            # Preserve the pilot's historical seed sequence exactly.
            context_seed = int(plan["context_seed"]) + index
        context = build_context(
            name=f"{row['street']}-{index}",
            split=row["split"],
            street=row["street"],
            board=board,
            holding=tuple(row["holding"]),
            support=support,
            samples=int(
                row.get(
                    "world_samples",
                    plan.get("world_samples_by_stratum", {}).get(
                        f"{row['street']}:{'facing' if row.get('facing', False) else 'open'}",
                        plan.get("world_samples", 4),
                    ),
                )
            ),
            deals_per_sample=int(row.get("deals_per_sample", plan.get("deals_per_sample", 2))),
            seed=context_seed,
            facing=bool(row.get("facing", False)),
        )
        yield context, key


def iter_specs(plan, context_filter=None):
    """Stream context construction without retaining all hidden worlds."""

    return _iter_specs(plan, context_filter=context_filter)


def split_specs(plan, context_filter=None):
    """Materialize contexts for legacy callers and small diagnostics."""

    return tuple(_iter_specs(plan, context_filter=context_filter))
