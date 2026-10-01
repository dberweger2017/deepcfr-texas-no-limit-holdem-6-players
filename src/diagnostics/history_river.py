"""Common-range HU20 restricted river diagnostics; no solving during play.

Project the saved uncapped policy onto the existing restricted game by action
identity and renormalize retained mass. Report zero retained mass explicitly.
Neither hidden representative cards nor policy likelihoods define the range law.
"""
from collections import Counter
from itertools import combinations
from math import fsum

import numpy as np

from src.blueprint.river_game import RiverGame
from src.blueprint.search import DECK
from src.game.observation import replay

LAW = 'dr2x2-hu20-uniform-board-compatible-product-v1'
PROJECTION = 'retain-equal-concrete-actions-renormalize-uniform-if-zero-v1'


class CommonRiverGame(RiverGame):
    table_players = 2
    free_fold = False

    def __init__(self, root_history, ranges, **kwargs):
        # The base compiler/settlement/BR machinery is identical to #142.
        start = root_history[0]
        if (start.stacks != (2000, 2000) or start.small_blind != 50
                or start.big_blind != 100 or start.chip_unit != '0.01'):
            raise ValueError('Common river requires the frozen HU20 game')
        super().__init__(root_history, ranges, law_label=LAW, **kwargs)


def common_ranges(root_history):
    board = replay(root_history, 0, ()).board
    pairs = tuple(tuple(sorted(p)) for p in combinations(
        (card for card in DECK if card not in board), 2))
    return {seat: tuple((pair, 1.0) for pair in pairs) for seat in (0, 1)}


def restricted_distribution(source, view, restricted_menu):
    menu, probabilities, trained = source.distribution(view)
    if not restricted_menu or len(menu) != len(probabilities):
        raise ValueError('Saved policy/menu differ')
    if (not np.isfinite(probabilities).all() or any(p < 0 for p in probabilities)
            or abs(fsum(probabilities) - 1) > 1e-8):
        raise ValueError('Invalid saved policy probability')
    masses = {choice.action: p for choice, p in zip(menu, probabilities, strict=True)}
    if any(choice.action not in masses for choice in restricted_menu):
        raise ValueError('Restricted game adds an action absent from saved policy')
    retained = tuple(masses[choice.action] for choice in restricted_menu)
    total = fsum(retained)
    if total < 0 or not np.isfinite(retained).all():
        raise ValueError('Invalid projected policy mass')
    probabilities = (tuple(p / total for p in retained) if total else
                     (1 / len(retained),) * len(retained))
    return probabilities, {'trained': bool(trained), 'zero_retained_mass': not total,
                           'removed_mass': max(0.0, 1 - total)}


def policy_profile(game, source, guard=lambda: None):
    profile = {}
    counts = Counter()
    for node in game.nodes:
        if node.actor is None:
            continue
        rows = []
        seat_index = game.seats.index(node.actor)
        for pair in game.holdings[seat_index]:
            guard()
            view = replay(node.history, node.actor, pair)
            row, telemetry = restricted_distribution(source, view, node.menu)
            rows.append(row)
            counts['queries'] += 1
            counts['trained' if telemetry['trained'] else 'missing'] += 1
            counts['zero_retained_mass'] += telemetry['zero_retained_mass']
            counts['removed_mass_positive'] += telemetry['removed_mass'] > 1e-12
        profile[node.id] = np.asarray(rows, dtype=np.float64)
    return profile, dict(counts)
