"""Exact uniform-opponent equity histograms for per-flop preview buckets."""

from itertools import combinations
from pathlib import Path

import numpy as np

from src.blueprint.search import DECK
from src.diagnostics.flop_check_analysis import emd_clusters, equity_quantiles, uniform_river_equities

BLOCKED = np.iinfo(np.uint16).max


def build_equity_features(flop, path, *, bins=20, seed=202610020004, check=lambda: None):
    """Exhaust all final boards; cache their equity for both turn/river orders.

    Histograms describe the distribution of river equity over uniform legal
    runouts, rather than point equity rounded into a bucket. Opponent holdings
    are uniform and card-compatible at each final board. Each street's K is
    shared across that flop's runouts. River quantile boundaries are pooled
    over all final-board/holding contexts; ties remain together.
    """
    flop = tuple(flop)
    if len(flop) != 3 or len(set(flop)) != 3 or bins < 2:
        raise ValueError("Need a valid flop and at least two histogram bins")
    cards = tuple(c for c in DECK if c not in flop)
    holdings = tuple(combinations(cards, 2)); indices = {tuple(sorted(h)): i for i, h in enumerate(holdings)}
    card_indices = {c: i for i, c in enumerate(cards)}
    pairs = tuple(combinations(cards, 2))
    river = np.full((len(pairs), len(holdings)), np.nan, dtype=np.float32)
    flop_hist = np.zeros((len(holdings), bins), dtype=np.uint16)
    turn_hist = np.zeros((len(cards), len(holdings), bins), dtype=np.uint16)
    for row, pair in enumerate(pairs):
        check(); concrete, equity = uniform_river_equities(flop + pair)
        mapped = np.asarray([indices[tuple(sorted(h))] for h in concrete])
        # DECK is sorted by rank/suit, while lexical card-string sorting is
        # different. Root indices use a canonical unordered card-pair key.
        river[row, mapped] = equity
        band = np.minimum((equity * bins).astype(int), bins - 1)
        np.add.at(flop_hist, (mapped, band), 1)
        for card in pair:
            np.add.at(turn_hist[card_indices[card]], (mapped, band), 1)
    turn_totals = turn_hist.sum(axis=2); valid_turn = turn_totals > 0
    flop_probability = flop_hist / flop_hist.sum(axis=1)[:, None]
    turn_probability = turn_hist[valid_turn] / turn_totals[valid_turn][:, None]
    arrays = {"flop": np.asarray(flop), "cards": np.asarray(cards),
              "holdings": np.asarray(holdings), "river_pairs": np.asarray(pairs),
              "flop_histograms": flop_probability.astype(np.float32),
              "turn_histograms": turn_hist, "river_equities": river,
              "flop_equities": np.nanmean(river, axis=0)}
    valid_river = np.isfinite(river)
    for k in (50, 200):
        check(); flop_labels, flop_centers = emd_clusters(flop_probability, k, seed=seed + k)
        check(); turn_labels, turn_centers = emd_clusters(turn_probability, k, seed=seed + k + 1)
        turns = np.full(valid_turn.shape, BLOCKED, dtype=np.uint16); turns[valid_turn] = turn_labels
        rivers = np.full(river.shape, BLOCKED, dtype=np.uint16)
        rivers[valid_river] = equity_quantiles(river[valid_river], k)
        arrays.update({f"flop_k{k}": flop_labels.astype(np.uint16), f"turn_k{k}": turns,
                       f"river_k{k}": rivers, f"flop_centers_k{k}": flop_centers,
                       f"turn_centers_k{k}": turn_centers})
    path = Path(path); temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)
    return {"histogram_bins": bins, "seed": seed, "ks": [50, 200],
            "river_boards_unordered": len(pairs), "root_holdings": len(holdings),
            "clustering": "mean centroids; cumulative-L1 assignment; at most 100 iterations",
            "river_quantiles": "pooled across this flop's runouts; equal equities stay together"}


class EquityBuckets:
    def __init__(self, path, k):
        if k not in (50, 200):
            raise ValueError("Protocol K must be 50 or 200")
        self.arrays = np.load(path); self.k = k
        self.flop_cards = set(self.arrays["flop"])
        # NPZ indexing decompresses an entire member, so cache these small
        # label arrays once rather than decompressing per solver holding.
        self.flop_labels = self.arrays[f"flop_k{k}"]
        self.turn_labels = self.arrays[f"turn_k{k}"]
        self.river_labels = self.arrays[f"river_k{k}"]
        self.hands = {tuple(sorted(h)): i for i, h in enumerate(self.arrays["holdings"])}
        self.cards = {c: i for i, c in enumerate(self.arrays["cards"])}
        self.pairs = {tuple(sorted(h)): i for i, h in enumerate(self.arrays["river_pairs"])}

    def __call__(self, row, holding):
        board = row["board"]; hand = self.hands[tuple(sorted(holding))]
        if set(board[:3]) != self.flop_cards:
            raise ValueError("Equity preview belongs to a different flop")
        if len(board) == 3:
            value = self.flop_labels[hand]
        elif len(board) == 4:
            value = self.turn_labels[self.cards[board[3]], hand]
        elif len(board) == 5:
            value = self.river_labels[self.pairs[tuple(sorted(board[3:]))], hand]
        else:
            raise ValueError("Invalid preview street")
        if value == BLOCKED:
            raise ValueError("Blocked holding has no equity bucket")
        return int(value)
