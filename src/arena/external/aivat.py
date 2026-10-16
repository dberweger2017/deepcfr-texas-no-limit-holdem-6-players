"""AIVAT (Burch et al., 2018) for verified Slumbot hands: card luck and own-action luck removed.

Value function: check-down `pot * equity - own contribution`, exact equity, raises valued as
called. Method and evidence: docs/slumbot-aivat.md.
"""

from fractions import Fraction
from functools import lru_cache
import itertools
import math

import numpy as np

from src.arena.external.slumbot import CONTRACT, TOKEN

STACK = CONTRACT.stack
RANKS = '23456789TJQKA'
CARD_INDEX = {r + s: 4 * i + j for i, r in enumerate(RANKS) for j, s in enumerate('cdhs')}
STREETS = ('deal', 'flop', 'turn', 'river')
DEALT = (0, 3, 4, 5)
DEAL_POT = 150
# Kickers per category, as `src.game.showdown.hand_value` orders them.
KICKERS = {8: 1, 7: 2, 6: 2, 5: 5, 4: 1, 3: 3, 2: 3, 1: 4, 0: 5}


def _straight_high(mask: int) -> int:
    for high in range(12, 3, -1):
        if mask >> (high - 4) & 31 == 31:
            return high + 2
    return 5 if mask & 0x100F == 0x100F else 0


def _top(mask: int, k: int) -> int:
    """The top k ranks of a 13-bit rank mask, as 2..14 nibbles from bit 16 down."""
    ranks = [r + 2 for r in range(12, -1, -1) if mask >> r & 1][:k]
    return sum(rank << (16 - 4 * i) for i, rank in enumerate(ranks))


_MASKS = range(1 << 13)
POPCOUNT = np.array([m.bit_count() for m in _MASKS], dtype=np.uint8)
STRAIGHT = np.array([_straight_high(m) << 16 for m in _MASKS], dtype=np.uint32)
TOP = {k: np.array([_top(m, k) for m in _MASKS], dtype=np.uint32) for k in (1, 2, 3, 5)}
HIGH_BIT = np.array([1 << (m.bit_length() - 1) if m else 0 for m in _MASKS], dtype=np.uint16)
# Adding 8 - n to a rank's count nibble sets that nibble's top bit exactly when the count is at least n.
_AT_LEAST = {n: sum((8 - n) << (4 * r) for r in range(13)) for n in (2, 3, 4)}
_FLAGS = sum(8 << (4 * r) for r in range(13))


def _rank_masks(counts: np.ndarray) -> dict[int, np.ndarray]:
    """For n = 2, 3, 4, the 13-bit mask of ranks held at least n times."""
    masks = {}
    for n, add in _AT_LEAST.items():
        flags = (counts + np.uint64(add)) & np.uint64(_FLAGS)
        mask = np.zeros(len(counts), dtype=np.uint16)
        for r in range(13):
            # Rank r's flag sits at bit 4r + 3; shifting by 3r + 3 moves it to bit r.
            mask |= ((flags >> np.uint64(3 * r + 3)) & np.uint64(1 << r)).astype(np.uint16)
        masks[n] = mask
    return masks


def seven_card_values(suits: list[np.ndarray], counts: np.ndarray) -> np.ndarray:
    """Seven-card values ordered like `hand_value`: category at bit 20, kickers as nibbles below."""
    ranks = suits[0] | suits[1] | suits[2] | suits[3]
    flush = np.zeros(len(counts), dtype=np.uint16)
    for mask in suits:
        flush = np.where(POPCOUNT[mask] >= 5, mask, flush)
    multiple = _rank_masks(counts)
    pairs, trips, quads = multiple[2], multiple[3], multiple[4]
    top_pair, top_trip = HIGH_BIT[pairs], HIGH_BIT[trips]
    second_pair = HIGH_BIT[pairs & ~top_pair]
    full_pair = HIGH_BIT[pairs & ~top_trip]
    # Lowest category first; each stronger one overwrites the rows it applies to.
    value = TOP[5][ranks]
    value = np.where(pairs != 0, (1 << 20) | TOP[1][pairs] | TOP[3][ranks & ~top_pair] >> 4, value)
    value = np.where(second_pair != 0, (2 << 20) | TOP[1][pairs] | TOP[1][second_pair] >> 4
                     | TOP[1][ranks & ~top_pair & ~second_pair] >> 8, value)
    value = np.where(trips != 0, (3 << 20) | TOP[1][trips] | TOP[2][ranks & ~top_trip] >> 4, value)
    value = np.where(STRAIGHT[ranks] != 0, (4 << 20) | STRAIGHT[ranks], value)
    value = np.where(flush != 0, (5 << 20) | TOP[5][flush], value)
    value = np.where((trips != 0) & (full_pair != 0), (6 << 20) | TOP[1][trips] | TOP[1][full_pair] >> 4, value)
    value = np.where(quads != 0, (7 << 20) | TOP[1][quads] | TOP[1][ranks & ~HIGH_BIT[quads]] >> 4, value)
    return np.where(STRAIGHT[flush] != 0, (8 << 20) | STRAIGHT[flush], value).astype(np.uint32)


def decode(value: int) -> tuple[int, ...]:
    """A packed value as the `hand_value` tuple it stands for."""
    category = value >> 20
    return (category, *(value >> (16 - 4 * i) & 15 for i in range(KICKERS[category])))


@lru_cache(maxsize=4)
def _combinations(n: int, k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(n), k)), dtype=np.uint8).reshape(math.comb(n, k), k)


def _encode(cards) -> tuple[list[np.ndarray], np.ndarray]:
    """Suit masks and packed rank counts of fixed cards (scalars) or card columns (arrays)."""
    cards = np.asarray(cards, dtype=np.uint8)
    rank, suit = (cards >> 2).astype(np.uint64), cards & 3
    bit = (np.uint64(1) << rank).astype(np.uint16)
    suits = [np.bitwise_or.reduce(np.where(suit == s, bit, np.uint16(0)), axis=-1) for s in range(4)]
    counts = np.sum(np.uint64(1) << (np.uint64(4) * rank), axis=-1, dtype=np.uint64)
    return suits, counts


def equity_share(hole, bot, board) -> Fraction:
    """The holder of `hole`'s exact pot share against `bot` over every completion of `board`."""
    known = [CARD_INDEX[c] for c in (*hole, *bot, *board)]
    if len(set(known)) != len(known) or len(hole) != 2 or len(bot) != 2 or len(board) > 5:
        raise ValueError('Equity needs two distinct two-card hands and at most five board cards')
    rest = np.array([c for c in range(52) if c not in known], dtype=np.uint8)
    runouts = rest[_combinations(len(rest), 5 - len(board))]
    boards = np.concatenate([np.broadcast_to(np.array([CARD_INDEX[c] for c in board], dtype=np.uint8),
                                             (len(runouts), len(board))), runouts], axis=1)
    board_suits, board_counts = _encode(boards)
    values = []
    for cards in (hole, bot):
        hole_suits, hole_counts = _encode([CARD_INDEX[c] for c in cards])
        values.append(seven_card_values([b | h for b, h in zip(board_suits, hole_suits)], board_counts + hole_counts))
    mine, theirs = values
    share = 2 * int(np.count_nonzero(mine > theirs)) + int(np.count_nonzero(mine == theirs))
    return Fraction(share, 2 * len(runouts))


def street_contributions(action: str) -> tuple[list[int], list[int]]:
    """Matched contribution at each street close, and final contributions. Seat 0 is the big blind."""
    paid, street_bet, actor, closing = [100, 50], [100, 50], 1, []
    for token in TOKEN.findall(action):
        if token == '/':
            if paid[0] != paid[1]:
                raise ValueError('A street closed with unmatched contributions')
            closing.append(paid[0])
            street_bet, actor = [0, 0], 0
            continue
        if token == 'c':
            paid[actor] += street_bet[1 - actor] - street_bet[actor]
            street_bet[actor] = street_bet[1 - actor]
        elif token not in ('k', 'f'):
            paid[actor] += int(token[1:]) - street_bet[actor]
            street_bet[actor] = int(token[1:])
        actor = 1 - actor
    return closing, paid


def _dealt_streets(terminal: dict) -> list[int]:
    """Per-player contribution when each dealt street's cards came."""
    action = terminal['action']
    closing, paid = street_contributions(action)
    if paid == [STACK, STACK] and not action.endswith('f'):
        # Both all-in: every remaining street is dealt into the same pot.
        closing = closing + [STACK] * (3 - len(closing))
    if len(terminal['board']) != DEALT[len(closing)]:
        raise ValueError('The board does not match the streets that were dealt')
    return closing


def street_equities(terminal: dict) -> list[Fraction]:
    """The client's exact equity on each street the hand reached, from the deal on."""
    hole, bot, board = terminal['hole_cards'], terminal['bot_hole_cards'], terminal['board']
    return [equity_share(hole, bot, board[:DEALT[s]]) for s in range(len(_dealt_streets(terminal)) + 1)]


def chance_corrections(terminal: dict, equities: list[Fraction] | None = None) -> list[dict]:
    """Per chance event: pot, equity before and after, correction `pot * (after - before)`.
    The deal's equity before is exactly 1/2 by symmetry."""
    closing = _dealt_streets(terminal)
    equities = street_equities(terminal) if equities is None else equities
    pots = [DEAL_POT] + [2 * c for c in closing]
    before = [Fraction(1, 2)] + equities[:-1]
    return [{'event': STREETS[s], 'pot': pots[s], 'equity_before': before[s], 'equity_after': equities[s],
             'correction': pots[s] * (equities[s] - before[s])} for s in range(len(pots))]


def action_values(public: dict, equity: Fraction, actions) -> dict[str, Fraction]:
    """Check-down value of each encoded client action at one decision."""
    closing, paid = street_contributions(public['action'])
    street = DEALT.index(len(public['board']))
    me = public['client_pos']
    if street and len(closing) < street:
        # Slumbot may omit the separator that ends the last closed street.
        if paid[0] != paid[1]:
            raise ValueError('A street closed with unmatched contributions')
        closing = closing + [paid[0]]
    before_street = closing[street - 1] if street else 0
    values = {}
    for action in actions:
        if action == 'f':
            values[action] = Fraction(-paid[me])
            continue
        after = paid[me] if action == 'k' else paid[1 - me] if action == 'c' else before_street + int(action[1:])
        values[action] = (2 * equity - 1) * max(after, paid[1 - me])
    return values


def action_corrections(terminal: dict, decisions, equities: list[Fraction] | None = None) -> list[dict]:
    """Per decision: chosen action's value minus the policy-weighted average.

    `decisions`: (journal record, {encoded action: probability}) from the model that played.
    """
    equities = street_equities(terminal) if equities is None else equities
    rows = []
    for record, policy in decisions:
        public = record['public']
        if list(public['hole_cards']) != list(terminal['hole_cards']) or public['client_pos'] != terminal['client_pos']:
            raise ValueError('A decision belongs to another hand')
        # AveragePolicy admits rows up to 1e-8 off; corrections still use the probabilities as stored.
        if abs(math.fsum(policy.values()) - 1) > 1e-8:
            raise ValueError('Policy probabilities do not sum to 1')
        if not policy.get(record['selected'], 0) > 0:
            raise ValueError('The journal action has no probability under the model')
        street = DEALT.index(len(public['board']))
        values = action_values(public, equities[street], policy)
        average = sum(Fraction(p) * values[a] for a, p in policy.items())
        rows.append({'street': street, 'selected': record['selected'],
                     'correction': values[record['selected']] - average})
    return rows


def hand_corrections(terminal: dict, decisions=()) -> list[float]:
    """Corrections in chips: [deal, flop, turn, river, then own actions per street]."""
    equities = street_equities(terminal)
    row = [0.0] * 8
    for i, event in enumerate(chance_corrections(terminal, equities)):
        row[i] = float(event['correction'])
    for decision in action_corrections(terminal, decisions, equities):
        row[4 + decision['street']] += float(decision['correction'])
    return row


def aivat(terminal: dict) -> float:
    """Chip result minus chance corrections, unweighted."""
    return float(terminal['winnings'] - sum(c['correction'] for c in chance_corrections(terminal)))


def cross_fitted(results, corrections) -> tuple[list[float], list[list[float]]]:
    """Results minus corrections scaled by least-squares weights fitted on the other half
    (alternate hands). Returns estimates and each half's weights."""
    y = np.asarray(results, dtype=float)
    x = np.asarray(corrections, dtype=float)
    halves = [np.arange(len(y)) % 2 == h for h in (0, 1)]
    estimates, weights = np.empty(len(y)), []
    for half, other in (halves, halves[::-1]):
        design = np.column_stack([np.ones(int(other.sum())), x[other]])
        beta = np.linalg.lstsq(design, y[other], rcond=None)[0][1:]
        estimates[half] = y[half] - x[half] @ beta
        weights.append(beta.tolist())
    return estimates.tolist(), weights
