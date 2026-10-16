"""AIVAT: exact equities, zero-mean corrections, cross-fitting and the analysis output."""

from fractions import Fraction
import random

import numpy as np
import pytest

from src.arena.external.aivat import (CARD_INDEX, DEALT, STACK, _encode, action_corrections, action_values, aivat,
                                      chance_corrections, cross_fitted, decode, equity_share, hand_corrections,
                                      seven_card_values)
from src.arena.external.analysis import DECK, allin_equity
from src.arena.external.slumbot import settle
from src.game.showdown import hand_value


def fast_value(cards):
    suits, counts = _encode([CARD_INDEX[c] for c in cards])
    return decode(int(seven_card_values([np.array([s]) for s in suits], np.array([counts]))[0]))


def hand(action, hole, bot, board, seat=0):
    return {'action': action, 'client_pos': seat, 'hole_cards': hole, 'bot_hole_cards': bot, 'board': board,
            'winnings': settle(action, seat, hole, bot, board)}


def test_fast_values_equal_the_showdown_evaluator():
    edge_cases = [
        ['Ah', '2c', '3d', '4s', '5h', 'Kc', 'Kd'],    # wheel over a pair
        ['Ah', '2h', '3h', '4h', '5h', '6h', 'Kh'],    # seven hearts: six-high straight flush
        ['Ac', 'Ad', 'Ah', 'Kc', 'Kd', 'Kh', '2s'],    # two trips make aces full of kings
        ['Ac', 'Ad', 'Kc', 'Kd', 'Qc', 'Qd', '2s'],    # three pairs: the queen is the kicker
        ['9c', '9d', '9h', '9s', 'Kc', 'Kd', 'Kh'],    # quads beside trips: king kicker
        ['Tc', 'Jc', 'Qc', 'Kc', 'Ac', '9c', '8c'],    # royal flush inside a seven-card flush
        ['2c', '3d', '4h', '5s', '7c', '8d', '9h'],    # no straight across the gap
    ]
    rng = random.Random(2026101601)
    hands = edge_cases + [rng.sample(DECK, 7) for _ in range(20000)]
    assert {hand_value(tuple(h))[0] for h in hands} == set(range(9))
    for cards in hands:
        assert fast_value(cards) == hand_value(tuple(cards)), cards


def test_equity_matches_the_exact_reference_after_the_flop():
    rng = random.Random(2026101602)
    for size in (3, 3, 4, 4, 5):
        cards = rng.sample(DECK, 4 + size)
        hole, bot, board = cards[:2], cards[2:4], cards[4:]
        # Both divide the same exact count, and division rounds correctly.
        assert float(equity_share(hole, bot, board)) == allin_equity(hole, bot, board)
    # The existing exact cases of the all-in adjustment.
    assert equity_share(['Ah', 'Ad'], ['Kh', 'Kd'], ['2c', '7s', '9d', 'Jc']) == Fraction(42, 44)
    assert equity_share(['2c', '3d'], ['4h', '5c'], ['As', 'Ks', 'Qs', 'Js', 'Ts']) == Fraction(1, 2)


def test_preflop_equity_is_exact_and_antisymmetric():
    # Matches allin_equity's exact count (minutes in pure Python, so pinned).
    assert equity_share(['Ah', 'Ad'], ['Kh', 'Kd'], []) == Fraction(707495, 856152)
    rng = random.Random(2026101603)
    cards = rng.sample(DECK, 4)
    assert equity_share(cards[:2], cards[2:], []) + equity_share(cards[2:], cards[:2], []) == 1


def test_equity_before_each_card_is_the_exact_average_after_it():
    """The property that makes every correction average zero: equity is a martingale."""
    rng = random.Random(2026101604)
    for size in (3, 4):
        for _ in range(3):
            cards = rng.sample(DECK, 4 + size)
            hole, bot, board = cards[:2], cards[2:4], cards[4:]
            rest = [c for c in DECK if c not in cards]
            after = [equity_share(hole, bot, board + [card]) for card in rest]
            assert sum(after) / len(after) == equity_share(hole, bot, board)


def test_corrections_follow_the_dealt_streets_and_telescope():
    aces, kings = ['Ah', 'Ad'], ['Kh', 'Kd']
    board = ['2c', '7s', '9d', 'Kc', '3h']
    pre = equity_share(aces, kings, [])

    # Preflop fold: only the deal, into the blinds.
    [deal] = chance_corrections(hand('b300f', aces, kings, []))
    assert (deal['event'], deal['pot'], deal['correction']) == ('deal', 150, 150 * (pre - Fraction(1, 2)))

    # A flop fold: the deal, then the flop into the 600-chip pot.
    events = chance_corrections(hand('b300c/b400f', aces, kings, board[:3]))
    assert [(e['event'], e['pot']) for e in events] == [('deal', 150), ('flop', 600)]
    assert events[1]['equity_before'] == pre and events[1]['equity_after'] == equity_share(aces, kings, board[:3])

    # Preflop all-in: street corrections telescope to the all-in adjustment, plus the deal's.
    allin = hand('b300b20000c///', aces, kings, board)
    assert allin['winnings'] == -20000
    assert [e['pot'] for e in chance_corrections(allin)] == [150, 40000, 40000, 40000]
    adjusted = pre * 40000 - 20000
    assert aivat(allin) == float(adjusted - 150 * (pre - Fraction(1, 2)))
    # The same hand written without its trailing street separators.
    assert aivat(hand('b300b20000c', aces, kings, board)) == aivat(allin)

    # The other seat sees the opposite corrections.
    mirrored = hand('b300b20000c///', aces, kings, board, seat=1)
    flipped = dict(mirrored, hole_cards=kings, bot_hole_cards=aces, winnings=-mirrored['winnings'])
    assert aivat(mirrored) + aivat(flipped) == 0

    with pytest.raises(ValueError, match='board does not match'):
        chance_corrections(dict(hand('b300c/b400f', aces, kings, board[:3]), board=board[:4]))


def test_action_values_follow_the_contributions_after_each_action():
    three_quarters = Fraction(3, 4)
    # Big blind facing a raise to 300.
    facing = {'action': 'b300', 'board': [], 'client_pos': 0}
    assert action_values(facing, three_quarters, ['f', 'c', 'b900']) == {'f': -100, 'c': 150, 'b900': 450}
    # Flop raises are street-local; same values with the preflop separator omitted.
    flop = {'action': 'b300c/', 'board': ['2c', '7s', '9d'], 'client_pos': 0}
    values = action_values(flop, three_quarters, ['k', 'b200'])
    assert values == {'k': 150, 'b200': 250}
    assert action_values(dict(flop, action='b300c'), three_quarters, ['k', 'b200']) == values


def test_action_corrections_average_zero_under_the_policy_and_refuse_impossible_actions():
    hole, bot, board = ['Ah', 'Kd'], ['Qs', 'Qc'], ['2c', '7s', '9d', 'Kc', '3h']
    terminal = hand('b300c/kb200c/kk/kk', hole, bot, board)
    public = {'action': 'b300c/k', 'board': board[:3], 'client_pos': 0, 'hole_cards': hole}
    policy = {'c': 0.5, 'f': 0.25, 'b600': 0.25}
    record = lambda selected: {'public': dict(public, action='b300c/kb200'), 'selected': selected}
    facing = dict(public, action='b300c/kb200')
    rows = {a: action_corrections(terminal, [(record(a), policy)])[0] for a in policy}
    assert all(r['street'] == 1 for r in rows.values())
    # Exactly zero on average over the policy's own draws.
    assert sum(Fraction(p) * rows[a]['correction'] for a, p in policy.items()) == 0
    values = action_values(facing, equity_share(hole, bot, board[:3]), policy)
    assert rows['c']['correction'] == values['c'] - sum(Fraction(p) * values[a] for a, p in policy.items())
    # The flop decision lands in the flop-actions column, after the four card columns.
    row = hand_corrections(terminal, [(record('c'), policy)])
    assert row[4:] == [0.0, float(rows['c']['correction']), 0.0, 0.0]
    assert row[:4] == [float(e['correction']) for e in chance_corrections(terminal)]
    with pytest.raises(ValueError, match='no probability'):
        action_corrections(terminal, [(record('b900'), policy)])
    with pytest.raises(ValueError, match='another hand'):
        action_corrections(terminal, [({'public': dict(facing, hole_cards=bot), 'selected': 'c'}, policy)])


def test_cross_fitted_weights_come_only_from_the_other_half():
    rng = np.random.default_rng(2026101605)
    luck = rng.normal(0, 100, size=(4000, 2))
    results = 5 + 3 * luck[:, 0] + rng.normal(0, 10, 4000)
    estimates, weights = cross_fitted(results, luck)
    assert all(abs(w[0] - 3) < 0.05 and abs(w[1]) < 0.05 for w in weights)
    assert abs(np.mean(estimates) - 5) < 1 and np.var(estimates) < np.var(results) / 50
    # Changing only the odd hands moves only the even hands' estimates, through their weights.
    changed = results.copy(); changed[1::2] += 7 * luck[1::2, 1]
    again, _ = cross_fitted(changed, luck)
    assert np.allclose(np.array(again)[1::2], changed[1::2] - luck[1::2] @ np.array(weights[1]))
    assert not np.allclose(np.array(again)[0::2], np.array(estimates)[0::2])


class FakeSlumbot:
    """A Slumbot transport for tests: fresh seeded deals, Slumbot checks or calls, seats alternate."""

    def __init__(self, seed):
        self.rng, self.hands = random.Random(seed), 0

    def post(self, endpoint, body):
        if endpoint.endswith('new_hand'):
            cards = self.rng.sample(DECK, 9)
            self.holes, self.board = [cards[0:2], cards[2:4]], cards[4:]
            self.client, self.hands = self.hands % 2, self.hands + 1
            self.tokens, self.paid, self.bet, self.actor, self.street = [], [100, 50], [100, 50], 1, 0
            self.acted, self.done = [False, False], False
        else:
            self.apply(body['incr'])
        while not self.done and self.actor != self.client:
            self.apply('k' if self.bet[0] == self.bet[1] else 'c')
        response = {'token': 'fake-token', 'action': ''.join(self.tokens), 'client_pos': self.client,
                    'hole_cards': self.holes[self.client], 'board': self.board[:DEALT[self.street]]}
        if self.done:
            bot = self.holes[1 - self.client]
            response.update(bot_hole_cards=bot, baseline_winnings=None, winnings=settle(
                response['action'], self.client, response['hole_cards'], bot, response['board']))
        return response

    def apply(self, token):
        me, other = self.actor, 1 - self.actor
        self.tokens.append(token)
        if token == 'f':
            self.done = True
            return
        if token == 'c':
            self.paid[me] += self.bet[other] - self.bet[me]; self.bet[me] = self.bet[other]
        elif token != 'k':
            self.paid[me] += int(token[1:]) - self.bet[me]; self.bet[me] = int(token[1:]); self.acted[other] = False
        self.acted[me] = True
        if not (self.acted[other] and self.bet[0] == self.bet[1]):
            self.actor = other
        elif self.paid == [STACK, STACK] or self.street == 3:
            self.tokens += ['/'] * (3 - self.street); self.street, self.done = 3, True
        else:
            self.tokens.append('/')
            self.street, self.bet, self.actor, self.acted = self.street + 1, [0, 0], 0, [False, False]


def test_analysis_corrects_the_model_s_own_actions_end_to_end(tmp_path):
    from src.arena.external.analysis import analyze
    from src.arena.external.journal import PrefixJournal
    from src.arena.external.session import hu200_average, play_hand
    from src.arena.external.slumbot import SlumbotConnection
    from tests.test_hu200_evaluation import fixture_spec
    # One stored key: the client samples uniformly from its menu.
    spec = fixture_spec(tmp_path / 'model')
    model = hu200_average(spec['path'], spec['sha256'])
    connection, player = SlumbotConnection(FakeSlumbot(17), model), model.policy(29)
    journal = PrefixJournal(tmp_path / 'journal.jsonl', {'fixture': True})
    hands = [play_hand(connection, player, f'h{i}', journal) for i in range(8)]
    journal.close()

    with_model = analyze(tmp_path / 'journal.jsonl', model=model, workers=2, aivat=True)['aivat']
    cards_only = analyze(tmp_path / 'journal.jsonl', workers=2, aivat=True)['aivat']
    assert with_model['own_actions_corrected'] and not cards_only['own_actions_corrected']
    assert 'action_luck_bb_per_100' not in cards_only
    assert with_model['card_luck_bb_per_100'] == cards_only['card_luck_bb_per_100']
    raw_width = analyze(tmp_path / 'journal.jsonl', workers=1)['bb_per_100']['half_width_95']
    ratio = (with_model['bb_per_100']['half_width_95'] / raw_width) ** 2
    assert with_model['plain_variance_vs_raw'] == pytest.approx(ratio)
    assert with_model['plain_hands_worth'] == pytest.approx(1 / ratio)

    # Recompute every hand's corrections directly from the played hands and the model.
    from src.arena.external.slumbot import SlumbotAdapter, verify_record
    rows = []
    for h in hands:
        decisions = []
        for record in h.decisions:
            view = verify_record(record)
            menu, probabilities, _ = model.distribution(view)
            policy = {}
            for item, p in zip(menu, probabilities):
                encoded = SlumbotAdapter().encode_action(item.action, view)
                policy[encoded] = policy.get(encoded, 0) + p
            decisions.append((record, policy))
        rows.append(hand_corrections(h.terminal, decisions))
    raw = [h.terminal['winnings'] for h in hands]
    assert sum(len(h.decisions) for h in hands) > 8 and any(any(r[4:]) for r in rows)
    assert with_model['action_luck_bb_per_100']['mean'] == pytest.approx(np.mean([sum(r[4:]) for r in rows]))
    assert with_model['bb_per_100']['mean'] == pytest.approx(np.mean([x - sum(r) for x, r in zip(raw, rows)]))


def test_a_failure_ends_the_failed_hand_even_when_its_id_is_reused(tmp_path):
    from src.arena.external.analysis import analyze
    from src.arena.external.journal import PrefixJournal
    from src.arena.external.session import hu200_average, play_hand
    from src.arena.external.slumbot import SlumbotConnection
    from tests.test_hu200_evaluation import fixture_spec
    spec = fixture_spec(tmp_path / 'model')
    model = hu200_average(spec['path'], spec['sha256'])
    fake, player = FakeSlumbot(23), model.policy(37)
    scratch = PrefixJournal(tmp_path / 'scratch.jsonl', {'fixture': True})
    failed = play_hand(SlumbotConnection(fake, model), player, 'x0', scratch)
    scratch.close()
    journal = PrefixJournal(tmp_path / 'journal.jsonl', {'fixture': True})
    for record in failed.decisions:
        journal.append(record)
    journal.failure('timeout-ambiguous')
    completed = play_hand(SlumbotConnection(fake, model), player, 'x0', journal)
    journal.close()
    assert failed.decisions and failed.terminal['hole_cards'] != completed.terminal['hole_cards']
    result = analyze(tmp_path / 'journal.jsonl', model=model, workers=1, aivat=True)
    assert result['hands'] == 1 and result['failures'] == {'timeout-ambiguous': 1}
    assert result['aivat']['own_actions_corrected']


def test_policies_within_the_loader_tolerance_are_accepted():
    hole, bot, board = ['Ah', 'Kd'], ['Qs', 'Qc'], ['2c', '7s', '9d', 'Kc', '3h']
    terminal = hand('b300c/kb200c/kk/kk', hole, bot, board)
    record = {'public': {'action': 'b300c/kb200', 'board': board[:3], 'client_pos': 0, 'hole_cards': hole},
              'selected': 'c'}
    assert action_corrections(terminal, [(record, {'c': 0.5 + 5e-9, 'f': 0.25, 'b600': 0.25})])
    with pytest.raises(ValueError, match='do not sum to 1'):
        action_corrections(terminal, [(record, {'c': 0.5 + 2e-8, 'f': 0.25, 'b600': 0.25})])
