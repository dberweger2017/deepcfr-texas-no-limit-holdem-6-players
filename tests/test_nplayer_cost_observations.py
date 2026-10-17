import copy
import unittest

from scripts.extract_nplayer_cost_observations import acting_observation


class ActingObservationTests(unittest.TestCase):
    def fixture(self):
        state = dict(actor=1, finished=False, holes=[[0, 1], [2, 3], [4, 5]],
            board=[6, 7, 8], street='flop', pot=500, wager=100, min_raise=100,
            players=[[1800, 100, 200, False]] * 3,
            legal={'check': False, 'call': 100, 'raise_to': [200, 1900]},
            rewards=None)
        return dict(actors=[0, 1], actions=[['raise', 100], ['call', 0]],
            states=[dict(state, actor=0), state, dict(state, board=[6, 7, 8, 9, 10])],
            stacks=[2000] * 3, button=0, sb=50, bb=100, deck=list(range(52)))

    def test_opponent_cards_future_actions_and_outcome_are_not_inputs(self):
        hand = self.fixture()
        before = acting_observation(hand, 1)
        changed = copy.deepcopy(hand)
        changed['deck'].reverse()
        changed['states'][1]['holes'][0] = [30, 31]
        changed['states'][1]['holes'][2] = [32, 33]
        changed['states'][1]['rewards'] = [999, -999, 0]
        changed['states'][2]['board'] = [20, 21, 22, 23, 24]
        changed['actions'][1] = ['raise', 1900]
        self.assertEqual(acting_observation(changed, 1), before)
        self.assertEqual(before['hole'], [2, 3])
        self.assertEqual(len(before['public_actions']), 1)

    def test_own_cards_and_observed_actions_remain_visible(self):
        hand = self.fixture()
        before = acting_observation(hand, 1)
        hand['states'][1]['holes'][1] = [40, 41]
        hand['actions'][0] = ['raise', 200]
        self.assertNotEqual(acting_observation(hand, 1), before)

    def test_refuses_wrong_acting_seat(self):
        hand = self.fixture()
        hand['states'][1]['actor'] = 2
        with self.assertRaises(ValueError):
            acting_observation(hand, 1)
