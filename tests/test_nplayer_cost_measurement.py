import json
from pathlib import Path
import tempfile
import unittest

from scripts.measure_nplayer_blueprint_cost import summarize_trace


class TraceCostSummaryTests(unittest.TestCase):
    def fixture(self, folder, hands):
        path = Path(folder) / 'trace.jsonl'
        header = {'format': 'nplayer-privileged-policy-replay-v1',
                  'hands': 2, 'policies': [{}, {}, {}]}
        path.write_text('\n'.join(json.dumps(v) for v in [header, *hands]) + '\n')
        return path

    def hand(self, number, streets):
        return {'hand': number, 'actors': [0] * len(streets),
                'actions': [['call', 0]] * len(streets),
                'states': [{'street': s} for s in streets] + [{'street': None}]}

    def test_counts_decisions_without_using_rewards(self):
        with tempfile.TemporaryDirectory() as folder:
            hands = [self.hand(0, ['preflop', 'flop']), self.hand(1, ['preflop'])]
            path = self.fixture(folder, hands)
            expected = {'hands': 2, 'decisions_by_street': {'preflop': 2, 'flop': 1}}
            self.assertEqual(summarize_trace(path, 2), expected)
            hands[0]['rewards'] = [999999, -999999, 0]
            self.assertEqual(summarize_trace(self.fixture(folder, hands), 2), expected)

    def test_rejects_missing_or_duplicate_hands(self):
        with tempfile.TemporaryDirectory() as folder:
            for hands in ([self.hand(0, [])], [self.hand(0, []), self.hand(0, [])]):
                with self.subTest(hands=hands), self.assertRaises(ValueError):
                    summarize_trace(self.fixture(folder, hands), 2)

    def test_rejects_incomplete_action_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            hand = self.hand(0, ['flop'])
            hand['actions'] = []
            with self.assertRaises(ValueError):
                summarize_trace(self.fixture(folder, [hand, self.hand(1, [])]), 2)
