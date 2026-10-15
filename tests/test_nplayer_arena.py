"""Statistics cluster all six duplicates; the final reader requires verification."""
import copy
import unittest
from scripts.analyze_nplayer_arena import summarize, validate_blocks

PERMS = ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0))


def fixture():
    rows = []
    for block, win in enumerate((100, 200, 300)):
        for permutation, ids in enumerate(PERMS):
            rows.append({'block': block, 'permutation': permutation, 'identities': list(ids),
                         'rewards': [win if i == 0 else -win // 2 for i in ids]})
    return rows


class StatisticsTests(unittest.TestCase):
    def test_block_means_and_seat_denominators(self):
        result = summarize(fixture(), big_blind=100)
        self.assertEqual(result['blocks'], 3)
        self.assertEqual(result['hands'], 18)
        self.assertEqual(result['singleton']['mean_bb100'], 200)
        self.assertEqual(result['pair_per_player']['mean_bb100'], -100)
        for seat in result['singleton_by_seat']:
            self.assertEqual(seat['mean_bb100'], 200)
            self.assertAlmostEqual(seat['standard_error_bb100'], 100 / 3**0.5)
        # Eighteen correlated hands must not be used as eighteen independent observations.
        self.assertAlmostEqual(result['singleton']['standard_error_bb100'], 100 / 3**0.5)
        self.assertLess(result['singleton']['lower95_bb100'], 0)

    def test_missing_duplicate_or_bad_settlement_fails(self):
        rows = fixture()
        with self.assertRaises(ValueError): validate_blocks(rows[:-1])
        rows[0]['rewards'][0] += 1
        with self.assertRaises(ValueError): validate_blocks(rows)

    def test_repeated_identity_or_permutation_fails(self):
        rows = fixture()
        rows[0]['identities'] = [0, 0, 2]
        with self.assertRaises(ValueError): validate_blocks(rows)
        rows = fixture()
        rows[1] = copy.deepcopy(rows[0])
        with self.assertRaises(ValueError): validate_blocks(rows)


class NativeHarnessTests(unittest.TestCase):
    def test_scripted_duplicates_replay_reproduce_and_reject_corruption(self):
        import json
        import os
        from pathlib import Path
        import subprocess
        import tempfile
        from scripts.check_nplayer_policy_replay import check
        from scripts.analyze_nplayer_arena import verified_readout
        binary = Path(os.environ.get('NPLAYER_ARENA_BINARY', 'native/hu20-trainer/target/release/examples/nplayer_arena'))
        if not binary.exists(): self.skipTest('build nplayer_arena')
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for bot in ('random', 'check-call', 'tight-aggressive', 'pot-pressure'):
                for reverse in (False, True):
                    a, b = f'bot={bot}', 'bot=check-call'
                    if reverse: a, b = b, a
                    traces = [root/f'{bot}-{reverse}-{i}.jsonl' for i in (0,1)]
                    for trace in traces:
                        result = subprocess.run([str(binary), '-', '3', '7701', str(trace), a, b],
                                                capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(traces[0].read_bytes(), traces[1].read_bytes())
                    receipt = check(traces[0])
                    self.assertEqual(receipt['hands'], 18)
                    receipt_path = root/'replay.json'
                    receipt_path.write_text(json.dumps(receipt))
                    self.assertEqual(verified_readout(traces[0], receipt_path, traces[1])['blocks'], 3)
            lines = traces[0].read_text().splitlines()
            row = json.loads(lines[1]); row['rewards'][0] += 1
            lines[1] = json.dumps(row)
            bad = root/'bad.jsonl'; bad.write_text('\n'.join(lines)+'\n')
            with self.assertRaises(AssertionError): check(bad)
            with self.assertRaises(ValueError): verified_readout(bad, receipt_path, traces[1])
            stop = root/'stop'; stop.touch()
            result = subprocess.run([str(binary), '-', '3', '7701', str(root/'partial.jsonl'), a, b],
                capture_output=True, text=True, timeout=30,
                env=dict(os.environ, NPLAYER_STOP_FILE=str(stop)))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('clean stop before block 0', result.stderr)


if __name__ == '__main__':
    unittest.main()
