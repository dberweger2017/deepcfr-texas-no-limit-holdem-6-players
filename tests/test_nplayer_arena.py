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


if __name__ == '__main__': unittest.main()
