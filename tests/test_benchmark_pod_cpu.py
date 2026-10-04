import unittest

from scripts.benchmark_pod_cpu import data, parse_cpu_max, process_counts


class BenchmarkPodCpuTests(unittest.TestCase):
    def test_cgroup_quota_parsing(self):
        self.assertEqual(parse_cpu_max("2720000 100000\n"), 27.2)
        self.assertIsNone(parse_cpu_max("max 100000\n"))

    def test_process_counts_never_exceed_affinity(self):
        self.assertEqual(process_counts(4), [2, 4])
        self.assertEqual(process_counts(18), [2, 4, 8, 16, 18])
        self.assertEqual(process_counts(112), [2, 4, 8, 16, 32, 112])

    def test_input_is_deterministic(self):
        self.assertEqual(data(size=4096), data(size=4096))
        self.assertEqual(len(data(size=4096)), 4096)


if __name__ == "__main__":
    unittest.main()
