"""Linux integration checks of guard stops using a sleeping child, never heat."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.machine_qualification import watchdog


@unittest.skipUnless(Path('/proc/meminfo').exists(), 'Linux guard')
class GuardTests(unittest.TestCase):
    def exercise(self, second, *, command=None, seconds=10):
        row = {'package_c': 40, 'available_bytes': 100 * 1024**3,
               'swap_used_bytes': 0}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'job'
            args = argparse.Namespace(output=out, seconds=seconds,
                                      command=command or ['sleep', '30'])
            cpus = sorted(os.sched_getaffinity(0))[:8]
            # The production wrapper uses 0-7 on this inventoried host. Tests
            # can also run in a CI container with a different allowed CPU set.
            real_popen = watchdog.subprocess.Popen
            def popen(command, **kwargs):
                command[2] = ','.join(map(str, cpus))
                return real_popen(command, **kwargs)
            with patch.object(watchdog, 'package_sensor', return_value=Path('/unused')), \
                    patch.object(watchdog, 'sample', side_effect=[row, second]), \
                    patch.object(watchdog.subprocess, 'Popen', side_effect=popen):
                code = watchdog.run(args)
            return code, json.loads((out / 'result.json').read_text())

    def test_hot_reading_kills_the_child(self):
        code, result = self.exercise({'package_c': 85, 'available_bytes': 100 * 1024**3,
                                      'swap_used_bytes': 0})
        self.assertEqual(code, 1)
        self.assertEqual(result['stop_reason'], 'package temperature >=85 C')
        self.assertLess(result['exit_code'], 0)

    def test_sensor_failure_kills_the_child(self):
        code, result = self.exercise(OSError('sensor vanished'))
        self.assertEqual(code, 1)
        self.assertIn('sensor/resource read failed', result['stop_reason'])
        self.assertLess(result['exit_code'], 0)

    def test_low_memory_kills_the_child(self):
        code, result = self.exercise({'package_c': 40, 'available_bytes': 5 * 1024**3,
                                      'swap_used_bytes': 0})
        self.assertEqual(code, 1)
        self.assertEqual(result['stop_reason'], 'available RAM below 6 GiB')
        self.assertLess(result['exit_code'], 0)


if __name__ == '__main__':
    unittest.main()
