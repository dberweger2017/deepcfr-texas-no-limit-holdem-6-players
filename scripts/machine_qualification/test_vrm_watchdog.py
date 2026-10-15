"""Synthetic sensor/stop checks; no test deliberately heats the host."""
import tempfile
import unittest
from pathlib import Path

from scripts.machine_qualification.vrm_watchdog import ThermalWindow, motherboard_sensors, request_stop


class VrmTests(unittest.TestCase):
    def test_thirty_continuous_seconds_and_reset(self):
        window = ThermalWindow()
        self.assertFalse(window.update(100, 95))
        self.assertFalse(window.update(129.9, 96))
        self.assertFalse(window.update(130, 94.9))
        self.assertFalse(window.update(131, 96))
        self.assertFalse(window.update(160.9, 95))
        self.assertTrue(window.update(161, 95))

    def test_sensor_identity_is_label_not_hwmon_number(self):
        with tempfile.TemporaryDirectory() as root:
            device = Path(root) / 'hwmon99'
            device.mkdir()
            (device / 'name').write_text('nct6687\n')
            (device / 'temp3_label').write_text('Thermistor 15\n')
            (device / 'temp3_input').write_text('76000\n')
            (device / 'fan1_input').write_text('341\n')
            self.assertEqual(motherboard_sensors(Path(root)),
                             (device / 'temp3_input', device / 'fan1_input'))
            (device / 'temp3_label').write_text('CPU\n')
            with self.assertRaises(RuntimeError):
                motherboard_sensors(Path(root))

    def test_missing_fan_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            device = Path(root) / 'hwmon0'
            device.mkdir()
            (device / 'name').write_text('nct6687')
            (device / 'temp3_label').write_text('Thermistor 15')
            (device / 'temp3_input').write_text('76000')
            with self.assertRaises(RuntimeError):
                motherboard_sensors(Path(root))

    def test_stop_file_is_not_a_signal_or_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            stop = Path(root) / 'stop'
            request_stop(stop, 'VRM hot')
            self.assertEqual(stop.read_text(), 'VRM hot\n')
            with self.assertRaises(FileExistsError):
                request_stop(stop, 'changed')


if __name__ == '__main__':
    unittest.main()

@unittest.skipUnless(Path('/proc/meminfo').exists(), 'Linux supervision')
class SupervisorTests(unittest.TestCase):
    def test_vrm_requests_clean_stop_and_records_failure(self):
        import argparse
        import json
        import sys
        from unittest.mock import patch
        from scripts.machine_qualification import vrm_watchdog as guard
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            stop = root / 'stop'
            args = argparse.Namespace(output=root / 'guard', stop_file=stop,
                seconds=10, max_rss_gib=1, min_disk_gib=0, clean_stop_grace_seconds=5,
                command=[sys.executable, '-c',
                    'import os,time; from pathlib import Path; '
                    'p=Path(os.environ["NPLAYER_STOP_FILE"]); '
                    '\nwhile not p.exists(): time.sleep(.01)\n'])
            row = {'package_c': 40, 'available_bytes': 100 * 1024**3, 'swap_used_bytes': 0}
            with patch.object(guard.base, 'package_sensor', return_value=root / 'cpu'), \
                 patch.object(guard, 'motherboard_sensors', return_value=(root/'vrm', root/'fan')), \
                 patch.object(guard.base, 'sample', return_value=row.copy()), \
                 patch.object(guard.base, 'read_int', side_effect=lambda p: 76000 if p.name=='vrm' else 341), \
                 patch.object(guard.ThermalWindow, 'update', return_value=True):
                self.assertEqual(guard.supervised(args), 1)
            result = json.loads((args.output/'result.json').read_text())
            self.assertEqual(result['exit_code'], 0)
            self.assertEqual(result['stop_reason'], 'VRM >=95 C for 30 continuous seconds')
            self.assertIsNone(result['hard_stop_reason'])
            for line in (args.output/'resources.jsonl').read_text().splitlines():
                self.assertEqual(json.loads(line)['vrm_c'], 76)
                self.assertEqual(json.loads(line)['fan1_rpm'], 341)
