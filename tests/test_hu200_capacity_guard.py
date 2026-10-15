"""Campaign safety checks exercise elapsed heat and owned stop requests."""
from scripts.guard_hu200_capacity import VrmWindow, motherboard_sensors, stop_files
import pytest


def test_heat_requires_thirty_continuous_seconds_and_resets_on_gaps():
    window = VrmWindow()
    for now in range(30):
        assert not window.update(now, 95)
    assert window.update(30, 95)
    assert not window.update(31, 94.9)
    assert not window.update(32, 96)
    assert not window.update(62, 96)
    for now in range(63, 92):
        assert not window.update(now, 96)
    assert window.update(92, 96)


def test_sensor_identity_and_missing_fan_refuse_load(tmp_path):
    device = tmp_path / 'hwmon9'
    device.mkdir()
    (device / 'name').write_text('nct6687\n')
    (device / 'temp3_label').write_text('Thermistor 15\n')
    (device / 'temp3_input').write_text('40000\n')
    with pytest.raises(FileNotFoundError):
        motherboard_sensors(tmp_path)
    (device / 'fan1_input').write_text('341\n')
    assert motherboard_sensors(tmp_path) == (device / 'temp3_input', device / 'fan1_input')


def test_stop_requests_are_per_job_and_preserve_first_reason(tmp_path):
    jobs = [{'stop_file': str(tmp_path / str(seed) / 'stop')} for seed in (1, 2, 3)]
    stop_files(jobs, 'VRM hot')
    stop_files(jobs, 'Later reason')
    for job in jobs:
        from pathlib import Path
        assert Path(job['stop_file']).read_text() == 'VRM hot\n'
