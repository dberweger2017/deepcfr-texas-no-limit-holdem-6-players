"""Campaign safety checks exercise elapsed heat and owned stop requests."""
from scripts.guard_hu200_capacity import VrmWindow, motherboard_sensors, stop_files
import pytest
from pathlib import Path


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


def synthetic_run(monkeypatch, tmp_path, *, cpu_stop):
    import os
    from pathlib import Path
    from types import SimpleNamespace
    from scripts import guard_hu200_capacity as guard
    jobs = [dict(name=str(i), cpus=[2*i, 2*i+1], rss_gib=1,
                 stop_file=str(tmp_path / ('seed-' + str(i)) / 'stop'), command=['never-executed'])
            for i in range(3)]
    plan = dict(jobs=jobs, max_seconds=100, aggregate_rss_gib=4,
                swap_baseline_bytes=0, clean_save_grace_seconds=3600)
    monkeypatch.setattr(guard.os, 'uname', lambda: SimpleNamespace(nodename='cygnus-x1'))
    monkeypatch.setattr(guard, 'motherboard_sensors', lambda: (tmp_path/'vrm', tmp_path/'fan'))
    monkeypatch.setattr(guard.base, 'package_sensor', lambda: tmp_path/'cpu')
    calls = [0]
    def sample(sensor):
        calls[0] += 1
        return dict(package_c=85 if cpu_stop and calls[0] > 1 else 30,
                    available_bytes=100*1024**3, swap_used_bytes=0)
    monkeypatch.setattr(guard.base, 'sample', sample)
    monkeypatch.setattr(guard.base, 'read_int', lambda path:
        341 if path.name == 'fan' else 40000 if calls[0] == 1 or cpu_stop else 95000)
    monkeypatch.setattr(guard.base, 'group_rss', lambda pid: 1024)
    monkeypatch.setattr(guard.os, 'killpg', lambda pid, sig: None)
    monkeypatch.setattr(guard.shutil, 'disk_usage', lambda path: SimpleNamespace(free=200_000_000_000))
    ticks = iter(range(1000))
    monkeypatch.setattr(guard.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(guard.time, 'sleep', lambda seconds: None)
    stopped = []
    class Process:
        pid = os.getpid()
        code = None
        def poll(self):
            if not cpu_stop and all(Path(j['stop_file']).exists() for j in jobs):
                self.code = 3
            return self.code
        def wait(self):
            return self.code
    monkeypatch.setattr(guard.subprocess, 'Popen', lambda *a, **kw: Process())
    def terminate(proc):
        stopped.append(proc)
        if proc.code is None:
            proc.code = -15
    monkeypatch.setattr(guard.base, 'terminate_group', terminate)
    result = guard.supervise(plan, tmp_path / 'guard')
    return result, jobs, stopped


@pytest.mark.skipif(not Path('/proc/self/stat').exists(), reason='Linux process identity')
def test_cpu_85_stops_all_three_owned_groups_immediately(monkeypatch, tmp_path):
    from pathlib import Path
    result, jobs, stopped = synthetic_run(monkeypatch, tmp_path, cpu_stop=True)
    assert result['status'] == 'failed' and result['hard_stop_reason'] == 'CPU >=85 C'
    assert result['elapsed_seconds'] < 5 and len(stopped) == 3
    assert all(Path(j['stop_file']).read_text() == 'CPU >=85 C\n' for j in jobs)


@pytest.mark.skipif(not Path('/proc/self/stat').exists(), reason='Linux process identity')
def test_vrm_clean_stop_of_all_three_is_still_a_failed_attempt(monkeypatch, tmp_path):
    from pathlib import Path
    result, jobs, stopped = synthetic_run(monkeypatch, tmp_path, cpu_stop=False)
    assert result['status'] == 'failed' and result['hard_stop_reason'] is None
    assert result['stop_reason'] == 'VRM >=95 C for 30 continuous observed seconds'
    assert len(stopped) == 3 and result['exit_codes'] == [3, 3, 3]
    assert all(Path(j['stop_file']).exists() for j in jobs)
