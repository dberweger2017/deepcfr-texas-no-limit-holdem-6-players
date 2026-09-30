from scripts.mature_cpu_rental_guard import check_quote, owned_pods
from scripts.verify_mature_cpu_linux import export_transport_difference


def test_cutoff_names_cannot_capture_another_owners_pod():
    ours = {'name': 'doctor-research-mature-cpu3c-1', 'id': 'ours'}
    other = {'name': 'doctor-research-mature-cpu3c-10', 'id': 'other'}
    assert owned_pods([ours, other], {ours['name']}) == [ours]


def test_shape_rate_and_cpu_only_are_required():
    config = {'cpu_id': 'cpu3c', 'vcpus': 2, 'ram_gb': 4}
    pod = {'cpu': {'id': 'cpu3c', 'vcpuCount': 2, 'memory': 4}, 'cost': .062}
    assert check_quote(config, pod, .11)
    for changed in ({'gpu': {'count': 1}}, {'cost': .12}, {'cost': 0},
                    {'cpu': {'id': 'cpu5c', 'vcpuCount': 2, 'memory': 4}},
                    {'cpu': {'id': 'cpu3c', 'vcpuCount': 4, 'memory': 8}}):
        assert not check_quote(config, dict(pod, **changed), .11)


def test_only_known_export_os_byte_is_exempted(tmp_path):
    a, b = tmp_path / 'mac.gz', tmp_path / 'linux.gz'
    header = bytes.fromhex('1f8b08000000000002')
    a.write_bytes(header + bytes([19]) + b'identical-payload')
    b.write_bytes(header + bytes([3]) + b'identical-payload')
    assert export_transport_difference(a, b)
    b.write_bytes(header + bytes([3]) + b'changed-payload')
    assert not export_transport_difference(a, b)
    b.write_bytes(header + bytes([255]) + b'identical-payload')
    assert not export_transport_difference(a, b)


def test_linux_worker_is_launched_from_driver_not_runtime():
    from pathlib import Path
    source = (Path(__file__).parents[1] / 'scripts/mature_cpu_linux_setup.sh').read_text()
    launch = source.index(' -m scripts.mature_cpu_linux_worker')
    assert source.rfind('cd /workspace/driver', 0, launch) > source.rfind('cd /workspace/runtime', 0, launch)
    assert '/workspace/runtime/.venv/bin/python -m scripts.mature_cpu_linux_worker' in source
