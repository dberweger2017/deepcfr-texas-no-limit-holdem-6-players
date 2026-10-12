from scripts.guard_hu20_k50_first_pass import GIB, violation


def test_campaign_admission_boundaries_keep_fixed_swap_and_deadline():
    sample = dict(pressure_level=1, free_percent=15, swap_bytes=1000+512*1024**2,
                  disk_free_bytes=16*GIB, ac=True)
    baseline = dict(deadline=100, swap_bytes=1000)
    assert violation(sample, 7*GIB-1, baseline, 99) is None
    assert 'swap' in violation(dict(sample, swap_bytes=sample['swap_bytes']+1), 0, baseline, 99)
    assert 'six-hour' in violation(sample, 0, baseline, 100)
    assert 'RSS' in violation(sample, 7*GIB, baseline, 99)
    assert 'disk' in violation(dict(sample, disk_free_bytes=16*GIB-1), 0, baseline, 99)
    assert 'pressure' in violation(dict(sample, pressure_level=2), 0, baseline, 99)
    assert 'pressure' in violation(dict(sample, free_percent=14), 0, baseline, 99)
    assert 'AC' in violation(dict(sample, ac=False), 0, baseline, 99)
