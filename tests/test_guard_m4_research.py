from scripts.guard_m4_research import GIB, sustained


def sample(at=0, **changes):
    return dict({'monotonic': at, 'family_rss_bytes': 0, 'pressure_level': 1,
                 'free_percent': 30, 'swap_bytes': 0, 'disk_free_bytes': 20*GIB}, **changes)


def test_transient_and_sustained_each_memory_guard():
    for change, key in [({'family_rss_bytes':13*GIB+1},'family_rss'),
                         ({'pressure_level':4},'critical_pressure'),
                         ({'free_percent':7},'free_memory'),
                         ({'swap_bytes':4_000_000_001},'swap_growth')]:
        held = {}
        assert sustained(sample(10, **change), sample(), held)[1] == []
        assert sustained(sample(69, **change), sample(), held)[1] == []
        assert sustained(sample(70, **change), sample(), held)[1] == [key]
        assert sustained(sample(71), sample(), held)[1] == []
        assert held == {}
        assert sustained(sample(72, **change), sample(), held)[1] == []


def test_warning_thresholds_and_swap_baseline():
    base = sample(swap_bytes=2_000_000_000)
    reading = sample(90, family_rss_bytes=13*GIB, pressure_level=2, free_percent=8,
                     swap_bytes=6_000_000_000, disk_free_bytes=16*GIB)
    assert not any(sustained(reading, base, {})[0].values())


def test_disk_is_immediate_and_breaches_are_independent():
    held = {}
    assert sustained(sample(0, free_percent=7), sample(), held)[1] == []
    assert sustained(sample(59, pressure_level=4), sample(), held)[1] == []
    assert sustained(sample(60, pressure_level=4, disk_free_bytes=16*GIB-1), sample(), held)[1] == ['free_disk']
