"""Capacity forecasts cannot silently extrapolate to an unmeasured billion-node table."""
import pytest
from scripts.estimate_hu_export_capacity import estimate


def fixture():
    measurements = [{'model':'hu100-11042440','phase':'final','operation':op,'entries':3255387,
                     'seconds':seconds,'sampled_peak_family_rss_bytes':125000000,'kernel_command_peak_rss_bytes':95000000}
                    for op,seconds in [('export',30),('audit',60),('extract',28)]]
    training = [{'process_rss_bytes_after_save':924909568,'write_seconds':9.774,'checkpoint_bytes':114633184,
                 'diagnostics':{'entries':3255387}}]
    return measurements, training


def test_next_growth_has_explicit_headroom_and_bounded_extrapolation():
    measured, training = fixture(); result = estimate(measured, training, 6510774)
    assert result['advisory_only'] and result['fits_rss_estimate']
    assert result['export_audit_extract_reserve_bytes'] == 2.2 * 125000000 * 2
    assert result['one_set_export_audit_extract_reserve_seconds'] == pytest.approx(472)
    assert result['training_save_reserve_bytes'] > 2 * 924909568 * 2
    with pytest.raises(ValueError, match='2x'):
        estimate(measured, training, 1000000000)
    assert not estimate(measured, training, 6510774, rss_gib=1)['fits_rss_estimate']


def test_incomplete_or_invalid_measurements_cannot_quote():
    measured, training = fixture()
    with pytest.raises(ValueError, match='Complete'):
        estimate(measured[:-1], training, 3255387)
    measured[0]['seconds'] = float('nan')
    with pytest.raises(ValueError, match='Invalid'):
        estimate(measured, training, 3255387)
    with pytest.raises(ValueError):
        estimate([], training, 3255387)
