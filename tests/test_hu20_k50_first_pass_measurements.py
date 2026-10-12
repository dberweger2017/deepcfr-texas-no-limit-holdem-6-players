from collections import Counter
import pytest
from scripts.hu20_k50_first_pass_measurements import quantiles


def test_histogram_quantiles_equal_expanded_numpy_linear_quantiles():
    import numpy as np
    histogram=Counter({0:3,1:2,17:1,100:4})
    values=[n for n,count in histogram.items() for _ in range(count)]
    result=quantiles(histogram)
    for key,p in [('min',0),('p10',.1),('p25',.25),('median',.5),('p75',.75),('p90',.9),('p99',.99),('max',1)]:
        assert result['quantiles'][key]==pytest.approx(float(np.quantile(values,p)))
    assert result['mean']==pytest.approx(np.mean(values))
    assert result['zero_fraction']==.3 and result['count']==10


def test_empty_histogram_has_no_quantile_claim():
    assert quantiles(Counter())=={'count':0,'mean':None,'quantiles':None}
