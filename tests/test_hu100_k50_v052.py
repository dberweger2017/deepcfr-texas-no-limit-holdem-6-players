from scripts.guard_hu100_k50_v052 import GIB, sustained
from scripts.run_hu100_k50_v052 import ARMS, configuration
from scripts.evaluate_native_hu100_baseline import make_plan


def sample(**kwargs):
    return dict(pressure_level=1,free_percent=50,swap_bytes=100,disk_free_bytes=32*GIB,**kwargs)


def test_transient_thresholds_reset_and_warning_is_allowed():
    since = {}
    good = sample()
    assert sustained(good,14*GIB,100,0,since) is None
    assert sustained(good,14*GIB,100,59,since) is None
    assert sustained(good,12*GIB,100,59.5,since) is None
    assert not since
    warning = {**good,'pressure_level':2}
    assert sustained(warning,14*GIB,100,60,since) is None
    assert sustained(warning,14*GIB,100,119.9,since) is None
    assert sustained(warning,14*GIB,100,120,since)=='rss'


def test_each_sustained_guard_and_immediate_disk():
    for key,value,reason in [('pressure_level',4,'critical_pressure'),('free_percent',7,'free_memory'),
                             ('swap_bytes',4_000_000_101,'swap_growth')]:
        since = {}; bad = {**sample(),key:value}
        assert sustained(bad,0,100,10,since) is None
        assert sustained(bad,0,100,69.9,since) is None
        assert sustained(bad,0,100,70,since)==reason
    assert sustained({**sample(),'disk_free_bytes':16*GIB-1},0,100,0,{})=='disk below 16 GiB'


def test_explicit_k50_scripted_plan_and_translation_arms():
    model = dict(name='k50',path='ignored.gz',sha256='a'*64,
        format='holdem-hu100-stored-cfr-average-research-v1',
        abstraction='hu100-native-reopening-ordered-history-equity-k50-v1',card_buckets='results/tables')
    cfg = configuration(model,None,4096)
    plan = make_plan(cfg,'pot_pressure',4096,20261013101103)
    assert plan.models[0].abstraction==model['abstraction']
    assert plan.models[0].card_buckets==model['card_buckets']
    assert plan.scenarios[0].stacks==(10000,10000)
    assert ARMS['k50_off'][1] is None and ARMS['v052_off'][1] is None
    assert ARMS['v052_on'][1]==dict(max_states=512,max_events=128)
