"""Real admission boundaries for the 60-minute M1 pilot."""
from scripts import run_hu200_feasibility as p


def test_actual_m1_guards_keep_fixed_swap_baseline():
    s=dict(pressure=1,free_percent=70,swap_bytes=1_500_000_000,ac=True,disk_free_bytes=30*p.GIB)
    assert p.violation(s,1_500_000_000,1*p.GIB) is None
    assert p.violation({**s,'swap_bytes':2_100_000_000},1_500_000_000,0)=='swap limit'
    assert p.violation({**s,'swap_bytes':3_000_000_001},3_000_000_000,0)=='swap limit'
    assert p.violation({**s,'pressure':2},1_500_000_000,0)=='system pressure'
    assert p.violation({**s,'ac':False},1_500_000_000,0)=='AC power'
    assert p.violation({**s,'disk_free_bytes':p.DISK_FLOOR},1_500_000_000,0)=='disk floor'
    assert p.violation(s,1_500_000_000,p.HARD)=='hard family RSS'


def test_quote_reserves_closeout_and_caps_entries_by_memory():
    last={'diagnostics':{'entries':500_000},'completed_nodes':1_000_000,'write_seconds':2}
    q=p.quote(last,'next',20_000_000,2,12)
    assert q['entry_ceiling']==10_000_000 and q['upper_seconds']>=p.CLOSEOUT
    large=p.quote(last,'next',100_000_000,2,12)
    assert large['forecast_family_bytes']<=p.SOFT
    assert large['entry_ceiling']==(p.SOFT-100_000_000)//110
