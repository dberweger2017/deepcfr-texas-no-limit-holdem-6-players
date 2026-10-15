from scripts.plan_hu200_blend_resources import GIB, project


def test_projection_charges_temporaries_and_archive_without_three_seed_caches():
    scan = dict(union_keys=1000, action_slots=2000, cache_array_bytes=35*1000+8*2000+768,
                output=dict(rows=10), inputs=[dict(entries=100, action_slots=200)]*3,
                menus=[dict(names=['fold', 'call'], keys=1000)])
    sample = {name: dict(bytes=size) for name,size in [('checkpoint',200),('current',100),('average',150)]}
    result = project(scan,sample,trace_bytes=500)
    retained = sum(result['projected_outputs'].values())
    assert result['future_peak_additional_bytes']['audit'] == retained+result['audit_workspace_bytes']
    assert result['future_peak_additional_bytes']['play'] == retained+result['blend_cache_bytes']+result['seed_cache_bytes'][0]+500
    assert result['required_free_bytes'] == max(result['future_peak_additional_bytes'].values())+18*GIB
    assert result['future_peak_additional_bytes']['archive'] == 2*(retained+500+256*1024**2)
