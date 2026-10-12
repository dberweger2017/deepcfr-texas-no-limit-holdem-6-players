"""Write the PR238 report only after both complete final replay and arithmetic."""
import json
from pathlib import Path
from time import time
from scripts.hu20_k50_first_pass_assets import write
from scripts.run_hu20_k50_first_pass_training import BASE


def read(path):
    return json.loads(path.read_text())


def fmt(value):
    return '—' if value is None else f'{value:,.2f}'


def interval(row):
    ci=row['ci95']
    return '—' if ci is None else f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'


def generate():
    final=read(BASE/'final-readout.json')
    closed=read(BASE/'guards/operations/final-numerical-readout/receipt.json')
    if final['status']!='verified' or closed['status']!='complete' or (BASE/'guards/campaign-failure.json').exists():
        raise ValueError('No report before completed final replay/readout')
    posted=read(BASE/'canonical-plan-posted.json')
    quote=read(BASE/'pilot-quote.json')
    primary=final['contrasts']['primary']['overall']
    rows=['# HU20 K50 5B first pass','',
          f"**K50 stronger at 5B: {primary['bb_per_100']:+.2f} BB/100 {interval(primary)}.** "
          +final['primary_decision']+'. Primary and secondary results below use the frozen, '
          'equally weighted three-seed/two-seat duplicate-block schedule. '
          'All 393,216 final hands independently replay before numerical readout. '
          'This answers the equal-node first-pass question; the owner decides next.','',
          '[Protocol](../hu20-k50-5b-first-pass.md) · [canonical plan posted before final play]('+posted['url']+') · '
          '[exact plan](hu20-k50-5b-first-pass-artifacts/canonical-final-plan.json).','',
          '## Direct results','',
          'Each contrast contains 32,768 duplicate blocks per seed, three fixed seeds, two rotated seats, '
          '196,608 hands. Both seats and the three seeds receive equal weight within each shared deal block. '
          'Intervals are nominal paired Student-t 95%, conditional on these trained policies; '
          'they do not include population uncertainty over future training seeds. '
          'Raw chips equal BB/100 because the big blind is 100 chips. '
          'Secondary and splits are descriptive; only the primary aggregate selects the predeclared recommendation.','']
    for contrast,title in [('primary','PRIMARY: K50 5B − matched-seed v1 5B'),('secondary','SECONDARY: K50 5B versus v0.4.2 O10B')]:
        r=final['contrasts'][contrast]
        rows += ['### '+title,'','| Split | BB/100 | Nominal 95% interval |','| --- | ---: | --- |']
        stats=[('Aggregate',r['overall'])]
        stats += [('Seed '+str(2026100600+int(seed)),value) for seed,value in r['lineages'].items()]
        stats += [('Seat '+seat,value) for seat,value in r['positions'].items()]
        stats += [('Seed '+seed+' / '+seat,value) for seed,values in r['seed_by_seat'].items() for seat,value in values.items()]
        rows += [f"| {name} | {value['bb_per_100']:+.2f} | {interval(value)} |" for name,value in stats]
        rows += ['']
    rows += ['The primary rule was frozen before final hands: lower bound > 0 means stronger/recommend the longer '
             'protocol; upper bound < 0 means weaker/recommend stopping full-game K50; otherwise inconclusive/no extension. '
             'No LBR, pressure/scripted panel, seed selection, failed-hand rerun, outcome-driven extension, adoption or release.','',
             '## Exact whole-table training density','',
             '| Policy / seed | Actual nodes | Iterations | Keys | Mean traverser visits/key | Zero-visit fraction |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    models={}
    for label in ('v1','k50'):
        for seed in (2026100601,2026100602,2026100603):
            r=read(BASE/'audits'/f'{label}-{seed}-5b.json');m=r['model'];d=r['density']
            models[f'{label}-{seed}-5b']=r
            rows += [f"| {label} / {seed} | {m['actual_nodes']:,} | {m['iteration']:,} | {d['count']:,} | {d['mean']:.2f} | {d['zero_fraction']:.4%} |"]
    rows += ['','K50 has 1.29–1.31 times the matched v1 keys, below the anticipated ~1.4. '
             'Mean visits/key are ~117 versus 151–154 at the same requested 5B node budget. '
             'Training work was not adjusted. Complete-iteration overshoots are shown.','',
             '[All current/average full-row audits and model hashes](hu20-k50-5b-first-pass-artifacts/terminal-audits.json). '
             'Checkpoint keys are irreversible hashes without a street map. The owner accepted reached-key '
             'quantiles with disclosed coverage; no complete-table street distribution is inferred.','',
             '## Street coverage and reached-key visit quantiles','',
             'The following covers distinct stored keys reached in the **primary frozen direct matches**. '
             'Each stored key counts once per policy. The secondary coverage is retained separately in the full '
             'readout receipt. Missing reached keys are separate from stored zero-visit keys. '
             'Unclassified stored keys remain outside these street quantiles.','',
             '| Policy | Classified stored keys | Full-table coverage | Unclassified stored keys | Missing reached keys |',
             '| --- | ---: | ---: | ---: | ---: |']
    coverage=final['contrasts']['primary']['coverage']
    for name,c in sorted(coverage.items()):
        missing=sum(s['missing_reached_keys'] for s in c['streets'].values())
        rows += [f"| {name} | {c['unique_stored_reached_keys']:,} | {c['classified_fraction']:.4%} | {c['unclassified_table_keys']:,} | {missing:,} |"]
    for name,c in sorted(coverage.items()):
        rows += ['','### '+name,'',
                 '| Street | Stored reached keys | Mean visits | Min | p10 | p25 | Median | p75 | p90 | p99 | Max |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for street in ('preflop','flop','turn','river'):
            s=c['streets'][street]
            d=s['stored_reached_keys'];q=d['quantiles'] or {}
            rows += [f"| {street} | {d['count']:,} | {fmt(d['mean'])} | "+' | '.join(fmt(q.get(k)) for k in ('min','p10','p25','median','p75','p90','p99','max'))+' |']
    rows += ['','## Correctness, cost and resource evidence','',
             'Current-main native source 045c793a4abd2e371d4d77de663fd3dd2f00316b is unchanged. '
             'Binary SHA256 aa7d3a5ef197f113c82a3e861abdc3081af9f5262f96f991e093d6d5a7a0add6; 14 native tests pass. '
             'Fresh v1 seed 602 at 1B reproduces archived bytes (a0ef695d6807aa4554ba01b32d27f64c39ccdc72d03be5a21c99cf67e52ad716), '
             'including max_entries 1B. Fresh K50 4M versus its own 2M resume has 785,675 exact rows, '
             'including IEEE signed zero and all three table hashes. Each of the six terminal checkpoint/current/average sets passes the full existing export auditor.','',
             'The cost-only 2048-block/seed/contrast pilot inspected costs and pairedSD only. '
             f"Primary SD {quote['primary_aggregate_paired_sd_bb_per_100']:.2f} BB/100 projected half-width {quote['projected_primary_half_width_bb_per_100']:.4f} BB/100; "
             f"total {quote['projected_total_seconds']:.2f} s including 25% play/replay headroom and 1,800 s archive/closeout. Both limits passed. "
             'Counts/roots/input hashes/decision were frozen on PR before any final hand. '
             '[Pilot quote](hu20-k50-5b-first-pass-artifacts/pilot-quote.json).','',
             'Independent replay rekeys each acting policy with its own schema/tables and own observation, '
             'checks menus, public history/legal bounds, probability normalization, selected actions, '
             'deal coordinates/seeds, public event digest and exact zero-sum settlement for every final hand. '
             'Independent math.fsum/Student-t arithmetic agrees with the existing production reporter for aggregate, seed and seat splits. '
             'Additional seed-by-seat estimates preserve aggregate weighting.','',
             'The original fixed six-hour budget started 10:20:26UTC /12:20:26Madrid and ends 16:20:26UTC /18:20:26Madrid. '
             'No reset. Continuous whole-family RSS < 7 GiB, normal pressure, free memory >= 15%, fixed-baseline '
             'swap growth <= 512 MiB, disk >= 16 GiB and AC remained binding. Actual closed-stage resource receipt follows archival closeout.','',
             '## Admission, provenance and preservation','',
             'PR236 merged 04:29:05UTC; the 06:30 Madrid admission check found no campaign processes or denied metadata. '
             'The owner superseded the 02:02 ETA with 06:00 Madrid then 30-minute checks. Only this thread timer was modified. '
             'M4 is the scientific machine; M1 only coordinates.','',
             'Initial 21.87 GiB disk prompted a preventive 38 GiB forecast hold, not a guard failure. '
             'The owner authorized individually reserved stages while provisioning; guarded admission then had 40.50 GiB. '
             'No other PR root/process/schedule, synced deletion/offload or paid compute was used. '
             '[Initial storage formula](hu20-k50-5b-first-pass-artifacts/storage-admission.json).','',
             'Setup failures are retained: remote PATH, HTTPS keychain, quoting/missing test paths, unused wrapper indentation, '
             'canonical-post local time-module error (existing exact comment read back; no final hands started), '
             'and an older direct test double lacking the newly logged key contract. The corrected fixture preserves runtime bytes '
             'and 11 focused direct checks pass; two archive inventory/hash-refusal checks pass. These were source/setup checks, '
             'not failed scientific attempts or hand reruns.','',
             'Archive local/member/whole-hash verification, separate native and independent Drive acceptance, '
             'RESULTS_INDEX restoration records and final exact-head checks remain pending. Originals are preserved; '
             'the PR remains draft until closeout is complete.','']
    destination=Path('docs/reports/hu20-k50-5b-first-pass.md')
    destination.write_text('\n'.join(rows).rstrip()+'\n')
    write(BASE/'report-generated.json',{'at':time(),'status':'reported-verified-readout','report':str(destination)})
    out=Path('docs/reports/hu20-k50-5b-first-pass-artifacts/final-readout.json')
    out.write_text(json.dumps(final,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':generate()
