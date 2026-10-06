"""Publish all independently verified contrasts without selecting lineages or outcomes."""
import gzip
import json
from pathlib import Path
from scripts.run_hu20_zero_mass_fallback import ROOT,FAMILIES,read,write,guard

NAMES={'A-Tprime-vs-T':'Primary A: T′ vs T','B-Oprime-vs-O':'Primary B: O′ vs O',
       'secondary-Tprime-vs-R1':'Secondary: T′ vs shipped R1','secondary-Oprime-vs-R1':'Secondary: O′ vs shipped R1',
       'secondary-Tprime-vs-O':'Secondary: T′ vs O'}

def display(e):
    return f"{e['bb_per_100']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]" if e['ci95'] else f"{e['bb_per_100']:+.2f} [interval unavailable]"

def generate():
    guard();audits={f:read(ROOT/'final'/f/'run/audit.json.gz') for f in FAMILIES}
    assert all(a['status']=='verified' and a['independent_arithmetic_matches'] for a in audits.values())
    pilot={f:read(ROOT/'pilot'/f/'run/audit.json.gz') for f in FAMILIES}
    assert all(a['status']=='verified' for a in pilot.values())
    final=read(ROOT/'final-plan.json.gz');quote=read(ROOT/'quote.json.gz')
    lines=['# M1 zero-mass current fallback comparison','',
        'The six #165 1B checkpoints were re-exported from merged #178 with `--zero-mass current`. T′ and O′ differ from the exact original T and O averages only at stored zero-mass keys. Positive-mass averages, training and missing-key behavior are unchanged. No release decision.','',
        '| Contrast | Three-lineage BB/100 [95% interval] | Predeclared label |','| --- | ---: | --- |']
    for f,a in audits.items():e=a['primary_overall'];lines.append(f"| {NAMES[f]} | {display(e)} | {e['label']} |")
    a,b=[audits[f]['primary_overall'] for f in FAMILIES[:2]]
    reading='T’s fallback '+('improves direct play against its original export' if a['label']=='better' else 'hurts direct play against its original export' if a['label']=='worse' else 'has no detectable direct difference from its original export')+'. '
    reading+='O’s fallback '+('also improves its matched original' if b['label']=='better' else 'hurts its matched original' if b['label']=='worse' else 'has no detectable direct difference from its matched original')+'. '
    reading+='These are direct pairwise results for the retained policies; nondetection does not establish equivalence or general poker strength.'
    direct_reading=reading
    lines+=['',reading,'','## Every retained lineage','',
        '| Contrast | Training seed | BB/100 [95% interval] | Label |','| --- | --- | ---: | --- |']
    for f,result in audits.items():
        for l,e in result['lineages'].items():lines.append(f"| {NAMES[f]} | {2026100600+int(l)} | {display(e)} | {e['label']} |")
    widths=[(e['ci95'][1]-e['ci95'][0])/2 for f in FAMILIES[:2] for e in [audits[f]['primary_overall'],*audits[f]['lineages'].values()]]
    lines+=['','## Frozen method and audit','',
        'Labels were declared in [#179 before the pilot](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/179#issuecomment-6015294307): better if lower bound >0, worse if upper bound <0, otherwise no detectable difference. All intervals are nominal Student-t 95%, conditional on three retained training lineages. Seats and lineages are averaged inside independent deal blocks; no multiplicity-adjusted claim. One BB is 100 chips, so mean net chips per hand equals BB/100.','',
        'The #175 direct runner, loader, duplicate deals/seats, separate action streams, reporter and independent auditor have unchanged source hashes. A storage adapter gzips only metadata and logs. Final root **202610063501** is disjoint from excluded pilot **202610063401**. Counts come from outcome-blind maximum block SDs across each primary’s three lineages and aggregate, targeting ≤3.5 BB/100 with headroom for requested ≤4. All secondary families use the larger primary count. No sample extension.','',
        '| Family | Paired deal blocks | Final hands |','| --- | ---: | ---: |']
    for f in FAMILIES:lines.append(f"| {NAMES[f]} | {final['families'][f]['plan']['blocks']:,} | {audits[f]['hands_replayed']:,} |")
    lines+=['',f"Canonical bundled final plan SHA256 `{read(ROOT/'final-plan-hash.json.gz')['sha256']}`. Achieved primary aggregate/lineage half-widths range **{min(widths):.2f}–{max(widths):.2f} BB/100**, {'all meeting' if max(widths)<=4 else 'not all meeting'} the requested ≤4. Counts and roots remain frozen.",'',
        f"Every **{sum(a['hands_replayed'] for a in audits.values()):,} final hands / {sum(a['decisions_checked'] for a in audits.values()):,} actions**, plus **{sum(a['hands_replayed'] for a in pilot.values()):,} pilot hands / {sum(a['decisions_checked'] for a in pilot.values()):,} actions**, independently replay and reproduce all aggregate, lineage and position estimates and labels. Pilot arithmetic was audited only after counts froze; pilot scores were never used to size or select outcomes.",'',
        f"M1 only, free compute, at most three workers, unchanged 6-GiB worker guard and 8-GiB free-disk stop floor. Startup-inclusive final play quote **{quote['play_seconds']/60:.1f} minutes**, independent replay/report allowance **{quote['replay_report_seconds']/60:.1f}**, **{quote['with_50_percent_headroom_seconds']/60:.1f}** with 50% headroom; two-hour final cap. Completed final play: **{read(ROOT/'final-play/complete.json.gz')['seconds']/60:.2f} minutes**; final independent report/replay: **{read(ROOT/'final-audit/complete.json.gz')['seconds']/60:.2f} minutes**. All outputs, failures and partials remain compressed; no file was deleted or evicted. The owner independently freed M1 space during preparation. No M4 compute or RunPod.",'','## Export identity','',
        'All six checkpoint SHA256s and original T/O/R1 sizes and hashes match #165’s frozen manifest. Shipped R1 is `4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf`. Native release source is merged main `0a3765f34335b8ca8efea7d79e25c7fd19d94657`; relevant native/loader files match exactly. Each `cfr_average.audit` checks every checkpoint accumulator, emitted probability, visit count, stored regret/current policy and coverage.','',
        '| Export | Audited keys | Zero-mass keys | Export SHA256 |','| --- | ---: | ---: | --- |']
    for m in read(ROOT/'models.json.gz'):
        if m['arm'] not in ('Tprime','Oprime'):continue
        audit=read(ROOT/'export-audits'/(m['name']+'.audit.json.gz'))
        lines.append(f"| {m['name']} | {audit['all_nodes_verified']:,} | {audit['counts']['zero_mass']:,} | `{m['sha256']}` |")
    lines+=['','## Conditional native pressure','']
    pressure=ROOT/'pressure/run/audit.json.gz'
    if pressure.exists():
        p=read(pressure);assert p['status']=='verified'
        lines+=['A primary aggregate was better, triggering the predeclared **12,288 paired blocks** on fresh root **202610063601**, four arms, both seats and three matched seeds. #165’s native-pressure opponent, loader and arena play are unchanged; only four-arm enumeration/reporting is new. No release gate.','',
            '| Native-pressure contrast | BB/100 [95% interval] |','| --- | ---: |']
        for name,e in p['contrasts'].items():lines.append(f"| {name.replace('prime','′')} | {display(e)} |")
        effect=p['contrasts']['Tprime-T'];ci=effect['ci95']
        reading=('The fallback recovers some of T’s native-pressure loss on this new root.' if ci and ci[0]>0 else 'The fallback worsens T’s native-pressure result on this new root.' if ci and ci[1]<0 else 'This check does not detect a change in T’s native-pressure result.')
        lines+=['',reading+' The earlier ~24 BB/100 O−T difference from #165 is a different-root description; it is not a recovery target or release threshold.', '',
            f"All **{p['hands']:,} pressure hands / {p['decisions_checked']:,} actions** independently replay. Raw chips reproduce every absolute, aggregate, lineage/position estimate and street coverage count. Complete pressure detail and hashes remain in the archive."]
    else:
        assert (ROOT/'pressure-not-triggered.json.gz').exists()
        lines+=['Neither primary aggregate was labeled better, so the conditional pressure check was not triggered.']
    lines+=['','## Validation and archive','',
        'Review of #178 found no unresolved findings; both full CI shards, test aggregation and GitGuardian were green before merge. Local validation: 28 native parity/bench tests, 165 diagnostic tests, Rust tests, 16 existing measurement tests and two real-arena pressure replay tests pass. Retained attempts include an initial concurrent Git-ref refresh failure (corrected before verified merged build), native tests initially skipped before the explicit built rerun, and the small synthetic fixture corrected to the arena’s inference minimum. No poker run was repeated for an outcome.','',
        'The complete member-hashed ZIP, exact checkpoint/policy inputs, frozen source/binary/environment, plans, pilot/final raw traces, logs, failures, audits and restoration instructions are under `~/Local/Research-Cloud/PR-179-HU20-zero-mass-fallback/`. Archive/member readback receipts and hashes are indexed in [RESULTS_INDEX](../../RESULTS_INDEX.md). Originals stay; no synced files are deleted or evicted.']
    text='\n'.join(lines)+'\n';target=Path('docs/reports/hu20-zero-mass-fallback.md');target.write_text(text)
    with gzip.open(ROOT/'report.md.gz','xt') as f:f.write(text)
    comment='**M1 independent direct replay/audit complete.**\n\n'+'\n'.join(lines[4:11])+'\n\n'+direct_reading+'\n\n'
    comment+='Every final/pilot action and settlement independently replayed; nominal Student-t 95% deal-block intervals condition on the three retained lineages. No sample extension or release decision. The full lineage/position results, hashes and audit evidence are retained.\n'
    with gzip.open(ROOT/'direct-results-comment.md.gz','xt') as f:f.write(comment)
    print(text)

if __name__=='__main__':generate()
