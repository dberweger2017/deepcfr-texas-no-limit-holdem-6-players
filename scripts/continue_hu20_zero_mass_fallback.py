"""Continue the frozen authorized phases while the agent sleeps between check-ins."""
import contextlib
import gzip
import json
from pathlib import Path
import subprocess
import time
from scripts.run_hu20_zero_mass_fallback import ROOT,FAMILIES,read,write,guard,execute,pressure_plan,pressure_run,digest

LABELS=('Primary A: T′ vs T','Primary B: O′ vs O','Secondary: T′ vs shipped R1','Secondary: O′ vs shipped R1','Secondary: T′ vs O')

def state(stage):write(ROOT/(f'controller-{stage}.json.gz'),{'stage':stage,'at':time.time(),'free_disk_bytes':guard()})
def post(name,body):
    with gzip.open(ROOT/(name+'.md.gz'),'xt') as f:f.write(body)
    guard();result=subprocess.run(['gh','pr','comment','179','--body-file','-'],input=body,text=True,capture_output=True,check=True)
    write(ROOT/(name+'-posted.json.gz'),{'url':result.stdout.strip(),'posted_at':time.time()})

def results():
    audits=[read(ROOT/'final'/f/'run/audit.json.gz') for f in FAMILIES]
    assert all(a['status']=='verified' and a['independent_arithmetic_matches'] for a in audits)
    body='**M1 independent direct replay/audit complete; frozen zero-mass fallback results.**\n\n'
    body+='| Contrast | Lineage scope | BB/100 [95% interval] | Predeclared label |\n| --- | --- | ---: | --- |\n'
    for name,a in zip(LABELS,audits,strict=True):
        for scope,e in [('three-lineage aggregate',a['primary_overall']),*[(str(2026100600+int(l)),e) for l,e in a['lineages'].items()]]:
            ci=e['ci95'];interval=f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else '[unavailable]'
            body+=f"| {name} | {scope} | {e['bb_per_100']:+.2f} {interval} | {e['label']} |\n"
    labels=[a['primary_overall']['label'] for a in audits[:2]]
    plain=[]
    for arm,label in zip(('T','O'),labels,strict=True):
        plain.append(f"{arm}'s fallback "+('improves direct play against its original export' if label=='better' else 'hurts direct play against its original export' if label=='worse' else 'has no detectable direct difference from its original export')+'.')
    pilot=[read(ROOT/'pilot'/f/'run/audit.json.gz') for f in FAMILIES]
    widths=[(e['ci95'][1]-e['ci95'][0])/2 for a in audits[:2] for e in [a['primary_overall'],*a['lineages'].values()] if e['ci95']]
    body+='\n'+' '.join(plain)+' Nondetection is not equivalence; these are direct results conditional on the three retained policies.\n\n'
    body+=f"Nominal paired Student-t 95% intervals over **36,864 independent duplicate deal blocks per family**; seats/lineages averaged inside blocks. Achieved primary aggregate/lineage half-widths **{min(widths):.2f}–{max(widths):.2f} BB/100**; no sample extension.\n\n"
    body+=f"All **{sum(a['hands_replayed'] for a in audits):,} final hands / {sum(a['decisions_checked'] for a in audits):,} actions**, plus **{sum(a['hands_replayed'] for a in pilot):,} pilot hands / {sum(a['decisions_checked'] for a in pilot):,} actions**, independently replay and reproduce aggregate/lineage/position intervals and labels. #175 runner/report/auditor hashes unchanged. M1/free only, ≤3 workers, 6-GiB guard, 8-GiB disk floor. No release decisions.\n"
    body+='\n'+('A primary aggregate is better, so the predeclared small four-arm native-pressure check follows on fresh root 202610063601.' if 'better' in labels else 'Neither primary aggregate is better, so the conditional pressure check is not triggered.')+'\n'
    post('direct-results-before-pressure',body)

def main():
    assert (ROOT/'freeze-comment-posted.json.gz').exists()
    state('final-play-start');execute('final','play')
    state('pilot-audit-start');execute('pilot','audit')
    state('final-audit-start');execute('final','audit')
    results();state('direct-audited')
    pressure_plan()
    if (ROOT/'pressure/plan.json.gz').exists():
        p=read(ROOT/'pressure/plan.json.gz')
        post('pressure-plan',f"Triggered native-pressure check frozen before its play: fresh root **{p['root']}**, **12,288 paired blocks**, T′/T/O′/O × both seats × three matched lineages = **294,912 hands**. #165 arena play/loader/opponent unchanged. Canonical plan SHA256 `{digest(p)}`. M1/free only, ≤3 workers, 6-GiB worker guard, 8-GiB disk floor, separate two-hour cap. T′−T and O′−O only; all actions/settlements and raw-chip arithmetic independently audited. No sample extension or release decision.\n")
        state('pressure-play-start');pressure_run('play');state('pressure-audit-start');pressure_run('audit')
        a=read(ROOT/'pressure/run/audit.json.gz');assert a['status']=='verified'
        body='**Independent native-pressure replay complete.**\n\n| Contrast | BB/100 [95% interval] |\n| --- | ---: |\n'
        for k,e in a['contrasts'].items():
            ci=e['ci95'];body+=f"| {k.replace('prime','′')} | {e['bb_per_100']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] |\n"
        e=a['contrasts']['Tprime-T'];ci=e['ci95']
        body+='\n'+('The fallback recovers some of T’s pressure loss on this new root.' if ci[0]>0 else 'The fallback worsens T’s pressure result on this new root.' if ci[1]<0 else 'This check detects no change in T’s pressure result.')
        body+=f" All **{a['hands']:,} hands / {a['decisions_checked']:,} actions** independently replay; raw chips reproduce every absolute, contrast, lineage/position interval and street coverage count. The earlier ~24 BB/100 O−T pressure difference was on another root; no recovery or release threshold is imposed here. No release decision.\n"
        post('pressure-results',body)
    with gzip.open(ROOT/'report-generation.log.gz','xt') as log,contextlib.redirect_stdout(log):
        from scripts.report_hu20_zero_mass_fallback import generate
        generate()
    state('research-phases-complete')

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(ROOT/(f'controller-failure-{time.time_ns()}.json.gz'),{'error':str(e),'at':time.time(),'stage_markers':[p.name for p in ROOT.glob('controller-*.gz')]})
        raise
