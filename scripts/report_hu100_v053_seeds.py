"""Write the short three-seed qualification report from verified receipts only."""
import argparse
import json
from pathlib import Path
from time import time

from scripts import evaluate_hu100_direct as direct
from scripts.report_m4_resources import summarize
from scripts.train_hu100_v053_seeds import BASE,ROOT,SEEDS,read


def interval(row):
    return f"{row['bb_per_100']:+.2f} [{row['ci95'][0]:+.2f}, {row['ci95'][1]:+.2f}]"


def table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',
        *['| '+' | '.join(map(str,row))+' |' for row in rows]])+'\n'


def render():
    summary=read(BASE/'summary.json');plan=read(BASE/'canonical-plan.json')
    if not summary['all_final_replayed_and_reproduced']:
        raise ValueError('No report before the global verification barrier')
    old=read(ROOT/'docs/reports/hu100-v2-frozen-5b-artifacts/summary.json')
    output=ROOT/'docs/reports/hu100-v053-three-seeds.md'
    assets=ROOT/'docs/reports/hu100-v053-three-seeds-artifacts';assets.mkdir(exist_ok=True)
    resources=summarize(BASE/'guards')
    (assets/'resource-summary.json').write_text(json.dumps(resources,indent=2,sort_keys=True)+'\n')
    parts=['# HU100 frozen menu-v2: three-seed v0.5.3 qualification\n',
        '**Training and internal evaluation only. The owner decides whether this is the better overall package and whether to release or merge.**\n',
        'Two fresh training seeds repeat #260 → #272. Three fresh direct matches use released v0.5.2 with translation **on (512/128)**, and v2 with translation **off**. These are reported facts, with no improvement-release pass/fail gate. No tag, release, default change, Slumbot or credential work.\n',
        'Intervals are nominal paired Student-t 95%, over independent duplicate blocks for each fixed policy pair. They do not estimate a training-seed population or external playing strength.\n',
        '## Exact lineages and audits\n']
    rows=[['2026100601 (#272)',332894788,104400,97290578,5000006023,1794658]]
    for seed in SEEDS:
        first=read(BASE/str(seed)/'stage1-verified.json');final=read(BASE/str(seed)/'training-verified.json')
        a=first['header'];b=final['header']
        rows.append([seed,a['native_state']['completed_nodes'],a['iteration'],first['keys']['entries'],b['native_state']['completed_nodes'],b['iteration']])
    parts += [table(['Seed','Stage-1 actual nodes','Stage-1 iterations','Frozen entries','Final actual nodes','Final iterations'],rows),
        'Stage 1 copies #260’s **97,290,497** nominal cap, 500M milestone and 1B target, ending at its first complete-iteration capacity stop. Filenames identify targets; the table records actual work. Stage 2 hash-pins each seed’s own terminal, passes its actual entries to `--max-entries`, enables `--freeze-at-cap` and targets 5B. Every final sorted key list matches its stage-1 list by count and streaming SHA256.\n',
        'HU100, two 10,000-chip stacks, 50/100 blinds, no ante/rake/raise cap; linear CFR, opponent-sampled average, one root per seat, ordered history, menu v2 and #163’s unchanged hash-pinned K50 tables. Current-main native build is byte-identical to #272 (`4f6abcbb954c3c744448c2a3542ab3164e0e1a015cb05ecebeda077525991cd4`).\n']
    density=[];missing=[]
    for seed in SEEDS:
        first=read(BASE/str(seed)/'stage1-verified.json')
        final=read(BASE/str(seed)/'training-verified.json')
        audited=read(BASE/str(seed)/'5b.audit.json')
        for stage,row in [('capacity terminal',first['telemetry'][-1]),('frozen 5B',final['telemetry'][-1])]:
            d=row['diagnostics'];n=d['entries']
            density.append([seed,stage,f"{d['traverser_visits']/n:.6f}",f"{100*d['visits_histogram']['0']/n:.4f}%",f"{100*d['zero_average_mass_keys']/n:.4f}%"])
        f=final['telemetry'][-1]['frozen_table']
        for i,street in enumerate(f['streets']):
            missing.append([seed,street,f['traverser_decisions_by_street'][i],f['missing_traverser_by_street'][i],
                f"{100*f['missing_traverser_share_by_street'][i]:.2f}%",f['opponent_decisions_by_street'][i],
                f['missing_opponent_by_street'][i],f"{100*f['missing_opponent_share_by_street'][i]:.2f}%"])
        state=final['header']['frozen_table']
        parts.append(f"Seed **{seed}**: all **{audited['all_nodes_verified']:,}** current/average rows pass `src.diagnostics.cfr_average.audit`, with `--zero-mass uniform`. Discarded updates **{state['discarded_updates']:,}**; discarded traverser visits **{state['discarded_traverser_visits']:,}**. [Audit]({assets.name}/{seed}-5b.audit.json), [lineage and membership]({assets.name}/{seed}-training-verified.json).\n")
    parts += [table(['Seed','Save','Stored visits/key','Zero visits','Zero average mass'],density),
        table(['Seed','Street','Traverser decisions','Missing','Missing %','Opponent decisions','Missing','Missing %'],missing),
        'Missing decisions cover the frozen counterfactual training interval, separately for the traverser and opponent roles. Retained keys stay fixed; absent keys play uniformly. Stored-key density is not full-game coverage or a convergence certificate.\n',
        '## Primary scripted panels beside #272\n',
        f"Each new seed uses **{plan['counts']['panel']:,} blocks per opponent**, paired frozen-v2 off minus released-v0.5.2 translation-on. #272’s original column uses its original 16,384 blocks; those hands are not reused in either new panel.\n"]
    rows=[]
    for opponent in direct.PANEL_OPPONENTS:
        rows.append([opponent,interval(old['primary'][opponent]['contrast']),
            *[interval(summary['panels'][str(seed)]['opponents'][opponent]['contrast']) for seed in SEEDS]])
    parts.append(table(['Opponent','2026100601 (#272)','2026101501','2026101502'],rows))
    full=[]
    for seed in SEEDS:
        panel=summary['panels'][str(seed)]
        for op,row in panel['opponents'].items():
            full.append([seed,op,interval(row['arms']['candidate']),interval(row['arms']['baseline']),interval(row['contrast'])])
        flags=[op for op,v in panel['historical_upper_below_minus20_flags'].items() if v]
        parts.append(f"Seed **{seed}**, #272 historical rule: **{'meets' if panel['historical_pr272_rule_outcome'] else 'does not meet'}**. Upper <−20 flags: **{', '.join(flags) or 'none'}**. The rule requires no primary upper bound below −20 and pot-pressure’s lower bound >−20 BB/100; it uses unrounded limits. #272 met it. This comparison is not a release gate; no flag does not establish 20 BB/100 non-inferiority.\n")
    parts += [table(['Seed','Opponent','Frozen v2 [95%]','v0.5.2 on [95%]','Paired difference [95%]'],full),
        '## Fresh fair direct matches\n',
        f"Each match uses **{plan['counts']['direct']:,} fresh duplicate blocks**. v0.5.2’s translation is **on**, 512 states/128 events; menu-v2 translation is unsupported and stays off. Each side owns its schema. Results are descriptive and never gated. #272’s old both-off direct result is not pooled.\n",
        table(['Training seed','Fresh frozen v2 − released v0.5.2 on, BB/100 [95%]'],
              [[seed,interval(summary['direct'][str(seed)])] for seed in (2026100601,*SEEDS)])]
    coverage=[]
    for seed in (2026100601,*SEEDS):
        for key,row in sorted(summary['coverage'][f'direct-{seed}'].items()):
            counts=row['lookup'];total=row['decisions'];miss=counts.get('missing-key',0)
            coverage.append([seed,key,total,row['native_lookup'].get('missing-key',0),miss,f'{100*miss/total:.2f}%',counts.get('zero-mass',0),counts.get('positive-mass-known-key',0),row['translation_modes'].get('translated',0)])
    parts += [table(['Seed','Policy/street','Decisions','Native missing','Effective missing','Effective missing %','Zero mass','Positive mass','Translated'],coverage),
        'Coverage counts the actual key selected after translation, with exact native missing counts retained separately. An exact-native miss can be covered by a positive-mass translation witness; its visits belong to that selected key. Missing-key and zero-mass fallbacks remain part of the policy package; this table does not isolate a causal translation or menu effect. Full reached visit bands and all panel coverage are in the compact summary.\n',
        '## Costs, prospective declaration and verification\n']
    counts=plan['counts']
    final_costs={label:read(BASE/'final'/label/'costs.json') for label in read(BASE/'all-final-verified.json')['labels']}
    fixed=sum(c['fixed_seconds'] for c in final_costs.values());work=sum(c['block_work_seconds'] for c in final_costs.values())
    hands=sum(c['primary']['hands'] for c in final_costs.values())
    parts.append(f"Canonical [plan]({assets.name}/canonical-plan.json) and [PR-description readback]({assets.name}/plan-posted.json) precede every final hand. Whole prospective cost **{counts['projected_whole_seconds']/3600:.3f} h**, including elapsed training/preparation/pilots, **{counts['fixed_seconds']/60:.2f} min** fixed evaluation admission, **{counts['scaled_block_seconds']/60:.2f} min** per-block work, **{counts['additional_preparation_seconds']/60:.2f} min** fixed future cache preparation, 25% future-work margin and 30-minute closeout reserve. Only per-block work scales. Actual final evaluation: **{fixed/60:.2f} min fixed + {work/60:.2f} min per-block**.\n")
    parts.append(f"All **{hands:,} original final hands** independently replay and exactly reproduce before any readout. Reproduction adds no independent sample. Cost-only pilots are physically disjoint and excluded. No failed-hand rerun, outcome-driven extension or scientific retry. The original 14-hour deadline remains fixed. [Full costs]({assets.name}/final-costs.json), [global barrier]({assets.name}/all-final-verified.json).\n")
    r=resources['all_operations']
    parts.append(f"M4 only, detached tmux, both locks. All **{r['samples']:,} five-second readings** retained. Sampled peak family RSS **{r['peak_family_rss_bytes']/2**30:.3f} GiB**; minimum free **{r['minimum_free_percent']}%**; minimum disk **{r['minimum_disk_free_bytes']/2**30:.3f} GiB**; maximum fixed-baseline swap growth **{r['maximum_swap_growth_bytes']/1e9:.3f} GB**; pressure levels **{r['pressure_levels']}**. Guard stopped: **{resources['stopped']}**. Same 60s RSS/critical/free/swap and immediate disk rules as the protocol. [Resource summary]({assets.name}/resource-summary.json).\n")
    failures=read(BASE/'setup-failures.json')
    parts += ['## Failures and evidence\n',*['- '+row['failure']+'\n' for row in failures],
        '\nThese setup/metadata/unit-test corrections precede final evaluation; their original receipts/logs survive. Every additional failure or stop is included in the archive/index, rather than replaced.\n',
        'Per-seed raw evidence and separate average-only ZIPs use embedded size/SHA256 manifests and complete local member readback. Shared evidence retains every closed guard reading, original-seed direct traces, source/build receipts, costs, prospective declarations and all failures. [RESULTS_INDEX](../../RESULTS_INDEX.md) records exact hashes, Drive locators, restoration members and commands. Local archive integrity and upload/cloud acceptance are reported separately. Originals remain; no synced deletion or forced offload.\n',
        'No release, tag, default change or merge. The owner decides.\n']
    output.write_text('\n'.join(parts))
    exports={'summary.json':summary,'canonical-plan.json':plan,'plan-posted.json':read(BASE/'plan-posted.json'),
        'all-final-verified.json':read(BASE/'all-final-verified.json'),'final-costs.json':final_costs,
        'build-receipt.json':read(BASE/'build-receipt.json'),'first-build.json':read(BASE/'first-build.json'),
        'budget.json':read(BASE/'budget.json'),'evaluation-preparation.json':read(BASE/'evaluation-preparation.json'),
        'setup-failures.json':failures,'evaluation-inputs.json':read(BASE/'evaluation-inputs.json')}
    for seed in SEEDS:
        for name in ('stage1-verified.json','training-verified.json','5b.audit.json','treatment.json','superseded-checkpoints.json','archive-receipt.json','average-archive-receipt.json'):
            path=BASE/str(seed)/name
            if path.exists():exports[f'{seed}-{name}']=read(path)
    for name,value in exports.items():
        (assets/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    print(render())
