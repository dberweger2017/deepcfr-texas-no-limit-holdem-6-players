# HU100 4B and independent 2B direct ladders: prospective protocol

## Status and authorization

Owner authorizes this campaign on the free M4, one PR on
`feature/hu100-4b-seed-ladder`. Isolated checkout
`~/Local/hu100-4b-seed-ladder-20261010`, current-main base
`ff984da9a16bf1e76e891fab239d0785188a93af` (#224 streamed exports).
Initial read-only qualification finds insufficient disk; no training, pilot
or final hand has started. [Admission receipt](reports/hu100-4b-seed-ladder-artifacts/storage-admission.json).
This is preventive admission, not a guard breach or failed-science retry.
Keep the PR draft until storage admits the campaign. No roadmap result entry
or claim of completed science is appropriate yet.

Required prior reads: AGENTS, ROADMAP, artifact storage, development, rules,
#223 report/protocol, #215 independent stages, #207 growth, compact table and
streamed-export reports. Native trainer source must remain unchanged. A trainer
defect stops the campaign and is reported on the PR; no local trainer fix.

## Sequential training and exact gates

Train one process at a time, fresh from each seed, in order 2026100601,
2026100901, 2026100902. Recipe:
`train --stack-bb 100 --seed S --roots-per-seat 1 --average-rule opponent-sampled`,
with telemetry, unchanged linear CFR, v1 abstraction, native menu and uniform
missing/zero-mass fallback. First complete iteration at/past each target.
Continue only this campaign's fresh state between milestones; do not retrieve
archived state to resume.

Seed 2026100601: 1B gate, 2B gate, then 4B. Seeds 2026100901 and 2026100902:
1B gate then 2B. Gate checkpoint SHA256s:

- 2026100601 /1B: `cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec`.
- 2026100601 /2B: `e84039c7a934a966a2c01f237748a23951124a8b665edc2585ef4809e5681676`.
- 2026100901 /1B: `5f4898d839011d62fda20e3df7e1546150ca1d72dd666654a6d4bfcaaf63a6e7`.
- 2026100902 /1B: `c21e6dd51694b379b28cb5dcc2ab595fa154092f1b69a4cfd1a9149fc85b23e4`.

Use reference `--max-entries 57658644` through 1B, and reference 67419934
at original-seed 2B. Before 4B or independent-seed 2B, set capacity to
floor((7 GiB -200,000,000 bytes)/101 bytes per entry) = **72,437,552 entries**.
#223's guard receipt identifies its 6.08 GiB peak as *final direct evaluation*;
train/save at2B peaks at4,799,266,816 bytes for54,626,283 entries,
**87.86 B/entry**. The101 B/entry bound provides15% headroom over that entire
operation peak, plus200 MB for the added controller and fixed workspace.
It also exceeds #207's largest measured97.8 B/entry once the fixed reserve
is included. This distinguishes training capacity from match schedule memory. Reassess whether the
reference 2B cap is safe before launch using its *actual endpoint*, 54.63M
entries; that reference gate target terminates below the cap.

This conservative ceiling may stop original-seed training before 4B. A clean
entry-cap stop saves at a complete iteration and its terminal is evaluated
against 2B. No raising the ceiling to reach a desired outcome. If the gate
cannot safely be run with its reference cap, use the conservative cap and
verify identical decompressed rows and a header equal except for declared
`config.max_entries`. Whole-file mismatches are not accepted without this
complete comparison against hash-verified reference bytes. Stop on any
other mismatch; do not retry science.

Stream-export current and average, uniform zero-mass, and fully audit every
evaluated checkpoint. All gate export hashes are checked against indexed
references where available. #207/#215/#223 archives remain restoration
dependencies; do not duplicate their indexed models in this campaign ZIP.

## Guards and admission

Whole-family RSS soft7 GiB/hard9 GiB, total swap <=3,000,000,000 bytes,
normal pressure, >=15% reported system free, free disk >=16 GiB on each
volume storing campaign outputs, and AC power. Use checked macOS ancestry
and creation-verified owned children; an unreadable owned process fails closed.
Native entry-cap stop is the preventive memory endpoint. Resource soft-stop
support must be verified before reliance; no ad hoc killing as a valid save.
Any actual guard breach, exactness mismatch, information leak, invalid action,
numerical failure or accounting error stops and is reported on the PR with
a SOMA. No scientific retry.

Measure disk before every step. Reserve retained originals, outputs/staging,
raw primary and reproduction traces, guard logs and immutable ZIP copies;
count hardlinks once, exclude hash-verified indexed models from ZIPs, and
reserve no sort workspace for strictly sorted checkpoints under #224.
No cleanup of other PRs, synced deletion or force offloading.
Initial historical-size estimate is an admission forecast, not a timing quote.

## Fixed direct evaluation

Use unchanged #223 `scripts.evaluate_hu100_direct`:
100 BB reset each hand, blinds50/100, no rake, each deal played twice with
seats swapped, private policy action streams, native menu only, translation
off. Policies receive legal observations only. Fully independently replay
and deterministically reproduce every final hand/action/probability/key/menu/
settlement; report decision coverage and traverser-visit bands by policy/street.

Primary family, Bonferroni FWER .05 over three two-sided paired Student-t
intervals (confidence 1-.05/3 =98.333333%); each contrast improves only when
its own lower bound >0:

1. 2026100601: 4B or clean capacity terminal versus2B.
2. 2026100901: 2B versus1B.
3. 2026100902: 2B versus1B.

If original seed stops at2B without further training, its self-comparison
cannot establish improvement; explicitly report that limitation.

Descriptive ordinary95%: 2B2026100601 versus2B2026100901 and versus2B2026100902.
Display independent-seed 2B-minus-1B gains alongside #223's +29.51
[26.82,32.20] BB/100, identifying its earlier schedule and nominal95% interval.
Do not pool hands across seeds or interpret conditional deal intervals as
training-seed population intervals.

Proposed distinct timing root202610104401 and final root202610104402, with
a distinct scenario name for each contrast; scripted pilot/final
202610104403/202610104404 if admitted. These are reservations only: verify
physical deal disjointness against *every* earlier HU100 schedule, including
#223 and later merged/current relevant campaigns, and between new contrasts,
before any pilot/final play.

Timing-only pilot32 blocks/contrast, full loading/play/replay/reproduction;
inspect costs and sizes only, never winnings/variance. Target primary
half-width about3 BB/100. Historical #223 largest95% half-width3.01 at524,288
blocks gives planning SD about1,113 BB/100. Adjusted planning requirement
ceil((2.394*1113/3)^2) is about789k blocks; propose1,048,576 blocks/contrast.
Freeze actual sample for each contrast from measured pilot costs, memory and
storage before any final hand. Contrasts may have different frozen counts.
For schedule/replay memory, #223's terminal pilot-to-final whole-family
growth is4,807 B/block; reserve5,200 B/block plus100 MB over each new full-pair
pilot peak. The2 KiB/block planning allowance previously used by #223
underestimated its measured final growth. Preserve primary precision before
reducing descriptive samples for disk admission. No extensions, pooling, checkpoint selection or outcome-driven scope
change. Any prospectively smaller admitted sample must disclose its projected
precision limitation.

## Quote, review and preservation

Before final play, post on the PR the measured pilot-based total quote:
training/loading/save/export/audit, direct play/replay/reproduction, descriptive
matches, optional scripted panel, packing/readback and proportionate closeout.
Use measurement for the budget. If full scope looks well over about10h,
finish training plus the primary family first, then descriptive matches.
The optional scripted panel is omitted from this campaign's fixed scope so
training and all five direct contrasts take priority. Its authorized optional
settings would be #215 translation512 states/128 events, 4,096 duplicate
blocks/opponent, descriptive only. No optional play will be added after outcomes.

One independent source review before final play and one end evidence review.
No extra metadata seals. ZIP into
`~/Local/Research-Cloud/PR-<n>-hu100-4b-seed-ladder/`, with member-hashed
manifest, local readback, failures/partials and source/retrieval provenance.
Record actual Drive folder/archive IDs, bytes, hashes, manifest and separate
native upload/cloud acceptance in RESULTS_INDEX per artifact-storage rules.
Keep originals; never put models in Git.

Stage intended files and run `python -m scripts.check_repository_artifacts`
before each commit. Commit/push completed subtasks. PR comments only for issues,
each with one immediate SOMA; normal quote/status belongs in the PR body.
Merge only with green checks and no open findings; on landing add one short
Current position entry and update v0.5 groundwork. No recipe/menu/abstraction
change, paid compute, release, tag or publication.
