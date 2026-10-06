# CFR+ floor control match

The exact #165 no-floor **T** (1B-node traverser-reach average) and **O** (1B-node opponent-sampled average) exports play shipped v0.4.0 **R1**. The exact **0.4.0-shield** exports also play T directly with matching training seeds, controlling node budget and extraction to isolate the floor recipe. These are direct opponents, rather than differences against a shared scripted rival. [Protocol](../hu20-floor-control.md), [predeclaration](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/175#issuecomment-6013362287), [outcome-blind sizing and frozen hash](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/175#issuecomment-6013456971).

Positive BB/100 favors the first policy. Labels were declared before the pilot: **better** if lower >0, **worse** if upper <0, **no detectable difference** otherwise. Student-t 95% intervals use **49,152 independent duplicate deal blocks per family**; both swapped seats and the three lineages are averaged within each block. Each lineage interval also uses paired deal blocks. Intervals are nominal, without multiplicity adjustment, and conditional on these three training lineages.

| Contrast | Lineage scope | BB/100 [95% interval] | Predeclared label |
| --- | --- | ---: | --- |
| Primary A: T vs R1 | three-lineage aggregate | +8.12 [+5.50, +10.73] | better |
| Primary A: T vs R1 | seed 2026100601 | +8.51 [+5.20, +11.82] | better |
| Primary A: T vs R1 | seed 2026100602 | +8.20 [+4.89, +11.50] | better |
| Primary A: T vs R1 | seed 2026100603 | +7.65 [+4.35, +10.95] | better |
| Primary B: shield vs T | three-lineage aggregate | -14.26 [-16.90, -11.62] | worse |
| Primary B: shield vs T | seed 2026100601 | -16.20 [-19.54, -12.85] | worse |
| Primary B: shield vs T | seed 2026100602 | -14.33 [-17.70, -10.96] | worse |
| Primary B: shield vs T | seed 2026100603 | -12.26 [-15.60, -8.91] | worse |
| Secondary: O vs R1 | three-lineage aggregate | +11.45 [+8.83, +14.07] | better |
| Secondary: O vs R1 | seed 2026100601 | +11.78 [+8.49, +15.07] | better |
| Secondary: O vs R1 | seed 2026100602 | +11.82 [+8.55, +15.09] | better |
| Secondary: O vs R1 | seed 2026100603 | +10.75 [+7.47, +14.02] | better |

For these tested policies, the floor explains shield’s loss: T beats R1, while shield loses to matched-seed T. Both no-floor 1B averages beat R1, so the shared budget/averaging change is not sufficient to explain the loss.

Pilot root **202610061501** is excluded from the final root **202610061601**. The pilot uses 2,048 blocks per pairing; only SDs, costs, completeness and failures were inspected before freezing. Final counts, roots and canonical bundled plan hash `7e1cf06d29b2988faf9c262f7c81bdf7c12c4c5e2b59c4da702046e9d4779036` were posted before final play, and stayed fixed. Primary interval half-widths are **2.62–3.37 BB/100**, meeting ≤4 for both aggregates and all six individual primary pairings. No outcome-driven extension or checkpoint selection.

The original #173 direct runner, policy loader, observation boundary, paired reporter and independent auditor are unchanged from execution source `8bbfc457`. Every policy is verified by size, SHA256, training seed, iteration, schema and extraction/checkpoint provenance before play. Shield retains its original archived O alias; folder prefixes keep it distinct from #165 O. No training or policy re-extraction occurs. Both players receive only their own observation and separate deterministic action RNGs.

All **884,736 final hands / 4,397,269 actions**, plus **36,864 pilot hands / 183,756 actions**, pass native replay during play and the independent action/settlement replay. Raw chips independently reproduce every aggregate, lineage and position estimate, interval and label. Every final coordinate, deal seed, actor, own hole cards, board, pot, legal action, settlement, chip conservation and public event hash is checked. All nine final and nine pilot worker results are complete, with no play/audit failure. A pre-play root search first encountered a historical broken symlink; the preserved correction records unreadable paths and continues the readable-plan search without changing play.

All play, reporting and independent replay use the M4 with three workers and free local compute. No RunPod or rental. The frozen guards are a two-hour aggregate deadline, 6-GiB per-worker RSS limit and ≥15-GiB free disk. The pilot quote was 25.58 minutes with 50% headroom, including play and replay/report. Final play took **7.93 minutes**; the elapsed time from the frozen final admission through play, reporting and all pilot/final audits was **10.46 minutes**. Actual worker peaks and final resource snapshots are retained in the research archive.

The measured differences describe these fixed trained policies and matchups. A nondetection is not equivalence, and the T-versus-R1 control cannot separate the 10× training budget from average-versus-current extraction. Training uses equal node budgets and matching initial seeds; the floor can change trajectories and completed iteration counts. No release decision, tag or model promotion follows.

The member-verified research ZIP is `~/Local/Research-Cloud/PR-175-HU20-floor-control/hu20-floor-control-complete-20261006.zip`. Its embedded `ARCHIVE-MANIFEST.json` pins every research member by size and SHA256; every member is checked by ZIP readback. The snapshot includes all exact inputs, source, environment/native binary, pilot/final plans, raw traces, logs, audits, preparation failures, report and restoration instructions. Independent raw-trace hashes match the M4 auditor after retrieval; all ten policy copies still match the frozen manifests. Archive creation/verification receipts remain beside the ZIP as closeout metadata. Originals stay on both hosts; no synced file is deleted or evicted. [Research index and restoration](../../RESULTS_INDEX.md).

Verified main ZIP: **147 members / 2,100,003,481 logical bytes**, ZIP **2,018,260,354 bytes**. SHA256 `f350afc81cb8090d24ca125413df8d58259e1d1787912193d13de291904a6962`; embedded manifest SHA256 `d7a4aaa73f704bdd586f32bb969d10bc88887a31647918c383e66435724af16f`. All members pass readback. Final publication/validation/upload receipts and updated report metadata are retained in a separate member-hashed closeout ZIP in the same folder.
