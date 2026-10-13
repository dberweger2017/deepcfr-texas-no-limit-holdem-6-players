# First HU200 K50 model

Fresh K50 seed **2026101302** reached its declared entry-cap terminal at **752,593,175 nodes /493,496 iterations /85,373,907 entries**. Current and opponent-sampled average exports both pass every-row accumulator/current-regret audit. **1B and 3B were not reached.** No matches or playing-strength claim; Slumbot evaluation remains separate.

[PR250](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/250) · [Protocol](../hu200-k50-first-protocol.md) · [Milestones](hu200-k50-first-artifacts/milestones.json) · [Resource accounting](hu200-k50-first-artifacts/resource-summary.json)

## Recipe and recovery

Apple M4, 16 GiB /10 cores; all research through `ssh m4`, no M1 research or paid compute. Native source is current-main base `30c4a9dcd2da8e936e23d278cd912c4bac276b3b`, including #247’s two-core traversals. The path-independent build receipt and exact executable are retained. Native/game/information/regret code was unchanged by this campaign.

`hu200-native-reopening-ordered-history-equity-k50-v1`, 200 BB, 50/100-chip blinds, no rake, linear CFR weighting, opponent-sampled average, one root per seat and the existing menu. Fresh main lineage, no resume. PR163’s three tables were separately restored into ignored nonsynced storage after live merged-owner verification, with whole-archive and full table SHA256 checks. PR246’s root and uploads were unchanged.

Fresh 20M and 10M→20M recovery checkpoints were byte-identical: `4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`, 270,143,830 bytes. This gate preceded the fresh main run. [Build](hu200-k50-first-artifacts/build-receipt.json), [inputs](hu200-k50-first-artifacts/inputs.json), [recovery](hu200-k50-first-artifacts/gates-complete.json).

The predeclared cap was `floor((11,000,000,000−1,400,000,000−500,000,000)/(96.9×1.1)) = 85,373,862 entries`. The terminal overshot by **45 entries** at a complete iteration. The table grew faster than the brief’s 1.5–2.5B projection; the fixed cap stayed unchanged. The target-named `1000000000.*` files contain the **752,593,175-node terminal**, not a reached 1B checkpoint.

## Stored-key density

| Save | Actual nodes | Iterations | Entries | Mean traverser visits/key | Zero-visit keys | Nodes/s including writes | Write s | Cumulative peak family GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 500M | 500,000,140 | 321,139 | 69,084,815 | 1.4335 | 46.45% | 1,085,186 | 240.83 | 7.331 |
| Cap terminal | 752,593,175 | 493,496 | 85,373,907 | 1.7446 | 44.29% | 838,734 | 303.45 | 8.244 |

Main train/save supervision took **15.34 minutes**. Native elapsed including writes was 897.30 s; the two atomic writes total 544.27 s. Visits describe stored keys, not all possible information sets. Terminal zero-average-mass fraction is 36.20%, distinct from zero traverser visits because the average is opponent-sampled.

1B, 1.5B, 2B, 2.5B and 3B are **unreached**. The full recoverable set contains the 500M and terminal checkpoints. `export --zero-mass uniform` produced current/average at terminal; existing `src.diagnostics.cfr_average.audit` checked all **85,373,907 rows** against checkpoint accumulators and current regrets. No 1B export can exist under this stop.

## Guards and preservation

Every five-second reading is retained. RSS >13 GiB, critical pressure, free memory <8%, or swap growth >4,000,000,000 bytes require 60 continuous seconds; warning is allowed. Disk <16 GiB stops immediately. psutil enumeration handles unowned helper AccessDenied; PID creation identities prevent signaling unrelated/reused processes. Unobserved gaps cannot establish a sustained breach.

Through scientific closeout: sampled family peak **8.244 GiB**, minimum free **51%**, maximum swap growth **0.385 GB**, minimum disk **38.995 GiB**, pressure levels [1]. No guard stop. Archive/upload readings and final acceptance are added to the indexed closeout; these are sampled peaks, not kernel maxima.

Initial preflight found the attached M1 shell, a main-checkout environment without psutil and a locked M4 Git credential helper. All were corrected before the main scientific run, with no failed hands or scientific retry. [Retained setup notes](hu200-k50-first-artifacts/setup-notes.json). Five guard regressions and the repository artifact check pass; final reviewed-head CI is required before merge.

## Artifacts

Full evidence and a separate **terminal-average-only ZIP** (one model, since 1B is unreached), each with an embedded size/SHA256/member manifest and complete local readback, are stored in `~/Local/Research-Cloud/PR-250-hu200-k50-first/`. [RESULTS_INDEX](../../RESULTS_INDEX.md) and the final model index contain accepted Drive locators, archive/member hashes, retrieval commands, upload evidence and any cleanup receipts. Later Git receipts record archive/upload closure after this scientific snapshot. No remote-byte verification is implied. No binaries enter Git.

Both archives are now accepted by native upload state and independent Drive membership/ID/name/size. [Full evidence](https://drive.google.com/file/d/1Lrsw5tYJ6lQExmBJpZEQ6xILqOcsVvKW/view?usp=drivesdk) · [Terminal average](https://drive.google.com/file/d/1DMz8kmY4DIQfO-NSIdEiOu9k-tl1hzrB/view?usp=drivesdk). [Model and retrieval index](hu200-k50-first-artifacts/model-index.json) binds every required model and table. [Final resource summary](hu200-k50-first-artifacts/resource-summary-final.json) includes archive/upload operations; all readings remain within the guards.
