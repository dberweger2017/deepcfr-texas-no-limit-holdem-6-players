# HU200 200M pipeline model

This owner-authorized task trains and verifies one fresh HU200 model for a first
live Slumbot baseline in a later PR. A **200M-node model is far below any release
budget and exists to exercise the pipeline**. This PR makes no Slumbot requests,
runs no external matches and makes no strength claim or release/default change.

## Declared run and acceptance

The execution source is current main
`d5c113f76efa0b4c63dc5956b278a4acbe53f718` (#247). Build with
`python -m scripts.build_native_trainer` and retain its toolchain/binary receipt.
Use seed **2026101301**, 200 BB, opponent-sampled averaging, linear weighting,
one root per seat, v1 cards and the existing min/pot/conditional-jam menu.
Run fresh to 200,000,000 requested nodes with `--milestones 100000000` and
distinct recoverable 100M/200M checkpoint paths. Complete iterations may
overshoot the requested node counts.

Before the main run, directly compare all compressed bytes of an uninterrupted
fresh 20M checkpoint with a fresh 10M checkpoint resumed to 20M, using the same
recipe and seed. Export current and average at 200M with `--zero-mass uniform`.
The existing full native-export audit in `src.diagnostics.cfr_average.audit`
must check every checkpoint and both export rows. It validates stored totals,
regrets and serialization; it does not reconstruct historical increments or
prove convergence.

Hash-verify the terminal average, load it through the existing compact
`AveragePolicy` path, require schema
`hu200-native-reopening-ordered-history-card-v1`, the HU200 game and 200 BB.
Play exactly 32 duplicate blocks /64 internal hands against `check_call`, with
translation off. Replay all engine actions/events/settlements and reproduce all
policy actions, then independently replay every hand with the native engine,
including legal menus, keys and settlements. This checks loading and play only;
no winrate or strength result is reported.

Run on the owner's Apple M1 at nice 15, using the shared M1 research lock and
psutil process enumeration with unrelated `AccessDenied` handled. Sample the
whole owned process family, including the supervisor, every 250ms plus host
query time. Soft/hard RSS limits are 6/7 GiB. Soft RSS requests a complete-
iteration training stop and blocks dependent work; hard RSS, critical/unknown
memory pressure, free memory below 15%, or disk below 16 GiB stop the owned
worker. Normal/warning pressure is allowed. Kernel command RSS peaks are retained
separately from sampled family peaks. A 60-minute local cap bounds this campaign;
there is no unattended continuation, paid compute, M4 access or other-PR work.

## Initial admission

Initial admission refused both build and training. It found only
8,872,615,936 free disk bytes after the first 5.9 GiB observation, below the
required 16 GiB. Pressure was normal, memory free 54%, and process enumeration
found no competing research worker. The owner freed disk space; execution admitted at 41,858,174,976 free bytes,
normal pressure and 52% free memory. This task removed no local or synced
evidence. The original admission refusal is preserved.

The reproducible build completed in 14.77s; native source tree
`25aafbce93956aa54cf885c7275cbe73581810e0`, binary SHA256
`8c50f67b6628a33f3fddd02a692a15ff06c9bfde82b9e6db0ba594095d32b94a`.
The [build receipt](hu200-200m-model-artifacts/build-receipt.json) records the
toolchain, linker and lockfile. All four focused HU200 tests passed in 5.14s.
The uninterrupted fresh 20M run and 10M/resume-to-20M run are directly
byte-identical: 227,890,921 bytes, SHA256
`c60977f8cff235886914377a2c8483a25af1bb894d731aaded4e60ab11c475f5`,
20,002,603 actual nodes /8,687 iterations. [Recovery receipt](hu200-200m-model-artifacts/recovery.json).
The fresh run saved both milestones, reaching **200,001,087 actual nodes /
111,844 iterations /31,582,921 entries**. [Training receipt](hu200-200m-model-artifacts/training.json).
Native elapsed was 462.15s, including 102.76s /146.36s for the 100M /200M
saves. Supervised train/save wall time was 470.47s. Sampled family peak was
2.64 GiB; kernel command peak was 2.62 GiB. Export and full verification remain
in progress.

## Closeout requirements

Final reporting will place actual nodes, iterations, entries, mean traverser
visits/key, zero-visit fraction, native traversal/save and supervised wall
times, and sampled/kernel peak RSS next to #216's 100M measurement.

The full checkpoint/export/audit/smoke/build/recovery evidence and exact campaign
helper will be packaged with an embedded member manifest under
`~/Local/Research-Cloud/PR-<n>-hu200-200m-model/`. A separate ZIP contains only the
terminal average export. Local archive/member readback, current native upload
status and independent Drive metadata acceptance are distinct checks. The model
index and `RESULTS_INDEX.md` will record bytes, SHA256, Drive locators, member
paths and retrieval commands. The local average stays in this worktree's ignored
`results/hu200-200m-20261013/` directory for the later PR, after hash verification.
No binaries enter Git and no synced files are deleted or evicted. Merge requires
completed acceptance checks, a reviewed diff and green current-head CI.
