# HU20 average-policy play and turn/river search

Owner-approved protocol, October 2, 2026. No outcomes have been generated for this task.

## Sequencing and resources

M1 performs development, tests and smokes lasting a few minutes. M4 remains reserved for #145: admit this task only after its main run finishes, final report is pushed, owned solver processes exit, and no follow-up claims the machine. During active implementation check #145 every 30 minutes. Ambiguous ownership requires owner clarification. No unattended monitor is created.

The new task has 24 cumulative M4 compute hours including pilots, failed attempts and verification. Part A has at most six. At admission record reclaimable memory, owned process-family RSS and swap baseline; RSS must fit 80% of headroom and never exceed 10 GiB. Stop at swap growth above 1 GiB. Preserve Ollama and unrelated services. No rental, training, promotion, automatic merge or destructive cleanup.

## Part A

[Planned inputs and counts](../configs/diagnostics/hu20-turn-search-part-a.json) reuse the six #141 exports and all 13 opponent definitions: 2,048 paired blocks each for original-cap2 bounded LBR and native pressure, 256 for every other panel. Three lineages, both positions and current/average yield 82,944 hands. Root 202610020801 is reserved; pilot 202610020802 and arena 202610020803 are disjoint. Verify against prior manifests before admission.

Prior primary SD is 317.8 BB/100 at the independent shared-deal-block level. Target 80% power for 20 BB/100 at two-sided alpha .05; the normal approximation needs 1,982 blocks. Compute noncentral-t power for the actual frozen count. A separate timing pilot excludes payoffs from sizing. Freeze counts before final outcomes; use 1.5x timing headroom and the largest balanced primary count fitting six hours if 2,048 does not fit. Do not extend after outcomes.

Average is the default. Switch all lineages to current only when the paired 95% upper bound of average-minus-current is below -10 BB/100 for LBR, -20 for native pressure, or -10 for selective-stackoff. Average lineage contrasts within shared deal blocks before Student-t intervals. Publish all outcomes, seed/position details and decision before arena execution. Zero average mass remains uniform and separate from missing-key fallback.

## Search and range law

Blueprint play ends after the flop. Solve exact cards from the start of turn/river betting. Construct public ranges from the base policy and public history, never actual rival cards/deck/seeds. For opponent action factors compare epsilon 0 and .01 using max(likelihood,epsilon); normalize after conditioning. Own factors stay unchanged, incompatible cards stay impossible, and the existing off-menu likelihood kernel remains declared separately. This is a robustness model, not a true opponent posterior or a change to played blueprint probabilities.

Retain current-round full holding matrices on-tree. Insert exact off-menu sizes and re-solve, locking prior hero matrices, including hypothetical holdings. Newly inserted actions have zero probability at locked nodes. River ranges condition on actual turn matrices/actions, followed by a fresh river-root solve. Hypothetical LBR queries reconstruct public policy state and cannot mutate live state. One public solve serves all queried holdings. Cache identities include bot seat, public root, model/range law, work, menu and locks.

Prefer the native menu so cap2 LBR alternatives are on-tree. Reduced-menu tradeoffs must include extra speculative solves. Use original-cap2/K4/five-soft-second LBR and retain actual batch completion, preparation overruns and zero-likelihood telemetry. A 30-second end-to-end watchdog falls back to the base policy; never publish partial profiles or execute illegal actions. Count failures and zero support by cause.

## Calibration and validation

[Calibration settings](../configs/diagnostics/hu20-turn-search-calibration.json) freeze the candidate axes. Stage native candidates first; retain every attempted cell and failure. Quality is the turn strategy with its internal river continuation in the full native tree, against #145 original reference ranges; also report own-law sensitivity. No requirement to solve every river runout. Validate 32 outcome-blind sampled river roots separately across eight strata, both seats and three lineages.

Select fastest native cold end-to-end p95 among settings with p95 latency <=30 s and weighted mean/p95 residual <=.5% pot. Break ties by median then epsilon .01. Only if none qualifies use mean <=1%, p95 <=2%, and residual <10% of blueprint e_bp at every eligible root. Ask the owner only if that fails. Report the entire curve, every root, unavailable reference and exclusion. Never exclude failed runtime behavior from the qualification denominator.

Pin external upstream 9d1509fe5077d019825f833eed04b16d342dfda1. AGPL code/harness remain outside this MIT repository; preserve the original #145 binary. Fingerprint each new external source/build. M1 may cross-compile x86_64 Linux (cargo-zigbuild) or use short native ARM Linux checks. No emulated x86 builds/runs. Otherwise build x86 Linux on the approved RunPod pod and run macOS/Linux parity there before production. Cross-compilation is not runtime parity. Freeze tolerances and record thread-level float differences. Normalize ru_maxrss bytes on macOS versus KiB on Linux.

## Arena and publication

Assume a RunPod arena. Publish Part A and calibration before issuing a live quote. Forecast actual decisions, turn reach, LBR hypothetical lines, river solves, startup/build/parity, verification, storage and shutdown reserve. Use independent workers. No paid allocation before owner-approved quote; do not shrink useful scientific counts to fit M4. Freeze fresh base-only versus base+search paired schedules on all 13 panels/three lineages with the planned Part A counts.

Retain complete/partial hands, every failure and input/source/output hashes. Independently native-replay observations, actions and settlements. Provide status.md and TensorBoard sidecar; PR milestones Part A, chosen curve, arena 25/50/100%, final report. Report full tails, street coverage, latency and fallback causes. Ending-street losses are descriptive, not causal attribution. Update ROADMAP and RESULTS_INDEX. Shipping a playable AGPL-dependent bot remains an owner licensing decision.
