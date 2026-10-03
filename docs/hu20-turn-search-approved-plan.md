# HU20 average-policy play and turn/river search

Copy of the owner's approved implementation plan. Saved October 2, 2026 for the recurring continuation in draft PR #148. The operational protocol is [hu20-turn-search-protocol.md](hu20-turn-search-protocol.md); current progress and evidence are in [reports/hu20-turn-search.md](reports/hu20-turn-search.md) and `/Users/dberweger/Local/hu20-turn-search-20261002/status.md`.

## Summary and fixed decisions

Create `feature/hu20-turn-search` and one draft PR. Complete Part A, calibrate turn search, validate sampled river re-solving, then publish a RunPod arena quote.

Use #136 and #141–#146 as evidence and implementation precedents. Use #145’s final report once published. Stored average is the default base. Allow 24 cumulative M4 compute hours, with at most six for Part A. Expect the arena to require RunPod.

No training, promotion, automatic merge, emulated x86-64 work on M1, or paid compute without an approved quote.

## Part A: average versus current

- Reuse #141’s three hash-pinned B500M current/average pairs, extraction semantics and all 13 separate opponent panels.
- Plan 2,048 paired blocks each for original-cap2 bounded LBR and native pressure; 256 per remaining panel. Each block covers both policies, both positions and three lineages.
- Size primary LBR for 80% power at 20 BB/100, two-sided 5%, using prior paired block variance. Run a separate timing pilot; if six hours cannot accommodate the target, freeze the largest balanced primary sample that fits and report expected power.
- Freeze schedules, counts, hashes and analysis before outcomes. Average lineage contrasts within shared deal blocks before calculating paired 95% Student-t intervals.
- Retain average unless an aggregate average-minus-current paired 95% upper bound falls below −10 BB/100 for LBR, −20 for native pressure, or −10 for selective-stackoff.
- Preserve zero-average-mass uniform handling separately from missing keys. Publish all panels, lineages and positions, plus the base-policy decision, before the search arena.

## Player, ranges and solver

- Add an observation-only `distribution(observation)` policy and action-sampling wrapper. Configuration identifies artifacts, executable, menu, range law, fixed work, threads, compression, caches and deadline.
- Pin upstream commit `9d1509fe5077d019825f833eed04b16d342dfda1`. Keep AGPL code and the fingerprinted harness outside the MIT repository; preserve #145’s original binary.
- Add a play mode that omits diagnostic best-response work and exports current-round matrices over all holdings. Turn solving retains internal river continuation without dumping every river runout.
- Play the base blueprint preflop/flop. Construct turn ranges from public history and the base policy.
- Freeze opponent-action likelihood variants ε=0 and ε=0.01: use `max(observed_likelihood, ε)` and renormalize compatible ranges. Apply this only to search’s opponent range factors; preserve own-action likelihoods, card incompatibility zeros and the declared off-menu kernel.
- Solve from the round’s beginning. Retain on-tree profiles; insert exact off-menu opponent wagers and re-solve. Freeze previously used hero matrices across holdings, assigning zero prior probability to newly inserted actions.
- At the river root, condition ranges on observed turn actions using the matrices actually used, then re-solve.
- Cache full matrices by public history, bot seat, artifact/range identities, settings, inserted actions and prior hero locks. Neither hypothetical holdings nor action RNG affect solve construction.
- Search-aware LBR reads holdings from the same cached solution. Additional public lines trigger additional solves; hypothetical queries never mutate live state.
- Prefer the native menu, which contains LBR’s cap2 alternatives. Include additional LBR solves in any reduced-menu comparison.
- Enforce a 30-second end-to-end decision budget. On startup failure, timeout, resource refusal, unsupported support/holding or invalid output, return the legal base distribution. Retain failures and cause-specific zero-support/fallback counters.

## Calibration, Linux preparation and arena

- During M1 development, prepare reproducible Linux build instructions and fixtures. Cross-compile for x86_64 Linux using cargo-zigbuild if practical; otherwise build on RunPod at the beginning of the approved paid run. **Never build or run through x86-64 emulation on M1.**
- Short native ARM Linux checks in Colima are permitted. Cross-compilation alone does not establish runtime parity.
- Record source, lockfile, compiler, architecture and executable hashes. Normalize RSS units correctly on macOS and Linux.
- Freeze a staged curve over native/reduced menus, 1/2/4/6 threads, compression, fixed iterations and both likelihood-floor variants. Include preparation, parsing and speculative LBR solves.
- Gate quality on the exported **turn strategy with internal river continuation**, evaluated against full-native-tree deviations. Evaluate against #145’s original reference law and report sensitivity under the declared search range laws.
- Select the **fastest native-menu configuration meeting cold decision p95 ≤30 seconds and weighted mean/p95 residual ≤0.5% pot**. Rank speed by cold end-to-end p95, then median; use ε=0.01 as the remaining tie-breaker.
- Use the relaxed gate only if no native configuration meets the strict gate: mean ≤1%, p95 ≤2%, and residual below 10% of blueprint `e_bp` on every eligible root. Ask the owner only if this also fails.
- Publish the full curve, every root, exclusions, missing references, support gaps and failures regardless of which gate qualifies.
- Separately validate river re-solving on 32 outcome-blind sampled roots covering eight strata, both bot seats and three lineages. Check conditioning, legality, latency and full-native river quality; do not evaluate every runout.
- Publish Part A and calibration results before arena execution; issue the RunPod quote immediately after calibration.
- Freeze fresh paired base-only versus base-plus-search evaluation on all 13 panels and three lineages, starting with 2,048 LBR/native-pressure blocks and 256 per remaining panel. Preserve paired deals and coupled action streams.
- Quote independent workers including all solves, setup/build, parity, replay, verification, storage and shutdown reserve. Do not shrink the arena to fit M4.
- After quote approval, perform macOS↔x86_64 Linux parity on the actual pod before production. Check native menus, payouts, indexing, locks, range updates and strategies using frozen tolerances; record repeat-request determinism and thread-level floating differences.

## Validation, reporting and safeguards

- Test hidden-world invariance, public-only requests, hypothetical-query isolation, cache identity, exact wager insertion, hero locks, hand reset and river conditioning.
- Test opponent-only likelihood flooring, structural card zeros, ε=0 equivalence and cause-specific telemetry.
- Exercise malformed output, missing executable, timeout/process cleanup, memory refusal and legal fallback. Independently replay every arena hand and verify settlement and arithmetic.
- Retain LBR batch completion, soft-budget overruns and zero-likelihood traces. Report returns, latency, fallback causes, street activity, coverage and tails. Ending-street losses remain descriptive.
- Keep M1 work short. During active implementation, check #145 status and PR every 30 minutes.
- Admit M4 compute only after #145 finishes, its final report is pushed, its processes exit and no follow-up owns the machine. Ask if ownership is unclear.
- Count pilots, failures and verification toward 24 hours. Measure reclaimable memory, stay below 10 GiB owned RSS, retain #145’s swap-growth guard and preserve unrelated services.
- Commit/push coherent subtasks; retain manifests, failures and hashes; provide `status.md`, TensorBoard and milestone PR comments. Update ROADMAP and RESULTS_INDEX.
- Flag distribution of a playable AGPL-dependent bot as an owner licensing decision. No destructive evidence cleanup is included.

## Owner-approved recurring continuation

After development/review fixes, use a Codex recurring message in this chat every 30 minutes rather than a polling loop. Each wake reads this file and current status, checks #145 and M4 release evidence, then returns when M4 is still occupied. Resume the authorized plan only once the full admission conditions above hold. Check ownership claims and #145's recurring follow-up as well as process state; an exited solver alone does not release the machine. No duplicate worker, automatic restart after a guard failure, budget reset or paid allocation is authorized by the recurring check.

The branch and draft PR already exist. Continue them; do not create a second PR. Full live-hand matrices and solutions must survive hypothetical-query cache eviction. Report live turn conditioning gaps with a frozen tolerance of zero, p99/max latency and timeout fallback rates by host, and explicit arena `arm: base|search` alongside the actual base `strategy`.

## Owner-approved calibration continuation — October 3, 2026

After calibration-01's preserved RSS guard stop, the owner approved the [resource readmission amendments](hu20-turn-search-resource-readmission.md#approved-modifications-for-calibration-02) on [PR #148](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/148#issuecomment-5966496087) and directly instructed this chat to continue. Count free/inactive/speculative/file-backed memory at fresh admission; family cap is min(8 GiB, 80% reclaimable minus 0.5 GiB sidecar). Retain measured native allocation headroom minus 256 MiB, original swap/disk/10-GiB guards and stop on a further guard failure. Preserve calibration-01 separately. First apply the original 30-second selection; absent a native qualifier, permit a separately measured strict-quality native research setting at cold p95 ≤120 seconds. Report both deadlines, use the chosen deadline for research arena and quoting, and ask if no configuration qualifies even there. Candidate axes, quality gates, roots, river sample, arena counts and 24-hour cumulative journal remain unchanged. No paid compute without a separately approved quote. Post the resumed-worker milestone promptly with its pushed commit.

## Owner-approved PR guidance and final-stage budget amendment

At each blocker or unexpected event, read PR #148 comments/reviews before choosing the next action: an issue may already have been anticipated. The owner directly specified that comments from the owner account explicitly marked **Owner approved** or **Owner-approved** communicate decisions reviewed and agreed by the owner. Treat that approval within the action/budget/scope the comment actually states. When documenting execution of an approved change, name and link the owner-approved comment; do not label unapproved proposals as approved.

The [Owner-approved final-stage budget amendment](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/148#issuecomment-5967372814) instructs stopping calibration-02 cleanly before final outcomes, preserving/charging every screen row, dropping iteration counts whose screen-scaled p95 cannot meet the deadline (keep 25/50/100 at 30 seconds including borderline 100), and sharing only exactly seat-independent requests/matrices with tests. Keep two finalists per floor if a timing-only forecast fits, otherwise one fastest per floor. Push the amended final-stage and conditional 120-second worst-case forecast, leaving at least three M4 hours for sampled rivers, and a new immutable settings file before resuming from retained screen; never rerun completed screen rows or stitch calibration-01. If the forecast still does not fit the original 24-hour cumulative allowance, stop and ask rather than shrinking roots or lineages. Other gates, guards, selection and separate paid-compute approval remain.

The authorized stop occurred at 238 retained screen rows, all 128 native screening rows complete, zero final rows. Worker/sidecar/native processes exited; total journal use 12053.465135375325 seconds. This is an owner-requested timing amendment, not a new guard failure. All prior evidence and the remaining eighteen reduced-menu screen coordinates are retained/explicit.
