# N-player Phase 0 measurements and Phase 1 smoke

The owner cleared PR289's independent traversal/information/regret review at
`5df55c52d6b9679e71cf9eaa24dd1f29f523468f` on October 15, 2026. Both CI shards
pass. This work starts on `feature/nplayer-size-smoke`, stacked on that reviewed
parent while it remains unmerged. The owner decides every merge.

## Size gate, fixed before measurement

Use the reviewed generic traversal and cash keys, pinned #163 K50, 50/100-chip
blinds and the exact HU v1 min/pot/conditional-all-in menu. A simulator wrapper
records each decision key reached by traversal without changing sampling,
policies or updates. Count every distinct reached key, including opponent-only
keys on streets excluded from average storage. Separately count retained
trainer entries and average/regret payloads. Prove instrumented and plain
training have identical tables, counters and checkpoints before measuring.

One fixed measurement seed **2026101521**; this is a growth/cost study, not a
strength or training-seed-population claim. Six configurations:
N=2 and N=3 at 20/100 BB with full-street averages, plus N=3 at both depths with
preflop-only averages. Each has a **20,000,000-node target**, reported at complete
iteration boundaries after 100k, 1M, 2M, 5M, 10M and 20M nodes. The entry cap is
**12,000,000**; reaching it is a reported capacity terminal, never permission
to enlarge the run. Record allocation requests, hash capacity, action payloads,
observed RSS and keys/entries by street. The observer's own memory is separate.
Compare N=2 generic keys with the unchanged production HU K50 trainer at both
depths using the same 20M target and seed, with their different key/traversal
semantics stated explicitly. Production HU receipts include retained training allocations (including scratch
and worker-pool storage), entry count, save cost and checkpoint bytes; its
segmented store does not expose one aggregate hash capacity.

First admit two **100,000-node** three-player pilots, 20/100 BB, seed
**2026101520**, <=120 seconds each. Proceed only if a conservative linear quote
fits **10 minutes per size configuration** and **60 minutes total** for the size
study including HU references. Otherwise report the measured limitation and
bring a concrete revised budget to the owner. Main measurement jobs have an
on-host 600-second cap each, 16-GiB family RSS and 32-GiB disk floor. No automatic
node/entry/time extension or outcome-based rerun.

Project tables from measured growth slopes and marginal new keys, showing
linear upper scenarios rather than presenting a fitted saturation as fact.
Account for allocator/hash-capacity steps, checkpoint sorting, K50 residency,
observer overhead and recovery/export peaks. Recommend a Phase 2 depth, menu,
averaging scope and node budget for cygnus-x1's 125 GiB, with explicit margin.
This recommendation does not authorize a long campaign.

## Later smoke gate

Only after size results and a written resource admission: one fresh
**200,000,000-node** three-player **20-BB** smoke, seed **2026101531**, followed
by a **100,000,000-node** prefix and resume to the identical endpoint. Full
compressed checkpoint bytes must match. Averaging scope and fixed entry cap
are chosen from the completed size study and declared before smoke execution.
No strength selection or threshold tuning. Quote the whole fresh+split,
load/save/export/loader verification cost and cap it before starting.

Add a minimal N-tagged export and an observation-only loader, then seat three
separate policy instances in a small deterministic replay/legal-action check.
State precisely how excluded-street averages and missing keys behave; never
silently label a current-policy continuation as a full average. Preserve
legacy HU bytes and formats. No search, release, M4 use or long training.

All host work requires an idle-campaign check, <=8 assigned CPUs and the
on-host **85°C** watchdog (stricter than the owner's 90°C maximum). M1 is for
edits/light checks. Retain all attempts, resource receipts and unique evidence;
archive/cleanup remains separate from this open PR.
