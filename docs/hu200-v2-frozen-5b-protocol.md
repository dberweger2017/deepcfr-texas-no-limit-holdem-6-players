# HU200 densely trained overbet menu

Owner-authorized October 14, 2026. One fresh lineage, M4 only, detached tmux;
no research depends on the M1 remaining awake. Source base is current main
`e2bcdc63`. The owner decides merge; no release or default change.

## Fixed recipe

Copy #250/#265: HU200, stacks 20,000 chips each, blinds 50/100, no rake/ante,
seed 2026101302, R=1, linear CFR (no training options), opponent-sampled lifetime
average, #163's pinned flop/turn/river K50 tables. Change only `--menu v2`:
min, half, pot, 1.5-pot, 2-pot and the existing conditional jam, with exact
clamping/deduplication and v2 history labels. Translation off, uniform missing
keys and `export --zero-mass uniform`.

Build with `python -m scripts.build_native_trainer`. Fresh recoverable 20M and
10M→20M checkpoints must have identical size/SHA256. Sample native RSS at 50 ms
for train/save measurement, additionally retaining every five-second guard
reading. Measured B/entry = (max(train peak, save peak) − 1,384,000,000)/entries.
Cap = floor((11,000,000,000 − 1,384,000,000 − 250,000,000)/(B/entry × 1.1)).
Keep this measured cap fixed. As in #265, run fresh until its complete-iteration
cap, save recovery and exact node count, then resume with `--freeze-at-cap` and
the actual cap-reaching entries. Resume to 5B cumulative nodes with an
intermediate 2B save. This is equivalent to freezing immediately after the
cap-reaching iteration in a fresh `--freeze-at-cap` run; no intervening update
occurs. Preserve freeze/2B/5B checkpoints and every telemetry receipt. Export
5B current/average and audit every row with `src.diagnostics.cfr_average.audit`.
Report nodes, iterations, entries, stored visits/key, zero-visit and zero-mass
shares, discarded visits/updates, role/street missing-decision shares and RSS.

## Evaluation and decision

Pinned v1 frozen 5B average: 5,000,000,031 nodes, 3,748,464 iterations,
85,373,907 entries, 2,756,501,598 bytes, SHA256
`43aa3b3cd3c7365c5be1b177dcc8a7c04e01c8470403c38327aa845922726512`.
Use #263's schema-owned HU200 harness. Primary panel: 16,384 independent
four-hand paired blocks per opponent (random, check/call, tight aggressive,
loose aggressive, 1.5× pot, half pot, matched 1× pot). Shared physical deals and
logical-role action streams; rotated seats. Key contrasts: 1.5× pot and half pot.
Descriptive direct duplicate: 131,072 two-hand blocks, with both sides' coverage.
Different-menu head-to-head cannot decide the recommendation.

Cost-only pilot: 1,024 direct blocks and 64 paired blocks per opponent; expose
only timing, memory and paired SDs. Reserve distinct pilot/final roots against
current main and all open PRs. After pilots, post and read back the canonical
plan in the PR description before any final hand. A single pre-outcome reduction
may select 65,536 direct /8,192 panel blocks if the full counts cannot fit the
fixed October 14 22:00 Madrid deadline. Use a 1.25× padded cost projection and
reserve 30 minutes for archival closeout. If neither option fits, stop incomplete.
Never reduce/extend counts after final outcomes. Replay every final hand
independently and exactly reproduce all hands/decisions before any readout.

Nominal block-paired Student-t 95% intervals, no multiplicity adjustment.
Recommend a Slumbot confirmation only if the 1.5× pot contrast lower bound is
strictly positive AND no panel contrast upper bound is below −20 BB/100.
All other findings descriptive; one training seed and a restricted frozen table.

Only if this rule passes: prepare, do not start, a hash-verified v2 5B average,
separate cache and 5,000-hand launcher on db-ocean. Account proposal
`deepcfr-k50v2-5b` (16 characters), fresh seed, existing checkout/venv. Verify
client and analysis support K50 menu-v2 HU200; fix/test if necessary. Never touch
running slumbot tmux sessions, runs, accounts or credentials.

## Guards and evidence

Check idle campaign processes and acquire both shared research locks before
science. Every five-second reading retained. Only continuously observed 60-second
breaches stop on family RSS >13 GiB, critical pressure, free memory <8%, swap
growth >4,000,000,000 bytes. Gaps reset timers. Disk <16 GiB stops immediately.
Validate PID creation identities before signals. Time/numerical/hash/replay
failures stop dependent work; keep partials and failures, no scientific retries.
Native training has a bounded clock with closeout reserves. RSS can grow after
freeze as opponent-only keys acquire regret arrays.

Archive full evidence and a separate average-only ZIP with embedded manifests
and complete local readback to `~/Local/Research-Cloud/PR-<n>-hu200-v2-frozen-5b/`.
Exclude caches and already indexed table/v1 model dependencies; record their
retrieval provenance. Index exact model/archive hashes, Drive IDs and separate
local integrity/native upload/cloud metadata receipts in RESULTS_INDEX.md.
Retain all originals; never delete or evict synced files. Final report, roadmap
entry and overbet-menu open question accompany the PR with green CI.
