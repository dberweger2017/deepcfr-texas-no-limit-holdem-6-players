# Frozen HU200 v0.5.5 blend: corrected raw-input protocol

PR297; owner authorized continuation October 15 after the normalized-export stop.
One blend only; no training, Slumbot, server, credential, release, tag, default
change or outcome tuning. The owner decides merge and any later Slumbot use.

## Fixed normalization before any merge

Read raw training checkpoint row `[key, names, regrets, Node.average, visits]`.
For each seed choose **W = T*(T+1)/2**, total iteration weight per seat. The
production `Trainer::step_with` starts `t = iteration + 1`, runs one root per
seat and sets `average_weight=t`; `Traversal::visit` adds `t*policy[a]` at
sampled opponent nodes. No discount applies. This scheduled normalizer is
well-defined. Do not sum mass over every node: that would weight tree shape
and traversal counts. A common factor of two across all seeds is irrelevant
to final row normalization, but this protocol uses per-seat weights explicitly.

- Seed 2026101302: T=16,670,844, W=138,958,528,171,590, checkpoint `c5929bda17457911347e47cd77a83dd8b34618a5542731bed96ece1fb4288b6f`.
- Seed 2026101421: T=16,745,107, W=140,199,312,593,278, checkpoint `c70e8e01b3b27c7fa8b8e0561bdbe5b77b321125de646f04614c82ecacacc676`.
- Seed 2026101422: T=16,827,728, W=141,586,223,234,856, checkpoint `e265c384686f003521e49158c5162e57175d3b53e405140cb494d30a8ec9660c`.

Also record mass across the initial small-blind root infosets (all 169 private
preflop classes), where the opponent-sampled traversal accumulates once per
iteration. Compare it with the schedule; report floating-point differences
and missing root keys. Root mass is diagnostic, not a second candidate rule.

Require all three full checkpoint hashes and pinned #163 card hashes. Export
each checkpoint using the existing frozen native exporter, `--zero-mass
uniform`, and require its exact original full average SHA256. Stop dependent
work on any discrepancy. Retain every failure; no failed-hand retry.

## Union gate and merge

Before construction, scan strictly sorted unique keys from three bounded
lookahead streams; count keys in exactly one, two and three seeds, union,
action slots and per-menu counts. Project compressed checkpoint/current/average,
exact cache backing sizes, audit SQLite workspace, evidence/archive duplication
and all retained input/download space. Require the 16-GiB disk floor plus an
explicit 2-GiB margin at each peak; stop and report if capacity fails.
For compression cost only, the scan retains up to 100,000 keys whose first
12 digest bits are zero. The same fixed formula produces this unplayed sample;
the existing exporter measures its current/average sizes. Project checkpoint,
current and average bytes from that sample with a 1.5 multiplier and header
allowance; the full cache array projection is `35*union_keys + 8*action_slots +
768` bytes plus manifest allowance. Audit workspace and archive duplication
are separate peak reservations. The SQLite ledger reserves 1.5 times the
exact menu-weighted TEXT payload estimate plus 128 MiB. Initial traces reserve
4 GiB, revised only from cost-only pilot byte sizes before finals. Archive
reservation doubles retained blend checkpoint/current/average and trace/log
bytes; caches retire before sealing. No sample outcomes or alternate blends
exist.

For each union key/action, raw blend accumulator is `fsum(A_s/W_s)` over
present seeds, checked identical action menus. A missing key contributes zero.
Output the trainer's JSONL-v2 checkpoint under distinct research seed/model
identity **hu200-v055-sequence-blend-2026101529701**, synthetic seed
**2026101529701**, and a blend manifest; header iteration is the
format-required synthetic marker **1**, with no training-iteration claim; retain game/schema/card-table
identity (the strict shared game identity is not a seed/model identity).
The source checkpoint and average hashes are new; none of the seed model
identities are reused. Set regrets and visits to zero, document that no training/regret
interpretation is asserted, and omit seed recovery/frozen counters. Mark
synthetic provenance in a separate manifest and header extension. Do not
represent this as a trained seed or a resumable experiment. Export and audit
with existing normalization and `--zero-mass uniform`; zero-mass rows are
uniform. No alternative rule is selected after outcomes.

## Fixed play, global verification barrier and reporting

Direct: **131,072 duplicate blocks against each of the three seeds**. Scripted
panel: **16,384 blocks per each of seven opponents**, with blend and all three
seeds paired on common deals and identical logical-role streams, using the
#263 harness. Reusing the same blend arm against each seed must reproduce its
streams exactly. Translation 512 states /128 events on both sides. Scripts:
random, check_call, tight_aggressive, loose_aggressive, pot_pressure (1.5x),
half_pot_pressure and matched_pot_pressure (1x).

Fresh roots will be reserved against main and every other open head before
pilots/finals. Pilot direct roots are 20261015297010–12; shared pilot panel
root is 20261015297020. Final direct roots are 20261015297101–03; shared
final panel root is 20261015297200. Distinct script scenario names define
distinct deals within a panel root; all four models share each script
scenario's deals and role streams. Cost-only pilots: 128 direct blocks per seed and 32 blocks per
script/seed, separate roots from finals. No reduced final counts; pilot reports
only cost, variance and coverage. Declare a resource/time envelope from pilots
before final hands. The quote is fixed primary/reproduction/audit setup
plus target blocks times measured primary/reproduction/audit block work per
pilot block, with a **1.5 planning multiplier**. Cache rebuilds and archive
reserves are separate. Trace hashing remains in scalable audit work.
Every final hand independently replays and completely
reproduces; verify all comparisons and stream hashes before reading any result.
The two-model harness runs three pairs per script, with the blend arm
repeated. Before readout, its hand and deterministic decision fingerprints
must match across all three pairs. There are 1,703,936 unique primary hands
and 2,162,688 stored primary records including the repeated blend arms; each
stored hand also independently replays and reproduces. The reuse adds no
independent samples to any interval.

Report nominal paired Student-t 95% intervals and a clearly labeled descriptive
21-contrast scripted family adjustment. No pass/fail or milestone gate.

Report blend-versus-seed played missing-key share per street, separately for
exact absent keys and unresolved missing keys after translation. Fixed-model
uncertainty does not measure the training-seed population. No Slumbot verdict.

## Machine and storage

All model construction/play on the 16-GiB M4, one job. Both exclusive research
locks and the existing five-second guards: 13-GiB family RSS, 8% free memory,
critical pressure and 4,000,000,000-byte swap growth sustained for 60 observed
seconds; immediate 16-GiB disk floor. PID creation-time ownership checks apply.
The full intake starts a fixed 16-hour wall envelope, with its last two hours
reserved for preservation/archiving; the intake stage itself is capped at six
hours. No final-count reduction or scientific retry follows a stop. Cost
pilots must admit the full remaining counts before final play.

All four caches are written as the existing six NumPy arrays through bounded
file streams, passing exported float64 probabilities unchanged. Its bytes
are tested against the original cache builder. Retain the blend cache, but
build only one seed cache at a time for its complete pilot/final comparison
group, then retire that inactive rebuildable cache with input pins and exact
removal receipts. Rebuild it for a later phase if needed. This keeps disk
cache residency at blend plus one seed. No research original is removed.

No synced deletion or forced offloading. Only inactive rebuildable caches may
be removed after dependency/provenance checks, with an exact receipt. M1 is
limited to source/PR/report/archive administration. Do not assume enough disk
for hydration of the full source ZIPs. Revisit ETA after input admission,
union count and cost pilots; no finite final ETA is claimed while storage is
unresolved. Keep originals and all open-PR dependencies.

The active runtime environment is `~/Local/hu20-k50-5b-first-pass-20261012/.venv/`;
retain it while PR297 runs. Seed1421 checkpoint, the frozen exporter and card
tables have verified read-only APFS clones under PR297's own ignored
`results/hu200-v055-blend/protected-inputs/`, protecting active dependencies
during the owner's storage work. Original indexed files are unchanged.
