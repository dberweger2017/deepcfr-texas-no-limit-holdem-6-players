# HU100 menu v2: frozen continuation to 5B

Owner-authorized October 14, 2026. One change: freeze #260's exact
97,290,578-key terminal table and continue its existing lineage to 5B
cumulative nodes. The owner decides the merge and any release.

## Inputs and training

Build current main (`6df0dab9`) with `python -m scripts.build_native_trainer`
and retain the receipt. Source checkpoint SHA256:
`202058df8aff994612b441d26ffc4e70475ee790cc06a799635e4389e219db2f`.
If absent on M4, regenerate #260's exact recipe and require byte identity;
do not retrieve that checkpoint from Drive. HU100, seed2026100601, K50
#163 tables, menu v2, linear CFR, opponent-sampled average, R=1, no raise
cap, no ante/rake, 50/100 blinds, two 10,000-chip stacks. Regeneration uses
#260's original nominal cap97,290,497 and 1B target; its complete-iteration
capacity terminal has332,894,788 nodes /104,400 iterations.

Resume with `--freeze-at-cap --max-entries 97290578 --nodes 5000000000`.
Save the final complete iteration, export current and average with
`--zero-mass uniform`, and audit every row with
`src.diagnostics.cfr_average.audit`. Report stored visits/key, zero visits,
discarded updates and both players' frozen missing decisions by street.
Membership stays fixed; missing keys play uniformly forever. Additional
training need not make the policy strong or its deep streets supported.

## Evaluation and predeclared decision

Use the HU100 duplicate/shared-deal harness with each policy's own schema.
Primary: frozen v2 (translation off) minus released v0.5.2
(translation512 states /128 events on) against random, check/call,
tight-aggressive, loose-aggressive and1.5× pot pressure. Include half-pot
and matched1× pot through the existing seven-opponent harness.
Requested16,384 independent duplicate blocks per opponent.

Descriptive learning check: frozen v2 minus #260's unfrozen terminal on
the same panel/deals. Descriptive direct match: frozen v2 versus v0.5.2,
translation off for both,131,072 duplicate blocks, reached-key coverage
for both sides. No cross-menu direct strength gate.

Cost-only pilot:1,024 direct blocks and64 blocks per panel contrast.
Keep outcomes sealed; expose only costs and paired dispersion. Separate
model/cache loading and input/table hashes from per-block play, raw-trace
hashing, replay and reproduction; scale only the per-block component.
Retain all requested counts if the corrected projection fits7 hours.
Otherwise reduce once before outcomes to8,192 panel blocks /65,536 direct
blocks; stop if that option cannot fit. Include a25% planning margin and
30-minute closeout reserve. Post and read back canonical JSON in the PR
description before final hands. Every final hand independently replays
and exactly reproduces before any readout; no failed-hand reruns,
outcome-driven extension or pilot pooling.

Nominal paired Student-t95%, conditional on fixed models/seed. Recommend
a v0.5.3 candidate iff no primary panel contrast's upper bound is below
−20 BB/100 AND pot-pressure's lower bound is greater than−20 BB/100.
Everything else is descriptive. This is no publication authorization.

## Machine, guards and evidence

M4 only, detached tmux overSSH. Hold both shared research locks; refuse
another campaign. Keep every5-second reading. Stop on60 seconds of
family RSS>13 GiB, critical pressure, free memory<8%, swap growth>4 GB.
Disk<16 GiB stops immediately. Verify PID/create-time before signals.
Fixed entries can still acquire regret storage; retain measured peaks.

Archive raw evidence, partials, failures, source/build and guard receipts
with an embedded manifest and complete local ZIP readback in
`~/Local/Research-Cloud/PR-<n>-hu100-v2-frozen-5b/`; create a separate
average-only ZIP. Reference existing accepted input archives, record
retrieval hashes/commands and separate native/independent cloud acceptance
in RESULTS_INDEX. No synced deletion or forced offloading. Research
binaries never enter Git. Report results, update ROADMAP's current
position and overbet-menu's open question, and leave a green PR to main
for owner review. PR comments are concrete issues with a SOMA.
