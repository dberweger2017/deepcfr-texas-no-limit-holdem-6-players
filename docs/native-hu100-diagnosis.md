# HU100 coverage and aggressive-loss diagnosis

Owner-requested diagnostics only: no training, extraction, policy changes, new
arena deals, checkpoint selection or merge. Read #197's recovery/baseline and
#200's exact final head `acef17133bad6f26c4b2265772e64f40abe9a9fd` while open;
#200 subsequently merged as `3d92a93`. Its original roots remain untouched.
Read-only retrieval from the accepted Research-Cloud archive is authorized by
this task; it does not authorize moving or changing open-PR evidence.

## Prospective analysis

Use all five indexed average exports and every candidate hand in #200's final
sample: 5 checkpoints × 5 opponents × 4,096 hands = **102,400 hands**. Reused
uniform arms remain reference evidence, excluded from decision associations.
Timing pilot uses only #200's disjoint pilot traces, all five model scans, replay,
and native parity; report cost/completeness only, suppress pilot poker outcomes.
Freeze admission after timing and before reading final settlements. If the full
sample cannot fit, retain an incomplete diagnostic; do not select losing hands
or shrink the sample after seeing results.

For every candidate hand, replay public events, all actions and settlement in
Python. Emit native fixtures for the entire sample, checking every native key,
menu, bounds, pot, actor and settlement against Python. Join trace probabilities,
lookup class, menu, average mass and traverser visits to the indexed exports.
Verify all selected archive members and whole ZIP SHA256 before using them.

Missing keys receive two independent annotations: whether a preceding opponent
action was outside the training menu, and abstract-key support. An exact all-menu
path is a witness. A matching row in any verified checkpoint is a stored-training
witness (including keys unseen at an earlier checkpoint). For remaining off-menu
histories, enumerate supported menu paths whose complete ordered history tokens
match, including alternate exact raise sizes. Accept only a matching actor,
street, player flags and menu names. Card descriptor is held fixed; the key does
not encode exact pot/stacks. Exhaustion proves no supported public betting path
to that payload; a 5,000-state/signature cap or 120-second total search cap means
**unresolved**, never unreachable. Cache only by the complete public signature.
Native/Python discrepancies are a separate correctness failure, not silently
classified as missing coverage. Replay representative paths for all discovered
classes and losses, with deterministic first-occurrence selection.

Retain opponent/street/button-position/pot bands (<4, 4–16, 16–64, ≥64 BB),
visits (0, 1, 2–9, 10–99, ≥100), average-mass bands (0, (0,100), [100,10k),
≥10k), and mean fold/passive/raise probabilities. Decision summaries include
wins, ties and losses and attach final hand payoff; repeated decisions weight
that payoff repeatedly, so these associations are not additive or causal.
Also partition **all hands**, including opponent folds before any target action,
into ever-missing, any-zero-no-missing, all-positive, and no-target-decision exposure. These
disjoint hand contributions sum to each panel's total BB/100. No post hoc
significance testing or causal/strength claim. Average mass is iteration-weighted
opponent-sampled accumulation, not independent visits; traverser visits can be
zero at a positive-mass key.

## M1 resource and verification plan

A single absolute **1,800-second deadline** starts before archive retrieval and
covers pilot, frozen analysis, native build/parity, focused tests, independent
arithmetic verification and evidence archive readback. Source editing and later
human-facing review/upload metadata do not restart scientific compute. Record
actual M1 identity, physical memory, pressure, available disk, power and original
swap baseline at admission. Choose whole-family RSS min(4 GiB, measured free
headroom minus 3 GiB), disk floor max(8 GiB, current available minus 6 GiB), and
maximum swap growth 0.25 GiB. Require normal pressure, ≥15% system free and AC;
fresh stage admission reserves the whole RSS ceiling plus 2 GiB. One sequential
worker, one-use budget/phase claims, two build jobs, no other jobs stopped.
Existing supervisor samples the complete owned family every five seconds.

Pilot quote doubles model-scan and scaled replay/native/arithmetic costs and reserves 240 seconds
for verification/archive; science stops 240 seconds before the absolute deadline.
A root phase lock prevents analysis/archive overlap. No count extension or retry. Partial outputs, stopped
panels, failures and resources remain available and are archived. Independently
review methodology, code and final report; add focused regressions for demonstrated
correctness issues. Keep compact report/receipts/CSV in Git and large evidence
with member manifests/restoration provenance in a new Research-Cloud folder.
