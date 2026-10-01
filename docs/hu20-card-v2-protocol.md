# HU20 private-card abstraction v2 A/B protocol

I test whether preserving more private-card distinctions improves a 100M-node
HU20 current blueprint at equal completed traversal work. This is an opt-in
schema experiment, not a new default, model promotion or follow-on to 500M.
Equal nodes controls traversal work; it does not imply equal wall time or equal
convergence. Additional keys and fewer visits/key remain part of the result.

## Invariants and prospective gates

The merged base includes #139 and #142 (`571bcb26`). Keep the native HU20 game,
20BB reset stacks, blinds, no rake/ante, preflop 169-class representation,
ordered betting-history encoding, native-reopening min/pot/conditional-jam
menu, K1 external-sampling CFR/current extraction and trainer random streams.
Only postflop card representation changes. Schema/identity registration is
necessary so artifacts cannot be mistaken for v1; do not change trainer math.

Before training or strength outcomes:

1. Audit all retained #139 holdings and its five highlighted same-board
   collisions, without loading a policy or re-estimating their equities.
   Require all five to separate. Preserve sample provenance/hashes.
2. Count old/new buckets and information keys under identical deterministic
   public contexts; report counts by street, refinement and residual collisions.
   This sample measures uniform-card coverage, not learned occupancy.
3. Run a from-zero M1 resource preflight sequentially for v1 and v2 with the
   first fixed seed. Prefixes at 250k/500k/1M/2M complete traversal nodes;
   one worker, at most 20 minutes/path, 6 GiB peak, 8 GiB free disk.
   Retain entries, visits, per-street growth and nodes/sec. No strength outcomes.
4. Freeze exact descriptor source/schema, three seeds (2026093001/2/3),
   100M-node stopping rule and practical resource/evaluation budgets before
   paid training or opening comparisons. Complete iterations may overshoot;
   report the actual work. Stop rather than silently coarsen or add training.

## Candidate card distinctions

Keep v1's category/top band/draw/paired-board tuple as a prefix, ensuring a
refinement. Add own-card contribution to the best made hand, hole participation
in made groups, board-relative made-group ranks, kicker bands, suited-card/nut
potential and distinct straight-out quality. Use only own cards and current
public board. No equity table, opponent policy, hidden cards, future deck,
learned clustering or betting-history redesign. Exact tuple definitions and
source hash will be frozen after their correctness audit, before training.

## Training and recovery

Train three independent v2 lineages from zero on RunPod; one worker per pod,
no M4/#136 work. Compare matching existing v1 B100M lineages, never pick a seed.
The retained v1 recipe has a 3M-entry bound; assess its feasibility explicitly
before admission. Resource/recipe changes require an explicit decision, not
silent compensation for v2. No node budget increase after outcomes.

Reuse pinned Linux engine/build and verified complete-iteration checkpoint,
hash-before-load, fresh-process recovery and next-iteration state equality.
Do not infer recovery from successful loading. Retrieve/hash checkpoints,
current exports, iteration/resource logs and manifests before rental teardown.
Use exact owned pod names and independent rental cutoffs. Exclude credentials
from committed evidence. Freeze a finite cost/rental cap from live prices and
measured resource projections before launch; no unrelated pod changes.

## Fresh paired comparison

Keep the existing evaluator, observations, settlement, opponents and native
replay. Freeze a fresh evaluation root distinct from #139/#136 and earlier
reports. Compare both positions and all three matching lineages within each
independent deal block. Keep uniform/styles, native-pressure and frozen
selective-stackoff separate, with bounded LBR only if its outcome-free runtime
projection passes the frozen practical budget. Retain all results, not just
positive panels. No river search in either arm.

Report per-seed/position BB/100 and paired uncertainty, full-stack wins/losses,
large-wager opportunities/actions and opponent folds/continues; subgroup returns
are whole-hand returns, not bet EV. Include street-specific trained/fallback
exposure, entries/visits per key, nodes/sec, RAM, compressed sizes and hashes.
Missing keys retain uniform fallback; do not add a translator or nearest lookup.

## Delivery

One focused draft PR with schema/tests, model-free/preflight evidence, frozen
configuration, Linux recovery/hash evidence, paired tables, raw replay records,
resources and reproduction commands. No automatic merge, promotion, training
beyond 100M, paid follow-on or M4 access.
