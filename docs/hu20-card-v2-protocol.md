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

## Frozen card distinctions

Keep v1's category/top band/draw/paired-board tuple as a prefix, ensuring a
refinement. Add own-card contribution to the best made hand, hole participation
in made groups, board-relative made-group ranks, kicker bands, suited-card/nut
potential and distinct straight-out quality. Use only own cards and current
public board. No equity table, opponent policy, hidden cards, future deck,
learned clustering or betting-history redesign. The exact tuple implementation is `src/blueprint/cards_v2.py`, frozen SHA-256
`190d530ce66d65a031334d95600ce0f005d81324a7353170dcebf1d64a6ccd92`,
version `hu20-contribution-kicker-draw-descriptor-v2`. Its source at `334cd7d`
and all five highlighted separation checks passed before this freeze. Rank
bands are 2–7 / 8–T / J–Q / K–A; own contribution, grouped private ranks,
board-relative group rank, kicker bands, flush stage/private count/nut band
and straight completion count/private contribution/high band/backdoor form
the fixed refinement. Residual collisions remain part of this experiment.

## Training and recovery

Train three independent v2 lineages from zero on RunPod; one worker per pod,
no M4/#136 work. Compare matching existing v1 B100M lineages, never pick a seed.
The retained v1 recipe has a 3M-entry safety bound. The measured 2M prefix
projected 14–17M v2 entries at100M, with considerable uncertainty. I approved
a 20M safety ceiling on October1 before rentals; it does not change CFR
updates, and exceeding it stops the run. No node budget increase after outcomes.

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
beyond 100M, paid follow-on or M4 compute.

## Approved rental and evaluation freeze

`configs/diagnostics/hu20-card-v2-run.json` freezes three CPU5 memory pods
(`cpu5m`, 8vCPU/64GB), one worker each, $10 total hard cap, 18,000 seconds
maximum from controller start (so each later pod has less than five hours),
and an admitted compute+disk price <=$0.57/pod-hour ($8.55 maximum nominal
allocation). No automatic retries of creation. A recorded setup repair can restart only
inside the original absolute cutoff, including all previous spending. Independent
exact-name network watchdog on awake M1 survives controller/SSH loss; it
retries teardown during provider outages. No client can guarantee a billing
cutoff during provider/network failure; report any such failure explicitly.
No provider credential is copied to pods. Memory45GiB /80% of actual cgroup,
swap-growth0.5GiB and disk8GiB guards protect each owned process tree.

Keep 25/50/75/100M complete-iteration checkpoints (including stored averages),
final current exports and all iteration/resource logs. Never resume the
preflight as a production lineage. Before each production run, compare
every decompressed final/current/next-iteration byte between the100k-node
M1 reference, Linux direct run and fresh-process midpoint resume. Distinct
gzip framing is recorded separately. Linux Python3.11.15, Rust1.90.0,
NumPy1.26.4, SciPy1.17.1 and pinned engine5db20e3d are recorded.

Evaluation root202610010701 and outcome-free timing root202610010702, plus
fixed panel indices, are distinct from prior comparisons. Thirteen panels
are the retained #141 uniform/passive/original-cap2 minraise/pressure, six
styles, native-pressure, frozen selective-stackoff and bounded LBR. Each
standard panel has256 independent paired-deal blocks, two positions per arm;
LBR has128blocks, four chance samples and its unchanged5-second soft guard.
The already validated exact rank/cache executor changes cost, not actions.
Each seed first runs four disjoint validation blocks per arm/panel without
chip outcomes. Admit LBR only if1.25 × the sum of per-arm maximum hand times
× final hands projects all panels within1800seconds/seed. If even the cheap
panels exceed that bound, stop the comparison rather than shrink its counts.
Any deadline interruption remains incomplete, with its original records.

Large exports may exceed M1 headroom, so final evaluation uses the same
64GB Linux pods inside the approved rental cutoff, not M4. Hash-verified
M1 cached v1 inputs are transferred read-only. The independent four-cell
Doctor Research experiment uses its own schedule and ownership; this A/B
report is not a factorial comparison. No additional compute is implied.

### Setup correction before any training

The first three pods exited before tests/training because uv0.8.22's bundled
catalogue predates Python3.11.15. All were deleted; estimated compute+disk
upper cost$0.0315037 and setup logs/archives remain retained. Pin uv0.12.21
with its published Linux asset SHA256 and use the public versioned Python
build mirror; verify installer hashes before execution. The installer
correction changes no descriptor, CFR work, seed, Python version or outcome
schedule. A manual repaired attempt retains the original19:54:53UTC cutoff
on October1 and includes prior spending in admission. No model results
were opened to make this change.

## Reproduction commands

Use Python3.11.15 with the pinned native engine and NumPy/SciPy versions
in the run plan. Read the complete protocol before any paid rental. The
following audit/tests/recovery commands use small generated artifacts on M1:

```sh
python -m scripts.audit_hu20_cards_v2 --out results/v2-audit-reproduction
python -m pytest -q tests/test_hu20_cards_v2.py tests/test_hu20_card_v2_campaign.py \
  tests/test_blueprint_hu20.py tests/test_blueprint_native_reopening.py
python -m scripts.hu20_platform_pilot run \
  --plan configs/diagnostics/hu20-card-v2-recovery.json --out results/v2-reference
python -m scripts.hu20_platform_pilot run \
  --plan configs/diagnostics/hu20-card-v2-recovery.json --out results/v2-resumed \
  --resume results/v2-reference
python -m scripts.hu20_platform_pilot compare --left results/v2-reference \
  --right results/v2-resumed --out results/v2-recovery-comparison.json
```

The two retained resource-prefix plans run with
`python -m scripts.preflight_hu20_cards_v2 --plan PLAN --out NEW_DIRECTORY`.
They are independent from production and must never seed a100M lineage.

For an independently approved Linux allocation, the controller CLI requires
`--plan`, `--key` (a private local RunPod config), `--root`, `--reference` and
`--baseline`. The baseline directory must contain the exact B100M policy
AND training-checkpoint pairs in `hu20-stackoff-v1.json`; every byte hash is
checked before transfer and load. Use a new empty output root. An approved
manual setup recovery additionally passes the **original** `--deadline` and
`--prior-cost`; omitting them is not authority to reset a failed allocation.
Credentials stay on M1 and are excluded from results. The setup/worker scripts
are experiment tooling, not a general account rental service.

After verified teardown, reproduce independent arithmetic without loading
models:

```sh
python -m scripts.summarize_hu20_cards_v2 \
  --root results/card-v2-rental-20261001-repair --out results/v2-independent-report
```

Raw generated `hands.jsonl.gz` entries replay with
`replay_row(row)` from `scripts/play_robustness.py`. The recorded source,
model/checkpoint hashes, panel root, block, rotation, exact actions and
event digest define each native hand. Final checkpoints and exports are
retained outside Git; public compact manifests and raw hands identify them.
The report states artifact availability explicitly rather than implying that
the repository contains every trained model binary.
