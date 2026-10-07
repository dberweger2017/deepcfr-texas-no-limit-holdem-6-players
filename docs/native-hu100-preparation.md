# Native heads-up 100 BB preparation

This is engineering preparation for v0.5, not permission to run a campaign.
No training, resource pilot, poker evaluation, benchmark, model download or
worker operation was performed for this PR. M1/#190 and M4/#188 remain occupied;
their roots, environments and active inputs are protected. No unattended work
is scheduled. An operator may use the commands below only after the owner
resumes the relevant stage on an admitted idle worker.

## Scope and baseline

The same native external-sampling traversal now accepts `--stack-bb 20|100`:
two equal stacks of 2,000/10,000 chips, blinds 50/100, chip unit 0.01, no ante or
rake. It keeps linear CFR, one root per seat per iteration, independently derived
deal/action streams, opponent-sampled average and uniform zero-mass/missing-key
fallback. The campaign commands explicitly select `--average-rule opponent-sampled`;
the historical CLI default remains traverser-reach for reproducibility. No CFR+
floor, DCFR, history compression, equity tables, new bet sizes or multiplayer work.

The v1 card descriptor drops kickers and secondary ranks; the ordered history
buckets wager sizes. The menu remains minimum raise, pot and conditional jam,
with native reopening and no free fold. It is a deliberately limited baseline
at 100 BB: deeper play brings more histories and potentially severe sparsity,
and the menu does not approximate every legal no-limit size. #190's equity-table
validation is a separate dependency chain, not a prerequisite or an input here.
[Earlier abstraction lessons](reports/hu20-abstraction-lessons.md) forbid reading
a finer key's equal-node comparison as representation quality. This PR changes
only supported stack geometry and its required game identity.

`src/game/` already supports these tables through the pinned engine; its rules
and observation interface need no behavior change. Native terminal targets
subtract the selected initial stack, in BB, while policy keys still use only
owner cards and public history. Native parity fixtures compare legal bounds,
menus, exact keys and settlement against that engine, including off-menu raises,
short all-ins, unmatched refunds and a tied board. Equal-stack heads-up remains
the invariant underlying the native reopening shortcut.

| Artifact contract | HU20 (unchanged) | HU100 (research only) |
| --- | --- | --- |
| Game | `hu20-native-reopening-20bb-52card-no-ante-rake-v1` | `hu100-native-reopening-100bb-52card-no-ante-rake-v1` |
| Key schema | `hu20-native-reopening-ordered-history-card-v1` | `hu100-native-reopening-ordered-history-card-v1` |
| Training/current format | `holdem-hu20-native-reopening-blueprint-v1` | `holdem-hu100-native-reopening-blueprint-v1` |
| Stored average format | `holdem-hu20-stored-cfr-average-diagnostic-v1` | `holdem-hu100-stored-cfr-average-research-v1` |

Identity checks bind game, schema, equal starting stacks, blinds, chip unit,
menu and card descriptor. Average loading requires an explicit HU100 schema;
HU20 remains its default. Arena registry adapters reject mismatched scenarios
before dealing; observation keys reject them again. HU100 is absent from the
release/UI catalog. Existing v0.4.0/v0.4.1 pins and fallback contracts are unchanged.

## Tooling and reproducibility

The historical scaling, learning-curve and O@10B controllers embed frozen
sources, host permissions, roots and scientific gates. They must not be launched
for this campaign. `scripts.prepare_native_hu_campaign.py` writes new manifests,
job argument arrays and `commands.txt`, without launching any process except Git
reads. It reuses `scripts.hu20_scaling_supervise.py` with explicit resource limits;
old defaults remain unchanged. That supervisor is a macOS operator tool; Linux
admission would require a separately reviewed equivalent, not silent reuse of
`sysctl`/`pmset` guards.

Each stage uses a new ignored nonsynced root. Its manifest pins source commit,
binary SHA256, Cargo lock/requirements hashes, Python/platform, seed, stack,
recipe, limits and exact commands. It refuses dirty tracked source and an
existing output root. Record `cargo -V`, `rustc -Vv`, `python -m pip freeze`, actual
host, engine commit/binary hashes and available RAM/disk in a preflight receipt.
Verify binary/source binding with a fresh build, not a filename. Generated
commands do not confer authorization. Check #188/#190's live status and obtain
an idle slot; never stop, renice or modify another worker to make room.

Future setup, on that idle worker in a clean checkout of this PR's approved SHA:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
cargo build --locked --release --manifest-path native/hu20-trainer/Cargo.toml
cargo test --locked --manifest-path native/hu20-trainer/Cargo.toml --lib
.venv/bin/python -m pytest tests/test_native_hu100_preparation.py tests/test_native_hu_campaign_preparation.py -q
```

Normal CI also builds the release binary, so previously optional native parity
and training/export tests execute there. CI's larger tests are not a local
research authorization.

## Stage 1: HU20 reference regression

Use seed **2026100601**, 1B requested traversal nodes, 100M/500M milestones, the
exact old checkpoint header and export recipe. Do not add `--recovery` for this
reference replay: it intentionally adds metadata and would change the exported
file hash. Native checkpoints without that option retain the old bytes. Run
from scratch; the published reference average need not be downloaded to compare
its pinned SHA256.

```sh
.venv/bin/python -m scripts.prepare_native_hu_campaign --stage hu20 --out results/native-hu/hu20-regression-01
cat results/native-hu/hu20-regression-01/commands.txt
```

After separate stage authorization and preflight admission, execute the
generated fail-fast script with `bash results/native-hu/hu20-regression-01/commands.txt`.
A failed phase prevents export/audit from starting. Each phase has its own startup-inclusive
900-second absolute phase deadline: train, then export/full audit. They use one
Rayon thread, aggregate owned-process RSS <5.5 GiB, swap growth ≤0.5 GiB, free
disk ≥15.5 GiB and AC power. Deadline/RSS/disk/power failures, nonfinite updates,
invalid play/accounting or failed identity/hash checks stop dependent work.
Retain all failures; no automatic retry or quota escalation.

The final average must equal **142,677,367 bytes**, SHA256
`571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`,
the v0.4.1 pinned reference. `audit_native_hu_checkpoint` checks that hash and
independently recomputes every average/current row from accumulators. Check file
size too. Any mismatch blocks HU100. This is a serialization/training regression,
not a new arena result; the [release report](reports/v0.4.1-release.md) and
[learning curve](reports/hu20-learning-curve.md) supply historical context only.
The generated foreground commands share a 1800-second absolute session deadline.
If the quoted limits cannot admit this regression, stop and seek a revised
resource plan. Past M4 timings do not authorize the run or relax its limits.

## Stage 2: small HU100 resource and coverage pilot

Only after Stage 1 passes and the owner resumes this stage on an idle worker:

```sh
.venv/bin/python -m scripts.prepare_native_hu_campaign --stage pilot --out results/native-hu/hu100-pilot-01
cat results/native-hu/hu100-pilot-01/commands.txt
```

After authorized admission run `bash results/native-hu/hu100-pilot-01/commands.txt`.
Printed commands target **10M total nodes**, with 100k/1M/5M milestones, seed
2026100601, one root per seat, 2M soft entries and 900 seconds of training.
The separate export/audit phase has 900 seconds. Same family RSS/disk/swap/AC
limits as Stage 1; $0 paid compute. A milestone is the first complete iteration
at or beyond its requested node count, not an exact node cutoff. Store both
requested and completed nodes and overshoot. If even 100k work exhausts the
budget, stop and keep that evidence; do not increase RAM or nodes automatically.

Audit the final current/average set before interpreting the pilot. For earlier
milestones, repeat export and audit under the same supervisor with separate
fresh job/guard/audit paths and the same aggregate phase cap. Set
`--target-nodes` in each audit job to that milestone’s requested node count,
not the final 10M target. No arena, LBR,
benchmark or poker-strength comparison is part of this resource pilot.
At every milestone record elapsed time, sampled peak family/process RSS, swap,
free disk, checkpoint/export bytes, entries, traverser visits per key histogram,
positive/zero average mass and cumulative decision/traverser visits by street
(preflop/flop/turn/river). `native_state` and the audit summary provide these
counters; they are visitation diagnostics, not coverage of all information sets.
Keep the raw 5-second resource samples; short-lived peaks can escape sampling.

A pilot that does not reach its target, fails an audit, breaches a guard or has
missing telemetry is incomplete. It cannot trigger the extension. Zero visits
on any street demand diagnosis before extension. A low visits-per-key result
must be reported and considered, not treated as proof that 1B will fix sparsity.

## Stage 3: conditional proposal toward approximately 1B

**1B is a proposed node budget, not proof of sufficient training or convergence.**
Do not extrapolate HU20's ~10-minute/1.5-GiB numbers to HU100. The
[O@10B report](reports/hu20-o-10b.md) also shows why additional training and direct
strength evidence are different questions.

Prepare a measured quote using the completed pilot and all milestone samples.
For time use the slowest measured node rate including startup/checkpoint writes,
then add export/full audit/recovery allowance and at least **100% headroom**.
For RAM extrapolate new-key growth without assuming saturation; use the larger
of latest family RSS and observed bytes per key times projected entries, then
add export/audit peak and **50% headroom**. State that these are uncertain
projections; nonlinear growth still triggers the hard supervisor. Budget disk
for parent, every retained milestone, temporary atomic saves, current/average
exports, logs, audit and eventual archive duplicates while preserving ≥15.5 GiB.
The four extension milestones (50M/100M/250M/500M) add cumulative storage;
never delete an active parent to make the estimate fit.

A quote JSON must contain `source`, `binary_sha256`, `seed`,
`target_total_nodes` (1000000000), `parent_path`, `parent_sha256`,
`forecast_seconds`, `training_forecast_seconds`, `audit_forecast_seconds`,
`forecast_rss_gib`, `forecast_disk_free_gib`, `max_entries`,
and `pilot_audit_verified: true`. Also include the pilot plan/audit/resource
file hashes, completed-node/iteration counters, measured rates/growth,
headroom arithmetic, training/export/audit phase estimates and $0-cost idle-host
admission. Retain both raw measurements and calculation. `forecast_seconds`
is the complete extension cost, ≤7200 seconds; each phase forecast must fit its
printed cap, training ≤6300 and export/audit ≤900 seconds. RSS must stay below 5.5 GiB and free
disk above 15.5 GiB. Otherwise propose a smaller budget or different resources
to the owner; do not launch. The preparation tool performs basic quote admission,
not an independent scientific audit of the owner's resource calculation.

Get explicit owner authorization **after** posting the quote, limits, source,
parent and plan. Record `approved: true`, `quote_sha256` (whole quote-file hash),
`owner_instruction_url` (or durable chat evidence URI), and `approved_at` in an
approval JSON. Existing PR185 approvals do not apply. Only then prepare:

```sh
.venv/bin/python -m scripts.prepare_native_hu_campaign --stage extension --quote results/native-hu/quote.json --approval results/native-hu/owner-approval.json --out results/native-hu/hu100-extension-01
cat results/native-hu/hu100-extension-01/commands.txt
```

The tool verifies the bound quote/approval and parent hash/production HU100
identity, at least 10M completed nodes, full-pilot coverage baseline and nonzero
traverser visits on every street; it still does not run commands. Printed foreground commands resume
the pilot to the **total** budget, not another 1B, with complete-iteration saves
at 50M/100M/250M/500M/1B. Training has a 6300-second cap and export/audit 900;
the generated commands also share a 7200-second absolute session deadline.
After authorized admission run `bash results/native-hu/hu100-extension-01/commands.txt`. Resource forecasts, not positive
poker outcomes, govern admission. After reaching the fixed budget, stop, archive
and hand off; no automatic extension when results are weak or incomplete.

## Recovery and stop handling

HU100 checkpoints always carry `native_state` version 1 with completed nodes
and street counters. `coverage_start` identifies the iteration/node/visit baseline;
legacy recovery starts coverage at that baseline rather than inventing past
street counts. Native loads validate counter totals against stored visits and
completed nodes. HU20 can opt in via `--recovery`; legacy checkpoints require
a manifest-verified `--completed-nodes` value because their header has no lifetime
node count. Never substitute requested nodes for actual completed nodes.
Native resume restores seed, iteration, roots, game, average rule, regrets,
accumulators and visits; root RNGs derive from the restored iteration. It
requires `--resume-sha256`; explicit conflicting CLI configuration is rejected.
Resume is limited to linear CFR with no training options, native button-zero
root table and canonical menu rows. Duplicate/malformed keys, dimensions,
nonfinite/negative accumulators and mismatched game identity are rejected.

Example **future** recovery into a fresh attempt, after failure diagnosis and
same-scope owner resumption; fill the verified checkpoint SHA from its receipt:

```sh
native/hu20-trainer/target/release/hu20-trainer train --stack-bb 100 --resume results/native-hu/hu100-pilot-01/training/HU100-2026100601-5000000.json.gz --resume-sha256 VERIFIED_CHECKPOINT_SHA256 --nodes 10000000 --milestones 1000000,5000000 --average-rule opponent-sampled --max-entries 2000000 --max-seconds 900 --out 'results/native-hu/hu100-pilot-recovery-02/HU100-2026100601-{nodes}.json.gz'
```

Create its output directory first and put this argv in a fresh supervisor jobs
file; apply the same guard/deadline command as the original stage. Do not run
that trainer line unguarded. Never overwrite a prior attempt/checkpoint. Skip
milestones already reached. A hard kill can lose work after the last accepted
atomic save; ignore `.tmp` files and record the abandoned work. Verify the saved
hash, header and full rows before resuming. Changed source/binary needs its own
regression/parity and owner admission. Do not splice accumulators across games.

The native entry/time limits are **soft checks at iteration boundaries** and
can overshoot by one complete iteration. It saves a valid checkpoint with its
actual entry count reflected in the capacity, exits 3 and blocks downstream
jobs. A larger entry limit for recovery needs a reviewed quote; the original
cap may reject an oversized parent. The external supervisor covers owned
process-family RSS, disk, swap and the absolute deadline, including serialization,
and sends TERM then KILL to its own child group only. It does not modify workers
outside that group. Never interpret a resource stop as a successful target.

## Evidence, decisions and later science

Keep source/binary/environment receipts, all plans/commands, resource samples,
requested/completed counts, checkpoint/export hashes, full audits, partials and
failures. At future closeout use the designated Research-Cloud project folder
and owning campaign PR, with member manifests, archive hashes, upload acceptance,
retrieval commands and restore dependencies per [storage policy](artifact-storage.md)
and [RESULTS_INDEX](../RESULTS_INDEX.md). No binary or raw campaign log enters Git.
There is no campaign artifact or archive to retrieve from this preparation PR.

Later multi-seed strength confirmation needs fresh held-out roots, fixed
checkpoints, declared samples, direct comparisons and suitable HU100 opponents;
it is a separate protocol and authorization. The external v0.5 benchmark is
also separate: opponent, integration/license constraints and acceptance criteria
remain owner decisions. Neither this pilot nor the proposed 1B budget satisfies
v0.5 or authorizes a release, default-model change or public strength claim.
