# Verify native recovery and measure HU100 table growth

Owner-authorized capacity experiment, predeclared October 7, 2026. This campaign
uses merged [#196](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/196)
and its [preparation protocol](native-hu100-preparation.md). The owner has
authorized the HU20 recovery check and up to **10B total HU100 traversal nodes**
on the free M4, within the fixed limits below. The later instruction requires
**#188 to be merged before implementation, environment setup, builds, tests or
training start**. This preparation PR and lightweight scheduling/status work
are separately requested now. Nothing has been trained for this campaign.

The question is whether native HU20 recovery preserves the complete training
state and policy probabilities, then how the unchanged table representation
grows at 100 BB. A controlled capacity or time stop is a valid experimental
result. Neither completion nor growth establishes convergence or poker strength.

## Admission, schedule and ETA

- Initial T3 wake: **October 8, 2026, 04:00 Europe/Madrid**, or **02:00 UTC**.
  Madrid is on CEST (UTC+2) at this date. At 20:54:55 UTC on October 7 the
  calculated delay was **18,305 seconds /5 hours 5 minutes 5 seconds**.
  The scheduler returned `2026-10-08T02:00:00.424Z`, bound to this thread.
- Disable that initial timer on its first wake. Inspect durable state before
  doing work, and check current GitHub #188 state and the actual M4 process/
  completion receipts. An idle science worker alone is insufficient: #188 must
  be **MERGED** and its worker-side science, package and archive closeout must
  have finished. Never stop, renice or alter another job or its environment.
- If #188 is open, M4 occupied or SSH unavailable, maintain one bound
  **15-minute admission timer** until genuinely admitted or the deadline.
  Disabling/replacing a timer must be recorded with its exact ID. Reconcile
  scheduler listings with state after a lost response; never create a duplicate
  active timer or worker to recover uncertain acknowledgement.
- Once execution starts, disable admission checks and use one bound
  **30-minute execution timer**. These turns inspect compact status and anomalies;
  they do not repeatedly scan full logs or rerun expensive audits. Worker guards
  operate continuously and complete independently of M1, posting or agent wakes.
- **Hard finish: October 8 at 10:00 Madrid /08:00 UTC.** Disable every campaign
  timer at completion or deadline, including if #188 never becomes admissible.
  A late admission shortens the experiment; it never moves the deadline.

The earliest intended start is **04:00 Madrid**, conditional on merge, closeout
and resources. There is **no measured HU100 completion ETA yet**. The work window
is at most six hours and includes implementation, isolated setup, qualification,
independent review, HU20 reference/recovery, HU100 training, saves, exports and
audits. #196's reference train and export/full-audit phases each have a
900-second cap; the complete reference session has an 1800-second cap. Recovery
also needs separately bounded training, export and comparison time. Historical
HU20 timings are context, not a HU100 forecast. Publish an updated finish
estimate after qualification and measured pilot/save/export/audit timings.
The reporting target is **by 10:00 Madrid**, with any remaining archive upload
acceptance explicitly marked pending rather than represented as complete.

Local coordinator checkout: `~/Local/native-recovery-hu100-20261008`, branch
`feature/native-recovery-hu100`, initially based on #196 merge
`97d0cb9deb25e7937cf232f458dcea6fc71ff800`. Its ignored
`results/native-recovery-hu100/campaign-state.json` records the initial timer,
admission observations and launch history. After admission, create a separate
M4 checkout/environment through `ssh m4`; use fresh nonsynced ignored attempt
roots. M1 remains a lightweight coordinator, without large downloads or training.

Durable worker state must pin campaign/source/binary identity, phase, owned
PID/process group, parent hashes, launch receipt and terminal status. Acquire an
exclusive campaign launch lock before spawning work. Once a launch has been
recorded, uncertain liveness requires reconciliation against the recorded worker
and receipts; it does not authorize a fresh launch. Never automatically resume
a capacity stop. Preserve stale-lock, failure and interruption evidence.

## Frozen recipe and resource limits

Seed **2026100601**, linear CFR, one root per seat per iteration, independent
deal/action streams, **opponent-sampled** average, original v1 card abstraction,
ordered history, minimum-raise/pot/conditional-jam menu and **uniform**
zero-mass/missing-key fallback. HU20 uses equal 20-BB stacks; HU100 uses equal
100-BB stacks, 0.5/1 blinds, no ante/rake and #196's versioned HU100 identity.
No CFR+, equity tables, new action sizes or changed rules/information inputs.
Consult [rules](rules.md) before any game behavior or player-information change.

All phases retain #196's **5.5-GiB aggregate owned-process RSS ceiling**,
**swap growth ≤0.5 GiB**, **free disk ≥15.5 GiB** and **AC-power requirement**.
One native training thread and sequential heavy work; free M4 only, $0 paid
compute. Baseline swap is campaign-wide, so restarting a phase cannot reset away
earlier growth. Account for coordinator, supervisor and descendant processes.
Keep an external hard guard over training, serialization, exports and audits;
never run a native command outside the guard.

Add a lower controlled training stop that preserves measured serialization
headroom under the unchanged hard RSS ceiling. Reserve disk for every retained
checkpoint, atomic temporary save, exports, audit/log evidence and eventual
archive duplicate. Reserve time using actual save/export/audit timings and
conservative growth allowance, before the fixed hard deadline. An estimate that
does not fit ends the run; it is not grounds for increasing limits. Large table
growth alone does not require another owner approval within these boundaries.

## Stage 1 — HU20 correctness and recovery

1. Qualify the extended tooling with meaningful deterministic tests and
   independent correctness review before dependent training. Pin clean source,
   freshly built binary, Cargo lock, Python requirements/engine binary,
   environment and actual worker/resources. Resolve findings before proceeding.
2. Run #196's **fresh 1B reference protocol**, retaining 100M and 500M saves.
   Preserve its recipe and historical checkpoint header: **no `--recovery` on
   the uninterrupted reference**, which would alter the pinned average metadata.
   Record actual complete-iteration nodes/iterations and overshoot; milestone
   names and `config.max_nodes` are not lifetime-node evidence.
3. Independently recompute **every current and average export row** from regrets
   and accumulators. The reference average must be **142,677,367 bytes**, SHA256
   `571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`.
   A wrong size/hash, row, identity or missing telemetry blocks Stage 2.
4. Retrieve the **retained historical 500M checkpoint** into a fresh ignored M4
   input root, verifying its archive/member provenance. The required member hash
   is `a5318cd586c0b170a68334e4236111faddabaf7f686c071958757db888afab47`,
   iteration **1,095,942**. #182's original payloads were cleaned after verified
   archival; do not assume its historical `inputs/` path still contains it.
   Verify that the fresh reference's 500M checkpoint agrees with this retained
   input before using its actual-node receipt to establish the legacy recovery
   count. Never substitute the requested 500M for actual completed nodes.
5. Resume that retained checkpoint into a **fresh attempt** to the same **total
   1B endpoint**, with parent SHA256 and verified completed-node count supplied.
   Restore seed, iteration, roots, game, average rule, visits, regrets and all
   accumulators. Preserve the load/resume command, hash and coverage baseline.
6. Compare **all complete training rows**, not a sample: keys, menus, regrets,
   accumulators and visits, plus scientific header/configuration and actual
   final node/iteration counts. Compare both policies' complete probabilities.
   Independently audit the resumed current and average outputs too.

Recovery adds `native_state` and uses a legacy coverage baseline; historical
street counters cannot be invented. Report these recovery-only metadata
differences explicitly. Average export metadata also contains the different
checkpoint hash/header, so whole-file hash identity is not expected for that
resumed average. Any metadata exclusion must be narrow, named and validated;
it must never hide changed scientific configuration, table rows, node counts or
probabilities. State whether equality is byte equality or exact parsed/numeric
equality, including signed-zero handling. A correctness mismatch ends the
campaign and **blocks HU100**, even if later resource admission would be possible.

## Stage 2 — HU100 table growth

After Stage 1 passes, train a **fresh** HU100 policy with the frozen recipe.
Save at **100k /1M /5M /10M**, then **every 1B through 10B total nodes**.
Checkpoints land at the first complete iteration reaching each target; record
requested and actual completed counts. Stop at **10B, capacity or remaining
time budget**, whichever comes first. A controlled resource-ceiling stop is a
valid result, with no automatic retry and no increased allowance.

Audit the early pilot's current/average exports and coverage before continuing.
Correctness findings or zero street visitation demand diagnosis before dependent
training. An incomplete/failed pilot does not pass. Once the pilot passes,
continue within the already authorized 10B/resource/time limits without another
approval solely because table growth is large. If staged recovery is used to
pause for audit, every continuation uses a verified parent and fresh attempt;
record total endpoint, coverage continuity and source identity.

Retain continuous resource samples and record at **every checkpoint**:

- Requested/actual total nodes, overshoot and complete iterations.
- Current and sampled peak process/family RSS, swap/baseline growth, free disk
  and power; identify sampling interval and its limits.
- Actual table entries, increment since previous save, entries per node,
  throughput including startup/writes and recent throughput.
- Atomic checkpoint size, SHA256 and measured write duration. Publish a completed
  save receipt only after successful close/rename; interrupted `.tmp` files and
  abandoned saves remain partial, never valid checkpoints.
- Per-street decision/traverser visits, average positive/zero mass, visitation
  histogram and coverage-start provenance. These are visitation diagnostics,
  not coverage of all possible information sets.

Early pilots must be audited before continuation. Later checkpoints may be
saved faster than full audits fit: distinguish **verified**, **saved but
unaudited**, **incomplete target** and **interrupted save** in the report.
Do not silently label an atomic save as a verified inference policy. Use bounded
streaming diagnostics where needed; independently test their exactness before
acceptance. Save/export/audit allocation must also fit the same hard guards.
Retain every attempted phase, failure and partial file, including a guard kill
that prevents final serialization. The last accepted checkpoint remains evidence
of attained work, not a claim that interrupted work completed.

## Evidence, review and handoff

Update the [campaign report](reports/native-recovery-hu100.md) and evidence PR
as each subtask finishes; commit/push completed subtasks. Obtain independent
review of tooling/correctness and the final evidence, resolving blocking findings.
Do not merge this PR, publish/release/promote a model, run arenas or raise limits.
Update roadmap Current position when work lands; this PR remains unmerged at
handoff under the owner's explicit instruction.

Before any input copying/archival, check the owning PR's live status; respect
open roots and dependencies. #188's merge does not authorize changing its roots.
After terminal worker closeout, archive this campaign to a dedicated
`~/Local/Research-Cloud/PR-<owning-number>-native-recovery-HU100/` under the
[project Research-Cloud folder](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s).
Include source/binary/environment, parent provenance, commands/plans, complete
state/logs/resource samples, checkpoints/exports, failures/partials and audit
receipts. Seal a member size/SHA256 manifest, verify archive/member readback and
record archive hash, native upload acceptance, Drive URL/ID and exact retrieval
commands in [RESULTS_INDEX](../RESULTS_INDEX.md). Mark unconfirmed upload status
explicitly. Keep research binaries outside Git and retain originals; this task
does not perform cleanup or touch synced files destructively.

The final report must state HU20 equivalence (or the exact blocking mismatch),
HU100 table-growth measurements, stop reason and maximum verified/unaudited work,
resource/time compliance, artifacts and independent-review status. If admission
never occurs by the deadline, report **not run**, preserve the admission record
and disable all timers without inventing experiment results.
