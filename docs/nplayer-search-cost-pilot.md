# Three-player search cost calibration

Owner request, October 17, 2026: measure a small concrete task first, then size
the next experiment from its observed cost. Three to eight hours is a planning
estimate, not a grant to fill or an automatic extension. This is cost and
feasibility work, not a playing-strength comparison.

The first 1,000-hand task completed in 34.122 seconds. See the
[measured report](reports/nplayer-search-cost-pilot.md); search timing remains
pending current-model adapter qualification.

The immediate next task is the [small rollout probe](#immediate-task-small-rollout-probe)
below, sized at roughly eight hours of active implementation, checking,
measurement and reporting. The broader sampled-CFR/arena plan is retained as
later work. Eight hours is an estimate, not a deadline or compute allocation.

## First task: current-model baseline

Play exactly 1,000 three-player 20 BB hands with three independently seeded
instances of Phase 2 seed 2026101541's 1B-node all-street average. Use the retained
qualified `nplayer_policy` binary and K50 tables. The existing play command
verifies the model hash and replays every native transition. Record total
runtime, hand/decision counts, peak RSS, output size, source/input hashes and
the guard samples. Total runtime includes model/table loading and serialization;
it must not be presented as isolated inference time.

Average SHA256:
`990246065bfc1f8a13c5ca3456090689b5ab52105c860d4bca1531b3f5e3db70`.
Qualified binary SHA256:
`0f9f6e29f8d9fa79f32ba5098ab9aee8b3920c0c2de078208c60b26b46cbea2b`.
Input provenance is [PR292's member index](reports/nplayer-blueprint-phase2-artifacts/archive-model-members.json).
Cost-only seed `202610173401` and its hands are excluded from future confirmation.

This short observation uses one physical CPU, low CPU/I/O priority and at most
16 GiB address space on bb128. It leaves all existing jobs and inputs unchanged.
Its own ten-minute fail-safe is a maximum, not an expected duration. It stops
its own process group on CPU >=75 C, Thermistor 15 >=85 C, available RAM <6 GiB,
swap growth >256 MiB, NVMe free space after a 512 MiB output reservation <50 GB,
or an output above 256 MiB. Admission requires CPU <70 C and Thermistor 15 <80 C.
The bounded concurrent cost observation follows the owner's request to use
spare capacity; it does not change PR301's source, workload, deadlines or lock.
No general shared-host reservation or later parallel study is authorized by it.
No retry or modification of another job follows a failed admission or stop.

## Search qualification before timing comparisons

Today's supported N-player inference accepts the Phase 2 K50 format. The older
Python sampled/depth-limited search's `LiveBlueprint` and `SearchPlayer` refuse
K50, and the native N-player arena exposes no search player. Therefore this
baseline is not a three-way search benchmark. Do not substitute an old model,
uniform continuations or a heads-up adapter and label it a current-model test.

The old local CFR pilot also requires six seated players with exactly three
unfolded players at its flop root. Supporting the current three-seat model
therefore requires model integration, belief reconstruction and a bounded
solver; removing a format guard is insufficient.

## Immediate task: small rollout probe

Owner direction, October 17: obtain useful initial search data from a concrete
task expected to take around eight hours, rather than implementing the whole
search framework first. This task is a research-only command that reads the
current Phase 2 three-player 20 BB average and evaluates root actions on fixed
legal observations. It does not introduce a live arena player.

### Fixed deliverable and method

- Reuse the native N-player model loader, K50 keys, exact menu and rules. Keep
  the model and card tables loaded once. The command receives only each saved
  row's `observation`, excluding its source metadata and privileged trace.
- Use one method: **root-action Monte Carlo rollouts with fixed blueprint
  continuation**. For each legal root action, simulate complete hands with all
  later decisions made by the frozen blueprint using each acting seat's own
  observation. Aggregate values across sampled worlds before selecting an
  action. There is no additional optimization with access to a sampled opponent
  hand and no continuation selected using an unrevealed card.
- Construct an approximate posterior from 512 compatible joint holding
  proposals per position/repetition, excluding the observer's cards and public
  board. Weight proposals by the two opponents' observed-action likelihoods
  under the pinned blueprint, including any folded player's past actions.
  Reconstruct historical observations at their actual reveal boundaries.
  Document proposal probabilities and use stable weight normalization. This
  finite-particle approximation is not an exact or unbiased posterior estimator.
- Sample complete worlds from the weighted proposal set. All-in and side-pot
  behavior remains owned by the existing engine. Report effective sample size,
  maximum weight and lookup misses. Zero total likelihood fails the case;
  effective sample size below 16 marks it insufficient for decision analysis.
  Preserve either result without silently substituting uniform beliefs or
  adding particles until it looks acceptable.
- Audit the retained 43 observations and select at most 12 deterministically:
  the lowest two observation hashes in each street/unfolded-player stratum
  among reconstructable acting states with at least two legal menu actions.
  Retain all originals and report shortages/exclusions without replacement
  based on outcomes. No new hand generation is part of this task.
- Use three predeclared independent RNG repetitions and **16, 64 and 256
  rollout worlds per root action**, reusing prefixes within a repetition.
  Use matched worlds across actions to reduce comparison noise. With 12
  positions and at most five menu actions, this is at most 46,080 full-hand
  rollouts and 18,432 belief proposals. Declare and collision-check the actual
  seed manifest before execution, separately from future confirmation pools.
- Record per-action values, blueprint probabilities, preferred actions,
  variability across repetitions/budgets, completed rollouts, failures,
  coverage, CPU/wall time, memory and output bytes. Separate loading, beliefs,
  root simulation and verification costs. Reusing samples across budgets must
  not be reported as independent replication.

These measurements concern cost and decision stability against assumed
blueprint opponents. Maximizing noisy rollout estimates is not evidence of
playing gain. This probe measures one root action followed by complete-hand
blueprint continuation; it does not compare depth horizons, sampled CFR,
learned leaves or equilibrium quality. Those remain possible next experiments.

### Work estimate, with acceptance checks

1. **Inputs and fixed cases (about 0.5 h):** pin model/tables/source, declare
   seeds, audit observation eligibility and write the run manifest.
2. **Native probe (about 2.5 h):** implement observation reconstruction,
   likelihood-weighted worlds and per-action blueprint rollouts, using the
   existing native interfaces rather than porting the old solver.
3. **Focused correctness (about 2 h):** verify blueprint/menu parity; hidden
   card/deck invariance; replay of public histories; no duplicate cards; toy
   likelihood/weight calculations; legal transitions, terminal accounting,
   and reproducibility. Reuse pinned-engine replay for generated branches.
   Information or reconstruction failures block measurements.
4. **First small measurements (about 0.5 h):** execute the 16-world prefixes on
   the selected cases after checks. Record actual load and per-rollout costs.
5. **Complete the fixed sampling grid (about 1 h):** extend valid cases to
   64/256 worlds and complete the three RNG repetitions. Retain failed and
   low-support cases in the report, with no outcome-driven extensions.
6. **Readout and preservation (about 1 h):** produce the compact comparison,
   model/source/input pins, commands and restoration provenance. Retain raw
   samples, partials and failures outside Git. Follow the existing storage
   policy for eventual Drive/cache archival; never claim upload acceptance
   before the receipts exist.
7. **Review/fix allowance (about 0.5 h):** review the diff against the small
   scope, resolve correctness findings, run required checks and update PR334.

The sum is roughly eight hours of active work, including development. A stage
that finishes early does not create more work. This is not a promised elapsed
completion time or an eight-hour CPU run. Review/resource waits are reported
separately. No timer changes the task, triggers a restart or grants an extension.
Independent information-boundary review is required before treating prototype
results as qualified or starting dependent larger research. Engineering
measurements, if not yet independently reviewed, are explicitly provisional;
known unresolved correctness failures stop them.

### Resource placement: preserve bb128's current work

Read-only snapshot at **2026-10-17 20:38:37 UTC /22:38:37 Madrid**: PR301's
evaluation-first-pilot worker is on CPUs0–1; its supervisor holds
`/tmp/cfr-cygnus-exclusive.lock`. The two-second host sample was 9.8% busy,
with 124.58 GB available RAM, 105.82 GB free NVMe, CPU60 C and Thermistor15
61 C. These are observations, not unreserved capacity or a sustained-load test.
PR301's declaration gives its current evaluation an exclusive 24-hour clock
starting October17 at22:22:34 Madrid. No completion time is inferred from this.

- Implement and run tiny fixture tests locally on the M1. Leave the M4's
  existing search study alone.
- The full-model probe is planned for the M1, which has 16 GiB installed RAM,
  subject to an actual input-storage/memory-pressure admission before loading.
  The earlier 3.90 GiB sampled Linux RSS is a sizing reference, not a Mac peak
  guarantee. Limit this probe to one worker, low priority and one resident model;
  log resource use and stop its own process on unsafe memory/disk pressure.
  Declare exact platform-specific admission and guard settings before execution.
  Check the owning PR's status before retrieving retained inputs and verify
  hashes in ignored working directories.
- Do not launch a new bb128 job, take/ignore its lock, use a sibling of a reserved
  physical core, or change another campaign's affinity, priority or deadline.
  The earlier short baseline observation grants no ongoing sharing permission.
- If local admission fails, retain the implementation/checks and report the
  measurement dependency. Queue bb128 measurement after the exclusive
  reservation is released, or use a separately owner-approved shared reservation
  with combined CPU/RAM/disk accounting. Do not delay or stop PR301 to fit an
  overnight estimate. Recheck live jobs and reservation ownership at launch.
- Label timings by machine. M1 measurements cannot be substituted for bb128
  performance; a later bb128 calibration needs its own admission and clock.

Successful completion means the small command, tests, fixed-case measurements,
and a report answering which sampling budgets are cheap, stable and sufficiently
supported to justify a next study. A prerequisite failure leaves measurements
pending with preserved implementation/checks and an explicit dependency report.
Full arena integration and a 1,000-hand strength comparison are outside this
immediate task.

## Later implementation plan for v0.6

Owner-approved plan addition, October 17, 2026. The immediate target is the
existing three-player 20 BB Phase 2 full-average model. Keep interfaces suitable
for more players, but qualify three-player behavior first. PR334 contains the
completed baseline, observation extractor and this plan; the search code below
is pending implementation. Adding the plan starts no new compute campaign.

Use the current native N-player rules, policy loader and exact action menu.
The legacy Python search remains a correctness reference; it must not become a
second implementation of the N-tagged model format or its keys. Keep model
weights and K50 tables resident and shared across decisions. Preserve historical
heads-up behavior and keep search opt-in.

### 1. Current-model integration and baseline control

- Define a search interface accepting only the acting seat's observation,
  explicit configuration and a separate search RNG. Use
  `native/hu20-trainer/src/multi_policy.rs`, `multi_holdem.rs` and `nplayer.rs`
  for existing probabilities, keys, menu and rules.
- Verify the pinned model, table hashes, three-player/stack/blind identity and
  full-average scope before play. Continue to count existing missing-key
  blueprint fallbacks explicitly.
- **Acceptance:** search disabled reproduces the existing blueprint's seeded
  actions and outcomes; menus, key identities and probabilities agree exactly.
  Search work cannot advance the deal stream or blueprint action RNG.

### 2. Public history, beliefs and sampled worlds

- Reconstruct the betting-round root and current public state from the legal
  observation. Retain all public action amounts and board-reveal boundaries.
- Weight candidate holdings by their public action likelihoods under the
  declared policy model. Include folded players' latent holdings in card
  removal. Sample compatible joint holdings and undealt cards without reading
  the real hidden deal.
- Define public ranges for every player, including the searcher; the actual
  private hand must not silently alter other players' search information sets.
  Document approximations, zero-likelihood handling and any importance weights.
- **Acceptance:** sampled worlds reproduce all visible state and legal bounds;
  exact small-support enumeration validates the joint sampling law. Changing
  real opponent cards or future cards while holding the observation and search
  seed fixed leaves search inputs and outputs unchanged.

### 3. Bounded rollouts and sampled CFR

- Add action rollouts with fixed blueprint continuation as the first control.
  Then add sampled CFR, reusing the reviewed N-player traversal where applicable
  and checking additional root-conditioning/continuation mathematics explicitly.
- Configure solver cycles, range/world samples, node limits and precise
  street-boundary horizons separately. Work counts determine the task;
  wall/memory limits are fail-safes. Publish updates only after a complete cycle.
- At depth limits, support blueprint/fold/call/raise continuation styles as
  choices at each player's information set. Continuation choices must not see
  other holdings, other players' private choices or a board card revealed after
  the cutoff. Learned continuation values need separate training and held-out
  validation before becoming a comparison arm.
- Initially search only the first decision for the searcher in each betting
  round. Record an explicit blueprint fallback for later decisions until
  repeated-search consistency, including preserving earlier searcher policies,
  is implemented and validated.
- **Acceptance:** legal actions and chip accounting hold throughout; interrupted
  work cannot expose partial-cycle regrets. Unsupported situations have counted,
  seeded fallbacks; information leaks, invalid math or reconstruction mismatches
  fail the run rather than being hidden as ordinary fallbacks.

### 4. Correctness qualification and independent review

- Compare expected sampled regret updates with exact enumeration in small
  imperfect-information games, including root conditioning and continuation
  choices. Verify information-set sharing across indistinguishable worlds.
- Exercise short all-ins, closed raising, folds, side pots, street transitions,
  observation immutability and deterministic reproduction. Retain existing
  engine/key/loader regressions and verify new native traces independently in
  the pinned engine.
- **Acceptance:** focused tests, native library tests, repository checks and
  independent review clear information access and regret/sampling math before
  the timed solver grid or dependent larger research. No convergence,
  exploitability or playing-strength guarantee follows from these checks.

### 5. Small cost calibration

- Audit all 43 retained observations against declared eligibility before
  timing; preserve the originals and publish exclusion reasons and stratum
  counts. An insufficient eligible stratum is a coverage gap, not a zero-cost
  result. Any added cost-only cases require a declared selection rule.
- Start with 12 eligible decisions at the smallest complete work setting.
  Then freeze a feasible horizon/iteration/sampling grid and solver RNG repeats
  for the fixed eligible observation set, without selecting on winnings.
- **Acceptance:** report loading, belief construction, solving and verification
  separately, plus CPU time, median/tail latency, sampled memory, completed
  cycles/nodes, target visits, continuation coverage, action changes and failures.
  Keep the model resident to avoid charging every decision a fresh load.

### 6. Paired arena integration and measured quote

- Add the opt-in search player to the native three-player arena. Compare it
  with the unchanged blueprint on matched deals and six seat permutations,
  separating deal, action and search RNG streams. Record activation, unsupported
  decisions, misses and fallbacks per street/seat, then replay and reproduce.
- **Acceptance:** quote approximately 1,000 hands per eligible configuration
  from measured search frequency/cost and complete verification/storage work;
  use complete blocks, such as 996 hands. Exploratory screening remains separate
  from a fresh, predeclared strength confirmation.

Recommended implementation split: one follow-up PR for model/public-belief
integration and its reference tests, then one for the bounded solver and arena
integration. Each needs the relevant independent review before dependent runs.
This work targets v0.6; further HU200 diagnosis is not a prerequisite. Existing
campaigns and their resource reservations remain unchanged. Any larger shared
run needs a fresh combined resource admission and a measured task quote.

## Task-sized follow-up

1. From cost-only play, select a fixed outcome-independent set of postflop
   decisions, stratified by street and two/three live players. Search receives
   only the acting player's observation, never the replay trace or real deck.
2. After compatibility tests pass, time 12 fixed decisions at the smallest work
   setting. Freeze a grid of solver iterations, sampled holdings and precise
   street-boundary horizons from these cost measurements, without examining
   winnings. Depth and solve iterations are separate parameters.
3. Reuse the same eligible observations from the retained 43-case set across configurations
   and independent solver RNG seeds. Measure loading, range construction,
   solving, replay and storage separately; report median/tail latency, CPU time,
   memory, target-information-set visits, action changes and failures.
4. Quote approximately 1,000 full hands per eligible configuration from the
   observed searched-decision frequency and cost, including seat rotation and
   all verification/storage work. If using six-permutation blocks, declare
   the nearest complete count (for example 996 hands), not a partial block.
5. A pilot can support feasibility and exploratory comparisons; it cannot
   establish general strength or select a release. Future confirmation uses
   fresh hands, fixed counts and its own prospective analysis.

Keep complete partials and fail explicitly when a prerequisite is unavailable.
An unfinished adapter is an implementation gap, not an algorithm's timing
result. No existing M4/bb128 campaign is stopped, restarted or extended by this
plan. A larger shared run needs explicit combined CPU/RAM/disk admission.
