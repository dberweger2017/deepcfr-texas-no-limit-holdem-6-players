# Three-player search cost calibration

Owner request, October 17, 2026: measure a small concrete task first, then size
the next experiment from its observed cost. Three to eight hours is a planning
estimate, not a grant to fill or an automatic extension. This is cost and
feasibility work, not a playing-strength comparison.

The first 1,000-hand task completed in 34.122 seconds. See the
[measured report](reports/nplayer-search-cost-pilot.md); search timing remains
pending current-model adapter qualification.

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

## Implementation plan for v0.6

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
