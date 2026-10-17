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

The next engineering task is an observation-only adapter for the current
three-player policy, with exact key/menu/probability parity, hidden-card
invariance, correct public-range sampling and continuation-policy tests. An
independent review is required for changed regret/sampling math before a larger
study depends on it. Learned leaf values need separate training/held-out
validation before they are an eligible comparison arm.

## Task-sized follow-up

1. From cost-only play, select a fixed outcome-independent set of postflop
   decisions, stratified by street and two/three live players. Search receives
   only the acting player's observation, never the replay trace or real deck.
2. After compatibility tests pass, time 12 fixed decisions at the smallest work
   setting. Freeze a grid of solver iterations, sampled holdings and precise
   street-boundary horizons from these cost measurements, without examining
   winnings. Depth and solve iterations are separate parameters.
3. Reuse the same 32–64 decision observations across eligible configurations
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
