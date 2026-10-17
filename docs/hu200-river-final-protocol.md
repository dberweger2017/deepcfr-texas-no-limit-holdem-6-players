# HU200 river search final strength declaration

Owner-approved October 17, 2026, PR317. This declaration is committed before the
first final hand. Counts, roots, solver work, opponents and analysis are fixed;
no outcome-dependent tuning, rerun, extension or early success stopping. The
owner decides merge. No Slumbot play, turn search, training or default change.

## Counts, coordinates and unchanged policies

| Panel | Independent blocks | Root | Primary hands |
|---|---:|---:|---:|
|Direct search versus plain blend|131,072|20261017317900|262,144|
|random, paired search/plain on common deals|2,048|20261017317901|8,192|
|check_call|2,048|20261017317902|8,192|
|tight_aggressive|2,048|20261017317903|8,192|
|loose_aggressive|2,048|20261017317904|8,192|
|**pot_pressure, 1.5 times pot**|**2,048**|**20261017317905**|**8,192**|
|half_pot_pressure|2,048|20261017317906|8,192|
|matched_pot_pressure|2,048|20261017317907|8,192|

Total **145,408 independent blocks /319,488 primary hands**. Each direct block
has two seat rotations. Each panel block has both arms and both rotations,
common deal and logical-role action streams, yielding a paired contrast.
Use schedule version2, scenario `river-final-<panel>`, test split and the existing
`action_seed` law. Constant-memory `block_at` must match `build_schedule`.
Recheck roots against main and every other open PR head, and check every actual
final deal seed for uniqueness and exclusion of all 352 calibration deals before
playing. Calibration and all cost spots never enter the final sample.

Blueprint SHA256 is
`ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485`;
use its already verified read-only cache and pinned K50 tables. Freeze search to
qualified source `d0849fca`: 50 iterations, six native threads, full compatible
ranges/menu, zero global opponent floor, earlier hero/opponent locks, exact
wagers, 2 GiB native allocation, 300,000 public nodes and 120-second watchdog.
Translation 512/128 applies outside search. Native executable SHA256 remains
`9e1efe275c2eab216e3af9da20d55dea9436e88aacb057a41b86b32c171cf914`.
The tested hand player is unchanged. Policies receive only their legal own-seat
observations; no labels, seeds, opponent cards, outcomes or evaluator state.
Fresh players and private RNG streams per hand preserve within-hand consistency.

The existing ordinary joint-range CFR has no safe resolving gadget or safety
promise. Future equilibrium responses and blueprint-conditioned opponent ranges
may misrepresent scripted callers. That limitation is measured, never corrected
using these final outcomes. Timeout/error fallback remains the blueprint, with
every fallback/cause counted in the denominator.

## Resource admission and bounded storage

All model/cache/search/evaluation/statistics compute runs on the 16 GiB M4,
under the common exclusive lock and unchanged guard. Up to three worker searches
use the measured qualifying concurrency. No scientific compute runs on cygnus
while #301 is active; its role here is private-cable storage and file hashing.

Keep the 16 GiB free-disk floor, 13 GiB family guard, memory-pressure/swap rules
and all current samples unchanged. Before dispatching a block, reserve 512 MiB
per active block plus 256 MiB for journal growth above the disk floor. Admit at
most three and reduce dispatch concurrency only from available disk capacity,
without reading outcomes or changing any solve. Only that bounded set can stay
local while transfers are pending. The execution declaration fixes the safety
deadline before the first hand, including loading, primary play,
replay/reproduction, transfer, verification and primary readout. Take the larger
of calibrated compute and final-count-weighted serial storage cost (each with
25% headroom), then add 10% pipeline coordination allowance. Apply the same rule
to the upper cost estimates. Set the safety cap to at least 96 hours, or the
upper estimate rounded up to 12 hours plus another 12 hours, whichever is larger.
This is a fail-safe cap, not an additional sampling allowance. Restart retains
the original absolute deadline. A missed deadline is recorded; never extend a
completed outcome. The earlier compute-only quotes remain 73.4 h and an upper
block-cost projection of 87.4 h; qualified transport costs enter the committed
execution declaration before launch.

The owner explicitly authorizes transferring each completed block to the
existing PR317 research cache, verifying **every file's size and SHA256 on
arrival**, then removing that exact verified owned M4 block copy. Source model,
cache, earlier research, calibration originals and unrelated roots are protected.

Each block finishes independent hand replay, fresh-player ledger reproduction
with zero native solves, full per-hand inventory/checksums and block score/seed
checks before sealing a complete transfer manifest. Copy without overwriting
existing destination files. The remote standard-library helper hashes every
arrived member and the manifest, checks exact inventory and writes its arrival
receipt. Pin the helper's source SHA256. Persist the full manifest, arrival
receipt and retirement intent in a synchronous SQLite journal before removal.
Rehash all remaining local files first, then each file again immediately before
unlink. Never recursively delete a tree: unknown, changed or unverified files
stay intact. Record all retired names, sizes/hashes, destination and timestamps.

After interruption, verify the remote copy again and finish only a previously
verified retirement. Reuse completed verified blocks/hands without new solves.
Retain partial/unverified hands and stop automatic replay pending continuation
from recorded state; never silently start them again. Journal/completed-block
recovery is plumbing, not another research sample. Transfer/checksum/lock/disk
failures stop dispatch, preserve state and are fixed/continued. Numerical,
information, accounting or replay failures stop dependent research and remain
findings; no resampling to obtain a better outcome.

Preserve full ledgers, telemetry and checksums for every searched decision.
Native body rule remains `hu200-scientific-request-lowest-1pct-v1`: fixed
outcome-independent scientific-request sample plus every native failure,
losslessly compressed. No scientific retention changes. Final unpacked files,
source/control/failure provenance and receipts stay on bb128; final ZIPs with
manifests/readback upload using its deployed research cache tool. Macs' Drive
folders stay read-only. Every transfer and retirement is in the durable journal;
final snapshots/receipts join the research archives. No further approval is
needed for this explicitly authorized exact block retirement.

## Verification barrier and statistical analysis

Progress reports show counts, timing, storage and fallback operations only.
No interim win rates, confidence intervals or per-opponent gains inform work.
Every one of the fixed blocks must have a checked hand/seed/accounting result,
verified arrival and recorded local retirement. Cross the complete global
145,408-block/319,488-hand barrier before statistical readout.

The primary score is search's average chips across its two direct rotations,
scaled to BB/100 at a 100-chip big blind. The panel score is search minus plain
for the paired block. Independent blocks, not hands or replay copies, are the
sampling units. Report means, SDs, standard errors and two-sided Student-t95%
intervals. Report each panel contrast with both nominal95% and Bonferroni
seven-way95% intervals, plus both arm means/95% intervals. Report pot_pressure
prominently regardless of outcome. No pooling training-seed variation or claim
about a training-seed population; intervals are conditional on this blend and
these independently generated blocks, with a heavy-tail caveat.

A direct lower95% bound above0 establishes direct gain; an upper bound below0
establishes direct regression. Otherwise direct gain is not established. Report
panels with adjusted upper bounds below0 as negative effects, below−10 BB/100 as
material regression evidence, and separately whether−10 remains unexcluded.
Inconclusive intervals stay inconclusive. There is no automatic adoption or
release rule. Sizing uses calibration upper-SD1212.92 BB/100, approximately84.7%
power for+10 after rounding; panel counts are exploratory, not80%powered for
+10 per opponent. Zero observed sample variance does not exclude rare tails.

For direct and every opponent report searched-decision count/frequency, changed
same-uniform action count/fraction, model EV and EV-delta distributions, native
and complete decision timing, tree sizes, peak memory, fallback causes and
timeouts, including unavailable EV denominators. Model EVs are same solved
continuation estimates, not realized or causal action gains.

## Prespecified range-model check after the final barrier

Always examine pot_pressure, and each panel with a negative paired mean,
without changing future play. Attribute each paired hand's full outcome contrast
to its first differing action on the common prefix. Identical paths must settle
identically; a pre-river difference requires a wiring/information audit.
Partition contributions by first action change, facing price, pot, wager size
and immediate opposing response. These are realized whole-hand attributions,
not causal action EVs.

Select the five most negative paired hands with distinct public first-difference
contexts per selected panel; tie-break by block index then rotation. If fewer
exist, inspect all. At first difference and the largest later river wager, compare
search's conditional opponent holding law with an offline frozen-script public
posterior. Integrate the script's action randomness; never condition the
posterior on its realized RNG draws or actual hidden cards. Exclude hero/board
cards structurally. Compare strong/value-caller mass, holding support and
predicted future fold/call/raise rates under the solved versus frozen response
law. Actual hidden cards may be inspected only in this post-play diagnosis.
For the direct match the opponent is the blueprint; audit range conditioning
and consistency, rather than pretending a scripted reference applies.

Use retained request/ledger evidence. If a selected full body was not sampled,
reconstruct the **same pinned scientific request** for a diagnostic-only solve
and require its scientific/numeric hash and played matrix parity before use.
No alternate ranges/menu/iterations/locks enter a solve. Keep every diagnostic
failure and all selected losing/counterexample cases; no retuning, replacement
sample or final-hand extension. Report whether evidence supports mis-modeled
value callers/future folds, algorithm/finite-work effects or remains unresolved.
The primary readout alone does not complete these diagnostics or make the PR
ready for final review.

Transport qualification uses read-only copies of calibration blocks0–2 for
direct, check_call and pot_pressure, plus a copy of the first fixed24 sampled
body spot. No final deal is played. Measure full copy, remote per-file SHA256
and durable flush, and exact local retirement; original calibration/cost files
are unchanged. Fault fixtures exercise corrupt arrivals, changed local files,
unknown extras/symlinks and interruption after a verified unlink. Only newly
created owned qualification copies are retired. Keep all receipts and failed
attempts. Pin the passing qualification in the execution declaration.

The first transport qualification passes every checksum/retirement check, but
its per-file durable flushing exposes nontrivial storage latency. A dedicated
single storage writer therefore overlaps up to three searches, with at most
three completed blocks queued and free-disk admission still binding. No parallel
hashing workload is added on cygnus. A second storage-only qualification uses
read-only copies of all352 calibration blocks, with prospective final-count
weights for timing projections. No outcomes enter timing analysis, and no new
research hand is played. Earlier qualification files remain preserved. This
engineering refinement precedes final hands and changes no search or retention.

The bound is at most six unretired local blocks: three search futures (including
completed results awaiting a storage slot) and three submitted storage futures.
There is one storage writer and at most three active searches. Completed
storage bytes are already reflected in free-disk admission; the per-search
reservation is conservative for completed-but-not-yet-submitted results.

Each paired panel block checks its common prefix immediately before transfer:
observations/roles must match up to the first differing action, that difference
must be our river action, and identical action paths must settle identically.
This audit blocks a pre-river/RNG/role wiring error before further final play;
it does not use gains to tune or extend the study.
