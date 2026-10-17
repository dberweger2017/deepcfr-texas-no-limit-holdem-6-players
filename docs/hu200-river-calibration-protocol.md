# HU200 river evidence I/O and fixed calibration

Owner-approved October 17, 2026, in PR317. This prospective declaration precedes
all calibration hands. The owner chooses the final counts and venue after the
readout. No final strength hands, Slumbot play, training or default change.

## Unchanged search and ordered admission

Keep the audited blend/model/cache/K50 inputs, translation 512/128, river-only
ordinary joint-range solver, 50 iterations, 6 threads, full ranges/menu, prior
hero/opponent locks, 2 GiB allocation limit, 300,000 public-node limit and 120-second
work watchdog unchanged. All earlier gates and failures remain recorded.

1. Test retention, failure preservation and owned-array lifetime on M4.
2. Prove complete output parity on the original fixed 24 against source 214695a3:
   all final and ancestor scientific requests and compact numeric body hashes,
   scientific header semantics excluding runtime RSS,
   every normalized final profile probability/EV/mask, played policy/support,
   same-seed actions and fresh replay. Request decompression must recover its
   exact original bytes and SHA256. Do not omit a mismatching spot.
3. Rerun the same 24 cost gate once: p95 ≤10 s and at least 3 GB host available and
   physical-minus-family headroom, with the usual M4 locks/guards unchanged.
   Any miss remains a finding; no search weakening or sample change.
4. If qualified, measure two then at most three unchanged concurrent searches
   on the same 24. Use the existing aggregate memory/headroom admission and
   report contention, setup and throughput. Choose the fastest admitted
   measured worker count before calibration outcomes, without changing search.
5. Commit the calibration execution declaration before hands. Run exactly the
   fixed counts below, on M4, with a hard two-hour budget including load,
   playing, independent replay, fresh-player ledger reproduction and readout.
   The clock starts immediately before first calibration input/model loading.
   Preparation/retention/parity/cost/concurrency gates precede that clock.

No existing research originals are removed. Only newly generated successful
native scratch covered by the retention rule below can be removed. All research
that remains is preserved uncompressed on bb128 with manifest-bearing ZIPs;
Drive upload obeys the shared-uplink policy. Machine commands/paths stay in
ignored archival control records, not this contributor-facing protocol.

## Predeclared retention rule

`hu200-scientific-request-lowest-1pct-v1`: serialize the scientific request as
sorted, compact JSON (`allow_nan=False`), excluding only `mode`, `dump_path`,
`seconds`, `profile_format`. Hash UTF-8 bytes prefixed by
`hu200-river-native-body-v1\0`. Keep full native bodies exactly when the unsigned
SHA256 is below `floor(2^256/100)`, or when the native call fails. The rule is
fixed before solves and never reads wins/losses, terminal cards, EVs or quality
outputs. It is stable across directories, timing, process IDs and replay.

Every decision keeps its complete played policy/support/fallback ledger,
telemetry and checksums. Every native call records its scientific sample key,
request checksum, all generated compact header/numeric body sizes/SHA256s,
retention decision, response and failure information. All failed/partial native
bodies remain. Successful full requests are losslessly gzip-compressed with
fixed gzip metadata; their original size/SHA256 remains and decompression must
reproduce every byte. Selected successful bodies retain both `profile.jsonl`
and `profile.f64`; other successful numeric bodies are temporary parser input.
Their compact headers remain losslessly compressed for audit. Header semantic
comparison excludes only native `solver_peak_rss_bytes`, while original raw
header bytes/checksums and runtime telemetry remain recorded. A durable
receipt precedes removal. No parsed strategy, normalization or solver input is
changed by retention. Persistence failures are recorded as plumbing failures.

## Fixed calibration, separate from finals

| Panel | Independent blocks | Root |
|---|---:|---:|
| Direct search versus plain blend, duplicate seat rotations |128|20261017317100|
| random, both arms on common deals |32|20261017317101|
| check_call, both arms on common deals |32|20261017317102|
| tight_aggressive, both arms on common deals |32|20261017317103|
| loose_aggressive, both arms on common deals |32|20261017317104|
| pot_pressure, both arms on common deals |32|20261017317105|
| half_pot_pressure, both arms on common deals |32|20261017317106|
| matched_pot_pressure, both arms on common deals |32|20261017317107|

These roots and prospective final roots 20261017317900–20261017317907 are
checked against main and all other open PR heads before use. Calibration roots
and actual deal seeds are excluded from finals. Each direct block has two
hands, each scripted block four (two arms times two rotations). Both arms use
the same deal and logical-role action streams; policies receive only their own
immutable observation. Search uses the frozen blend before the river and a
fresh, internally consistent policy per hand. No turn search. All earlier hero
river matrices come from the policy actually played, never a counterfactual
blueprint replacement. Script implementations remain pinned repository code.

Every completed hand must independently replay legal actions/events/settlement,
then reproduce with fresh player/RNG state and the complete checked decision
ledger. Reproduction must invoke zero native solves. Full hand/decision streams
and native failures are retained with checksums. Replay/reproduction precede
any calibration statistical readout; no partial collection qualifies as a
completed 128+7×32 study. Numerical/information/accounting failures stop the run.
Plumbing repairs append provenance and continue recorded state; they do not
license rerunning, tuning or extending a completed research outcome.

## Variance-only planning readout and two venue quotes

Do not use mean effects or winning/losing hands to change the solver. Calibration
never enters the final test. Report verified counts, block SD/variance and its
uncertainty, river decisions per hand, fallback/timeout/changed-action counts,
per-panel/block timings and evidence bytes. Keep realized gain means and effect
intervals out of this engineering readout; all raw data remain archived.

Direct score is the average search return across its two rotations, scaled to
BB/100. Panel score is the paired search-minus-plain block return on common
deals. Plan for a +10 BB/100 direct effect at two-sided alpha .05 and about 80% power.
Report plug-in and normal-model one-sided 95% upper-SD sizing; the latter is
conservative only under that variance model, not a heavy-tail guarantee. Use
`ceil(((z(.975)+z(.8))*SD/10)^2)`, round up to a power of two with minimum 4096,
and propose the resulting counts, never start them. If measured direct variance
is zero or fewer than 20 blocks are nonzero, retain at least #306's historical
SD 879.355850 as a planning floor and disclose the sparse-tail limitation.
Suggest at least 256 paired blocks per script and at least direct/64, rounded
up to a power of two; report the implied per-opponent nominal and seven-way
adjusted precision rather than claiming the direct power target for the panel.

Quote final load, primary, complete replay/reproduction and evidence overhead,
with 25% headroom and assumptions exposed:

- M4: measured calibration cost and admitted throughput for up to three searches;
  do not divide by three without a measured admissible result.
- cygnus-x1: available only after PR301's compute finishes, within its eight
  busy-thread/thermal limits. Six fixed native threads permit one active native
  search plus runtime headroom, not two six-thread solves. Do not use PR301's
  current workload for a benchmark or assume a training rate is a measured
  search rate. If #301 is still active, label any hardware-proxy quote as
  conditional/unmeasured and require a post-301 fixed-spot build/parity/cost
  qualification before final hands on that venue.

The owner chooses counts and venue before final hands. A later final protocol
must commit all roots/counts, intervals, stopping/adverse-result rules and the
range-model diagnosis before playing. The PR remains draft while that planned
research and storage closeout are unfinished. Owner decides merge.

## Checker plumbing clarification before cost or calibration

The first retention parity attempt passes all normalized profile/EV/mask,
played-policy/support/action comparisons at spot 0, then stops because an
over-strict checker compares raw compact header hashes. The unchanged native
writer inserts `solver_peak_rss_bytes` into profile rows. Two previously retained
source 214695a3 runs already have different header hashes for this reason, with
identical numeric bodies; the new completed numeric body has that same SHA256.
No mathematical mismatch or strength outcome is observed. The failed checker
receipt remains. Compare header semantics excluding only runtime RSS, keep
compressed headers for audit, and reuse the completed hash-verified native
work for spot 0 rather than invoking that solve again. The remaining 23 spots,
then the single new cost gate, use the corrected evidence/checker plumbing.

## Storage admission clarification before calibration

The first two-search measurement completes with identical outputs and memory
headroom. The first three-search attempt stops on the unchanged free-disk guard,
before any decision completes; retain its partial native files and receipt.
A no-native diagnostic confirms parsed matrices own their data and the parser's
mapping is already released. No parser or solver change follows that diagnosis.

Keep all selected and failed native bytes losslessly gzip-compressed as final
research files, retaining their original size/SHA256 and compressed file pins;
this changes storage representation only. A durable receipt precedes removal
of the newly generated raw scratch. Failed/truncated bodies are compressed as
opaque bytes, never discarded or parsed to decide retention. Headers remain
compressed for every completed call. Existing originals are untouched.
Revalidate from completed hash-verified native outputs where possible, then
measure the changed I/O cost. Concurrent admission additionally reserves the
observed full per-worker temporary body/request size, the measured retained
fixed 24 payload and 512 MiB above the unchanged 16 GiB free-disk floor, checked
fresh before launch. A resource-stop continuation never replays a completed
research result; the stopped three-search attempt completed zero decisions.
Queue calibration direct blocks first, then scripts in table order, with
ascending block indices and no outcome-dependent scheduling.

## Execution declaration, committed before calibration

The [execution declaration](reports/hu200-river-search-artifacts/calibration-execution-declaration.json)
pins source `5b6b2f6c`, unchanged solver/configuration, fresh checked roots and
all three qualifying gate hashes. The fixed 24 pass exact parity. Sequential
p50/p95/max are 4.640/9.834/11.511 seconds; peak family RSS is 2.898 GB and minimum
host available is 6.222 GB. All 24 complete, with zero fallbacks or missing EVs.
The original 3-second failure remains unchanged.

Three workers are selected prospectively from measured throughput: 0.4194
versus 0.3187 decisions/s with two. Three-worker peak family RSS is 6.279 GB;
minimum host available is 6.074 GB. Concurrent p95 is 11.510 seconds, so this is
an offline throughput choice; the sequential interactive cost gate remains
9.834 seconds. Full output parity holds at both concurrent worker counts.

Calibration remains exactly 128 direct plus 32 blocks for each of seven scripts,
1,152 primary hands, hard 7,200 seconds including load and verification. Storage
admission reserves the fixed-24 observed bytes per decision for 1,152 decisions,
three maximum observed native scratch sets and 512 MiB failure margin above the
unchanged 16 GiB disk floor. This is a prospective estimate (1.64 searched
decisions per search-arm hand), not a decision cap or a guarantee about unseen
failures; the usual guard remains binding. No final hands are authorized.

### Launch plumbing continuation, same clock

The initial evaluator launcher fails while constructing the deadline wrapper,
before any action, terminal hand, native solve or completed block. It treated
an inherited static method as a bound method. The evaluator now captures and
calls that static check directly; search/game source is unchanged from the
qualified gates. A new real-policy wiring fixture checks both the external cap
callback and original expired-deadline behavior. An initial fixture run omitted
its required existing replay-ledger directory; that fixture setup is corrected,
and all 84 focused M4 checks pass.

The [continuation declaration](reports/hu200-river-search-artifacts/calibration-continuation-declaration.json)
pins repaired source, unchanged roots/counts/gates and the original absolute
calibration deadline, 1792225958.0844858. The initial attempt and every guard/test
receipt remain preserved. The clock is not reset; repair and revalidation consume
the approved two hours. There is no completed research hand to rerun.
