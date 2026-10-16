# PR301 scheduling amendment — owner approved October16

The October16 owner request seeks better use of bb128 without changing the
experiment. The owner explicitly approved this amendment as written on
October16, including the protocol wording correction, a two-audit first batch,
measurement before admitting three, unchanged guards/deadlines and a revised
ETA after that batch. Deployment requires the validated boundary handoff below.

## Approval provenance

[The protocol](hu200-capacity-cygnus-protocol.md) explicitly admits only one
export/audit/cache builder or verifier at once (execution section), says exports
and audits never overlap (checkpoint section), and requires retiring one seed's
disposable files before auditing the next (disk section). Its controller also
waits for all three seeds' audits before the next target. Owner request point4
reserves changing these restrictions for the owner. The original task likewise
required nonoverlapping exports/audits. These are scheduling restrictions, but
the owner has now specifically approved replacing these restrictions below.

## Approved schedule

- Keep the qualified binary, seeds, cap, R=1, target nodes, exact resume and twin
  gates, every-save byte/hash/CRC and full-row checks, evaluation sizes/roots,
  and global replay/reproduction barrier unchanged. No trainer or exporter
  source changes and no scientific retries.
- After all existing prefix/parity/memory gates pass, allow independent seeds
  to advance on separate queues. A seed starts its next target only after its
  own preceding checkpoint passes every existing audit. Different seeds may
  occupy different targets. Do not train from an unaudited checkpoint. Keep the
  all-three verified-input requirement before any5B continuation.
- Admit at most three concurrent one-CPU audit/export workers. While three
  two-CPU trainer reservations are active, admit at most two one-CPU helpers.
  Use disjoint physical CPU sets within0–7, with no sharing or oversubscription.
  There is no authorization to exceed eight busy CPU reservations.
- Start with two concurrent audits, compare throughput against the current
  single-worker observation, then admit three only if CPU/disk throughput,
  available RAM and projected output sizes justify it. Collect per-process CPU,
  I/O and thread counters plus disk busy counters through checkpoint validation,
  native export and full sorted validation. These are operations measurements
  of existing work, not extra scientific attempts.
- Give each streaming helper a512MiB family RSS ceiling. Three total1.5GiB,
  inside the existing2GiB helper allowance. Preserve38GiB/run and116GiB overall
  ceilings and the6GiB available-memory floor. Admission may serialize a job;
  it may not weaken a guard to obtain concurrency.
- Reserve the full prospective output bytes before every writer. Count all
  concurrent outputs, retained recoveries/averages, twin inputs, traces and
  unrelated occupancy against the existing259GB training plan and50GB free
  floor. Reallocate demonstrated file-size savings within that plan; do not
  enlarge it. When output reservations do not fit, reduce concurrency. Keep
  each old recovery until its own successor audits; preserve failures/partials.
- Preserve Thermistor15>=95°C for30 continuous observed seconds clean stop,
  CPU>=85°C immediate observed stop, swap-growth guard, RSS/disk guards and
  all fan/resource traces. A single owner supervises all concurrent jobs and
  broadcasts a stop to every owned job on a breach or failed audit. Do not run
  separate supervisors whose CPU/RSS accounting could each miss the others.
- Implement only campaign orchestration. Validate dependency order, CPU/RSS/disk
  reservations, global stop propagation and exact output equivalence before
  deployment. Preserve the live source tree; hand off at a successful stage
  boundary with a new pinned source and receipt. Carry forward the ORIGINAL
  deadline, receipts and completed work. Do not restart the controller through
  its current constructor, which would reset the clock and repeat gates.
- Evaluation cache scheduling remains as declared; this amendment concerns
  training/checkpoint operations only. No PR292, new HDD or other-machine work.

## Evidence and thread clarification

[Read-only30-second observation](reports/hu200-capacity-cygnus-artifacts/scheduling-observation-20261016.json):
the native export child consumes0.99946 CPU cores; its physical reads are zero
(cached), writes60.89MB, and host NVMe busy time is56ms/30.016s (~0.19%). This
phase is CPU-bound. Full-validator and parallel throughput remain unmeasured.
Each completed fresh500M audit peaks below49MB family RSS; the table is streamed,
so table-sized audit RAM was an unnecessarily conservative scheduling rationale.

The exact qualified trainer source hard-codes a two-worker Rayon pool and joins
the two R=1 seat traversals in parallel. Production delta grouping/application,
loading and checkpoint compression have serial sections. CPU affinity and
OMP/Rayon limits are already two per trainer; no missing `--threads2` flag needs
adding. A100% CPU snapshot can therefore occur with the correct binary/settings.
There was no live traversal to sample when this observation was taken, only an
audit/export. Capture per-thread CPU deltas during the next naturally scheduled
training stage before claiming sustained two-core utilization. Do not change R
or rebuild the trainer to increase utilization. The protocol's phrase "six busy
threads" should be corrected to "six reserved CPUs, two-worker traversal per
trainer, variable utilization" when this amendment is approved.

The owner reports PL1=PL2=125W, quiet CPU-sensor fans, a15-minute three-trainer
test peaking69°C VRM/59°C CPU, and stress-ng reaching the95°C VRM stop at eight
threads. These are owner observations, not a repeated stress test. Campaign
fresh-prefix audit peaks were CPU54–56°C and VRM57–60°C; no higher thread limit
is proposed.

## Conditional ETA

Pending measured concurrency, the revised planning target is
**Monday, October19, Madrid time**, with **Tuesday, October20** upload contingency,
instead of October20–21. This assumes roughly two-to-threefold audit concurrency
where disk admission allows it; it is not a measured throughput claim. Keep the
October19 02:29:18UTC training deadline and the original24h evaluation ceiling.
Requote after the first parallel cohort and evaluation cost pilot. The
October19/20 target is provisional until the parallel measurements are available.
