# Mature HU20 CPU comparison — frozen work, budget approval pending

Task 2 is separate from [draft PR #132](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/132).
Current merged main was pulled again at `8f50f1f` before this branch. No
optimization is vendored into this PR. Runtime points to #132's exactly
validated source `50326afcd4776308054e0c9efce8681de1e877eb`, explicitly unmerged,
with its candidate and literal-base benchmark harness. Python 3.11.14 and
engine 5db20e3 remain pinned. #129 provides the earlier small-table platform
parity evidence; this task measures mature state/resources independently.

**No rental may begin until the owner approves the $2 total pilot cap.**
The [plan](../configs/blueprint/runpod-mature-cpu-pilot.json), live API catalog
and capture provenance are committed. Recheck allocation, CPU-only status,
availability and hourly price before creating a pod. Do not exceed the frozen
rates; unavailable classes are explicitly pending, never replaced based on
performance. Two completed classes suffice for the requested 2–3 comparison;
do not hide any attempted failure.

| Class | Requested allocation | Live compute rate |
| --- | --- | ---: |
| CPU3 general-purpose | 2 vCPU / 8 GB | $0.080/h |
| CPU5 general-purpose | 2 vCPU / 8 GB | $0.092/h |
| CPU5 memory-optimized | 2 vCPU / 16 GB | $0.130/h |

At two hours each, total compute is at most $0.604; the $2 cap includes setup,
disk, transfers/retrieval, termination and billing uncertainty. Use 30 GB
disposable container disk, no GPU or persistent/network volume. One pod at a
time for this pilot, one trainer process per pod. Arm an independent M4
network-only shutdown lease before provisioning; each lease expires two hours
after creation. M1 only orchestrates lightweight API/Git/status/transfers.
Retrieve and verify completed/partial artifacts before operator termination,
then verify pod absence and no owned billable storage. Record posted itemized
cost if available; otherwise distinguish measured compute-time cost and
observed attributable debit from an unposted final invoice. Never publish
credentials or total account balances.

## Fixed workload

Use the exact retained first-seed B100M training parent in the plan: actual
100,000,029 complete lifetime nodes, iteration 246,212 and 1,496,914 entries.
Verify its hash before loading; preserve the original file. All settings,
seed, weighting, regret/current extraction and RNG semantics remain unchanged.
No poker-playing evaluation is performed, and pilot checkpoints are not
promoted or treated as a new strength experiment.

On M4 and each Linux class, execute the frozen benchmark harness in a fresh
process, candidate variant, **5M additional complete nodes**. Save the first
completed iteration crossing 2.5M, final checkpoint/current export and the
same next-iteration validation checkpoint. Resume that midpoint in another
fresh process to the same 5M boundary and next iteration. The resumed suffix
is duplicate validation work, not independent training. Preserve complete
iteration overshoot and any discarded work; never select an earlier endpoint
or rerun failed traversals. There is no persistent RNG between iterations;
compare seed/config/iteration-derived next roots and action RNG state hashes.

M4 uses the existing one-heavy-child/AC/caffeinate/resource workflow and `/tmp`
ownership coordination. RunPod setup uses one compiler job and the same native
engine revision. Capture physical host CPU, allocated CPU affinity/SMT sibling
mapping, cgroup CPU/memory limits and actual memory, rather than interpreting
all host cores/RAM as the allocation. Two vCPUs do not establish two physical
cores or useful two-worker throughput.

## Mandatory correctness and measured economics

Within each host, direct/resume/reload final/current/next artifact bytes and
every resumed non-timing iteration row must match exactly. Across Linux/M4,
compare the entire decompressed checkpoint/export bytes, all stored keys,
menus, regrets, strategy sums, visits, table/config/iteration, complete work
rows and next streams; no numeric tolerance. Compressed bytes are compared
too. #129's known export-only gzip OS byte 9 (macOS 19 versus Linux 3) is a
documented transport difference, not trainer divergence. Any other difference
requires a concrete cause; meaningful state divergence stops the pilot and
blocks a longer host recommendation. Do not normalize unexplained differences.

Report per class: direct/resume nodes/sec, whole-job wall time, load/midpoint/
final save/export/reload cost, process and aggregate/cgroup memory including
save/export peaks, new entries, bytes per table/entry, swap/disk, failures,
hardware allocation and recovery. Report compute dollars per million complete
nodes separately from the pilot's setup/recovery/retrieval-inclusive dollars.
Record duplicate validation work explicitly. Hash-seal logs after they close,
verify archives on M4, and retrieve only compact publication data to M1.

The 8 GB shapes are admitted only if observed M4 peak and a Linux limit check
leave room: guard owned RSS below min(10.5 GiB, 80% actual cgroup RAM). Swap
growth <=0.5 GiB and free disk >=8 GiB. Stop and retain the attempt on a guard,
invalid state, parity failure or cost/deadline problem; no blind retry.

## Deliverable and follow-on limit

Publish a separate draft PR/report with a measured host/RAM/worker
recommendation and a conditional three-lineage 100M→150M/200M/300M/500M plan,
including evaluation/storage/recovery costs and mature-growth uncertainty.
No automatic merge, substantial continuation, strategy change, promotion or
new rental beyond this approved pilot follows. The full campaign requires
separate owner approval of its final frozen budget/scientific plan.
