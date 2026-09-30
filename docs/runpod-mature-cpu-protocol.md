# Mature HU20 CPU comparison — six classes, $4 approved

Task 2 is separate from [draft PR #132](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/132).
Current merged main was pulled again at `8f50f1f` before this branch. No
optimization is vendored into this PR. Runtime points to #132's exactly
validated source `50326afcd4776308054e0c9efce8681de1e877eb`, explicitly unmerged,
with its candidate and literal-base benchmark harness. Python 3.11.14 and
engine 5db20e3 remain pinned. #129 provides the earlier small-table platform
parity evidence; this task measures mature state/resources independently.

**On September 30 the owner approved all six CPU classes and doubled the total pilot cap to $4, with concurrent independent pods allowed. No Linux outcomes or rentals preceded this amendment.**
The [plan](../configs/blueprint/runpod-mature-cpu-pilot.json), live API catalog
and capture provenance are committed. Recheck allocation, CPU-only status,
availability and hourly price before creating a pod. Do not exceed the frozen
rates; unavailable classes are explicitly pending, never replaced based on
performance. Attempt all six declared classes. Preserve unavailable allocations and failed attempts explicitly; do not substitute a different class or silently retry a failed workload.

| Class | Requested allocation | Live compute rate |
| --- | --- | ---: |
| CPU3 compute-optimized | 2 vCPU / 4 GB | $0.060/h |
| CPU3 general-purpose | 2 vCPU / 8 GB | $0.080/h |
| CPU3 memory-optimized | 2 vCPU / 16 GB | $0.110/h |
| CPU5 compute-optimized | 2 vCPU / 4 GB | $0.070/h |
| CPU5 general-purpose | 2 vCPU / 8 GB | $0.092/h |
| CPU5 memory-optimized | 2 vCPU / 16 GB | $0.130/h |

At two hours each, total quoted compute is at most **$1.084**; the **$4 total cap** includes setup, disk, transfers/retrieval, termination and billing uncertainty. Rates were rechecked against the provider CPU catalog before this amendment; allocation/availability and actual total hourly rate must be checked again at provisioning. Use 30 GB disposable container disk, no GPU or persistent/network volume.

Launch up to **six independent pods concurrently**, as close together as provider availability permits, with one heavy worker per pod. Parallelism is between isolated rentals, not multiple trainer workers inside a pod. Freeze one two-hour campaign rental cutoff before the first creation; each pod must stop no later than that cutoff, which also bounds every individual rental to two hours. Arm an independent M4 network-only shutdown watchdog for all six exact owned names before creation. Do not launch if the watchdog is not armed/healthy. It must discover a created pod by its exact owned name even if a creation response is lost. Preserve unknown-outcome creation attempts and reconcile them; do not blindly retry provisioning.

M1 only orchestrates lightweight API/Git/status/transfers. Linux training may overlap; retrieve and verify each archive sequentially on M4 before operator termination. On meaningful state divergence, stop all remaining owned workloads and preserve partial artifacts rather than completing extra training. At the hard rental cutoff, provider termination takes precedence over waiting for retrieval; report any unretrieved partial explicitly. Verify absence and no retained owned billable storage. Record posted itemized cost if available; otherwise distinguish measured compute-time cost and observed attributable debit from an unposted final invoice. Never publish credentials or total account balances.

Record actual CPU vendor/model/family/stepping/microcode, logical affinity, visible core/socket/SMT topology, cgroup CPU quota/period and memory/swap limits, kernel/architecture, provider flavor/data center, requested and actual shape, live compute and total hourly rate, image identity and pinned engine build. More visible host cores do not imply ownership. Flavor names do not prove clock speed, memory bandwidth, or dedicated physical cores. This is one instance per class; host variation remains a limitation, not an estimated population-wide class ranking.

The original three-class plan used by the already completed M4 reference is retained byte for byte in [reference-plan.json](reports/runpod-mature-cpu-artifacts/m4-reference/reference-plan.json). The amended paid matrix changes no reference workload, checkpoint, seed, runtime, RNG, scientific settings or recovery point. The M4 reference is not rerun.

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

The 4/8/16 GB shapes are admitted only if observed M4 peak and a Linux limit check
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
