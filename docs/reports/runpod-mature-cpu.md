# Mature HU20 CPU pilot — two verified classes; six-class comparison incomplete

September 30, 2026. [Draft PR #133](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/133) is separate from the [observation-reuse optimization, draft #132](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/132). Main was pulled to `8f50f1f` before branching. The owner approved **all six classes, a $4 total pilot cap and concurrent independent pods** before Linux outcomes. All initial rentals are now terminated. CPU5 general and memory completed mature Linux/M4 parity; CPU3 allocations were rejected and CPU5 compute lost part of its archive during premature controller cleanup. **This is a partial comparison, not a six-class winner.** No playing-strength outcome was evaluated.

## Frozen work and executed M4 reference

The [protocol](../runpod-mature-cpu-protocol.md) and [plan](../../configs/blueprint/runpod-mature-cpu-pilot.json) fix one retained first-seed B100M parent, 5M additional complete nodes, the first complete iteration crossing 2.5M as the recovery point, and a fresh-process resume to the same endpoint/next iteration. Python 3.11.14, engine `5db20e3`, and validated candidate runtime `50326afcd4776308054e0c9efce8681de1e877eb` are identical to #132. That runtime remains **unmerged**, explicitly identified rather than represented as current main. Driver revision `6f8799d` ran the M4 reference; verification/worker-test source is `5c18b60`. Original checkpoints were unchanged.

| Measurement | M4 result |
| --- | ---: |
| Parent actual complete nodes / entries | 100,000,029 / 1,496,914 |
| Direct additional nodes / overshoot / iterations | 5,000,174 / 174 / 10,923 |
| Final entries / new entries | 1,527,693 / 30,779 |
| Direct training throughput | 20,314.39 nodes/sec |
| Direct training / internal whole workload | 246.14 / 292.54 seconds |
| Fresh-process direct wall time including startup | 294.06 seconds |
| Resume validation additional work | 2,499,660 nodes; duplicate validation, not diversity |
| Resume throughput / fresh-process wall time | 20,192.58 nodes/sec / 171.38 seconds |
| Direct load / midpoint save / final save | 5.55 / 9.82 / 9.97 seconds |
| Direct current export / next checkpoint save | 7.29 / 9.87 seconds |
| Resumed reload-and-save validation | 9.79 seconds |
| Direct process peak / sampled aggregate peak | 2.067 / 2.053 GiB |
| Sampled swap growth / minimum free disk | zero / 33.38 GiB |
| Final compressed checkpoint / current export | 72,252,600 / 40,949,674 bytes |

This is one mature workload measurement, not a confidence interval or performance forecast at 500M. The one-hour reference clock started at `1790797797.6612968`; training/recovery finished at `1790798264.3007529`. Publication verification finished at `1790799264.813613`, before its unchanged deadline `1790801397.6612968`. All power samples were AC. One heavy child ran at a time. The M4 is released; New Guy has a read-only artifact-transfer window, with his computation on M1.

## Correctness and retained evidence

The direct and resumed final/current/next artifacts agree **byte for byte**, including full decompressed payloads. All 5,445 resumed non-timing iteration rows equal the direct suffix; all 10,923 direct iterations reconcile with completed work, attempted nodes and new entries. Next RNG roots/state hashes agree. Independent streaming verification rehashed all **26 archive files**, checked exact membership, preserved the parent hash, and verified all resource guards. No failed phase or discarded traversal occurred. Five Linux-worker guard/cleanup tests passed on M4; those tests are synthetic guard tests, not a Linux trainer result.

[Independent verification](runpod-mature-cpu-artifacts/m4-reference/independent.json), [direct result](runpod-mature-cpu-artifacts/m4-reference/direct.json), [recovery comparison](runpod-mature-cpu-artifacts/m4-reference/resume-verification.json), [reference supervisor](runpod-mature-cpu-artifacts/m4-reference/campaign.json), [worker tests](runpod-mature-cpu-artifacts/m4-reference/worker-guard-tests.log), and [manifest](runpod-mature-cpu-artifacts/m4-reference/manifest.json) are compact publication evidence. All eight transferred compact files matched independent M4 hashes in [transport verification](runpod-mature-cpu-artifacts/m4-reference/transport-verification.json).

The reference manifest SHA-256 is `cb10bf436d0b1eeff9cf79a63ba3b60b27f764601602a96bfba8418ade671b02`. Full checkpoints/iteration traces remain on M4:

```sh
scp -o HostName=100.122.216.94 -o BatchMode=yes -r \
  m4:/Users/dberweger/Local/runpod-mature-cpu-20260930/results/m4-mature-cpu-reference-20260930 ./
scp -o HostName=100.122.216.94 -o BatchMode=yes \
  m4:/Users/dberweger/Local/runpod-mature-cpu-20260930/results/m4-mature-cpu-reference-20260930-manifest.json ./
```

Check the manifest hash first, then every listed size/hash. Original parent path/hash and exact executed commands are in the plan and supervisor records. Verification is reproducible with `python -m scripts.verify_mature_cpu_reference --root <root> --out <fresh-sidecar>` before the recorded deadline; after expiry, verify hashes directly without resetting the attempt clock or rerunning training.

## Approved six-class comparison — execution and coverage

The live CPU catalog was rechecked on September 30. At 2 vCPU, CPU3 compute/general/memory shapes provide 4/8/16 GB for **$0.060/$0.080/$0.110 per hour**; CPU5 equivalents cost **$0.070/$0.092/$0.130 per hour**. [Catalog/provenance](runpod-mature-cpu-artifacts/live-cpu-catalog.json) is retained; actual allocation/availability/price must be rechecked before provisioning. At a two-hour cutoff, maximum quoted compute for all six is **$1.084**. The owner-approved **$4 total** includes setup, storage, retrieval and billing uncertainty. No account balance is published or presumed current.

All six may run concurrently, one heavy worker per pod, with an independently armed M4 shutdown watchdog and one fixed rental cutoff. M4 archive verification remains sequential. Capture actual CPU model/vendor/topology/affinity and cgroup quota/memory/swap, provider shape/data center/rate and native build identity. The smaller 4 GB shapes are admitted only after checking actual allocation against the existing RSS/headroom guard. No GPU, persistent volume or assumed core scaling. A single instance per class cannot characterize cloud host variance.

The [amended frozen protocol](../runpod-mature-cpu-protocol.md) records the prospective expansion. The original M4 [reference plan](runpod-mature-cpu-artifacts/m4-reference/reference-plan.json) and all measured reference outcomes remain unchanged. Retrieve/verify before operator termination, verify pod absence afterward, and preserve failures. Known #129 export gzip OS-byte differences must be distinguished from meaningful-state divergence, which stops the pilot. The observed allocation and executed results follow. Flavor names do not establish dedicated cores or guaranteed clock speeds.

## Retained startup attempt 1

The first six-class creation attempt used driver `c839877` and immutable cutoff `1790809835.502777`. CPU3 compute/general/memory creation returned HTTP 400 and created no owned pod. The three CPU5 shapes were allocated concurrently; setup built the pinned engine, but worker startup failed with `No module named scripts.mature_cpu_linux_worker` because the command ran from the frozen runtime checkout rather than the separate driver. **No trainer iteration or policy outcome ran.** The memory-shape setup log/archive is retained; the other owned pods were terminated promptly after this failure and their connection records retained. Provider absence was verified for all three. This is a deployment failure, not Linux/M4 state divergence or a performance result.

The reporting/launch repair changes only the working directory used to start the worker, preserves source `50326af`, parent/work/recovery/scientific settings, and records a focused regression. A fresh output attempt may use the **same original cutoff**, never reset it. Historical attempts and their rental costs remain in the $4 cap. Availability and provider rejection details are retained explicitly; the live catalog is not proof of allocatable capacity.

## Initial matrix: actual results

The first startup attempt did zero training. In corrected attempt 2, three CPU5 pods were allocated within four seconds and trained concurrently, each with one heavy child and the identical frozen workload. The original rental cutoff **1790809835.502777** (September 30 23:10:35 UTC / October 1 01:10:35 Madrid) was never reset. The coordinator and independent shutdown watchers are closed; provider absence was verified after termination.

| Planned class | Allocation / price per hour | Executed status |
| --- | --- | --- |
| CPU3 compute | 2 vCPU / 4 GB / $0.060 | Auto-placement and one prospectively declared EU-RO-1 placement returned HTTP 400; no pod/training |
| CPU3 general | 2 vCPU / 8 GB / $0.080 | Same two retained provisioning rejections; no pod/training |
| CPU3 memory | 2 vCPU / 16 GB / $0.110 | Same two retained provisioning rejections; no pod/training |
| CPU5 compute | 2 vCPU / 4 GB / $0.070 | Remote workload exited 0; archive truncated during retrieval; **complete parity/performance result unavailable** |
| CPU5 general | 2 vCPU / 8 GB / $0.092 | Complete archive, independent Linux/M4 and resume parity passed |
| CPU5 memory | 2 vCPU / 16 GB / $0.130 | Complete archive, independent Linux/M4 and resume parity passed after metadata-only verifier repair |

The CPU3 API response parser originally retained HTTP status but omitted the provider's `detail` field. Therefore the specific rejection cause is **unknown**; captured catalog HIGH availability does not prove capacity or explain rejection. The parser now preserves that bounded field for future failures. A separately guarded console provisioning investigation created no pod and ran no training: the exact ownership name could not be entered reliably, so admission was cancelled. All these records remain retained rather than relabeled as benchmark results.

## Measured mature performance and hardware

| Measurement | M4 | CPU5 general, 8 GB | CPU5 memory, 16 GB |
| --- | ---: | ---: | ---: |
| Direct nodes/sec | 20,314.39 | 17,662.72 | 17,770.10 |
| Direct training seconds | 246.14 | 283.09 | 281.38 |
| Fresh-process direct wall seconds | 294.06 | 358.45 | 356.44 |
| Fresh-process resumed wall seconds | 171.38 | 213.27 | 213.26 |
| Resumed nodes/sec | 20,192.58 | 17,782.02 | 17,787.49 |
| Load / midpoint save / final save seconds | 5.55 / 9.82 / 9.97 | 10.89 / 13.68 / 13.88 | 10.83 / 13.56 / 13.81 |
| Current export / next checkpoint save seconds | 7.29 / 9.87 | 12.87 / 13.79 | 12.84 / 13.73 |
| Resumed reload/save seconds | 9.79 | 13.49 | 13.39 |
| Direct process peak GiB | 2.067 | 1.848 | 1.849 |
| Sampled aggregate owned RSS peak GiB | 2.053 | 1.777 | 1.807 |
| Cgroup peak, including cache GiB | not comparable | 3.732 | 3.744 |
| Maximum sampled swap growth | zero | zero | 12,288 bytes |
| Minimum free disk GiB | 33.38 | 27.98 | 27.98 |
| Training-only dollars / million unique nodes | — | $0.001447 | $0.002032 |
| Direct + duplicate resume/check compute / million unique nodes | — | $0.002927 | $0.004122 |

Each verified direct run completed **5,000,174 nodes**, 174-node overshoot, **10,923 iterations**, and **30,779 new entries**, ending at 1,527,693 entries. Recovery repeats 2,499,660 nodes / 5,445 suffix iterations as validation, not independent learning. No completed traversal was discarded in those runs. Final trainer checkpoint/current export sizes are identical across platforms: 72,252,600 / 40,949,674 bytes. Process maximum and sampled aggregate peaks differ because sampling can miss a short serialization peak; retain both. Cgroup memory also includes cache and is not process RSS. All recorded worker guards passed.

All allocated CPU5 instances reported **AMD EPYC 4564P 16-Core Processor**, family 25/model 97/stepping 2, Linux 6.8.0-51 x86_64, Ubuntu 20.04 image `runpod/base:0.7.0-ubuntu2004`. Verified pods were in EUR-IS-1. Their affinities were **[7,23]** and **[4,20]**; topology maps each pair to two SMT threads of **one physical core**. CPU5 compute's preserved prefix reports [9,25], likewise a sibling pair. Cgroup CPU quota was `-1` with 100,000µs period; affinity, not all 32 visible host threads, bounds the allocated CPUs. Actual verified RAM limits were 8,000,000,000 and 16,000,000,000 bytes. The pinned native binary hash was identical between the two Linux builds; its architecture-specific bytes need not equal the M4 binary. [Hardware, quota, worker/resource and result records](runpod-mature-cpu-artifacts/linux-pilot/measurements.json) are retained with the full CPU inventory.

The memory shape was only 0.61% faster on this one workload and cost about 40% more per training node. This does not establish a general class ranking, memory-bandwidth effect, or significant difference. The M4 was faster than both for one worker. Larger CPU allocations may change placement, but this pilot does not validate intra-pod parallelism.

## Exact state parity, reporting failure and lost retrieval

Both complete Linux archives match the M4's **entire decompressed trainer/current/next state**, all keys, menus, regrets, strategy sums, visits, configuration, iteration/work and next RNG streams, with no numeric tolerance. All 10,923 direct rows and 5,445 fresh-process resumed suffix rows match. Trainer and next checkpoints are **byte-identical**; the inference export differs **only at gzip OS header byte 9**, macOS 19 versus Linux 3, as documented in #129. Within Linux, direct/resumed artifacts are byte-identical. Each independent verifier rehashed all 26 worker files and exact membership. This is correctness/resource evidence, not a new model promotion.

The original memory-pod verifier failed solely on raw `engine_origin` JSON formatting: the same URL/revision fields were serialized with different whitespace/order. It had already passed trainer state, RNG, work and archive checks. The controller treated that failure as a stop and **terminated the other rentals before every retrieval finished**. Consequently CPU5 compute's completed archive was truncated at **142,442,496 bytes**, and its remote checksum/full state were lost. That was a controller error; it is not evidence of trainer divergence. The partial archive and 15 complete small metadata members were preserved; parsing ends with `unexpected end of data`. It cannot substitute for complete parity or a measured class comparison. No gameplay or training was rerun to replace that evidence.

The metadata check now compares structured JSON identity; a regression rejects a genuinely changed revision while accepting formatting differences. Future controller verification waits until every archive transfer closes before cleanup. **The original failed verifier and combined transport verdict remain unchanged**, beside corrected sidecars. A bounded M4 reporting-only phase verified both complete archives using corrected source `0b1bb4b`, with all resource guards and original cutoff preserved. [Original failure](runpod-mature-cpu-artifacts/linux-pilot/original-metadata-verifier-failure.json), [corrected memory parity](runpod-mature-cpu-artifacts/linux-pilot/cpu5m/platform-verification.json), [general parity](runpod-mature-cpu-artifacts/linux-pilot/cpu5g/platform-verification.json), [transport repair basis](runpod-mature-cpu-artifacts/linux-pilot/cpu5m/transport-metadata-repair.json) and [partial retrieval](runpod-mature-cpu-artifacts/linux-pilot/partial-retrieval.json) keep these distinctions explicit.

Ten focused M4 worker/controller tests passed; full GitHub CI and GitGuardian passed on correction commit `0b1bb4b`. Linux setup also passed its five focused worker tests. A publication supervisor's first launch failed before any child because it was outside a Git checkout; its metadata startup failure is retained, and the corrected reporting launch changed only the working directory. No scientific source, workload, parent, RNG or recovery setting changed.

## Cost and shutdown

All six actually created pods—three setup-only failures and three corrected CPU5 allocations—are terminated; **zero owned pods or persistent billable volumes remain**. Their creation-to-confirmed-absence intervals give a conservative compute-time estimate of **$0.068865**, including setup, direct work, duplicate recovery, retrieval and failed startup. This uses the captured hourly rates and latest possible termination time; it is **not settled billing** and may exclude provider disk charges/rounding. The six pilot-specific billing queries returned no posted records. An empty ledger does not mean the run was free. [Per-pod billing query and estimate](runpod-mature-cpu-artifacts/linux-pilot/billing.json), [absence check](runpod-mature-cpu-artifacts/linux-pilot/absence.json) and all allocation/operator/watchdog records are published; no credentials or account balance are included. The owner-approved total cap remains **$4**.

## Archive, recommendation and conditional continuation

All closed initial attempts and complete/partial archives are sealed in the [final manifest](runpod-mature-cpu-artifacts/linux-pilot/final-manifest.json), whose hash and file count are in [seal.json](runpod-mature-cpu-artifacts/linux-pilot/seal.json). Compact publication transfers were independently matched against that M4 inventory. Ephemeral private keys and known-hosts are excluded. Large files remain on M4; retrieve the following without rerunning training:

```sh
scp -o HostName=100.122.216.94 -o BatchMode=yes -r \
  m4:/Users/dberweger/Local/runpod-mature-cpu-six-20260930/results/runpod-mature-cpu-six-20260930-attempt-2 ./
scp -o HostName=100.122.216.94 -o BatchMode=yes \
  m4:/Users/dberweger/Local/runpod-mature-cpu-six-20260930/results/mature-cpu-final-manifest.json ./
scp -o HostName=100.122.216.94 -o BatchMode=yes -r \
  m4:/Users/dberweger/Local/runpod-mature-cpu-six-20260930/results/runpod-mature-cpu-six-20260930 ./
```

Verify the manifest SHA and each listed archive/hash before analysis; the CPU5 compute archive deliberately remains partial. The cancelled console root and metadata-repair supervisors also remain in the established M4 root. Follow the cleanup archive policy when Google Drive upload verification is separately authorized; this report does not delete or evict originals.

**Conditional host recommendation:** CPU5 memory, **2 vCPU / 16 GB, one trainer per pod**, is a validated mature host with more growth headroom; three isolated lineage pods would free the M4 for diagnostics. CPU5 general is cheaper per node at 105M, but its future memory headroom is less certain. Neither is a six-class winner, and M4 remains faster per worker. The [150/200/300/500M proposal](../hu20-100m-to-500m-resource-plan.md) quotes measured baselines and capacity/evaluation/storage gates. It requires separate final budget/guard approval; the existing 3M-entry limit may stop before 500M. No large campaign, playing evaluation, strategy change or automatic merge is authorized here.

The owner has separately requested a **16-vCPU / 32-GB CPU5 compute** comparison, after the initial evidence is pushed. It is an extra engineering control, not a replacement for missing initial classes. Its protocol/results will be appended separately; none was launched for this initial report.
