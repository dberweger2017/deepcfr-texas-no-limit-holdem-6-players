# Mature HU20 CPU pilot — M4 reference complete, six-class pilot approved

September 30, 2026. [Draft PR #133](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/133) is separate from the [observation-reuse optimization, draft #132](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/132). Main was pulled to `8f50f1f` before branching. No rental has been created. On September 30 the owner approved **all six classes, a $4 total pilot cap and concurrent independent pods**, before any Linux outcomes. There is no Linux mature-state parity result or measured winning CPU class yet.

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

## Approved six-class comparison — launch pending

The live CPU catalog was rechecked on September 30. At 2 vCPU, CPU3 compute/general/memory shapes provide 4/8/16 GB for **$0.060/$0.080/$0.110 per hour**; CPU5 equivalents cost **$0.070/$0.092/$0.130 per hour**. [Catalog/provenance](runpod-mature-cpu-artifacts/live-cpu-catalog.json) is retained; actual allocation/availability/price must be rechecked before provisioning. At a two-hour cutoff, maximum quoted compute for all six is **$1.084**. The owner-approved **$4 total** includes setup, storage, retrieval and billing uncertainty. No account balance is published or presumed current.

All six may run concurrently, one heavy worker per pod, with an independently armed M4 shutdown watchdog and one fixed rental cutoff. M4 archive verification remains sequential. Capture actual CPU model/vendor/topology/affinity and cgroup quota/memory/swap, provider shape/data center/rate and native build identity. The smaller 4 GB shapes are admitted only after checking actual allocation against the existing RSS/headroom guard. No GPU, persistent volume or assumed core scaling. A single instance per class cannot characterize cloud host variance.

The [amended frozen protocol](../runpod-mature-cpu-protocol.md) records the prospective expansion. The original M4 [reference plan](runpod-mature-cpu-artifacts/m4-reference/reference-plan.json) and all measured reference outcomes remain unchanged. Retrieve/verify before operator termination, verify pod absence afterward, and preserve failures. Known #129 export gzip OS-byte differences must be distinguished from meaningful-state divergence, which stops the pilot. No mature Linux result or winning CPU class is claimed before execution.

## Retained startup attempt 1

The first six-class creation attempt used driver `c839877` and immutable cutoff `1790809835.502777`. CPU3 compute/general/memory creation returned HTTP 400 and created no owned pod. The three CPU5 shapes were allocated concurrently; setup built the pinned engine, but worker startup failed with `No module named scripts.mature_cpu_linux_worker` because the command ran from the frozen runtime checkout rather than the separate driver. **No trainer iteration or policy outcome ran.** The memory-shape setup log/archive is retained; the other owned pods were terminated promptly after this failure and their connection records retained. Provider absence was verified for all three. This is a deployment failure, not Linux/M4 state divergence or a performance result.

The reporting/launch repair changes only the working directory used to start the worker, preserves source `50326af`, parent/work/recovery/scientific settings, and records a focused regression. A fresh output attempt may use the **same original cutoff**, never reset it. Historical attempts and their rental costs remain in the $4 cap. Availability and provider rejection details are retained explicitly; the live catalog is not proof of allocatable capacity.

## Conditional future scaling plan

[Plan for owner review](../hu20-100m-to-500m-resource-plan.md) fixes proposed lineages/milestones and describes capacity, evaluation and storage gates. **No final paid host recommendation is possible before the approved six-class mature comparison.** No 500M continuation, poker evaluation, promotion or automatic merge is launched here.
