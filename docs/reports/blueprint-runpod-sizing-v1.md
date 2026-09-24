# RunPod blueprint sizing check

## First pod and hardware shown by RunPod

On 2026-09-24 the signed-in RunPod console deployed pod `k9rdph2fwhym87` (`blueprint-learning-check-20260924`) from the `runpod/base:0.7.0-ubuntu2004` template in region **US-CA-2**. The selected tier was **3 GHz memory-optimized, 8 vCPU, 64 GB RAM**, with a **20 GB disposable container disk** and no network volume. The console quoted **$0.44/hour CPU** and **$0.003/hour disk**. Its pod details identified the underlying processor as **AMD EPYC 7713 64-Core Processor**. That host model does not mean all 64 physical cores are assigned to this pod; the allocation shown was 8 vCPU.

The account's imported SSH key did not match any available private key on the local Mac or M4, so no commands or checkpoint transfer ran inside this pod. It was stopped with the console showing **$0.00/hour** continuing cost. The displayed balance moved from **$18.07 to $18.02** after delayed billing settled, an approximately **$0.05** charge. No scientific result or full topology measurement came from this first attempt.

The owner authorized adding the Mac's main Castl **public** key to the RunPod **account** SSH Public Keys list. The saved title is `castl main id_ed25519` and the fingerprint is `SHA256:UQqx9aybBRlDcmbVfyhDGJXJQzMvIcd4XZBFXIhjqy8`. This makes the same key available to subsequently created pods without copying the private key to RunPod or adding keys one pod at a time. Direct SSH authenticated successfully on the next pod.

## Active measurement pod

The replacement pod `xu414eguzakxfr` (`blueprint-sizing-20260924b`) in **US-MO-2** uses the same 3 GHz memory-optimized **8-vCPU/64-GB** tier, 20 GB disposable disk, and no network volume. Its live quote is **$0.44/hour CPU** plus **$0.003/hour disk**. The underlying host identifies as an **AMD EPYC 9354 32-Core Processor**: one socket, 32 physical cores, 64 hardware threads, one NUMA node, 32 MiB L2 and 256 MiB L3. The container image is `runpod/base:0.7.0-ubuntu2004`; the host kernel is Linux 6.14.

The container's `lscpu`/`nproc` show **64 host threads**, and `free -b` shows about **755 GiB host RAM**. These are host-visible values, **not** this pod's entitlement. The exposed cgroup v1 CPU quota is `-1` and no memory cgroup limit is mounted inside the container, so neither cgroup nor `free` independently verifies the 8-vCPU/64-GB allocation. Benchmarks use at most eight workers and a conservative 48-GiB aggregate process RSS guard. Raw hardware output is retained in the ignored local `results/blueprint-runpod-check-v1/blueprint-hardware/` directory.

The reviewed revision `7c29bd7e01e49faf5e756d8ee26d9d9734534eed` passed the eight blueprint tests on the pod. The transferred source checkpoint has the prescribed 252,235,464 bytes and SHA-256 `94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a`.

## Fixed 64-iteration worker comparison

Every arm loaded that same source, completed the same **341,389 traversal nodes**, advanced from iteration 8,733 to 8,797 and grew from 5,834,622 to 5,876,505 entries. All four output checkpoints are exactly identical: **254,045,478 bytes**, SHA-256 `2e692a8c5d015f06ee8bb7d1f8a9d7eb6dc4352ed22bf0a135cf551efed393ef`. This validates deterministic fixed-order merging across the measured worker counts.

| Workers | Work time | Nodes/work second | Load + work + save | Conservative parent + worker peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 42.94 s | 7,950 | 165.98 s | 4.80 GiB |
| 2 | 49.87 s | 6,845 | 175.11 s | 14.22 GiB |
| 4 | 55.05 s | 6,202 | 179.65 s | 23.67 GiB |
| 8 | 82.80 s | 4,123 | 206.19 s | 42.55 GiB |

The worker RSS sums include copy-on-write pages also visible in the parent and therefore overstate unique physical use, but they are a safe operational bound. Linux cgroup memory peaks were unavailable in this container. The four-worker bound stayed below the protocol's 45-GiB admission threshold for the eight-worker arm, and the eight-worker bound stayed below its 48-GiB runtime guard.

**More vCPUs do not currently accelerate this trainer at a 5.8-million-entry table.** `BlueprintTrainer.step` creates and shuts down a `ProcessPoolExecutor` every iteration, forking workers over the large current table. That repeated setup and related memory traffic is a likely explanation for the measured slowdown, although this check did not isolate costs inside each step. A 64-vCPU AWS rental would therefore be a poor default until a focused scalability change is measured. The M4 slice's approximately 12,957 nodes/work second was faster than this pod's best arm, but the runs differ in table growth and host, so this is a warning rather than a controlled cross-machine speed ratio.

The closer comparison is the M4's **last 64 iterations**, which processed 345,235 nodes in 26.45 step seconds at approximately **13,050 nodes/second** with a table growing toward 5.83 million entries. That is **1.64 times** this EPYC pod's one-worker rate. The segments are different, so this remains a sizing estimate, not a paired CPU benchmark. The pod's value here is its paid memory capacity, not a speed advantage at the present worker architecture.

## Export and fresh arena check

The original iteration-8,733 checkpoint exported both policies. The current export took 68.47 seconds and is 161,580,508 bytes; the average export took 40.26 seconds and is 116,480,491 bytes. The highest process RSS through export was **7,868,096,512 bytes (7.33 GiB)**. The runner records a process lifetime high-water mark, so its identical per-export peak fields do not establish each export's independent peak. Both model files and the uniform control have content hashes in the retained results. The full check, including load, export and both arenas, completed below its 48-GiB guard.

Each policy played 1,536 fresh paired-arena hands against the frozen three-style pool, with **zero invalid actions**. The paired advantage over the uniform abstract-menu control was **+33.46 BB/100** for current, 95% CI [−88.11, +155.04], and **+59.23 BB/100** for average, 95% CI [−80.08, +198.54]. Both comparisons are inconclusive. The candidate itself lost approximately 771 and 745 BB/100, respectively, to this pool. This early 8,733-iteration check is not a strength gate for the longer run. The card probe found trained first-to-act preflop keys for all 169 hand classes; current folds 72o and mostly calls AA, while average folds 72o about 86% of the time and mixes AA calls and raises. Those observations show differentiated behavior, not good poker.

| Street | Current trained / decisions | Average trained / decisions |
| --- | ---: | ---: |
| Preflop | 236 / 968 (24.4%) | 232 / 965 (24.0%) |
| Flop | 34 / 374 (9.1%) | 25 / 379 (6.6%) |
| Turn | 3 / 140 (2.1%) | 1 / 138 (0.7%) |
| River | 3 / 65 (4.6%) | 2 / 58 (3.4%) |

These counts are actual candidate policy queries during held-out games; the two policies reach different states, so their denominators differ. They are much lower than the fixed later-policy self-play lookup rates in the [coverage-growth check](blueprint-coverage-growth-v1.md), especially preflop and flop. The key retains the full ordered public action history, with raise sizes bucketed. A new opponent style can therefore reach many unseen keys even when its cards fall into trained buckets. More sampling should fill some keys, as the M4 slice already demonstrated, but the growing history space can also consume memory faster than coverage becomes useful. This is the primary architecture risk for a larger campaign. Candidate mean action latency was **0.046 ms current** and **0.040 ms average** on this host, with 95th percentiles 0.082 and 0.055 ms; model loading and memory are separate deployment costs.

## Recovery, cost and decision

The source, all four worker checkpoints, and the current, average and uniform policy files were verified by SHA-256 after retrieval. The 22 files in the learning output match the remote file count and total byte count; the four worker output directories, hardware record, and lifecycle record are retained locally under ignored `results/blueprint-runpod-check-v1/`. The pod was created at about **10:19:43 UTC** and stopped at **10:47:50 UTC** on September 24. After the stop, its details and the first pod's row each showed **$0.00/hour** continuing cost. The displayed balance was **$17.83** immediately after stopping, down from **$18.02** before this second pod; billing can settle later. At the quoted rates, the second pod's approximately 28-minute lifetime is about **$0.21** before any transfer charge. This is within the $3 test cap.

The correctness/recovery gates passed. The short arena did not establish playing strength, as anticipated. The two important sizing findings are that the current Python trainer has **negative multiworker scaling** at this table size and that a **5.83-million-entry table reaches few held-out decisions** against different opponent styles. Reserving a 64-vCPU, 512-GiB AWS machine for a long run on the unchanged trainer would pay for mostly unused cores and risk spending RAM on rare histories. One focused architecture decision is warranted before freezing that run: compare a shorter, still player-visible public-history summary against the present full-history key at equal traversal work, with held-out style-pool lookup coverage and legal play as the immediate checks. Separately, measure a worker-lifecycle change if a many-core host remains attractive. This should be one decision-oriented pilot, followed promptly by the larger memory-backed campaign, not a series of small poker-strength gates.
