# RunPod pod CPU benchmark

October 3, 2026. This report measures the CPU actually delivered by nine RunPod pod types, compared with the owner's M1. It informs host choice for CPU-bound solver work: the board-pooling batch (#149), the turn-search arena and the latency calibration in #148. Training, evaluation and playing strength are out of scope. The M4 was not measured because #148 was timing solves on it.

## Method

[`scripts/benchmark_pod_cpu.py`](../../scripts/benchmark_pod_cpu.py) uses only the standard library and finishes in under two minutes. It prints:

- **Single-core throughput:** zlib level 6 on a fixed, seeded 8 MiB pseudo-text block, best of three runs of 24 MiB each.
- **Aggregate throughput:** N worker processes (2, 4, 8, 16, 32 and all visible threads), each compressing 24 MiB twice. Inputs are built in a pool initializer before timing.
- **Usable CPUs:** scheduler affinity and the cgroup quota (`cpu.max`, or v1 `cfs_quota_us`). Inside GPU pods `nproc` and `lscpu` report every host thread, often 112–128, while the cgroup quota is the real allocation.
- **Pure-Python loop rate:** recorded but not compared. It depends on the interpreter: the pods ran Python 3.8, the M1 Python 3.14.

zlib runs in C, so it compares CPUs across Python versions. It is a general integer workload rather than our float-SIMD solver; the solver's own speed on each host still needs its parity and timing pilot before production use.

Each pod used image `runpod/base:0.7.0-ubuntu2004`. The script was passed base64-encoded in the container command, with `lscpu`, `nproc`, `free` and the cgroup memory/CPU limits. Output was read from container logs through the RunPod API, with no SSH or file transfer. Every pod was terminated right after its output was read, usually 1–5 minutes after creation. The committed script is a tidied version of the one the pods ran, with identical measurement semantics. Rerun on the M1 under Python 3.11, it gives 54.5 MB/s single core and 282.1 MB/s with eight processes, against 55.3 and 274.9 originally.

## Results

Ratios are relative to the M1's 55.3 MB/s single-core figure. "Peak aggregate" is the best measured process count. "Per $/h" divides peak aggregate MB/s by the hourly price. "Solver workers" is min(usable CPUs / 2, RAM / 5.37 GB): workers of two threads and at most 5 GiB, the shape used by #145 and #149.

| Pod | $/h | Host CPU | Single core (vs M1) | Usable CPUs | RAM | Peak aggregate | Per $/h | Solver workers |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple M1 (owner) | — | Apple M1 | 55.3 (1.00×) | 8 | 16 GB | 274.9 | — | — |
| cpu5c, 4 vCPU, secure | 0.14 | EPYC 4564P (Zen 4, 5.9 GHz) | **57.2 (1.03×)** | 4 vCPU | 8 GB | 173.0 | 1,236 | 1 |
| RTX 3090, community | 0.22 | EPYC 7663 | 38.6 (0.70×) | 23.8 | 62 GB | 857.9 | **3,900** | **11** |
| RTX 3070, community | 0.13 | EPYC 7663 | 36.1 (0.65×) | 18.7 | 24 GB | 531.8 | **4,091** | 4 |
| RTX 2000 Ada, secure | 0.24 | EPYC 7702 | 32.8 (0.59×) | 27.2 | 31 GB | 863.1 | 3,596 | 5 |
| RTX A4000, secure | 0.25 | EPYC 7702 | 32.9 (0.59×) | 15.3 | 62 GB | 480.4 | 1,922 | 7 |
| RTX PRO 4500, secure | 0.72 | EPYC 7663 | 37.5 (0.68×) | 23.8 | 62 GB | 830.7 | 1,154 | 11 |
| RTX 4090, secure | 0.74 | EPYC 75F3 | 41.8 (0.76×) | 13.6 | 62 GB | 532.3 | 719 | 6 |
| A40, secure | 0.49 | Xeon Gold 6342 | 34.9 (0.63×) | 7.65 | 50 GB | 211.4 | 431 | 3 |

The [raw results](runpod-pod-cpu-benchmark-artifacts/results.json) record every per-process-count measurement, pod ID, data center, listed vCPU/RAM, host thread count, cgroup limits and failure.

Oversubscription hurts. Running one process per visible host thread instead of per quota CPU cut aggregate throughput by 26–42% (for example, the 2000 Ada fell from 863 to 502 MB/s with 128 processes). Worker pools must be sized from the cgroup quota.

## Interpretation

- **Fast single solves (#148's 30-second turn-search latency): CPU5 pods.** Only the CPU5 EPYC 4564P matches the M1 per core; #133 also measured it at about 0.9× the M4 on the trainer. Every GPU-pod host was 24–41% slower per core. For ~5-GiB solves, cpu5g (4 GB per vCPU) is the matching shape. One cpu5c 4-vCPU pod is two physical cores, with SMT giving a 1.5× gain from two to four processes.
- **Parallel batches (#149 board pooling, arena hands): RTX 3090 community.** At $0.22/h it balances about 24 CPUs with 62 GB, enough for about 11 two-thread 5-GiB workers, with the second-best aggregate per dollar. The 3070 is marginally cheaper per MB/s but its 24 GB allows only about four such workers. The 2000 Ada is RAM-limited the same way.
- **Avoid high-end GPUs for CPU work.** The 4090, PRO 4500 and A40 deliver 3–9× less CPU throughput per dollar than the 3090 community pod.

## Limits

- **One pod per type.** RunPod assigns hosts per placement, so the same GPU type can land on a different CPU. The A4000 and 2000 Ada both drew EPYC 7702; the 3090, 3070 and PRO 4500 drew EPYC 7663.
- **Community cloud** uses third-party hardware. It suits resumable research jobs with atomic results, not unattended unrecoverable runs.
- **zlib is a proxy.** Solver parity, memory and per-solve timing must still be measured on the selected pod before production.
- **Not measured:**
  - cpu3c (8 vCPU, EUR-IS-1) reported RUNNING but emitted no log lines in about 4.5 minutes and was terminated.
  - RTX 4000 Ada and RTX A4500 had no stock.
  - The M4 remains unmeasured.

## Placement findings

- CPU pod requests (cpu5c, cpu5g, cpu3c and cpu3g at 8 vCPU) with a 10-GB container disk all failed with "no longer any instances available". A cpu5c 4-vCPU and a cpu3c 8-vCPU request with a 20-GB disk were placed immediately. One 3090 community request also failed at 10 GB and succeeded later at 20 GB. Use a disk of at least 20 GB.
- An RTX 2000 Ada request with `minVcpuCountPerGpu=32` failed, while the same type without the filter placed a 32-vCPU host.

## Cost

Estimated at about **$0.11** in total from listed hourly rates and pod lifetimes; settled billing was not checked. Every benchmark pod is terminated. Four older stopped pods from September 24–25 on the account were not created by this work and were left untouched.
