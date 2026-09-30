# RunPod HU20 platform pilot

## Current result

**Linux training and deterministic recovery passed. The fresh current-source
M4 reference is queued, so the requested comparison is not yet complete.**
The running [posterior audit #128](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/128)
retains exclusive heavy M4 use. Its source, clock and processes were not changed.

The Linux run used frozen harness `e18f0079a14addc90938acca8c30795e8af09691`,
merged base `7d31a39c80cc72deb772336ce251f3c4bdcd46c9`, Python 3.11.14 and
engine `5db20e3d5d6862b32a7402035c1340b622d3b005`. It followed the
[frozen protocol](../runpod-hu20-parity-protocol.md) and
[plan](../../configs/blueprint/runpod-hu20-parity.json). No trainer, game,
abstraction, action menu or RNG change was made. No poker evaluation ran.

## Instance, resources and cost

The owner authorized this small CPU rental. Live catalog selection chose
`cpu3c`, **2 vCPU / 4 GB RAM / no GPU**, region US-CA-2, official
`runpod/base:0.7.0-ubuntu2004`, with 20 GB disposable container disk and no
persistent/network volume. Pod `xdyx1equpj1u95` was created at
2026-09-30 15:14:39.674 UTC and its termination/absence was verified at
15:28:35.262 UTC: approximately **13 minutes 56 seconds**, including setup,
tests, both training paths, comparison and artifact retrieval.

The live compute rate was **$0.06/hour**. The final stable account debit
observed after termination was **$0.014613**, about **1.46 US cents**.
This is an attributable balance decrement while this was the sole active
pod, not an itemized invoice. The pod billing endpoint still returned no
posted record at publication; that empty response does not mean free use.
Compute rate × observed lifetime is $0.013926 before storage/rounding.
RunPod confirmed zero active hourly spend after termination. The independent
M4 network-only shutdown watchdog was armed before provisioning, then
released after operator termination was verified. Older exited pods were
untouched. No billable storage was intentionally retained for this pilot.

The physical host was an AMD EPYC 7713. Its 256 logical CPUs and ~1 TB
visible host RAM are **not** the pod allocation. The container had two
allowed logical CPUs (`116,244`, one SMT pair), memory limit
3,999,997,952 bytes, and no CPU quota beyond that affinity. Training used
one worker. No claim of useful two-worker scaling follows.

| Measurement | Direct from zero | Fresh-process resume |
| --- | ---: | ---: |
| Final completed nodes | 1,000,389 | 1,000,389 lifetime |
| Final iterations / entries | 2,626 / 118,978 | 2,626 / 118,978 |
| Training work in this path | 1,000,389 | 499,764 |
| Training time | 147.187 s | 73.316 s |
| Training throughput | 6,796.73 nodes/s | 6,816.54 nodes/s |
| Whole path wall time | 155.549 s | 82.314 s |
| Peak trainer RSS, including saves | 171.82 MiB | 182.43 MiB |
| Final checkpoint save | 2.008 s | 1.895 s |
| Current-policy export | 1.925 s | 1.704 s |

The completed-node boundary overshot by 389 nodes because the stopping rule
preserves complete iterations. The midpoint was 500,625 nodes / iteration
1,302. Both paths performed one further **289-node validation iteration**;
the resumed suffix and extra iterations are recovery checks, not independent
seeds or additional useful campaign work. No failed/discarded traversal was
reported. Container peak memory including the native build/page cache was
1,542,389,760 bytes (1.44 GiB); recorded cgroup swap was zero. Available disk
after retrieval staging was 20,098,625,536 bytes, above the 8-GiB floor.

## Exact state and recovery evidence

The Linux direct path and fresh-process resumed path are byte-identical for
the entire final training checkpoint, current-policy export, and next-
iteration checkpoint. Reload/save at the midpoint is byte-identical too.
All **1,324 resumed non-timing iteration records** equal the corresponding
direct suffix. Derived next deal/action seeds and initial Python action-RNG
state hashes agree. There is no persistent trainer RNG across completed
iterations; seed/config/iteration determine the next root streams.

Linux also agrees with the retained first fixed-seed M4 #115 reference:

| Artifact | Compressed equality | Entire uncompressed payload |
| --- | --- | --- |
| 1M training checkpoint | Identical | Identical |
| Next-iteration training checkpoint | Identical | Identical |
| Current-policy export | One header byte differs | Identical |

Training-checkpoint equality covers every stored information-set key, menu,
regret, strategy accumulator, visit/update count, table/config field and
iteration, without a numerical tolerance. This is not a selected-key sample.
The checkpoint SHA-256 is
`a212b7047201a5188d75e1a37819b70e787d6e16282e694dbd08509123797b1b`;
the next checkpoint is
`5d85d36d89cdf7d5dd13c63dcb303b074a997dd840ef88e89428d2e70a79d3d2`.

The inference exports differ **only at gzip byte offset 9**: M4's OS marker
is 19 (macOS), Linux's is 3 (Unix). Their deflate data and CRC/size trailer
are identical, as is every decompressed byte. Both platforms use the same
unchanged Python 3.11 `gzip.compress(..., mtime=0)` writer; its platform
header explains the transport difference. There is no unexplained policy
difference. The payload SHA-256 is
`359b74cea42802be646288b54a8d892c9bb0fe8e0d3c5f0675127590b5f07eff`.

The historical reference ran source `a87e9f8805d211e2b21dbb710339ada9083eedf6`.
It is useful interim evidence but does not replace the requested fresh
current-source M4 comparison. That reference is reserved after #128's
coordinator, workers, reporting and wrapper finish. Its queue only polls
small status files while waiting, expires 30 minutes after #128's hard
cutoff, and gives the reference at most one hour including verification.
It preserves the same harness/plan, checks AC, aggregate RSS/swap/free disk,
and stops on a foreign large process or failure instead of retrying.

Linux focused tests passed **3/3**. GitHub's full `test` check passed at the
frozen harness; its external GitGuardian check was pending. No tests,
compilation, trainer/model loads or evaluation ran on the travelling M1.

## What this implies for future three-seed training

Linux is a promising training host: this fixed prefix and deterministic
recovery agree exactly with the retained M4 trainer state. A final host
recommendation waits for the fresh current-source M4 check. This pilot is
not a playing-strength result or a safe scaling/memory study for large
tables.

For planning only, three independent single-worker processes could use
three `cpu3g` pods, each **2 vCPU / 8 GB RAM**, whose captured catalog rate
was $0.08/hour per pod. Eight GB provides more headroom than this pilot's
four GB for the retained 100M tables and save/export overhead; it has not
been validated for a 300–500M table. Three separate pods avoid assuming that
three SMT threads scale linearly on one host. Availability, prices and
late-table throughput must be measured again before any campaign.

The following are **constant-prefix-throughput projections**, using
6,796.73 nodes/s. They exclude setup, recovery saves, validation/evaluation,
table-growth slowdown and storage. They are not campaign runtime promises.

| Work per seed, from zero | One-worker wall time per seed | Three seeds sequentially | Three independent pods, compute subtotal |
| --- | ---: | ---: | ---: |
| 20M nodes | 0.82 h | 2.45 h | $0.20 |
| 100M nodes | 4.09 h | 12.26 h | $0.98 |
| 300M nodes | 12.26 h | 36.78 h | $2.94 |
| 500M nodes | 20.43 h | 61.30 h | $4.90 |

With three independent workers the ideal wall time is the per-seed column,
not one third of it. A continuation uses only the additional work in this
formula. The saved #116 100M policies contain roughly 1.5M entries per seed
and their complete M4 campaign peaked at 3.23 GiB owned-job RSS; that evidence
cannot certify future Linux table growth or save/export peak. A longer paid
campaign needs a frozen scientific question, a representative resource
measurement, durable off-pod recovery checkpoints, verified transfer hashes,
a wall-time/cost shutdown watchdog and explicit owner authorization.
**No 300–500M continuation, TP20, abstraction A/B or other campaign was launched.**

## Artifacts and reproduction

Compact results, full artifact inventories, environment/cgroup records,
setup commands, independent suffix/header checks and provider termination
evidence are under [runpod-hu20-parity-artifacts](runpod-hu20-parity-artifacts).
The large Linux artifacts were retrieved before the disposable pod was
terminated. Their opaque tar transport hash was verified on M1; payload and
member verification belongs to M4.

M4 retained archive:
`/Users/dberweger/Local/hu20-linux-pilot-pr129.tar`.
Local transfer mirror:
`/Users/dberweger/Local/runpod-hu20-parity-artifacts/hu20-linux-pilot.tar`.
Archive SHA-256:
`ca096bc89593f1f9dee4106389fb698a5fedb2bc9352c89d769d60db7018cb39`.

Retrieve without loading a model on the travelling M1:

```bash
scp -o HostName=100.122.216.94 -o BatchMode=yes \
  m4:/Users/dberweger/Local/hu20-linux-pilot-pr129.tar ./hu20-linux-pilot.tar
shasum -a 256 ./hu20-linux-pilot.tar
```

On a free M4 or Linux host, extract the archive, then use the exact run/compare
commands in the protocol. Reproduction is a deliberate validation run, not
authorization for another rental or a larger training experiment.
