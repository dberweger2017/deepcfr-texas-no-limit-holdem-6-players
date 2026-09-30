# RunPod HU20 platform pilot

## Current result

**The platform pilot passed: fresh current-source Linux/M1 training and
deterministic recovery agree exactly, with retained historical M4 agreement.**
The owner explicitly authorized replacing the waiting fresh M4 reference
with the M1 before that reference began. The idle M4 queue was cancelled
before performing any training. No fresh current-source M4 run is claimed.
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

### Fresh current-source M1 reference

The owner-approved M1 reference used the same frozen harness revision,
Python **3.11.14**, engine commit, seed/config and node boundaries as Linux.
The travelling M1 was on AC, with 16 GB RAM. A separate venv used the
already-installed pinned CPython 3.11 engine binary, with its origin/build
hash recorded; no native rebuild or full local suite was needed.

| M1 measurement | Direct from zero | Fresh-process resume |
| --- | ---: | ---: |
| Final completed nodes / iterations / entries | 1,000,389 / 2,626 / 118,978 | Identical |
| Training time | 88.428 s | 42.684 s |
| Training throughput | 11,313.00 nodes/s | 11,708.54 nodes/s |
| Whole path wall time | 94.281 s | 48.612 s |
| Peak trainer RSS | 183.81 MiB | 193.50 MiB |
| Checkpoint save / export | 1.327 / 1.077 s | 1.209 / 0.949 s |

The complete sequential reference/tests/verification took **150.09 seconds**.
Peak aggregate owned-job RSS was **210.69 MiB**, swap growth was **zero**,
minimum free disk was **42.75 GiB**, and every sampled power record was AC.
All six supervisor phases exited zero. The focused tests passed **3/3**.
No M1 training/reference process or caffeinate lease remains running.

The fresh Linux/M1 comparison verifies every checkpoint/export byte after
decompression, all **2,626 non-timing iteration records**, all **1,324 resumed
suffix records**, completed/overshoot/next-iteration counts, and next-root
deal/action seeds plus initial action-RNG state hashes. The whole direct
work digest on both platforms is
`e5bf9a0afe36ed40270ca3b33c7423243f35aeafd5879e7af97e7001df91c592`.
M1 midpoint reload and final/current/next artifacts also reproduce its
uninterrupted run exactly. Both native builds retain engine revision
`5db20e3d5d6862b32a7402035c1340b622d3b005`; their binary hashes differ,
as expected across arm64/macOS and x86_64/Linux, with no trainer-state
divergence in this complete prefix.

Independent verification checked all **36 original Linux inventory members**
and then all **64 files** in the final combined reference inventory after
child logs closed. The final manifest SHA-256 is
`cbf2b2f35baa07074acf79d03a3a298a08727d99fddb13510fbae5ac9b624617`.
M1's zlib build/runtime is 1.2.12, Linux's 1.3.1. Despite that difference,
checkpoint deflate bytes and trailers match; the inference export differs
only at the gzip OS marker. This is observed equality, not an assumption
that all compressor versions always produce identical output.

### Linux resume and retained M4 agreement

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

The historical M4 reference ran source
`a87e9f8805d211e2b21dbb710339ada9083eedf6`. M1's training and next-iteration
checkpoints and current-policy export reproduce its compressed hashes
exactly. That retained agreement complements the fresh current-source
Linux/M1 comparison; it is not a fresh current-source M4 benchmark. The
owner-approved host amendment and idle M4 queue cancellation are retained.

Linux and M1 focused tests each passed **3/3**. GitHub's full `test` check
passed at the frozen harness; final publication CI runs again. The M1
performed only this specifically authorized bounded reference and its
verification. No full local suite, native compilation or poker-playing
evaluation ran there. The M1's broader research restriction remains in
force outside this pilot.

## What this implies for future three-seed training

**Recommend RunPod CPU as a valid training host for this recipe**, conditional
on preserving the pinned environment and exact recovery/hash discipline.
The fresh Linux/M1 fixed prefix and recovery agree exactly, and both match
the retained M4 trainer state. There is no unexplained numerical/state
difference. This pilot is not a playing-strength result or a safe scaling/
memory study for large tables.

This cheapest CPU pod delivered about **60% of M1's measured single-worker
prefix throughput**. Its prospective advantage is separately provisioned
independent workers and RAM while freeing the Macs; it is not demonstrated
per-worker speed or arbitrary core scaling. A late-table resource check is
still required before choosing a larger paid campaign. No fresh M4 speed
comparison is available.

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
terminated. Their transport and all retained member hashes were verified
under the owner's bounded M1 exception; large archives also remain on M4.

M4 retained archive:
`/Users/dberweger/Local/hu20-linux-pilot-pr129.tar`.
Local transfer mirror:
`/Users/dberweger/Local/runpod-hu20-parity-artifacts/hu20-linux-pilot.tar`.
Archive SHA-256:
`ca096bc89593f1f9dee4106389fb698a5fedb2bc9352c89d769d60db7018cb39`.

The M1 outputs and final combined manifest are retained at
`/Users/dberweger/Local/hu20-platform-parity-pr129/results/m1-platform-pilot`.
Their separate archive (excluding the already-retained nested Linux copy)
is `/Users/dberweger/Local/hu20-m1-pilot-pr129.tar` on M4, with local mirror
`/Users/dberweger/Local/runpod-hu20-parity-artifacts/hu20-m1-pilot.tar`.
Its SHA-256 is
`916f7657b5fa4310642e7cf5d9975e3d42318cc22888b294ff1c954975d0a130`.

Retrieve without loading a model on the travelling M1:

```bash
scp -o HostName=100.122.216.94 -o BatchMode=yes \
  m4:/Users/dberweger/Local/hu20-linux-pilot-pr129.tar ./hu20-linux-pilot.tar
shasum -a 256 ./hu20-linux-pilot.tar
scp -o HostName=100.122.216.94 -o BatchMode=yes \
  m4:/Users/dberweger/Local/hu20-m1-pilot-pr129.tar ./hu20-m1-pilot.tar
shasum -a 256 ./hu20-m1-pilot.tar
```

On a free M4 or Linux host, restore both archives into one directory:

```bash
mkdir pilot-restore
tar -xf hu20-m1-pilot.tar -C pilot-restore
tar -xf hu20-linux-pilot.tar -C pilot-restore
```

Then verify the final 64-file manifest or use the exact compare commands in
the protocol on the restored `direct`, `resumed`, and
`results/platform-pilot/direct` directories. Reproduction is a deliberate
validation run, not authorization for another rental or a larger training
experiment.
