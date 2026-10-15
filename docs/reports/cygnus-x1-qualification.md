# cygnus-x1 qualification

**In progress; not admitted for campaigns.** The owner’s latest deadline is
**October 15, 07:00 Madrid**; later test jobs stop by **06:50**. The extended
window includes extra RAM coverage after the primary checks and i7/GPU tests. Debian 13.7 /Linux 6.12.111,
Intel i7-10700KF, 128 GiB installed /125.71 GiB visible. Inventory-main source
`b91a78d2d0893e80380bc3da9a24142e591fc04b`; isolated coordinator checkout and
remote evidence root `/home/dberweger/cygnus-qualification-20261014`.
[Measured inventory](cygnus-x1-artifacts/inventory.json) ·
[Predeclared qualification bounds](../cygnus-x1-qualification-protocol.md).

The owner's October 14 brief authorizes hardware qualification, correctness
reproduction and short trainer benchmarks, with no campaigns, Slumbot or
changes to other machines. Beszel remains active with its existing 5% cap.
CPU jobs use an on-host watchdog, two-second package-temperature readings,
an 85 °C stop and affinity to eight physical cores (CPU0–7). Builds use four
Cargo jobs. The owner subsequently allowed more thermal margin; initial
qualification retains 85 °C until measurements justify reassessment. No
16-thread workload has run.

## Inventory

MSI Z490M-S01 /MEG Z490 Infinite X; AMI D.00 BIOS dated June 3, 2020.
Four Kingston KF3200C16D4/32GX dual-rank 32 GiB DIMMs, non-ECC:
A0 `ECBB2A36`, A1 `ECBB2B25`, B0 `F6BB3C82`, B1 `EEBB25AE`.
DMI reports 800 MT/s for each DIMM and a 64 GB array maximum despite the
128 GiB installation. These fields are suspect; DDR4-3200 operation is
**not verified**. The owner confirms the kit is rated for **3200 MT/s with
XMP**. This run is the **pre-XMP baseline**; DMI’s 800 MT/s field is not
accepted as a measurement of effective DRAM speed. The owner plans to enable
XMP and connect the remaining storage for a **separate follow-up PR**, with
a **full XMP-enabled RAM pass on October 15** and fresh inventory/storage tests. No XMP change belongs to this qualification.

Linux exposes PL1 **125 W /55.967744 s**, PL2 **250 W /0.002440 s**;
`intel_pstate`, powersave governor, turbo enabled. These are exposed settings,
not a measured power trace or proof of BIOS enforcement. No fan RPM/PWM
sensor is exposed. All core/package throttle counters were zero at inventory.

Only one physical disk is detected: WD PC SN730 512,110,190,592 bytes,
serial `19527B803570`, firmware `11110000`; ext4 system filesystem had
414.2 GiB available. SMART passes: 100% spare, 5% wear used, 28.83 TB read,
51.23 TB written, 13,124 power-on hours, 273 unsafe shutdowns, **zero media
errors /error-log entries /thermal-warning minutes**. `smartctl -x` exits 4
because its self-test-log read returns invalid-field; the separate
`nvme smart-log` succeeds with zero critical warning. This inventory failure
is preserved rather than reported as a clean exit. No self-test was started.
All four SATA ports report link down; the expected 2 TB HDD and any other
storage are **not visible or benchmarked**. No disk was mounted or formatted.

RTX 2070 Super /TU104 `10de:1e84`, 8,192 MiB VRAM reported by nouveau;
proprietary NVIDIA driver absent. Boot contains nouveau MMU/channel faults
from `gst-plugin-scan`. Secure Boot is enabled. GPU compute is unqualified.
Wi-Fi AX201 is the active uplink; wired Ethernet is down.

## Qualification progress and failures

1. Inventory complete with the self-test-log limitation above.
2. **RAM screen complete, partial coverage:** eight locked 14 GiB instances
   /112 GiB ran for **10,800.80 s** (18:40:30–21:40:31 UTC). Every instance
   completed ten patterns: Stuck Address, Random Value, XOR/SUB/MUL/DIV/OR/AND,
   Sequential Increment and Solid Bits. **Zero observed errors per instance
   (0–7)**. Block Sequential stopped at settings 215–217 of 256; later patterns
   were not reached. This is not a full pass or final RAM qualification.
   The raw guard receipt says `failed /wall-time budget exceeded /exit -15`;
   that is the accepted three-hour budget stop, not a detected hardware error.
   The group was sent SIGTERM; individual child wait statuses were not recorded
   because the parent terminated with them. No memtester processes remained.
   Package temperature **47–81 °C, median 63 °C**; all-logical-CPU sampled
   clock min/median/max **800/4,700/4,942 MHz**, including idle siblings.
   Peak sampled family RSS **112.025 GiB**, minimum available **11.554 GiB**;
   **zero swap growth and throttle-counter change**. All 5,367 samples are
   retained; maximum sample gap **2.106 s**.
   [Per-instance outcomes and original guard receipt](cygnus-x1-artifacts/memory-partial.json) ·
   [Full thermal/clock CSV](cygnus-x1-artifacts/memory-locked-temperatures.csv) ·
   [Resource summary and raw-stream hash](cygnus-x1-artifacts/memory-resource-summary.json).
   Five Linux watchdog stop tests pass, including the absolute cutoff.
   Full boot-time **memtest86+ remains the owner's job**. Additional selected
   pre-XMP patterns may run after the other mandatory checks if time remains.
3. **Disk scratch fio complete:** all six timed cases and full 8 GiB CRC32C
   verification pass; see below. Network complete (below); repository setup/tests, path-independent build,
   exact fresh/resumed 20M SHA256 gates and HU200 K50 throughput are pending,
   in that order. No dependent training has started.

## NVMe scratch-file performance

fio direct I/O, libaio, one job, 8 GiB extent; 30 seconds plus 3-second warmup
per timed case. Initialization precedes reads; end-of-job fsync is enabled,
not a durable sync for every write. Rates include neither initialization nor
the separate verification pass. No raw device was written.

| Case | MiB/s | IOPS | Completion latency p50 /p99, µs |
|---|---:|---:|---:|
| Sequential read, 1 MiB Q16 | 3,260.83 | 3,260 | 4,882 /5,603 |
| Sequential write, 1 MiB Q16 | 1,268.69 | 1,268 | 15,270 /25,297 |
| Random read, 4 KiB Q1 | 50.06 | 12,815 | 77.3 /91.6 |
| Random write, 4 KiB Q1 | 305.34 | 78,167 | 10.3 /12.1 |
| Random read, 4 KiB Q32 | 758.46 | 194,165 | 146.4 /374.8 |
| Random write, 4 KiB Q32 | 964.29 | 246,856 | 114.2 /337.9 |

Full 8 GiB CRC32C write/read verification completed with **zero fio errors**.
Total guarded wall time **220.92 s**, peak family RSS **41.50 MiB**.
Package **36–57 °C**; SSD **36.85–54.85 °C**; zero swap/throttle changes.
All-logical-CPU sampled clock min/median/max **800/800/5,099 MHz**, including
idle siblings. These short, repeatedly overwritten file measurements do not
establish sustained whole-drive write performance or a long endurance result.
Scratch remains retained; no other disks were visible.
[Detailed fio measurements](cygnus-x1-artifacts/disk-results.json) ·
[Full thermal/clock readings](cygnus-x1-artifacts/disk-temperatures.csv).

## Tailscale network

Single TCP stream per direction, sender bounded to 15 seconds; both ends count
identical application bytes. Thirty ICMP probes per peer, separate from throughput.
Both Tailscale paths are direct UDP, with the M4 using its LAN endpoint.
No peer file, package, service or configuration change; one-shot in-memory
listeners exited. These application rates are not an iperf3 result.

| Peer | cygnus → peer Mbit/s | peer → cygnus Mbit/s | RTT min /mean /max ms | Loss |
|---|---:|---:|---:|---:|
| db-ocean | 92.50 | 90.22 | 14.354 /15.820 /19.094 | 0/30 |
| M4 | 69.69 | 80.91 | 5.580 /7.745 /11.860 | 0/30 |

AX201 Wi-Fi is the active cygnus uplink; Ethernet is down. M4's existing native
trainer was active and untouched, so this is observed usable throughput, not
idle-peer or wired-link capacity. Every client/latency guard passed; package
temperature and sampled clocks/throttle counters accompany each measurement in
the [network receipt](cygnus-x1-artifacts/network-results.json) and per-job thermal
CSVs alongside it.

## Repository setup

Fresh nonsynced current-main checkout:
`ebd3d8d567e4248f9f1eb57edfaf25262338aa31`, clean at setup.
Python **3.11.17**, CPU-only **PyTorch 2.5.1+cpu** (CUDA build absent),
requirements-dev installed; Node **22.23.3** added within the owned qualification
root for the playback tests, verified against the official archive SHA256.
Native trainer built through `python -m scripts.build_native_trainer`, four
Cargo jobs, retaining its build receipt and exact first binary.
Setup passed in **84.52 s**, peak RSS **1.203 GiB**, package **34–59 °C**,
no swap/throttle changes. Sampled all-logical-CPU clock min/median/max
**797/800/4,897 MHz**, including idle siblings.
The full test suite now passes (below); local binary identity/path checks and exact
checkpoint recovery remain pending. The owner accepts the established #243
build method rather than repeating its two-location proof. Prior M4/RunPod
[#129](https://github.com/dberweger2017/CFR-NLHE-6max/pull/129) /
[#133](https://github.com/dberweger2017/CFR-NLHE-6max/pull/133) exact-state
parity informs the unchanged fresh/resumed #250 checkpoint gates; executable
bytes are architecture/toolchain dependent.
[Environment/toolchain/build receipt](cygnus-x1-artifacts/setup-results.json) ·
[Full thermal/clock readings](cygnus-x1-artifacts/setup-temperatures.csv).

## Test-suite environment failure (preserved)

First attempt: Rust **16 passed**; Python **1,365 passed, 1 failed, 2 skipped**,
54 subtests passed, then `--maxfail=1` stopped the suite. The failure is
`test_continuation_keeps_the_complete_parent_and_cumulative_iteration_weights`:
its environment capture calls `python -m pip freeze`, but the uv-created
virtual environment omitted pip. This is a diagnosed setup failure, not a full
suite pass or a memory fault. No training/recovery benchmark followed it.
The original guard result remains failed /exit 1, with no guard stop reason.
Wall **355.54 s**, peak RSS **0.819 GiB**, CPU **39–71 °C**, no swap/throttle
changes. Raw output is preserved with its hash; full thermal readings remain.
Corrective action: install pip in the existing venv, then rerun the entire suite
under a separate guard label. Future setup seeds pip at venv creation.
[Failure receipt](cygnus-x1-artifacts/tests-first-failure.json) ·
[Full thermal/clock readings](cygnus-x1-artifacts/repo-tests-temperatures.csv).

## Retry: neural output identity failure (diagnosed)

Pip repair passed; the next whole-suite attempt reached **1,595 passed,
1 failed, 3 skipped**, 54 subtests passed, before maxfail stopped it.
Rust remained **16/16**. The multi-street hidden-world invariant's flop case
fails `torch.equal` on values for equal visible observations. Printed values
appear equal; this alone does not establish rounding or an information leak.
Tracing identical model inputs and the first divergent arithmetic operation
is required before any dependent recovery or trainer benchmark.
[Original retry failure and resource receipt](cygnus-x1-artifacts/tests-retry-failure.json) ·
[Full thermal/clock readings](cygnus-x1-artifacts/repo-tests-retry-temperatures.csv).
The three skips are the Mac resource-profile check, absent bucket-evaluator
binary, and unset optional real-K50-table environment. No full-suite pass is claimed.

Diagnosis: all observations/candidates are equal, and every packed GRU time-step
has identical inputs across its four batch rows. Default MKL yields up to
**2.9802322387695312e−8** hidden-state difference across those rows. The
**unchanged** focused suite gives 12 passed /1 failed at two threads, one
thread, COMPATIBLE mode and ATen's default CPU capability. Intel's
`MKL_CBWR=AVX2,STRICT` restores identical GRU rows/output for all six probed
street/architecture cases and **13/13 unchanged invariant tests pass**.
This trace supports a numerical-reproducibility diagnosis for this case; it
does not establish every neural workload's reproducibility. No assertion,
tolerance or model code was changed. The whole suite will rerun with that
explicit environment, real K50 tables (after full size/SHA validation) and
the optional native bucket evaluator enabled.
[Diagnostic matrix and traces](cygnus-x1-artifacts/neural-numerics-diagnosis.json).
[Intel strict CNR documentation](https://www.intel.com/content/www/us/en/developer/archive/training/conditional-numerical-reproducibility-cnr.html)
describes the mode; [PyTorch numerical notes](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
explain limits of bitwise floating-point equivalence.

## Full suite passed with explicit numerical mode

At source `ebd3d8d567e4248f9f1eb57edfaf25262338aa31`, the entire suite
completes with **2,219 Python tests passed, 54 subtests passed, zero failures**
in **801.34 s**. Two skips are macOS-only checks (Mac resource profile and
setuid-ps regression). The real K50-table check and native evaluator ran.
Rust library **16/16 passed** in **27.42 s**; 20 six-player hands and 30 changing-
lineup session hands pass, with session chip accounting 210,000 in =59,750 out
+150,250 on table. Native executable still matches the retained build hash.
Python uses **`MKL_CBWR=AVX2,STRICT`**, two BLAS/OpenMP threads; this host's
neural bitwise tests require that recorded mode. No model/assertion change.
Guarded stage **841.65 s**, peak family RSS **1.038 GiB**, package peak **69 °C**
(median **61 °C**), no swap growth/throttle change. All-logical-CPU sampled
clock min/median/max **800/800/5,111 MHz**, including idle siblings.
Both earlier failed attempts remain preserved.
[Full-suite/environment/resource receipt](cygnus-x1-artifacts/tests-passed.json) ·
[Full thermal/clock readings](cygnus-x1-artifacts/repo-tests-strict-temperatures.csv).

## Native build identity

The clean checkout/native tree still matches the build receipt. Retained first
binary and post-suite executable are identical: **1,752,760 bytes**, SHA256
`d7049a7a024f142d1a82f75d3c673e427a80c516afd9c00b7769265dfad16436`.
Neither checkout nor fixed-staging absolute path appears in either binary.
This checks retained/current identity and uses #243's established build method;
a second checkout was omitted under the owner's amendment. Exact cross-platform
checkpoint recovery passes below, independently of binary identity.
[Identity/path/source receipt](cygnus-x1-artifacts/build-identity.json).

## Exact M4 checkpoint recovery gates passed

HU200 K50 v1, seed 2026101302, R1/opponent-sampled, ceiling 85,373,862.
Fresh **20,002,695 nodes /8,728 iterations /7,883,678 entries** and a fresh-process
**10M→20M resume** both produce **270,143,830 bytes**, exactly matching #250:
`4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`.
The actual 10M prefix is 10,001,913 nodes /4,260 iterations /4,416,935 entries,
148,448,690 bytes, SHA256
`fc102fc232fa4376627bbcd10af5f77a0b8fdb25f8a62cf81d224de5fde9e5db`.
This verifies full compressed checkpoint equality, not a selected-row comparison.

| Path | Native compute s | Save s | Native elapsed incl. save s |
|---|---:|---:|---:|
| Fresh 20M | 12.747 | 39.044 | 51.791 |
| Fresh 10M prefix | 5.726 | 21.542 | 27.268 |
| Resume suffix to 20M | 7.125 | 40.247 | 47.372 |

Outer stage **158.69 s** also includes process startup, table/parent reading and
SHA verification. Peak sampled family RSS **1.958 GiB**; package peak **62 °C**
(median **55 °C**), no swap growth/throttle change. All-logical-CPU sampled
clock min/median/max **800/800/5,119 MHz**, including idle siblings.
[Actual checkpoints/counts/timing/resource receipt](cygnus-x1-artifacts/recovery-gates.json) ·
[Full thermal/clock readings](cygnus-x1-artifacts/recovery-gates-temperatures.csv).
Native identity stage separately peaked at 38 °C and completed in 2.02 s;
its full thermal CSV accompanies the identity receipt. The matched 500M
cost-only benchmark may now proceed; no campaign is started.

## Matched 500M trainer benchmark

One fresh cost-only run, same HU200 K50 v1/R1/opponent-sampled seed/cap as #250:
**500,000,140 nodes /321,139 iterations /69,084,815 entries**. The final
**2,727,516,803-byte** checkpoint exactly matches M4 SHA256
`2c4c1484b7e4ebefca2bb364223d6cd12e886f98b20784b51ff6ac6678f4312d`.
No evaluation, continuation or campaign followed.

| Measurement | cygnus pre-XMP | M4 #250, same 500M work |
|---|---:|---:|
| Native compute, excluding save | 534.509 s | 219.922 s |
| Final checkpoint write | 422.885 s | 240.828 s |
| Native elapsed including write | 957.394 s | 460.750 s |
| Compute nodes/s | 935,438 | 2,273,532 |
| Nodes/s including write | 522,251 | 1,085,186 |
| Sampled peak family RSS | 6.990 GiB | 7.331 GiB |

Cygnus delivers **41.1% of M4's compute rate**
and **48.1% including saves**.
Its larger RAM capacity does not make this matched job faster. This is one
sample with different architecture/toolchain and unverified effective DRAM
speed, not an isolated XMP or compiler comparison. M4 #265's **2,023,708 nodes/s**
excluding writes is a **different frozen-table workload**, contextual only.

Wrapper wall **995.500 s**, outer guard **1,000.243 s**, separately include
startup/table loading, post-save SHA/diagnostic work and supervisor overhead.
Package **39–62 °C**, median **50 °C**; SSD **34.85–35.85 °C**; no swap growth
or throttle change. All-logical sampled clock min/median/max **800/800/5,007 MHz**
includes idle siblings; assigned CPU0–7 per-sample maximum median
**4,903 MHz** is a clock proxy, not effective
residency or the frequency of a specific traversal thread.
[Full cost/model/resource receipt](cygnus-x1-artifacts/trainer-benchmark.json) ·
[All thermal/clock readings](cygnus-x1-artifacts/benchmark-temperatures.csv).

A planning budget of **100 GiB for table growth**, plus the **1.289 GiB** pinned
tables, leaves **24.42 GiB** of the visible RAM for OS/runtime/headroom.
At an aging allowance of **110–130 bytes/key**, this suggests approximately
**0.83–0.98 billion retained keys**. The measured young-table peak is
**108.6 bytes/key including fixed inputs**;
it is not safe to assume that coefficient remains constant as regrets age.
This estimate follows the [compact-store layout](native-compact-table.md),
is unmeasured at that size, predicts no node budget and authorizes no campaign.

## Stepped i7 integer verification

Each `stress-ng --cpu-method int64 --verify` step ran 60 seconds under its own
85 °C guard, sampled every 0.5 s, cooling below 60 °C between steps.

| Workers | Bogo-ops/s, real time | Peak package °C | Median package °C |
|---|---:|---:|---:|
| 1 | 2,938.07 | 57 | 55 |
| 4 | 11,287.56 | 64 | 61.5 |
| 8 | 22,108.10 | 76 | 72 |

All verification checks pass, with zero failed/untrustworthy stressor metrics,
swap growth or throttle-counter change. Eight-worker scaling is **7.52×**
the one-worker rate for this particular integer method. Eight-worker sampled
all-logical-CPU clock median **4,700 MHz**; one/four-worker active-clock proxies
and full per-step clocks/resources are in the receipt. Governor remains
powersave/intel_pstate and sched-autogroup remains enabled; no tuning was done.
These short integer checks do not validate AVX, long heat soak or 16 threads.
[Measurements/resources](cygnus-x1-artifacts/cpu-steps.json) · full 0.5-second
thermal/clock CSVs `cpu-{1,4,8}-temperatures.csv` alongside the receipt.

## RTX VRAM smoke: 1 GiB passes, 4 GiB single buffer rejected

Forced the installed nouveau/NVK ICD and identified the actual discrete
**NVIDIA GeForce RTX 2070 SUPER (NVK TU104), vendor/device 10de:1e84**;
CPU fallback is refused. The exposed device-local heap is 8 GiB.
A **1 GiB non-host-visible device-local allocation** passes zero/all-ones/A5/5A
patterns, each fully copied back in 64 MiB chunks and compared byte-for-byte:
**4 GiB total readback, zero mismatches**, 4.028 s test /6.040 s guard wall.
The **1,016.82 MiB/s** rate includes fill/copy/synchronization/CPU comparison,
not peak PCIe bandwidth.

The declared **4 GiB single-buffer attempt failed before any pattern**:
`vkCreateBuffer` returned **VK_ERROR_OUT_OF_DEVICE_MEMORY (-2)**. It is
preserved as a failure; 4 GiB coverage is not claimed. The completed read-only query reports **2 GiB maxBufferSize and
maxMemoryAllocationSize**, so the requested single4GiB buffer exceeded both
limits. [Actual queried limits](cygnus-x1-artifacts/gpu-limits.json).
The same4GiB total bound will be tested as four simultaneously live1GiB
allocations, each receiving all four patterns; that result is still pending.
The first limit-probe launch raced deployment and exited2 without querying
the GPU; its raw failure is retained alongside the successful query.
GPU temperature/clock telemetry is unavailable. The unprivileged journal query
cannot see system/kernel messages, so its empty output does not establish
absence of new faults. CPU thermal/clock/throttle readings are retained.
This tests transfers/VRAM only; no CUDA or shader arithmetic was qualified,
and no proprietary driver, reboot or system configuration was changed.
[Original GPU outcomes/failure/resources](cygnus-x1-artifacts/gpu-results.json) ·
full `gpu-{1g,4g}-temperatures.csv` alongside the receipt.

PR163’s indexed K50 archive and all three extracted table members now pass
full SHA256/size checks in the ignored coordinator input directory. The tables
total **1,384,031,032 bytes**. [Retrieval and member receipt](cygnus-x1-artifacts/input-restoration.json).
They are now copied to bb128; all three full SHA256s and sizes passed on-host
verification before the real-table test was enabled.

After the fan curve/PL1 fix, rerun the stepped thermal check and 16-thread
memory, build/test and applicable throughput scaling tests. Trainer scaling
must respect its actual worker model; sixteen logical CPUs do not imply
sixteen native traversals.

Installing an NVIDIA driver remains an owner decision. A separate reviewed
root plan would need matching Debian kernel headers, an appropriate packaged
NVIDIA driver/DKMS and firmware, Secure Boot module signing/enrollment,
reboot and nouveau handoff, followed by `nvidia-smi`, VRAM and bounded CUDA
checks. [Debian’s driver instructions](https://wiki.debian.org/NvidiaGraphicsDrivers)
cover the packaged driver and Secure Boot enrollment. No driver install,
repository change has been performed; the bounded NVK VRAM checks above are the only GPU loads.
