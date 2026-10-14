# cygnus-x1 qualification

**In progress; not admitted for campaigns.** The owner’s latest deadline is
**October 15, 07:00 Madrid**; later test jobs stop by **06:50**. The extended
window includes extra RAM coverage after the primary checks and i7/GPU tests. Debian 13.7 /Linux 6.12.111,
Intel i7-10700KF, 128 GiB installed /125.71 GiB visible. Current-main source
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
   verification pass; see below. Network, repository setup/tests, path-independent build,
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

PR163’s indexed K50 archive and all three extracted table members now pass
full SHA256/size checks in the ignored coordinator input directory. The tables
total **1,384,031,032 bytes**. [Retrieval and member receipt](cygnus-x1-artifacts/input-restoration.json).
They have not been copied to or used on bb128; remote verification precedes use.

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
repository change or GPU load has been performed.
