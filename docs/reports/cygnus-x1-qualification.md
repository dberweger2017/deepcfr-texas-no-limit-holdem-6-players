# cygnus-x1 qualification

**In progress; not admitted for campaigns.** Debian 13.7 /Linux 6.12.111,
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
fresh inventory and RAM/storage tests. No XMP change or repeat run belongs
to this qualification.

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
2. Memory: first userspace attempt stopped after **36.65 s** because the
   8 MiB memlock allowance made memtester shrink each 14 GiB request one
   page at a time. Sampled family RSS was only **24.20 MiB**; no requested-RAM
   pass occurred. No memory error was observed. The same owner-reviewed
   root script now has a temporary-lock-limit launcher: it drops root before
   running eight 14 GiB instances /112 GiB total. All eight now have the full
   allocation locked; the full pattern pass is running. At **1,126.6 seconds**,
   all eight instances have passed nine patterns: address, random value,
   XOR/subtraction/multiplication/division/OR/AND and sequential increment,
   with zero observed errors. Package temperature is **63 °C**,
   peak **81 °C**; sampled family RSS peaks at 112.025 GiB, available RAM
   stays above 11.662 GiB, swap growth and throttle counters remain zero.
   [Interim receipt](cygnus-x1-artifacts/memory-progress.json). These are
   preliminary readings, not a completed pass.
   Four Linux guard checks pass: simulated overheat, sensor disappearance,
   low available RAM and swap growth each terminate the child. Full boot-time
   **memtest86+ remains the owner's job**, regardless of userspace results.
   Runtime correction: the observed solid-bits phase advances about once
   every 36 seconds. Later memtester patterns contain 1,280 settings in
   total on this 64-bit host, so the full pass is projected to need roughly
   12–15 more hours from the 32-minute reading. This is an extrapolation,
   not a measured completion time. The initial **three-hour cap remains
   active** and is expected to yield a partial pass around **23:40 Madrid**.
   A full-pass result cannot be claimed under that anticipated time stop.
   [Pattern-loop source](https://sources.debian.org/src/memtester/4.7.1-1/tests.c/).
3. Disk scratch fio, network, repository setup/tests, path-independent build,
   exact fresh/resumed 20M SHA256 gates and HU200 K50 throughput are pending,
   in that order. No dependent training has started.

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
