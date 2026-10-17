# Hardware used to date

Inventory of the M1, M4 and bb128 systems used for this project's development, training and evaluation through **17 October 2026**. The current hardware and software snapshot was collected on that date. Earlier qualification and performance results retain their original dates and configuration limits.

Evidence labels:

- **Observed:** read from the installed machine or its operating-system API on 17 October 2026.
- **Firmware:** decoded from installed firmware tables; available operating states, not measured sustained clocks.
- **Published:** manufacturer specifications or explicitly identified original research.
- **Historical:** an earlier project measurement, with its scope and configuration stated.
- **Unverified:** unavailable, inconsistent or not independently established by this inventory.

Memory and cache sizes use binary units (KiB, MiB, GiB); drive capacities and published bandwidth use decimal units (GB, TB, GB/s) unless stated otherwise. A capacity or API limit is not a successful allocation test.

## System overview

| Property | M1 | M4 | bb128 |
|---|---|---|---|
| System | MacBook Pro, 13-inch, 2020 | MacBook Pro, 14-inch, 2024; base M4 | MSI MEG Z490 Infinite X desktop |
| System model | MacBookPro17,1 | Mac16,1 | MS-B916, system version 1.1 |
| Logic board / platform | Apple M1 platform | Apple M4 platform | MSI Z490M-S01, MS-7C99, board version 1.1 |
| CPU | Apple M1 | Apple M4 | Intel Core i7-10700KF, Comet Lake |
| Architecture | arm64 | arm64 | x86_64 |
| Physical CPU cores | 8: 4 performance + 4 efficiency | 10: 4 performance + 6 efficiency | 8 homogeneous cores |
| Logical CPU threads | 8; no SMT | 10; no SMT | 16; two SMT threads per physical core |
| Installed RAM | 16 GiB unified | 16 GiB unified | 128 GiB DDR4 |
| GPU | Apple M1, 8 GPU cores | Apple M4, 10 GPU cores | MSI NVIDIA GeForce RTX 2070 SUPER, TU104 |
| GPU memory | Shared with CPU and OS | Shared with CPU and OS | 8 GiB dedicated GDDR6 |
| Neural Engine | 16 cores, Apple-published | 16 cores, Apple-published | No separate neural accelerator inventoried |
| Internal SSD | Apple AP0512Q, 500.28 GB | Apple AP0512Z, 500.28 GB | WD PC SN730, 512.11 GB |
| Additional storage | None included in this inventory | None included in this inventory | Seagate ST2000DM008, 2.00 TB SATA HDD |

The Apple model identities, CPU/GPU core counts, RAM capacities and installed drive models were observed. Neural Engine counts are published specifications; no local Neural Engine usage or arithmetic qualification was established. Apple's model specifications are listed in the sources below. Marketing configuration labels such as “512 GB SSD” can differ from the device-reported byte capacity.

## CPU topology, clocks and instruction sets

| Property | M1 | M4 | bb128 |
|---|---|---|---|
| CPU topology | One four-core performance group and one four-core efficiency group | One four-core performance group and one six-core efficiency group | One socket, one NUMA node, eight physical cores |
| Performance-core maximum clock | 3.204 GHz, decoded firmware table | 4.464 GHz, decoded firmware table | 3.80 GHz published base clock; up to 5.10 GHz maximum turbo |
| Efficiency-core maximum clock | 2.064 GHz, decoded firmware table | 2.892 GHz, decoded firmware table | Not applicable |
| OS-allowed CPU frequency range | Not exposed by the generic frequency sysctls used | Not exposed by the generic frequency sysctls used | 800–5,100 MHz, observed |
| SIMD | 128-bit NEON | 128-bit NEON; SME/SME2 also reported | 256-bit AVX2, FMA; SSE4.1/SSE4.2 |
| CPU FP16 support | Reported | Reported | No separate native FP16 arithmetic capability established by this inventory |
| CPU BF16 and I8MM | Both reported absent | Both reported present | Neither established as a native capability by this inventory |
| CPU SME / SME2 | Both reported absent | Both reported present; SME2.1 absent | Not applicable; no AMX |
| AVX-512 | Not applicable | Not applicable | Absent |
| CPU identification | Apple M1 | Apple M4 | Family 6, model 165, stepping 5 |

The firmware clock maxima are not simultaneous all-core guarantees, effective-clock residency measurements or a thermal test. The Apple CPU tables use different numeric scales between the captured generations; the decoded values are expressed above in GHz. The 24 MHz clock exposed elsewhere in the Apple device tree is a reference/timebase value, not the CPU or GPU operating frequency.

On bb128, observed SMT sibling pairs are `(0, 8)`, `(1, 9)`, `(2, 10)`, `(3, 11)`, `(4, 12)`, `(5, 13)`, `(6, 14)` and `(7, 15)`. Logical CPUs 0–7 therefore cover one thread per physical core in this snapshot. SMT was enabled.

## CPU caches and virtual memory

| Property | M1 | M4 | bb128 |
|---|---|---|---|
| Performance-core L1 instruction cache | 192 KiB per core | 192 KiB per core | 32 KiB per physical core |
| Performance-core L1 data cache | 128 KiB per core | 128 KiB per core | 32 KiB per physical core |
| Efficiency-core L1 instruction cache | 128 KiB per core | 128 KiB per core | Not applicable |
| Efficiency-core L1 data cache | 64 KiB per core | 64 KiB per core | Not applicable |
| Performance-group L2 | 12 MiB shared by four cores | 16 MiB shared by four cores | 256 KiB per physical core; 2 MiB across eight cores |
| Efficiency-group L2 | 4 MiB shared by four cores | 4 MiB shared by six cores | Not applicable |
| Shared CPU L3 | No conventional CPU L3 capacity established by this inventory | No conventional CPU L3 capacity established by this inventory | 16 MiB shared by all physical cores / SMT threads |
| System-level cache (SLC) | 8 MiB reported in original M1 research; distinct from CPU L2 and GPU-private caches | Capacity unverified | No separate system-level cache included in this inventory |
| CPU cache line | 128 bytes | 128 bytes | 64 bytes |
| OS base page | 16 KiB | 16 KiB | 4 KiB |

All CPU L1/L2 sizes and bb128 L3 sizes above are operating-system observations. For the Apple CPUs, the per-performance-level sysctls identify the P/E cache sizes; generic cache fields and some device-tree fields do not consistently describe the performance-core cache.

Additional observed bb128 cache geometry:

| Cache | Associativity | Sets | Sharing |
|---|---|---|---|
| L1 instruction, 32 KiB | 8-way | 64 per physical core | The two SMT siblings of that core |
| L1 data, 32 KiB | 8-way | 64 per physical core | The two SMT siblings of that core |
| L2, 256 KiB | 4-way | 1,024 per physical core | The two SMT siblings of that core |
| L3, 16 MiB | 16-way | 16,384 total | All 16 logical CPUs |

Linux reported a 2 MiB huge-page size, zero reserved explicit huge pages, and transparent huge pages set to `always`. This is a configuration snapshot, not evidence that every research allocation used huge pages.

The M1 SLC figure is external research evidence: 4,096 sets × 16 ways × 128 bytes = 8 MiB. It was not measured locally and is not a GPU-private L2 specification.

## System memory

| Property | M1 | M4 | bb128 |
|---|---|---|---|
| Installed bytes | 17,179,869,184 | 17,179,869,184 | 137,438,953,472, corresponding to four recorded 32 GiB modules |
| OS-visible memory | Physical capacity above; usable/application budgets are smaller | Physical capacity above; usable/application budgets are smaller | 134,984,122,368 bytes in the current snapshot, approximately 125.714 GiB |
| Reported memory family | LPDDR4 | LPDDR5 | DDR4 |
| Reported manufacturer | Micron | Hynix | Kingston modules in the historical DMI record |
| Memory ownership | Unified CPU/GPU/OS pool | Unified CPU/GPU/OS pool | CPU system RAM; discrete GPU VRAM is separate |
| Published bandwidth | Not established from the manufacturer sources used in this inventory | 120 GB/s SoC memory bandwidth | Intel lists 45.8 GB/s at the processor's officially supported memory configuration |
| Measured sustained bandwidth | Not measured here | Not measured here | Not measured here |

The LPDDR family labels are the operating-system readback. Exact LPDDR variant, bus timing and effective data rate were not independently read from either Mac.

The bb128 qualification recorded:

- Four occupied DIMM slots, two per channel across channels A and B.
- 32 GiB per DIMM, each recorded as dual rank.
- Module part number `KF3200C16D4/32GX` in all four slots.
- Non-ECC memory.
- Kit rated for 3,200 MT/s; the owner subsequently reported enabling XMP 3200.

The effective current DRAM rate, voltage and active timing tuple remain independently unverified. The earlier firmware/DMI fields reporting “800 MT/s” and a “64 GB maximum” were already flagged as inconsistent in the qualification record. Intel officially lists DDR4-2933 and a 128 GB maximum for this processor; the installed kit's XMP rating is a separate value.

At an actually active 3,200 MT/s, two 64-bit DDR4 channels give a calculated 51.2 GB/s transfer ceiling. This calculation is conditional on the effective rate and is not a measured bandwidth result.

## GPUs and compute APIs

| Property | M1 GPU | M4 GPU | bb128 GPU |
|---|---|---|---|
| Observed model | Apple M1 | Apple M4 | NVIDIA GeForce RTX 2070 SUPER, MSI subsystem |
| Architecture / family | Apple Metal family Apple7 | Apple Metal family Apple9 | NVIDIA Turing, TU104; CUDA compute capability 7.5 |
| GPU cores / SMs | Eight Apple GPU cores | Ten Apple GPU cores | 40 streaming multiprocessors (SMs) |
| CUDA cores | Not applicable | Not applicable | 2,560 |
| Tensor cores | No directly comparable NVIDIA Tensor-core count | No directly comparable NVIDIA Tensor-core count | 320, derived from 40 SMs × eight Turing Tensor cores per SM |
| Dedicated RT cores | Hardware ray-tracing acceleration not established | Apple publishes hardware-accelerated ray tracing | 40, derived from 40 SMs × one Turing RT core per SM |
| Installed GPU memory | Unified system memory | Unified system memory | 8 GiB GDDR6; 8,589,934,592-byte device-local heap in qualification |
| Memory interface | Shared SoC memory | Shared SoC memory | 256-bit GDDR6 |
| Published memory data rate | Not independently established | Not independently established | 14 Gbit/s per pin, reference specification |
| Published memory bandwidth | Not independently established | 120 GB/s shared SoC maximum | 448 GB/s reference VRAM bandwidth |
| Published reference GPU clocks | Not published in the cited Apple model specification | Not published in the cited Apple model specification | 1,605 MHz base / 1,770 MHz boost |
| Observed firmware GPU states | Six nonzero entries, 396–1,278 MHz | Fifteen nonzero entries, 338–1,578 MHz | Running clocks and VBIOS table not read |
| Published reference graphics-card power | Not separately established | Not separately established | 215 W; actual installed-board limit unverified |

Apple GPU cores and NVIDIA CUDA cores/SMs are different units; these counts do not provide a direct throughput comparison.

Apple GPU firmware frequency entries, excluding the zero/off entry:

- **M1, MHz:** 396, 528, 720, 924, 1,128, 1,278.
- **M4, MHz, in firmware order:** 338, 618, 796, 928, 952, 1,056, 1,053, 1,170, 1,152, 1,278, 1,204, 1,338, 1,326, 1,470, 1,578.

The M4 entries are not strictly ordered by frequency. They are preserved in firmware order rather than being represented as a monotonically increasing operating curve. Neither table measures clock residency or sustained performance.

The installed RTX was identified by non-unique product IDs: NVIDIA vendor/device `10de:1e84`, MSI subsystem `1462:3860`. These identify a product family/subsystem, not a card's unique serial number. The exact retail MSI variant, factory overclock, VBIOS version and active power/clock limits remain unverified; the reference values above are not asserted as installed-card readback.

### GPU cache and memory hierarchy

| Property | M1 | M4 | RTX 2070 SUPER |
|---|---|---|---|
| GPU-private L1 capacity | Unverified | Unverified | Turing combines L1 data cache and shared memory in a 96 KiB pool per SM |
| GPU-private L2 capacity | Unverified | Unverified | 4 MiB in published original research |
| Shared SLC | 8 MiB reported in original M1 research; separate from private GPU caches | Unverified | Not applicable to the Apple-style unified-memory hierarchy |
| Published compute L1/shared split | Not established | Not established | 64 KiB L1 + 32 KiB shared, or 32 KiB L1 + 64 KiB shared |
| Published Turing register file | Not applicable | Not applicable | 256 KiB per SM |
| GPU execution-group width | Not queried for an actual compiled pipeline | Not queried for an actual compiled pipeline | CUDA warp size 32; hardware/API property rather than a current NVK shader test |

### Live Metal properties

Both Macs returned the following device-level values through a Metal property query, without allocating large buffers or submitting GPU commands:

| Property | M1 | M4 |
|---|---|---|
| Unified-memory device | Yes | Yes |
| Reported Metal support | Metal 4 | Metal 4 |
| Highest Apple family included in the query | Apple7 | Apple9 |
| Recommended maximum working set | 12,713,115,648 bytes, approximately 11.840 GiB | Same |
| Maximum single buffer length | 9,534,832,640 bytes, approximately 8.880 GiB | Same |
| Maximum threadgroup memory | 32,768 bytes, 32 KiB | Same |
| Device maximum threadgroup dimensions | 1,024 × 1,024 × 1,024 | Same |
| Ray-tracing API support returned | Yes | Yes |

The dimension tuple is not permission to launch its product as the number of threads. Apple's feature tables give a 1,024-total-thread family limit; the actual compute pipeline may impose a smaller maximum. Pipeline execution width and pipeline-specific limits were not queried.

Ray-tracing API support on M1 does not establish dedicated hardware ray-tracing acceleration. The M4 model specification explicitly includes hardware acceleration. Apple's feature table places the full 64-bit atomic set at Apple9; this is a published family capability, not a local atomic-arithmetic validation.

### Installed bb128 graphics stack and PCIe

- Kernel graphics driver: `nouveau`.
- Vulkan driver: NVK, with qualification device name identifying the actual RTX 2070 SUPER/TU104 rather than a CPU fallback.
- Mesa Vulkan drivers: `25.0.7-2+deb13u1`.
- Vulkan loader package: `1.4.309.0-1`.
- Nouveau DRM userspace package: `2.4.124-2`.
- No proprietary NVIDIA driver was observed; `nvidia-smi` was unavailable.
- PCIe maximum capability observed in sysfs: x16 at 8.0 GT/s per lane, corresponding to PCIe 3.0.
- Current link snapshot: x16 at 2.5 GT/s per lane. No GPU load was applied to establish its active link rate.
- Current GPU clocks, reclocking behavior, temperature, board power and utilization were not established.

The hardware's CUDA compute capability does not establish a working CUDA installation. No CUDA arithmetic, Tensor-core arithmetic or neural training qualification was performed as part of this inventory.

### Historical GPU memory qualification

The earlier bb128 qualification queried a maximum buffer size and maximum memory-allocation size of **2,147,483,648 bytes (2 GiB)** under NVK. A single requested 4 GiB buffer was rejected. Four simultaneous 1 GiB device-local buffers passed a segmented transfer-pattern test:

- Four fill/readback patterns, with zero byte mismatches.
- 4 GiB simultaneously allocated test storage.
- 16 GiB total verified readback.
- 12.894 seconds test time; 14.088 seconds including the guard wrapper.
- 1,270.72 MiB/s end-to-end verified throughput.

That throughput includes GPU fill/copy, synchronization and CPU byte verification. It is not peak PCIe throughput or VRAM bandwidth. The test validates the declared memory-transfer patterns, not shader arithmetic, all VRAM cells, CUDA readiness or model numerics.

## Storage

| Property | M1 internal SSD | M4 internal SSD | bb128 NVMe | bb128 HDD |
|---|---|---|---|---|
| Model | APPLE SSD AP0512Q | APPLE SSD AP0512Z | WDC PC SN730 SDBPNTY-512G-1032 | Seagate ST2000DM008-2FR102 |
| Device-reported bytes | 500,277,792,768 | 500,277,792,768 | 512,110,190,592 | 2,000,398,934,016 |
| Decimal capacity | 500.28 GB | 500.28 GB | 512.11 GB | 2.00 TB |
| Binary capacity | Approximately 465.92 GiB | Approximately 465.92 GiB | Approximately 476.94 GiB | Approximately 1.819 TiB |
| Interface / class | Apple internal SSD controller / Apple Fabric | Apple internal SSD controller / Apple Fabric | PCIe NVMe SSD | SATA HDD |
| Main data filesystem | APFS | APFS | ext4 | ext4 |
| Reported device firmware | 591.40.3 | 2973.120 | 11110000 in historical qualification | Not independently recorded here |
| TRIM | Enabled | Enabled | Current trim schedule not inventoried | Not established |
| Health evidence | SMART “Verified” | SMART “Verified” | Historical SMART details below | No fresh health query included |
| Physical / logical sector | 4 KiB logical device block observed | 4 KiB logical device block observed | Not independently recorded in this summary | 4 KiB physical / 512-byte logical |

The Seagate exact-model datasheet specifies:

- 7,200 RPM.
- 256 MB multisegment cache.
- SATA 6 Gbit/s interface.
- **SMR recording technology.**
- Up to 220 MB/s sustained outer-track reads.
- 55 TB/year workload-rate limit and 2,400 power-on hours/year usage specification.

These are product specifications, not local sustained or random-write measurements. No fresh HDD benchmark was performed. The HDD is the installed local research-cache storage; this inventory does not certify a retention scheduler or cloud archive state.

Historical bb128 NVMe qualification recorded:

- 3,261 MiB/s sequential reads and 1,269 MiB/s sequential writes.
- Passing 8 GiB CRC32C integrity verification.
- SMART health passed, 100% available spare and 5% percentage-used counter.
- Zero media errors, zero error-log entries and no critical warning.
- 28.83 TB read and 51.23 TB written; 13,124 power-on hours and 273 unsafe shutdowns, as recorded at qualification time.
- Zero warning-temperature minutes and zero critical-temperature minutes.
- An unsupported self-test-log query returned an error; the separate NVMe SMART log succeeded. That failed query was not represented as a successful self-test.

Those drive counters are dated historical evidence, not current readback. No SSD benchmark or destructive storage test was run for the 17 October inventory.

The bb128 root partition was recorded as 484,679,090,176 bytes with a separate 26,406,289,408-byte swap partition; Linux reported approximately 24.6 GiB of usable swap. Free-space snapshots were approximately 19 GiB on M1, 28 GiB on M4, 98 GiB on bb128's system filesystem and 1.6 TiB on its HDD. These values are transient and are not available-capacity guarantees for subsequent runs.

## Network and external interfaces

| Property | M1 | M4 | bb128 |
|---|---|---|---|
| Published Wi-Fi family | Wi-Fi 6 / 802.11ax | Wi-Fi 6E / 802.11ax | Intel AX201 Wi-Fi 6, 2×2, 160 MHz capability observed through PCI inventory |
| Wired adapter inventoried | Not included in this snapshot | USB 10/100/1000 Ethernet adapter | Intel I219-V Ethernet |
| Direct M4–bb128 cable | Not applicable | 1,000 Mbit/s, full duplex, MTU 1,500 | 1,000 Mbit/s, full duplex, MTU 1,500 |
| Published high-speed external ports | Two Thunderbolt / USB4 ports, up to 40 Gbit/s | Three Thunderbolt 4 ports, up to 40 Gbit/s | Full external-port inventory not performed |

The direct Ethernet cable is additional to the existing overlay-network access. Addressing, routes, peer identities, interface hardware addresses and remote-access configuration are excluded from this document.

Recorded direct-cable measurements:

| Test | Direction | Result | Scope |
|---|---|---|---|
| Memory-to-memory TCP | M4 → bb128 | 940.87 Mbit/s; 117.61 MB/s | Earlier bounded eight-second transfer |
| Memory-to-memory TCP | bb128 → M4 | 934.36 Mbit/s; 116.80 MB/s | Earlier bounded eight-second transfer |
| Latency | Between M4 and bb128 | Approximately 0.5–0.6 ms, zero loss over five probes | Small sample, not a long reliability test |
| Verified SSH file transfer | M4 → bb128 | 1 GiB in 9.158 seconds; 117.24 MB/s; 937.92 Mbit/s | Earlier project transfer with matching SHA256 |

No new speed test was run for this inventory. The owner separately reported approximately 4 MB/s downloads from Google Drive; that is an owner observation, not an independently reproduced result in this report. The measured cable results do not establish end-to-end HDD-to-cloud throughput.

## Operating systems and toolchains

| Property | M1 | M4 | bb128 |
|---|---|---|---|
| OS release | macOS 27.2 | macOS 26.6 | Debian GNU/Linux 13.7, trixie |
| OS build / kernel | Build 26B5091g; Darwin 27.2 | Build 25G5065a; Darwin 25.6 | Linux 6.12.111+deb13-amd64 |
| Userspace architecture | arm64 | arm64 | x86_64 |
| C library | Apple system runtime; version not separately inventoried | Apple system runtime; version not separately inventoried | glibc 2.41 |
| Compiler | Apple Clang 21.0.0, clang-2100.3.34.2 | Apple Clang 21.0.0, clang-2100.0.123.102 | GCC 14.2.0, Debian 14.2.0-19 |
| Rust | rustc 1.96.0, Homebrew | rustc 1.89.0 | rustc 1.99.0 |
| Cargo | 1.96.0 | 1.89.0 | 1.99.0 |
| Default Python queried | 3.14.6 | 3.9.6 | 3.13.5 |

These are the queried system/toolchain defaults, not a claim that every experiment used these Python versions, toolchain versions or package environments. Research environments and individual campaigns retain their own source, binary, dependency and numerical-setting records. The repository's supported Python requirement is separate from the shell default.

## Power, cooling and operating-state snapshot

### M1 and M4

- Low Power Mode was disabled in the recorded battery and AC power profiles on both machines.
- Neither machine returned a recorded thermal or performance warning through the queried power-management interface.
- These responses are not a sustained thermal test and do not establish actual CPU/GPU clock residency, chip power, fan speed or available thermal headroom.
- Cooler details, fan RPM and power-adapter identity were not independently inventoried.

### bb128

| Property | Observed value |
|---|---|
| BIOS vendor | American Megatrends |
| BIOS version / date | D.00 / 2020-06-03 |
| CPU frequency driver | `intel_pstate`, active |
| Scaling governor | `powersave` |
| Energy-performance preference | `balance_performance` |
| Turbo | Enabled |
| Allowed performance range | 15–100% in the queried Intel P-state interface |
| Long-term package power limit, PL1 | 125 W |
| Short-term package power limit, PL2 | 125 W |
| PL1 time window | 55.967744 seconds |
| PL2 time window | 0.002440 seconds |

The governor's name is recorded literally; it does not mean turbo was disabled. Package power limits are CPU-package settings, not a measured wall-power figure, GPU power limit or total-system power budget.

The earlier qualification used PL1 125 W and PL2 250 W. The 17 October RAPL readback instead confirms PL1 = PL2 = 125 W. Historical results must not be labeled as measurements under the newer configuration.

The owner reported installing five additional cooling fans. Today's sensor snapshot exposed three nonzero tachometer channels, approximately 856 / 1,301 / 1,311 RPM. Tachometer channels are not a complete physical fan census. At that single reading, CPU package temperature was approximately 61°C and NVMe composite temperature approximately 37.9°C; one motherboard thermistor read approximately 77°C. That individual thermistor reading does not independently establish a VRM sensor mapping or a peak temperature.

Existing research guards have used an 85°C CPU stop and a 95°C sustained VRM stop in the applicable campaign, with the latter requiring 30 seconds above the threshold. These are project operating constraints, not hardware manufacturer ratings. Intel publishes a 100°C maximum junction temperature for the CPU. This document does not change any guard or authorize a load test.

PSU model/rating, CPU cooler model, physical fan placement and GPU board-power telemetry were not established.

## Historical training and qualification evidence

The following are recorded project measurements, not fresh benchmarks performed for this document.

### Matched 500M-node HU200 training measurement

The earlier hardware qualification compared the same 500,000,140-node workload and matching checkpoint content on bb128 and M4:

| Measurement | M4 | bb128 |
|---|---|---|
| Compute time excluding checkpoint writing | 219.922 seconds | 534.509 seconds |
| Checkpoint write time | 240.828 seconds | 422.885 seconds |
| Native elapsed time including checkpoint writing | 460.750 seconds | 957.394 seconds |
| Compute nodes/second | 2,273,532 | 935,438 |
| Nodes/second including checkpoint writing | 1,085,186 | 522,251 |
| Sampled peak process-family RSS | Approximately 7.33 GiB | Approximately 6.99 GiB |

The bb128 checkpoint contained 69,084,815 entries and was 2,727,516,803 compressed bytes. Its wrapper elapsed time was 1,000.243 seconds; wrapper time is distinct from native elapsed time. The run recorded no swap growth or throttle-counter increase, with CPU temperature 39–62°C and median 50°C.

On this particular earlier workload, M4's compute rate was approximately 2.43 times bb128's and its write-inclusive rate approximately 2.08 times bb128's. This comparison predates the current bb128 PL2 setting and the later owner-reported XMP/cooling changes. It is not a general speed ratio across different recipes, table sizes or software versions.

### Other recorded qualification results

- Fresh 20M-node training/resume and the matched 500M checkpoint passed the stated cross-machine checkpoint equivalence checks.
- The bb128 full test run passed 2,219 Python tests plus 54 subtests and 16 Rust tests with `MKL_CBWR=AVX2,STRICT` and the recorded two-thread BLAS setting.
- A default-MKL run had a bitwise GRU-row invariant failure; the original failure was retained. Passing the strict-MKL configuration did not establish equivalence for every numerical mode.
- One-, four- and eight-core integer-verification steps passed; recorded peak CPU temperatures were 57°C, 64°C and 76°C respectively.
- Two locked-RAM screens exercised 112 GiB, lasting approximately three hours and four hours 29 minutes. No errors were observed, but both stopped before a complete pass. Walking Zeroes was incomplete and later narrow-write tests were not reached.
- Those partial screens do not certify the whole RAM test suite or a complete post-XMP memory pass.
- The GPU transfer-pattern qualification and NVMe integrity/speed measurements are described in their respective sections above.

## Unverified details and measurement limits

The following remain explicitly unresolved:

- M1 and M4 GPU-private L1/L2 capacities; M4 system-level-cache capacity and associativity.
- Sustained CPU/GPU clock residency, package/board power and memory bandwidth on the current configurations.
- Exact effective LPDDR data rates and timings on the Macs.
- Effective bb128 DDR4 rate, active timings and memory voltage after the owner-reported XMP change.
- Exact MSI graphics-card retail variant, VBIOS identity, factory overclock, current reclocking behavior and active GPU power limits.
- Actual NVK shader arithmetic limits/performance and any CUDA or Tensor-core numerical qualification.
- Current HDD SMART health and measured sequential/random I/O behavior.
- Full post-XMP RAM qualification.
- Complete PSU, cooler, external-port, power-adapter and physical-fan inventories.

Unprivileged queries did not expose all requested telemetry. Missing details are left unverified rather than inferred from a neighboring product, a marketing clock or another machine's benchmark.

## Collection and publication scope

The 17 October collection used read-only operating-system and device-property queries: `system_profiler`, per-performance-level `sysctl`, IORegistry firmware-table reads, a Metal device-properties query, Linux procfs/sysfs, `lscpu`, `lspci`, `lsblk`, package-version queries and sensors. No GPU commands, stress tests, new large allocations, driver installs, BIOS changes or hardware-setting changes were performed for this inventory. A small Swift property-query program was compiled/run; it submitted no GPU work.

This public-facing document deliberately excludes credentials, tokens, account details, serial numbers, UUIDs, MAC/IP addresses, actual network hostnames, overlay peer identifiers, remote-access aliases/configuration, personal filesystem paths and raw machine dumps. M1, M4 and bb128 are the report's logical equipment labels. Non-unique manufacturer model/part numbers and PCI product IDs are retained as hardware specifications.

## Sources

- [Apple: MacBook Pro, 13-inch, M1, 2020 specifications](https://support.apple.com/en-us/111893).
- [Apple: MacBook Pro, 14-inch, M4, 2024 specifications](https://support.apple.com/en-au/121552).
- [Intel: Core i7-10700KF specifications](https://www.intel.com/content/www/us/en/products/sku/199325/intel-core-i710700kf-processor-16m-cache-up-to-5-10-ghz/specifications.html).
- [NVIDIA: GeForce RTX 2070 SUPER reference specifications](https://www.nvidia.com/content/nvidiaGDC/gb/en_GB/geforce/graphics-cards/rtx-2070-super.html).
- [NVIDIA: Turing architecture whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf), for SM structure and memory hierarchy.
- [Liu, VanAusdal and Burtscher: Performance Impact of Removing Data Races from GPU Graph Analytics Programs](https://userweb.cs.txstate.edu/~burtscher/papers/iiswc24a.pdf), Table I, for the RTX 2070 SUPER's SM count and cache configuration.
- [Apple: Metal feature-set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf), for Apple7/Apple9 capabilities and implementation limits.
- [Asahi Linux m1n1 device-tree parser](https://github.com/AsahiLinux/m1n1/blob/main/proxyclient/m1n1/adt.py), for the frequency/voltage-pair format used in Apple GPU performance-state tables.
- [SLAC: Access-Driven CPU-to-GPU Side-channel Attacks via System-Level Cache on Apple Silicon](https://arxiv.org/html/2608.09075v2), for original M1 SLC research.
- [Seagate: BarraCuda 3.5-inch datasheet](https://www.seagate.com/www-content/datasheets/pdfs/3-5-barracudaDS1900-14-2007US-en_US.pdf), exact ST2000DM008 model row.
- [Project hardware qualification report](https://github.com/dberweger2017/CFR-NLHE-6max/pull/283), for the dated CPU, RAM, disk, GPU-transfer, test-suite and matched-training results. The configuration changes documented above supersede its older inventory values where stated.

The current inventory supplements the historical evidence; it does not retroactively replace the configuration recorded for an earlier training run.
