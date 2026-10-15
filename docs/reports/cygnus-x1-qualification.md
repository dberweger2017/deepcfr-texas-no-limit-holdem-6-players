# cygnus-x1 /bb128 qualification — October 14–15

**All on-machine testing stopped at 06:50 Madrid, before the owner's 07:00 deadline.**
The machine reproduces the native trainer's exact M4 checkpoints and passes the
full suite with the recorded strict-MKL setting. It is ready for guarded CPU
builds, tests and bounded diagnostics. **RAM remains partially screened; no
campaign is authorized.** GPU transfers/4 GiB aggregate VRAM pass, while CUDA,
shader arithmetic and full 8 GiB VRAM remain unqualified.

Scientific main: `ebd3d8d567e4248f9f1eb57edfaf25262338aa31`.
[Protocol](../cygnus-x1-qualification-protocol.md) · [Artifact index](../../RESULTS_INDEX.md).
Every CPU load used an on-host 85°C watchdog, two-second readings and at most
eight physical cores; builds used four Cargo jobs. CPU stress used 0.5-second
readings. No 16-thread test ran. Beszel's existing 5% cap was left alone.

## Inventory

Debian13.7 /Linux6.12.111; i7-10700KF,8 cores/16 threads; MSI Z490M-S01 /MEG
Z490 Infinite X, BIOS AMI D.00 June3,2020. Four dual-rank non-ECC Kingston
KF3200C16D4/32GX32GiB DIMMs;125.714GiB visible. Slot serials A0ECBB2A36,
A1ECBB2B25,B0F6BB3C82,B1EEBB25AE. DMI800MT/s and64GB array maximum are suspect;
effective DRAM speed is unverified. The kit is rated3200MT/s **with XMP**.
This is pre-XMP; owner plans full XMP RAM/connected-storage tests in a separate PR.

Linux exposes PL1 **125W/55.967744s**, PL2 **250W/0.002440s**; intel_pstate,
powersave governor and turbo. These are settings, not measured enforcement.
No fan RPM/PWM is exposed. AX201Wi-Fi is active; wired Ethernet is down.

Only WD SN730 **512,110,190,592 bytes**, serial19527B803570/firmware11110000,
is visible; initial ext4 free414.2GiB. SMART: spare100%,wear5%,28.83TBread,
51.23TBwritten,13,124hours,273unsafe shutdowns, zero media/error-log entries
or thermal-warning minutes. `smartctl -x` exits4 on unsupported self-test-log
read; independent `nvme smart-log` succeeds. Four SATA links are down:
the expected2TB HDD and other storage are absent and untested. No raw-disk write,
format or mount was performed. RTX2070SuperTU10410de:1e84/8GiB is bound to
nouveau; proprietary driver absent, Secure Boot on. Boot has gst-plugin-scan
MMU/channel faults. [Complete inventory](cygnus-x1-artifacts/inventory.json).

## RAM: two accepted partial screens, zero observed errors

Both screens used eight **locked14GiB** instances,112GiB total. These are
partial tests, not a full memtester pass or evidence that every DIMM is sound.

| Screen | Duration | Completed in every instance | Incomplete /not reached | Package °C min/median/max |
|---|---:|---|---|---:|
| Original |10,800.80s|Stuck Address, Random Value, XOR/SUB/MUL/DIV/OR/AND, Sequential Increment, Solid Bits|Block Sequential215–217/256; later patterns not reached|47/63/81|
| Additional mask0x1ec00 |16,166.65s|Stuck Address, Checkerboard, Bit Spread, Walking Ones|Walking Zeroes123–127/128; narrow8/16-bit writes not reached; Block Sequential/Bit Flip omitted|40/61/68|

**Every instance recorded zero FAILUREs.** Original guard preserves failed/time
budget/exit−15; individual wait codes were not collected. Extra guard preserves
failed/absolute-deadline/exit143. All eight actual extra child wait codes are
**−15/SIGTERM**, and their process IDs were absent at final verification.
These are expected budget stops, not detected memory hardware failures.

Original peak RSS112.025GiB/minimum available11.554GiB,5,367 samples/max gap2.106s;
extra peak RSS112.025GiB/minimum available10.607GiB,8,009 samples/max gap2.093s.
Both have zero swap growth/throttle-counter change. Sampled clocks across all16
logical CPUs, including idle siblings: original800/4700/4942MHz, extra800/4700/4801.
Full boot-time **memtest86+ remains the owner's job**.
[Original outcomes](cygnus-x1-artifacts/memory-partial.json) ·
[Final extra outcomes/child signals/raw-stream hash](cygnus-x1-artifacts/memory-extra-final.json) ·
[Original thermal CSV](cygnus-x1-artifacts/memory-locked-temperatures.csv) ·
[Extra full thermal CSV](cygnus-x1-artifacts/memory-extra-temperatures.csv).

## NVMe scratch fio and Tailscale

Direct/libaio,one8GiB scratch file,numjobs1,30s+3s warmup per case; initialized
before reads, fsync at job end. Full8GiBCRC32C write/read verification passes.

| fio case |MiB/s|IOPS|Completion p50/p99 µs|
|---|---:|---:|---:|
|Sequential read1MiBQ16|3260.83|3260|4882/5603|
|Sequential write1MiBQ16|1268.69|1269|15270/25297|
|Random read4kQ1|50.06|12815|77.3/91.6|
|Random write4kQ1|305.34|78167|10.3/12.1|
|Random read4kQ32|758.46|194165|146.4/374.8|
|Random write4kQ32|964.29|246856|114.2/337.9|

220.92s guarded,41.50MiB RSS; CPU36–57°C,SSD36.85–54.85°C,no swap/throttle.
Short extent tests do not establish whole-drive sustained writes/endurance.
Scratch is retained, excluded from archives. [All fio/resource measurements](cygnus-x1-artifacts/disk-results.json).

|Direct Tailscale peer|Outgoing/incoming Mbit/s|30-probe RTT min/mean/max ms|Loss|
|---|---:|---:|---:|
|db-ocean100.96.144.26|92.50/90.22|14.354/15.820/19.094|0|
|M4100.122.216.94|69.69/80.91|5.580/7.745/11.860|0|

Each TCP direction ran15s, one stream, both ends counted identical bytes.
Wi-Fi/M4's existing research load qualify these rates; M4 used its LAN endpoint.
One-shot in-memory peer listeners exited; no peer files/install/config/services
were changed. [Network paths/counts/timing/resources](cygnus-x1-artifacts/network-results.json).

## Setup, tests and native identity

Non-synced repo-main: Python3.11.17,torch2.5.1+cpu,requirements-dev; verified
Node22.23.3 distribution for JS tests. Setup84.52s/59°C/1.203GiB RSS.
Maintained build command `python -m scripts.build_native_trainer`; receipt retained.
Rust1.99.0/LLVM23.1.1/GNUld2.44. Binary1,752,760bytes SHA256
`d7049a7a024f142d1a82f75d3c673e427a80c516afd9c00b7769265dfad16436`.
First/post-suite bytes match; checkout/staging paths are absent. The owner
waived a second-location rebuild after review of #129/#133 parity and #243's
established path-independent method; no new two-location proof is claimed.
[Build/environment](cygnus-x1-artifacts/setup-results.json) ·
[Identity/source/path receipt](cygnus-x1-artifacts/build-identity.json).

Full suite: **2219Python+54subtests passed**,2Mac-only skips; **16Rust passed**;
20six-player hands and30session hands pass chip accounting. Real K50 tables and
native bucket evaluator ran. Python801.34s,Rust27.42s,guard841.65s,CPU69°C peak,
RSS1.038GiB,zero swap/throttle. Explicit **`MKL_CBWR=AVX2,STRICT`**,two BLAS threads.
[Suite receipt](cygnus-x1-artifacts/tests-passed.json).

Preserved failures: first suite lacked pip in uv venv (1365passed/1failed);
second default-MKL suite had a bitwise GRU-row invariant failure
(1595passed/1failed). Identical inputs produced hidden differences up to
2.9802322387695312e−8. One thread/COMPATIBLE/default-ATen did not fix it;
strict AVX2 MKL passed six probe variants and13unchanged focused tests, then
all tests. No assertion/tolerance/model changed. Initial memlock-limited RAM
attempt shrank allocations and was stopped after36.65s/24.2MiBRSS; its1.8GiB
logs are preserved, not a memory pass/fault. [Numerical diagnosis](cygnus-x1-artifacts/neural-numerics-diagnosis.json).

## Exact recovery and matched HU200 K50 trainer cost

Pinned PR163 K50 tables total1,384,031,032bytes; full sizes/SHA256 verified on
coordinator and bb128 before use. [Input restoration pins](cygnus-x1-artifacts/input-restoration.json).
HU200K50v1/R1/opponent-sampled,seed2026101302,ceiling85,373,862:
fresh20M and10M→20M resume both produce20,002,695nodes/8728iterations/
7,883,678entries,270,143,830bytes and exact #250 SHA256
`4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`.
10M prefix:10,001,913nodes/4260iterations/4,416,935entries/148,448,690bytes,
SHA256`fc102fc232fa4376627bbcd10af5f77a0b8fdb25f8a62cf81d224de5fde9e5db`.
Fresh/prefix/resume compute12.747/5.726/7.125s,save39.044/21.542/40.247s;
outer158.69s,CPU62°C,RSS1.958GiB,no swap/throttle.
[All recovery counts/hashes/timing](cygnus-x1-artifacts/recovery-gates.json).

Single bounded500M cost-only run:500,000,140nodes/321139iterations/69,084,815entries;
2,727,516,803byte checkpoint exactly matches M4 SHA256
`2c4c1484b7e4ebefca2bb364223d6cd12e886f98b20784b51ff6ac6678f4312d`.

|Measurement|cygnus pre-XMP|M4 #250 matched work|
|---|---:|---:|
|Compute excluding save|534.509s|219.922s|
|Final save|422.885s|240.828s|
|Native elapsed incl save|957.394s|460.750s|
|Compute nodes/s|935438|2273532|
|Nodes/s incl save|522251|1085186|
|Peak family RSS|6.990GiB|7.331GiB|

Cygnus achieves41.1%of M4 compute rate/48.1%including saves. Wrapper995.500s,
guard1000.243s; CPU39–62°C/median50,SSD34.85–35.85°C,no swap/throttle.
Assigned CPU0–7 maximum-clock proxy median4903MHz, not effective residency.
M4 #265 frozen2,023,708nodes/s excludes writes and is a different workload.
A100GiB table budget +1.289GiB fixed tables leaves24.42GiB headroom;
110–130bytes/aging key suggests **0.83–0.98billion retained keys**, unmeasured.
Young-table peak108.6bytes/key includes fixed inputs; no node-budget promise.
[Benchmark/model/capacity receipt](cygnus-x1-artifacts/trainer-benchmark.json).

## i7 and RTX diagnostics

Three60s int64 verification steps, cooling below60°C between, all pass:
1/4/8workers **2938.07/11287.56/22108.10bogo-ops/s**, peaks **57/64/76°C**,
medians55/61.5/72°C;8/1scaling7.52×. Zero failed/untrustworthy metrics,
swap/throttle;8worker sampled all-logical median4700MHz. No AVX/long-soak/16thread
claim. [CPU results](cygnus-x1-artifacts/cpu-steps.json).

Forced nouveau/NVK ICD identifies actual discrete RTX2070SUPER10de:1e84;
CPU fallback refused.1GiB device-local non-host-visible buffer passes four
patterns/4GiB full readback,zero mismatches,4.028s test/6.040s guard/1016.82MiB/s.
Original single4GiB request failed `vkCreateBuffer`/OUT_OF_DEVICE_MEMORY−2.
Queried maxBufferSize/maxMemoryAllocationSize are **2GiB**. Four simultaneous
1GiB buffers subsequently pass all four rotating patterns: **16GiB readback,
zero mismatches**,12.894s test/14.088s guard/1270.72MiB/s. Rates include CPU
verification, not peak PCIe bandwidth. CPU peaks41/40°C,zero swap/throttle.
First limit-probe raced deployment and exited2; failure and successful retry
are retained. GPU clock/temperature unavailable; unprivileged journal cannot
establish absence of new kernel faults. No CUDA/shader/full8GiB qualification.
[GPU outcomes/limits/failures](cygnus-x1-artifacts/gpu-results.json).

## Verdict, owner work and evidence

Use guarded CPU builds/tests/bounded diagnostics with recorded numerical mode.
Before unrestricted research: owner reviews BIOS fan curve/PL1, performs full
boot-time memtest86+, then separate XMP/fullRAM/storage qualification. After the
thermal fix, rerun1/4/8/16thread thermal steps, memory/build/test and applicable
throughput scaling; trainer scaling must respect its actual worker model.
No campaign, Slumbot, XMP/driver change, reboot or cleanup occurred.

Proprietary driver remains owner decision: reviewed Debian packaged driver,
matching headers/DKMS/firmware, Secure Boot signing/enrollment, nouveau handoff
and reboot, then nvidia-smi/VRAM/bounded CUDA validation.
[Debian installation reference](https://wiki.debian.org/NvidiaGraphicsDrivers).

Main archive **528members/3,553,618,907bytes**, full member readback and whole
transfer/staging SHA checks pass; native upload/no-pending/no-conflicts and
independent Drive metadata acceptance pass. [Archive pins, model restoration
commands and observed Drive links](cygnus-x1-artifacts/archive-main.json).
Extra RAM/closed archive-guard/new source closeout is being sealed separately.
All originals retained; no remote Drive bytes were redownloaded.
Full thermal/clock/throttle CSVs and compact receipts accompany this report.
The recurring qualification check is disabled; no follow-on work is scheduled.
