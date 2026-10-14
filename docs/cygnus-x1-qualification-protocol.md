# cygnus-x1 qualification bounds

Owner-authorized October 14 hardware and runtime qualification, **not a training
campaign**. Test the machine in the requested order; report and investigate any
failed command/gate before dependent work. No Slumbot, driver install or changes
to other machines. Root steps remain in one reviewable script.

All CPU-heavy jobs on cygnus-x1 run under its local two-second coretemp package
watchdog, detached from SSH, with eight physical cores available (CPU0–7).
Initial thermal stop is 85 °C; the owner's later 90–95 °C guidance permits
reassessment, but does not remove the guard. Builds use four Cargo jobs;
BLAS/OpenMP are limited to eight threads. No sixteen-thread workload before
the owner confirms the fan/PL1 fix. Retain temperature, clock, throttle counters,
RSS, available RAM, swap and exit/stop reasons. A missing sensor fails closed.

1. Read-only DMI/CPU/PCI/block/SMART/kernel/sensor/power inventory. Root steps
   stay in one reviewable script. No disk formatting, mounting or raw writes.
2. A three-hour userspace memtester screen, eight concurrent 14 GiB instances: 112 GiB
   locked, leaving approximately 13.7 GiB for the OS. Temporary inherited
   memlock limit, tests run unprivileged. Three-hour wall cap, available RAM
   floor 6 GiB, swap growth ceiling 256 MiB. Per-instance allocation/error/exit
   results are mandatory; a partial screen is never called a full pass. The
   owner chose to retain the three-hour cap after the measured full-pass
   forecast exceeded it. An expected time stop with no memory errors is a
   partial result, followed by the remaining diagnostic stages; actual memory
   errors stop dependent work for diagnosis. Boot-time
   memtest86+ remains owner work.
3. Direct fio on one new 8 GiB NVMe scratch file: sequential 1 MiB Q16 and
   random 4 KiB Q1/Q32, read/write, 30 s per timed case with 3 s warmup.
   Initialize before reads and separately CRC32C-verify the full extent.
   No other disk is currently visible; unavailable tests remain explicit.
4. Thirty ICMP packets/Tailscale path discovery per peer, bounded forward and
   reverse TCP throughput. No peer installs, files, service or configuration
   changes. Any listener must be ephemeral and terminate after its test.
5. New nonsynced clone of then-current main, Python 3.11.17, CPU-only torch
   2.5.1 as in CI, requirements-dev. Guard dependency compilation, native
   builds/Rust library tests and the full Python suite. Also run check_game
   and check_session. Retain exact source/environment/build receipts.
6. Build via `python -m scripts.build_native_trainer` from two distinct
   absolute checkout paths; require equal binary bytes and no embedded local
   checkout/staging paths. Restore only PR163's SHA256-pinned K50 tables.
   HU200 v1, seed 2026101302, one root/seat, opponent-sampled averaging:
   fresh 20M checkpoint must be SHA256
   `4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`,
   and 10M→20M resume must reproduce it. A failed gate blocks benchmarks and
   any campaign admission.
7. Cost-only trainer benchmark: one fresh fixed **500M-node** HU200 K50 run,
   same seed/recipe/cap 85,373,862 as #250, one terminal save. This matches
   the published first M4 save at 500,000,140 nodes /69,084,815 entries.
   Bound total wall time to 1,800 s, sampled family RSS to 16 GiB, free disk
   floor 40 GiB, plus common guards. Separate native compute time, write time,
   wall time, nodes/s and RSS. An early stop remains incomplete. The #265
   frozen rate is a different fixed-table workload and is contextual only.
   Estimate capacity with aging allowances/reserves; promise no node budget.
   No evaluation, selection or continuation follows this benchmark.
8. GPU visibility only. Describe Debian packaged driver/DKMS/kernel-header
   and Secure Boot enrollment/reboot work; installation remains owner choice.

Keep raw/partial evidence and models in ignored roots; index and archive them
under the artifact-storage policy. Git holds source, compact receipts, the
report and concise thermal traces. Leave the feature PR reviewable and disclose
all failures, missing tests and future sixteen-thread reruns.

## Owner clarification: separate XMP/storage follow-up

The kit is rated for 3200 MT/s with XMP. Current PR283 tests establish the
pre-XMP baseline; the DMI speed field does not establish actual DRAM speed.
After these tests, the owner plans to enable XMP and connect storage, then
run the full XMP-enabled RAM pass on October 15 and repeat relevant storage
qualification in a separate PR. This plan
does not authorize changing firmware settings, rerunning tests or extending
PR283’s recurring task into that follow-up.

## Owner overnight amendment: all testing stops by 07:00

The owner extends the work window to **October 15, 07:00 Europe/Madrid
(05:00 UTC)** and authorizes longer RAM coverage, all listed runtime/trainer
checks, plus i7 and RTX 2070 tests. Later jobs have an on-host absolute
**06:50 Madrid /04:50 UTC cutoff**, leaving ten minutes for stopping/readout.
The current three-hour RAM screen remains unchanged; finish the primary
checks first, then use remaining time for additional locked-RAM patterns.
Selected-pattern or time-limited screens remain partial; no full-pass claim.
This amendment supersedes the earlier restriction on extra RAM testing here.

The i7 gets stepped 1/4/8-worker integer verification under the existing 85 °C
cutoff, with cooldown between stages; no 16-thread or AVX stress before the
cooling fix. GPU testing uses the installed nouveau/Mesa stack with actual
NVIDIA device identity checks, bounded VRAM fill/copy/readback and no software
fallback. Driver/API limitations and missing GPU telemetry stay explicit;
proprietary-driver installation/reboot remain separate owner decisions.
Recurring checks use 30 minutes during RAM, 15 minutes during other stages,
or 60 minutes for long quiet RAM work; no five-minute recurring checks.

Additional GPU smoke bounds: force the installed nouveau Vulkan ICD and require
PCI vendor/device `10de:1e84`, discrete-GPU type. Try 1 GiB device-local VRAM,
then 4 GiB only if the first smoke passes; four fill patterns (zero, all ones,
A5 and 5A), each read back in 64 MiB chunks and compared byte-for-byte. Each
fence waits at most five seconds; each job has a five-minute wall cap plus
the global cutoff. This is VRAM/transfer correctness, not CUDA or shader
arithmetic qualification. Missing GPU clock/thermal telemetry is explicit.
CPU verification samples temperature every 0.5 seconds at 1/4/8 workers.

The later RAM screen selects `MEMTESTER_TEST_MASK=0x1ec00`: Checkerboard, Bit
Spread, Walking Ones, Walking Zeroes, 8-bit Writes and 16-bit Writes, plus
memtester's mandatory Stuck Address phase. Indices were checked against the
Debian 4.7.1 source test list. The first screen's partial Block Sequential is
retained; the long Bit Flip phase is omitted to fit broader coverage before
the deadline. Stop at the absolute cutoff even if this selection is incomplete.

## Owner amendment: reuse established build/parity method

The owner requested checking prior M4/RunPod qualification before repeating the
two-location build. Reviewed merged [#129](https://github.com/dberweger2017/CFR-NLHE-6max/pull/129)
and [#133](https://github.com/dberweger2017/CFR-NLHE-6max/pull/133): they establish
exact trainer-state/recovery comparison across platforms, not equal executable
bytes across architectures. Merged [#243](https://github.com/dberweger2017/CFR-NLHE-6max/pull/243)
proved checkout-independent builds under one toolchain. Cygnus already uses its
maintained build entry point and retains the receipt. The second-checkout build
is therefore omitted here at the owner's request; compare the retained/current
local executable hash and scan for embedded checkout/staging paths instead.
The exact #250 M4 fresh20M and 10M→20M checkpoint SHA256 gates remain mandatory
and implement the prior cross-platform output-parity method on this current
HU200 recipe. No prior HU20 result is counted as a new cygnus test.
