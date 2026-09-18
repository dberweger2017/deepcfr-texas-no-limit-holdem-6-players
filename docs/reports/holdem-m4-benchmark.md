# M4 benchmark and training admission

**The M4 passes the declared host checks. Use it for the authorized full batch,
with two sequential seeds and unchanged training settings.** The owner gave the
conditional go to start training once preparation and benchmarking finished.
This report is resource/recovery evidence, not a playing-strength result.

The [protocol](../holdem-m4-benchmark.md) and [measurements](holdem-m4-benchmark.json)
retain both benchmark seeds, environments, stage timings and verification hashes.
The scientific plan remains [local-fullgame.json](../../configs/holdem/local-fullgame.json).

## Host and measured costs

Host: Apple M4, ten CPU cores, 16 GiB RAM, macOS 26.6, AC power. The isolated
checkout is `/Users/dberweger/Local/deepcfr-training`; it uses Python 3.11.14,
PyTorch 2.5.1, NumPy 1.26.4, SciPy 1.17.0 and the pinned Rust engine. Dependency
validation passed. Each workload used one Torch thread; benchmark supervisors
kept the laptop awake while running. No private keys were copied.

| Measurement | M1 four iterations | M4 same four iterations | M4 32 iterations |
| --- | ---: | ---: | ---: |
| Total seconds | 160.54 | 98.31 | 735.11 |
| Collect/fit/replay seconds | 113.14 | 66.89 | 596.71 |
| Collection seconds | 39.27 | 24.44 | 275.05 |
| Fitting seconds | 69.90 | 39.41 | 299.34 |
| Recorded peak training RSS, GB | 0.883 | 0.912 | 4.267 |
| Final retained records, all roles | 9,941 | 9,941 | 24,576 |

The matched four-iteration pilot is **1.63 times faster overall** on the M4 in
these observations. This is one host comparison, not a general hardware speedup
guarantee. The 32-iteration run completes in **12 minutes 15 seconds** and fills
all six replay buffers. Its last eight iterations average **18.51 seconds**.

Saving iteration 16 takes 57.05 seconds for 316.6 MB; iteration 32 takes 58.52
seconds for 330.7 MB. Checkpointing accounts for material time and memory growth.
The recorded 4.267 GB peak is the training telemetry's process high-water mark,
including earlier work; it is not a separately instrumented continuous bound
through the final checkpoint. Resource checks remained below their limits.

Project **2–3 hours per full seed**, approximately **4–6 hours for both sequential
seeds**, plus the declared final verification allowance. This includes headroom
for growing snapshot archives, larger evaluations, seed variation and other host
activity. Keep the existing six-hour per-seed caps and shared 30-minute final
verification/test cap; no budget increase or concurrency change is justified.
Retain the 7 GiB worker-RSS, 8 GiB output and free-disk limits. No paid compute.

## Reproduction and validation

The M1/M4 four-iteration runs match node counts, roots, terminal counts and each
role's admitted-record counts. Fitted/current/archive/replay fingerprints do
**not** match across machines. Floating-point fit metrics diverge and the OS and
Python builds differ; this does not establish cross-platform bitwise determinism
or isolate the numerical difference to one library.

A separate fresh-process rerun on the M4 reproduces its own four-iteration
training result and evaluation exactly. The completed 32-iteration checkpoint
also resumes on the M4 with **zero additional fitting steps**. Across these two
checks, all **12 original/recovered checkpoint, export, evaluation and raw-outcome
file hashes match**. The repeated four-iteration check adds 6,144 optimizer steps
on the same benchmark seed, not on either reserved full-run seed. All timings,
outputs and original files remain retained.

The M4 passed 22 focused runner/arena checks before benchmarking and 18 engine /
observation checks afterward. A subsequent supervisor-only fix tolerates a
checkpoint file being atomically renamed during disk sampling and retains peak
sampled RSS. Its 23 focused checks and full GitHub CI pass before launch. This
monitoring change does not alter collection, fitting or evaluation mathematics.

## Artifact access

All raw outputs, including checkpoints and inference exports, remain on the M4
under `results/m4-host-benchmark/`. The M1 retains the compact archive and its
verified extraction under the PR worktree's `results/m4-host-benchmark/`.
The archive contains measurements, manifests, scripts, logs and small outcomes;
large `.pt` files were not copied. All **84 inventoried files** match after transfer.
Archive size: 241,951 bytes. SHA-256:

```text
af5f4589870a7061186c1dbd8c7e8ca0ce4cb067132fa103735b2aae701aa5a1
```

No benchmark failed or hit a resource limit. Both benchmark supervisors exited;
no campaign results are implied by the tiny evaluations. The full batch's status
and progress belong to `results/local-fullgame/`, separately from these records.
Keep the PR draft until the full campaign's outcomes and artifact checks are
recorded. No policy is automatically promoted.
