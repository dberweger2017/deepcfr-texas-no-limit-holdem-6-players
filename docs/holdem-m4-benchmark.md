# M4 host benchmark

The owner authorized preparing the M4 over the existing `m4` SSH alias and running
benchmarks on September 18, 2026. This does not authorize the two full campaign
seeds, which remain reserved pending the owner's final go.

Use an isolated clone at `/Users/dberweger/Local/deepcfr-training`, a Python 3.11
virtual environment, PyTorch 2.5.1, NumPy 1.26.4, SciPy 1.17.0 and the pinned Rust
engine. No SSH keys or application credentials are copied. Run on AC power with
`caffeinate -i` only for the lifetime of the benchmark supervisor.

1. Repeat the retained four-iteration M1 timing plan exactly, seed 2026091801,
   on the M4. The local raw plan is retained with the earlier pilot; copy it as a
   benchmark input. Cap the worker at 600 seconds. Compare stage timings and
   deterministic iteration/replay/current-policy fingerprints, not timing-bearing
   files or environment-dependent checkpoint hashes. Report any mismatch.
2. Run the [32-iteration plan](../configs/holdem/local-m4-benchmark.json) on a
   separate benchmark seed, 2026091804. Same full-game recipe, 128 roots per role,
   256 fit steps, width 32 and replay capacity 4096. Save at 16/32 and evaluate
   only at 32 with eight blocks against styles and random. Cap this worker at
   900 seconds, with 120 seconds per training iteration. Report late-iteration
   throughput, replay occupancy, checkpoint costs and peak memory. A timeout is
   retained, not retried with a larger allowance based on its outcome.

The existing supervisor enforces 7 GiB worker RSS, 8 GiB output and at least
12 GiB free disk while running. Run one worker at a time. Neither tiny arena
selects a model or measures useful strength. These are host feasibility checks,
not extra campaign training seeds or architecture experiments.

Run focused runner/arena checks before benchmarking. Save source/environment,
commands, status, timings, reports, checkpoints and exports remotely. Retrieve
compact measurements and logs to the M1 and verify their hashes. Leave the
remote original artifacts intact; do not copy unrelated tens of GB of previous
research data. Revise only host/runtime estimates in PR #86 after measurement;
the scientific plan stays frozen. Future full training needs the owner's go.

## Same-host verification

The four-iteration comparison completed with equal integer workload counts but
different fitted/replay fingerprints across M1 and M4. Floating-point fitting
metrics differ; the platform and Python build also differ, so bitwise equivalence
across these two machines is not assumed. Before launching the full batch,
reproduce the four-iteration M4 run in a fresh process with a separate 600-second
cap, and resume the completed 32-iteration checkpoint without further optimizer
steps under a 300-second cap. Require same-host result and checkpoint/export
hash agreement. These are verification runs, not new seeds or outcome-based
retries. Run them after the timed benchmark to avoid CPU contention.
