# Snapshot-average readiness results

**Readiness passed: all eight seeds in both games meet the original limits.**
This closes roadmap milestone 3's small-game learning gate. It does not establish
Hold’em playing strength or complete milestone 4's learning exit.

The [frozen protocol](../snapshot-readiness.md) ran once at clean revision
`642aeab163aba990619f21c8775b3a40b030fe6a`, committed before these seeds were used.
The [machine-readable report](snapshot-readiness.json) contains the complete plan,
source/environment fingerprints, every final result and artifact hashes.
The [previous neural-average confirmation](neural-readiness.md) remains failed.
No seed was retried, checkpoint substituted, threshold relaxed or model promoted.

## Final results

Exploitability is the exact two-player best-response measure in **antes per hand**;
lower is better. Value error is the absolute difference between the profile's
player-zero value and the independently solved equilibrium value. These metrics
do not estimate a human Hold’em percentile or BB/100.

Only final iteration 100 in Kuhn and 480 in Leduc count. Both limits must pass
for every seed. The evaluated policy is the saved, reloadable snapshot mixture,
not a fitted strategy network or an evaluator-only policy table.

### Kuhn — 8/8 pass

Limits: exploitability ≤0.030; value error ≤0.030.

| Seed | Exploitability | Value error | Result |
| --- | ---: | ---: | --- |
| 503 | 0.020596 | 0.001966 | passed |
| 509 | 0.019929 | 0.001092 | passed |
| 521 | 0.015930 | 0.001728 | passed |
| 523 | 0.015410 | 0.002809 | passed |
| 541 | 0.008990 | 0.000246 | passed |
| 547 | 0.010918 | 0.000168 | passed |
| 557 | 0.011115 | 0.000669 | passed |
| 563 | 0.014989 | 0.000243 | passed |

Mean exploitability: **0.014735**; sample standard deviation: 0.004208; worst seed: **0.020596**. The mean does not replace the all-seed gate.

### Leduc — 8/8 pass

Limits: exploitability ≤0.150; value error ≤0.100.

| Seed | Exploitability | Value error | Result |
| --- | ---: | ---: | --- |
| 503 | 0.092855 | 0.009864 | passed |
| 509 | 0.091374 | 0.011708 | passed |
| 521 | 0.111323 | 0.015941 | passed |
| 523 | 0.103614 | 0.013829 | passed |
| 541 | 0.103063 | 0.012394 | passed |
| 547 | 0.097923 | 0.009584 | passed |
| 557 | 0.112756 | 0.019308 | passed |
| 563 | 0.091092 | 0.011982 | passed |

Mean exploitability: **0.100500**; sample standard deviation: 0.008623; worst seed: **0.112756**. The mean does not replace the all-seed gate.

## What this establishes

The declared snapshot-average training recipe reaches the required small-game
accuracy across eight fresh replicates per game. We can recover the full training
state and evaluate the actual exported average without fitting a second strategy
network. Local re-evaluation preserves every pass/fail decision.

This is a prospective readiness result, not a paired comparison with the previous
experiment: its seeds differ. It does not prove that every unseen seed will pass,
that wider networks are needed, or that these settings transfer to Hold’em.
Advantage approximation and sampled training still matter. Snapshot storage grows
with iteration count; the final exports are about 6.4 MB per Kuhn seed and 30.6 MB
per Leduc seed. Hold’em inference/storage costs need their own measurements.

## Execution and artifact audit

- All 16 jobs completed, with no missing seeds, timeouts, retries or worker errors.
  Each has exactly one scored final evaluation; no intermediate checkpoint selection.
- Eight single-threaded workers used a Runpod allocation of 16 logical CPUs,
  32 GB RAM and 20 GB temporary disk on an AMD Ryzen Threadripper 7960X host.
  The host exposes 48 logical CPUs, but the container affinity permits only 16.
  No GPU or network volume was used.
- Campaign wall time was **2,600.48 seconds (43 minutes 20 seconds)**. Training
  launched September 16, 2026 at **07:10:54 UTC**. Provisioning, packaging,
  retrieval and verification add to billed rental time.
- Python 3.11.16, PyTorch 2.5.1+cpu, NumPy 1.26.4 and SciPy 1.17.1.
  The six-job host smoke used non-confirmation seeds 101, 103 and 107.
- The complete 1,869,868,238-byte archive passed SHA-256 verification and safe
  extraction locally before shutdown. It contains reports, manifests, all
  scheduled checkpoints, inference exports, host smoke and non-secret operations.
- The audit checked the frozen plan, source fingerprints, all 16 job/training
  reports, final checkpoint descriptors and export hashes. All 16 final training
  states loaded, including replay, RNG and archive; their archived network weights
  matched the corresponding inference exports exactly. All 104 scheduled training
  checkpoint files are retained; only the 16 final checkpoints were restored.
- All 16 exported policies were independently re-evaluated locally against exact
  best responses. The maximum scalar/vector metric difference from Linux results
  was **5.43 × 10⁻⁸**, below the **10⁻⁶** evaluation tolerance. Every original
  readiness limit passes locally. No additional training or fitting occurred.
- Comparing locally reconstructed action probabilities with the checkpoint's
  Linux played-average diagnostic gives a maximum difference of **1.46 × 10⁻⁶**
  (Leduc 541). This exceeds the audit helper's initial 10⁻⁶ policy-array check;
  it is reported separately rather than called bitwise reproduction. The saved
  weights match exactly, and the exact evaluation differences above remain below
  the original audit tolerance. The readiness thresholds were not changed.

The implementation PR #62 passed 513 tests and CI before launch. Cleanup PRs
#63–64 removed obsolete workflows; 453 retained tests passed in a fresh minimal
environment. The cleanup did not change the solver source fingerprints. This
results PR changes documentation only; its relevant additional validation is the
artifact audit and report consistency checks.

## Rental and cost

The provider accepted stop at **08:06:24 UTC** and returned `EXITED`. The console
confirmed compute and container storage were not running, at **$0.00/hour**;
the disposable pod was then terminated, verified by **08:08:38 UTC**.
No network volume was created or left behind.

The all-in quote was **$0.563/hour**. Counting conservatively from **07:06 UTC**
through termination verification, including the stopped interval, gives
**$0.5876**, rounded up to **$0.59**. This is an estimate, not a
provider invoice, and stays within the **$1.50 / two-hour** campaign ceiling.
Prior conservative CPU usage was $2.08: the new total is **$2.67**,
leaving **$7.33** of the $10 CPU authorization. GPU funding is separate.

## Evidence and retrieval

Raw evidence is retained locally outside Git:

- Archive: `results/snapshot-readiness.tar.gz` (1,869,868,238 bytes).
- SHA-256: `6a59fa29bc9930886f78fb2031599d0908acb7105c8b64b367f714d760867330`.
- Extracted campaign: `results/snapshot-readiness-retrieved/snapshot-readiness/`.
- Raw report SHA-256: `47b7615c264b3c67bedfd0ea23824c34567906cbec7e119b8e691a3c3aba18eb`.
- Raw manifest SHA-256: `d0538a9a578f4ff38e69c699ae71eda87e49453acfb62d7b48c508d9d115e50a`.
- Resolved plan SHA-256: `455d15db17f03104dc3a08027aab274ce5ae21730c8e20335c34d1c6e341612a`.

The JSON report records each final export, checkpoint and training-report hash,
full source/environment provenance, verification differences and costs. Verify
the archive hash before extracting into a new directory. Large weights and replay
files are not committed or hosted elsewhere; retaining this local archive is
necessary for future artifact-level diagnosis. The terminated pod cannot serve it.

## Decision and next task

Close milestone 3's small-game gate and stop small-game tuning. Keep milestone 4's
**learning exit open**: its [first Hold’em report](holdem-baseline.md) completed
only ten of twelve jobs, hit an unequal-stack collection deadline, and established
no reliable improvement against its baselines.

Next, investigate that collection cost with bounded profiling, then declare a
meaningful multi-seed Hold’em training/evaluation baseline with realistic time
limits and recovery. Bring forward only the measurements needed from milestone 5
before committing to larger CPU/GPU campaigns. No new rental, training run or GPU
work is part of this result. No default model or professional-strength claim is
promoted, and the 1.0 release standard is unchanged.
