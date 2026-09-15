# Fresh readiness confirmation

**The confirmation failed. Milestone 3 remains open.** All sixteen training jobs
and eight paired controls completed, but one seed in each game exceeded its
predeclared exploitability limit. Every value-error check passed. No checkpoint
was substituted, seed retried, threshold relaxed, or model promoted.

The [frozen protocol](../neural-readiness.md) ran at revision
`d41a0540e7f6caa093799d382e3900f8503cf391` (PR #54). The
[machine-readable report](neural-readiness.json) retains all final results,
72 checkpoint evaluations, paired comparisons, provenance and artifact checks.
The earlier exploration screen remains failed under its own criteria.

## Final results

Exploitability is the exact two-player best-response measure in **antes per
hand**; lower is better. Value error is the absolute difference between the
profile's player-zero value and the independently solved game value. Neither
metric estimates a percentile of human Hold’em players.

Only iteration 100 in Kuhn and 480 in Leduc count toward readiness. Every one of
the eight seeds in each game must satisfy both limits.

### Kuhn — 7/8 pass

Limits: exploitability ≤0.030; value error ≤0.030.

| Seed | Exploitability | Value error | Result |
| --- | ---: | ---: | --- |
| 401 | 0.021084 | 0.000694 | passed |
| 409 | 0.022176 | 0.003670 | passed |
| 419 | 0.019093 | 0.001718 | passed |
| 421 | 0.013356 | 0.002030 | passed |
| 431 | 0.030425 | 0.000459 | failed |
| 433 | 0.015123 | 0.001171 | passed |
| 439 | 0.018664 | 0.002026 | passed |
| 443 | 0.008597 | 0.001584 | passed |

Across seeds, mean exploitability is 0.018565, sample standard
deviation 0.006532, and worst seed 0.030425.
The mean does not replace the all-seed gate.

### Leduc — 7/8 pass

Limits: exploitability ≤0.150; value error ≤0.100.

| Seed | Exploitability | Value error | Result |
| --- | ---: | ---: | --- |
| 401 | 0.102388 | 0.008155 | passed |
| 409 | 0.129444 | 0.007735 | passed |
| 419 | 0.115970 | 0.008549 | passed |
| 421 | 0.130194 | 0.012770 | passed |
| 431 | 0.105186 | 0.005078 | passed |
| 433 | 0.160619 | 0.006372 | failed |
| 439 | 0.106757 | 0.005113 | passed |
| 443 | 0.132636 | 0.011070 | passed |

Across seeds, mean exploitability is 0.122899, sample standard
deviation 0.019481, and worst seed 0.160619.
The mean does not replace the all-seed gate.

## What the diagnostics explain

The saved averages separate errors introduced while collecting a finite replay
memory from errors introduced when fitting a network to that memory. These
comparisons describe the observed policies; their differences are not an additive
causal decomposition of exploitability.

| Failed run | Exact played average | Replay-memory average | Fitted neural average | Limit |
| --- | ---: | ---: | ---: | ---: |
| Kuhn 431 | 0.021526 | 0.022083 | 0.030425 | 0.030 |
| Leduc 433 | 0.148821 | 0.153602 | 0.160619 | 0.150 |

- **Kuhn 431:** the played and replay averages both pass. The final network
  introduces a remaining strategy-fitting gap. Kuhn deliberately kept its
  frozen constant-rate recipe; no cosine Kuhn result was tested here.
- **Leduc 433:** the played average passes by only 0.001179, while the replay
  average already exceeds the limit. Perfectly reproducing that replay average
  would still fail. The neural policy is worse again, despite a small excess
  fitting MSE of 0.00005427. A low fitting loss alone does not certify strength.
- All sixteen final exact played averages pass their respective exploitability
  limits. This supports investigating a different way to extract the average
  strategy, but the narrow Leduc margin is not evidence of robust convergence.

Intermediate results are diagnostic only. Leduc 409 was above the limit at
iteration 360 and passed at 480; Leduc 433 moved from 0.091980 at 360 to
0.160619 at 480. Selecting earlier checkpoints would change the experiment.

## Paired Leduc comparison

Each fixed-rate control refits the final candidate's saved replay with the same
initialization and minibatch random stream. Only the strategy learning-rate
schedule changes: cosine 0.001 → 0.00001 versus constant 0.001, both 48,000
updates. These are paired final fits, not eight additional collection runs.
Negative deltas favor cosine. Regressions are retained, as required by the
prospective protocol; they are not an additional readiness veto.

| Seed | Cosine | Fixed rate | Cosine − fixed |
| --- | ---: | ---: | ---: |
| 401 | 0.102388 | 0.108754 | -0.006366 |
| 409 | 0.129444 | 0.128305 | +0.001139 |
| 419 | 0.115970 | 0.119486 | -0.003517 |
| 421 | 0.130194 | 0.150422 | -0.020228 |
| 431 | 0.105186 | 0.101930 | +0.003256 |
| 433 | 0.160619 | 0.179030 | -0.018411 |
| 439 | 0.106757 | 0.127939 | -0.021182 |
| 443 | 0.132636 | 0.149205 | -0.016569 |

Cosine improves 6/8 seeds and worsens seeds 409 and 431. Its mean paired delta
is -0.010235 antes/hand. Cosine passes the absolute exploitability limit in
7/8 final fits; fixed rate passes 6/8. All paired-control value errors pass.
This is evidence of a useful but incomplete improvement, not uniform dominance.

## Execution and verification

- All 16 training jobs completed their declared iterations; all eight controls
  completed. There were no job errors, timeouts or missing scored seeds.
- Eight single-threaded workers on a Runpod AMD EPYC 4564P allocation with
  16 vCPUs, 32 GB RAM and 20 GB temporary disk; no GPU or network volume.
- Campaign wall time: **3,265.14 seconds (54 minutes 25 seconds)**, including
  job orchestration and final controls. Launch: September 15, 2026 at
  18:23:52 UTC. Rental/setup began earlier and is included in cost accounting.
- Python 3.11.16, PyTorch 2.5.1+cpu, NumPy 1.26.4 and SciPy 1.17.1.
- The remote checkout was clean at the frozen revision. Six small host smoke
  jobs on non-confirmation seeds 101, 103 and 107 passed before launch.
- The collector retrieved and SHA-256/gzip-verified the complete archive before
  requesting provider stop at 19:19:42 UTC. The provider returned `EXITED`.
  The console subsequently confirmed compute and container storage were not
  running, at **$0.00/hour**. The pod was then terminated by 19:24:48 UTC.
  No network volume was created or left behind.
- Local audit checked the frozen plan, all 24 source-file hashes against the
  launch Git revision, all 16 final snapshot descriptors and training-report
  hashes, all 72 scheduled evaluation records, and all 24 exported policy files.
- All 24 exported candidate/control policies were loaded and re-evaluated
  locally against exact best responses. Maximum scalar evaluation difference
  from the remote reports was **6.82 × 10⁻⁸**, below the 10⁻⁶ audit tolerance.
  Every final pass/fail decision was independently checked against the plan.
  This was verification, with no additional fitting or training.

PR #54's implementation validation passed all 298 tests and CI, including
constant-mode equivalence, shared-replay controls and fresh-process recovery.
This results change does not alter training code or the frozen protocol.

The initial container required Python/pip and the existing SSH public key.
Provider API requests initially failed; an explicit User-Agent enabled the
GraphQL client using the pod-scoped credential. The cgroup v1 allocation probe
was corrected before training. These were setup corrections, not confirmation
retries. The temporary web terminal was disabled after SSH setup. No credential
is included in the archived evidence.

## Cost

The quote was **$0.563/hour**, including temporary storage. Charging that rate
conservatively from 18:15:37 through the latest termination-verification time
of 19:24:48 gives about **$0.65**. This deliberately includes the stopped interval
and is an estimate, not a provider invoice. It is within the **$2.50 / two-hour**
campaign ceiling.

Together with the previous $1.43 allowance, conservative CPU usage is **$2.08**,
leaving **$7.92** of the owner's $10 CPU authorization. The provider's rounded
account display was $9.49 before and $8.94 after; keep that account balance
separate from conservative experiment accounting. GPU funding remains separate.

## Evidence and retrieval

Raw artifacts are retained locally outside Git:

- Archive: `results/neural-readiness.tar.gz` (138,850,141 bytes).
- Archive SHA-256: `101e3dc9d8742bb08a936f9ed9f06d54338dd08da30fb5ac720256c36938f400`.
- Extracted campaign: `results/neural-readiness-retrieved/neural-readiness/`.
- Raw report SHA-256: `598678c7acef9137dbfc690dcd439e0b9a21a7bdf153de4bc11bf26dcf0a4a5f`.
- Raw manifest SHA-256: `86b142c4cf3a3586082019593613e38456520a140e5d29cfa845883a671ded65`.
- Committed config-file SHA-256: `c856f6430128896f65c43f1fb05a87a5f4cdc3c5053dfd25ed2a5767098a6de8`.
- Resolved plan SHA-256: `90532915446dd2a5dea3b8e728680a9ff9d5cbb659f4e4d4d05ad99ebd535da9`.

The JSON report retains the complete frozen plan, source/environment provenance,
per-seed final snapshot and export hashes, every final result and every scheduled
checkpoint evaluation. Raw job directories retain replay snapshots, candidate
and control policies, detailed fitting reports and manifests; the archive also
contains the host smoke and non-secret operational records. Verify the archive
hash before extracting it into a new directory. Large weights/replay files are
not in Git and cannot be fetched from the terminated rental; retaining this local
archive is required for future artifact-level diagnosis.

## Decision and next task

Keep milestone 3 open and substantial Hold’em training blocked. Do not rerun
these consumed confirmation seeds as another fresh test, extend iterations,
change thresholds or start another width/learning-rate sweep.

The next task is a **bounded strategy-extraction design decision**, including a
Single Deep CFR comparison: specify how stored advantage-policy snapshots would
produce a correctly weighted average, its inference/storage cost, and the small-
game acceptance checks. The current exact played average is a diagnostic over
an enumerated game; it is not an implemented scalable replacement policy.
Avoid promising that removing strategy fitting alone solves the narrow underlying
Leduc margin. Any follow-up learning campaign needs its own prospective protocol
and fresh confirmation data.

Independent milestone 4 engineering can proceed with **complete public-history
and variable-seat decision encoding**. That work addresses known Hold’em input
limitations without spending another rental or claiming the learning gate has
passed. No result here establishes professional poker strength or qualifies 1.0.
