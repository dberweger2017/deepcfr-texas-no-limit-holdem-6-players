# Longer six-player Hold'em experiment

**All three seeds completed 512 iterations with verified artifacts and final-boundary recovery. Useful poker learning remains unproven.** Every one of the 48 scheduled paired comparisons is inconclusive, and all three final policies lose heavily to the fixed style pool. No model is promoted.

This delivers the longer experiment evidence for the v0.5 research milestone. It is not a v0.5 release announcement, a milestone-4 learning pass, or evidence of the v1.0 professional standard. The [machine-readable report](holdem-longer-training.json) contains every seed, checkpoint, evaluation, artifact hash, manifest and resource summary.

## Protocol and execution

The [committed protocol](../holdem-longer-training.md) and [plan](../../configs/holdem/longer-05.json) were executed at clean source revision `2ea1a7d7925ef76b6256df11adc393651a2d1c92`. Seeds 307, 311 and 313 ran as three independent CPU processes. All six roles trained in six-player, 100 BB, no-rake no-limit Hold'em using first-decision expansion, sampled continuations and frozen value baselines. The learner uses the documented finite legal betting menu; it does not enumerate every possible chip-sized raise.

Each iteration collected 32 roots per role and reset width-32 role networks and Adam for 64 fitting steps at batch size 32 and learning rate 0.001. Replay retained 4,096 records per role. Full checkpoints were saved every 64 iterations, with average-policy exports and four fixed arena suites at 128, 256, 384 and 512. No recipe changes, training retries, omitted seeds, early outcome-dependent stopping or checkpoint selection occurred.

Training started on September 16, 2026 at 11:20:09 UTC; all jobs and final evaluations finished by 12:47:50 UTC, **87 minutes 41 seconds elapsed**. The original archive was ready around 12:51. Retrieval and recovery verification followed before termination at 13:20:56 UTC.

## Playing results

Each comparison uses 256 independent deal blocks, all six seat rotations and two paired arms: 3,072 table hands. The campaign completed **147,456 scheduled evaluation hands with zero invalid actions and zero failed hands**. That count includes correlated rotations, paired arms, repeated checkpoints and shared schedules; it is not 147,456 independent deals.

`styles` compares the average policy with uniform-candidate play against five fixed styles. `random` uses five random opponents, with the same uniform-candidate control. `previous` compares with the pinned seed-211 iteration-6 pilot against the style pool. `crossplay` compares with that pilot against five independent pilot copies. The weak pilot is an historical anchor, not a competitive strength benchmark.

### Final policies

All values are BB/100. Each bracket is a nominal paired 95% interval for the difference from the named control. Candidate results against styles and previous are the same hands; the paired control differs.

| Seed | Suite | Candidate | Difference from control | Paired 95% interval |
| --- | --- | ---: | ---: | --- |
| 307 | styles | -1189.58 | -19.34 | [-337.38, 298.71] |
| 307 | random | 60.97 | +36.36 | [-454.62, 527.34] |
| 307 | previous | -1189.58 | -37.53 | [-373.24, 298.17] |
| 307 | crossplay | -21.45 | +111.00 | [-466.17, 688.18] |
| 311 | styles | -1422.33 | -252.08 | [-560.81, 56.65] |
| 311 | random | -152.34 | -176.95 | [-656.97, 303.06] |
| 311 | previous | -1422.33 | -270.28 | [-570.41, 29.85] |
| 311 | crossplay | -19.82 | +112.63 | [-442.83, 668.09] |
| 313 | styles | -968.42 | +201.82 | [-143.51, 547.16] |
| 313 | random | 264.49 | +239.88 | [-249.79, 729.55] |
| 313 | previous | -968.42 | +183.63 | [-122.23, 489.48] |
| 313 | crossplay | 280.44 | +412.89 | [-162.27, 988.05] |

The absolute style-pool intervals are entirely below zero for all final seeds. The final random-opponent candidate intervals all include zero. Neither these results nor the comparisons with a weak historical model establish basic competence or improvement. They also do not prove that every larger network or longer run would fail.

### Complete learning curves

All scheduled checkpoints are retained below; values are candidate BB/100. Full candidate/control intervals and paired differences for all 48 comparisons are in the JSON report. Every paired interval includes zero. The curves show no consistent improvement across the three seeds.

| Seed | Iteration | Styles / previous | Random | Crossplay |
| --- | ---: | ---: | ---: | ---: |
| 307 | 128 | -940.82 | -52.44 | 147.01 |
| 307 | 256 | -1352.02 | -394.40 | -77.05 |
| 307 | 384 | -1260.58 | -297.82 | -336.49 |
| 307 | 512 | -1189.58 | 60.97 | -21.45 |
| 311 | 128 | -939.55 | 111.49 | 122.20 |
| 311 | 256 | -1179.62 | -355.50 | -47.95 |
| 311 | 384 | -1271.52 | -192.74 | -236.13 |
| 311 | 512 | -1422.33 | -152.34 | -19.82 |
| 313 | 128 | -1003.12 | -176.69 | -368.07 |
| 313 | 256 | -1278.19 | -205.37 | -78.48 |
| 313 | 384 | -1031.80 | 220.54 | 83.20 |
| 313 | 512 | -968.42 | 264.49 | 280.44 |

Intervals use the paired deal-block estimator, not individual hands. These are reused validation schedules, not fresh final-test confirmations or multiplicity-adjusted campaign claims. Six roles from one training seed are not six independent training seeds. No significance-based seed or checkpoint selection is justified.

## Fitting and sampling diagnostics

| Seed | Role fits | Clipped / total steps | Median loss after / before | Largest pre-clipping norm |
| --- | ---: | ---: | ---: | ---: |
| 307 | 3,072 | 196,608 / 196,608 | 0.998541498 | 224738624 |
| 311 | 3,072 | 196,608 / 196,608 | 0.998692450 | 92807440 |
| 313 | 3,072 | 196,608 / 196,608 | 0.998091760 | 1587202176 |

All **589,824 optimizer steps** were clipped at norm 1. Median sampled diagnostic loss fell by only about 0.13–0.19% within a fit. This motivates a fitting diagnosis; it does not establish that clipping caused weak play or that the full empirical objective barely moved. Diagnostics sample 128 retained records, while noisy importance-corrected targets can dominate the loss. Fresh networks receive 64 × 32 = 2,048 draws with replacement per fit from 4,096 retained records: expected distinct coverage is roughly 39%, not a complete replay pass.

| Seed | Largest inverse own sampling reach | Largest absolute corrected regret update (BB) |
| --- | ---: | ---: |
| 307 | 98304.00 | 40149548.48 |
| 311 | 30099.48 | 9078484.88 |
| 313 | 149883.73 | 47402757.32 |

These large regret updates are importance-corrected estimator values, not physical hand winnings. All completed fits stayed finite. More samples, different fitting budgets, estimator variance and representation remain distinct hypotheses. Replay street coverage and gradient alignment require separate measurements.

## Resources and cost

Host: AMD EPYC 4564P, **16 allocated vCPUs, 32 GB allocated RAM**, 50 GB container disk, CPU PyTorch 2.5.1 and Python 3.11. Three single-threaded training processes ran concurrently. Host-wide physical RAM is not the pod's allocated memory. Source and exact dependencies are preserved in the manifests.

| Seed | Job minutes | Collection seconds | Fitting seconds | Replay seconds | Peak process RSS (GiB) | Admitted / stored records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 307 | 84.12 | 1740.90 | 2345.61 | 89.71 | 4.784 | 308,490 / 24,576 |
| 311 | 87.34 | 1863.51 | 2403.03 | 97.18 | 4.828 | 319,725 / 24,576 |
| 313 | 87.61 | 1873.36 | 2411.15 | 95.15 | 4.856 | 326,451 / 24,576 |

Stage durations are accumulated within each process. They overlap across seeds and must not be summed as elapsed campaign time. Job runtime also includes checkpoint serialization, evaluation and other overhead. RSS is a per-process high-water mark, not simultaneous live memory. Checkpoint sizes/times and evaluation latency at every boundary are in the JSON report.

| Seed | Final styles candidate mean / p95 action latency (ms) | Final styles evaluation seconds |
| --- | ---: | ---: |
| 307 | 0.566 / 0.761 | 7.38 |
| 311 | 0.596 / 0.815 | 10.23 |
| 313 | 0.603 / 0.847 | 9.66 |

The verified quoted rate was **$0.567/hour** including container storage. Approximately 2 hours 6 minutes from provisioning through retrieval and termination cost about **$1.19** at that rate; reserve **$1.20** conservatively. This is a rate-based estimate, not a final invoice. Conservative cumulative CPU spending is **$3.87**, leaving **$6.13** of the owner's $10 authorization. No GPU budget was used. The rental was terminated and the console showed no remaining pod or billable network volume.

## Recovery, integrity and retained failures

After training, each final checkpoint was loaded in a fresh process on the same remote environment and resumed at its already-completed iteration 512. **Zero new training iterations** ran. Each reproduced nine deterministic files byte for byte: the checkpoint, average export, root/job result, learning curve and four raw-outcome files. All **27 comparisons matched**, with 684.50 seconds elapsed across sequential verification. This verifies final-boundary restoration and evaluation; it is not another seed, a continuation-learning experiment, or a cross-platform bitwise guarantee.

Local streaming verification checked **266 inventoried files** and **36 catalog artifacts** (24 full checkpoints and 12 average exports) within the original 284-file archive. Every model byte is retained. The first compact-extraction attempt hit its 1 GB safety guard because it included large raw outcome files. The corrected extractor omitted those files from extraction while still hashing their archived bytes. Both the initial failure log and successful verification remain local; no training job was retried.

The monitoring implementation was validated with 567 repository tests before the campaign. This report's 48 evaluation rows were checked directly against the retained evaluation JSON files; artifact inventory and recovery hashes are retained with the report. Publication changes documentation only and does not reinterpret model bytes.

## Artifact retrieval and comparisons

The owner retains these files under the repository's ignored `results/` directory. **They are local archives, not public downloads.** A fresh clone needs an owner-provided copy; publication hosting remains a release-packaging task. Do not claim publicly retrievable models until that is arranged.

| Archive | Bytes | Contents |
| --- | ---: | --- |
| `results/longer-05.tar.gz` | 6,623,622,300 | All three runs, checkpoints, average exports, hand outcomes, TensorBoard events, manifests, dependency records and operations logs |
| `results/longer-05-verification.tar.gz` | 10,578,536 | Recovery reports/scripts/logs and reproduced nonbinary outputs; byte-identical duplicate model files omitted |

Original archive SHA-256:

```text
882e59506b9162462ac76f2c46e9fc9cddbd7c908db5a4f24faac22802760a83
```

Recovery archive SHA-256:

```text
9cc4542f9783b29f5cc471af9d9b4a9fbf6cfedea84e231fe8e7d27a2e922ea4
```

Verify archive checksums before extracting. Extract only the files needed on machines with limited disk space. For example:

```sh
shasum -a 256 results/longer-05.tar.gz results/longer-05-verification.tar.gz
mkdir -p results/longer-restored
tar -xzf results/longer-05.tar.gz -C results/longer-restored \
  results/longer-05-seed-307/manifest.json \
  results/longer-05-seed-307/scenario-0-seed-307/average-512.pt
```

Use each catalog path relative to `results/longer-05-seed-SEED/scenario-0-seed-SEED/`, and verify its SHA-256 from the JSON report. The retained pilot reference is pinned by the executable plan. Load model bytes with the saved source/dependencies and supported `holdem-average-v1` adapter; preserve every seed as an anchor for later release comparisons.

## Decision

Finish the v0.5 evidence and artifact packaging without claiming useful learning. Before another longer campaign, use the fixed iteration-256 replay for a local **clip 1 / unclipped × 64 / 256-step** comparison, with matched initialization/minibatches, unchanged targets and all three seeds. Then compare current-policy and snapshot-average play under one declared paired arena schedule. Neither diagnosis alone demonstrates stronger poker.

Keep the production recipe unchanged until those measurements identify a justified next experiment. Scaling model width, renting a GPU, switching estimators or adding search is premature as the immediate response to this run. Milestone 4's useful-learning goal remains open; the longer-run research evidence is now available. No-rake multiplayer poker is zero-sum across players, but the two-player zero-sum CFR convergence guarantee does not extend automatically to this six-player setting.
