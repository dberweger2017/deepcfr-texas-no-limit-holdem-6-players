# First versus second-decision collection: online comparison

## Question and frozen recipe

Does expanding a second own decision improve learning from bootstrap at useful computational cost? The [frozen study](reports/holdem-collector-branching.md) showed lower conditional variance; the [local training pilot](reports/holdem-branching-pilot.md) verified the integrated path and measured manageable early costs. Neither established stronger poker.

Run [control](../configs/holdem/branching-online-first.json) and [candidate](../configs/holdem/branching-online-second.json), changing only `training.sampler`. All six roles train in six-player, 100 BB, unraked no-limit Hold'em with the existing finite bet menu and legal player observations. Bootstrap stays uniform over candidates; baseline, width, replay, exploration and fitting remain unchanged.

| Setting | Both arms |
| --- | --- |
| Independent training seeds | 719, 727, 733, paired across arms |
| Iterations | 512 per seed/arm |
| Collection | 32 roots per role per iteration; exploration 0.5; historical Q baseline |
| Fitting | Width 32, 64 fresh Adam steps, batch 32, LR 0.001, clip norm 1 |
| Replay | Separate 4096-record reservoirs per role, existing root normalization |
| Checkpoints | Every 64 iterations, including final |
| Average-policy export/evaluation | Iterations 128, 256, 384, 512 |
| Validation | 256 shared deal blocks, root seed 91919, style pool and random |
| Per-iteration ceiling | 250,000 nodes / 180 seconds, unchanged |
| Per-process ceiling | Three hours including evaluation/checkpoint work |

Each suite has 256 blocks × six seat rotations × two arena arms = 3072 hands. Across two suites, four checkpoints and six jobs: **147,456 scheduled table hands**, **48 full training checkpoints** and **24 average exports**. Arena arms remain trained average versus uniform-candidate control; the scientific comparison pairs the two trained averages' candidate outcomes on identical block schedules. Uniform outcomes supply a shared reproduction check. No new policy sees evaluation data during training.

## Primary decision

The primary endpoint is the **iteration-512 snapshot average against the style pool**, second-decision minus first-decision, in BB/100, for each of the three training seeds. Group all six rotations within their deal block. Retain the 256 paired block differences per seed and use a two-sided Student-t interval with Bonferroni family coverage 95% across the three final seed comparisons (per-comparison alpha 0.05/3).

A consistent improvement requires all three lower bounds above zero and an equally weighted mean seed improvement of at least **25 BB/100**. Report each seed even if the joint criterion fails. This criterion selects whether to pursue the collector; it does not promote a production model or demonstrate professional strength. The common deals and correlated rotations must not be counted as independent training runs. No precise population-level training-seed confidence claim is possible from three seeds; report their spread.

Earlier checkpoints and random-opponent results are descriptive diagnostics with pointwise block intervals, not alternative ways to pass. No best-seed/checkpoint selection, extra runs after a disappointing result or changing the acceptance threshold. No model is promoted automatically. Invalid play, non-finite training, a broken artifact/recovery contract or resource exhaustion stops the affected process; preserve its last complete boundary and failure. A failed arm/seed makes the planned complete comparison unavailable. Do not silently replace it or continue from an altered recipe.

## Cost and behavioral views

Report equal-iteration results and curves against measured cumulative collection/fitting/replay time, plus checkpoint, evaluation, setup and retrieval costs separately. A descriptive **1800-second training-work ceiling** selects the latest scheduled evaluation checkpoint whose cumulative `training-timing.total_seconds` is within the ceiling for each job. If none fits, label it unavailable; if all fit, report unused headroom. This coarse checkpoint comparison is not exact equal-cost training and does not replace the primary endpoint.

Retain collection counts by street and relative position, scheduled roots and roots containing postflop decisions. Distinguish numerous correlated branch records from more roots reaching a street. Also report replay size, inverse-reach tails, gradient/clipping statistics, actual evaluation decisions/all-ins and throughput. Card-generalization and coverage failures remain open; a branching benefit does not automatically solve them. Changing bootstrap, opponent sampling, action priors or replay weights is a separate intervention requiring explicit target semantics.

## Rental and host gate

The owner authorizes use of the remaining **$6.13 CPU budget**. This campaign reserves at most **$3.50 total and four hours from provisioning to termination**, including installation, calibration, training, evaluation, recovery, retrieval and storage. No GPU rental. The current candidate is Runpod's advertised 5-GHz compute instance: **16 vCPUs / 32 GB RAM at $0.56/hour**, plus a small container disk and an **80 GB network volume** for checkpoints, source and logs. The console quotes network storage at $0.07/GB/month ($5.60/month, approximately $0.0077/hour). Recheck the final quoted total before deployment. The volume must be removed after verified retrieval; its monthly rate is prorated, not a planned month-long rental.

Before campaign seeds, install pinned dependencies, run the focused collector/recovery tests and run the [first](../configs/holdem/branching-host-first.json) / [second](../configs/holdem/branching-host-second.json) host pilots concurrently, seeds 701/703/709 per arm. These six four-iteration jobs measure actual contention and memory; they are not strength tests. Each retains its 450-second limit and all outputs. Proceed only if all complete, observed memory leaves at least 50% of host RAM free at this early boundary, and the measured early rate projects 512 iterations within 120 minutes per worker before archive growth, and 1.30 times that worst early projection plus 45 minutes for checkpoints, evaluation, recovery and retrieval fits before the already-armed provider deadline. These are conservative host gates, not a guarantee of final cost. If they fail, stop provisioning work and revise the plan before using campaign seeds; retain the failed calibration.

### Host calibration amendment before campaign seeds

The original 90-minute projection gate **failed** on the rented Threadripper 7960X host: all six four-iteration jobs completed in 85.39 seconds elapsed, but mean iteration costs projected 73.79–107.25 minutes across workers. The three second-decision projections were 102.39, 107.25 and 91.56 minutes. Peak cgroup usage was 4.88 GB of the 32 GB allocation, below the original 50% ceiling. All 39 focused tests passed. An initial test command used a nonexistent filename and ran zero tests; the corrected invocation and both logs are retained.

Before any seed 719/727/733 starts, amend only the hardware admission estimate to 120 minutes plus an explicit deadline headroom calculation above. At the measured worst rate, 30% growth allowance plus 45 minutes gives **184.43 minutes** from campaign start through retrieval. This fits the existing rental cutoff if started by 00:00 UTC; target start is around 23:30 UTC. This is a revised resource estimate, not a retroactive pass of the original gate. The four-hour/$3.50 ceiling, three-hour job limits, 512 iterations, all scientific settings and decision criteria remain fixed. Retain the failed original gate and the revised admission record. If growth exhausts the allowance, preserve incomplete results rather than extend the rental or restart a seed.

Run one CPU thread per worker, six independent processes. A provider-level watchdog must terminate the specific disposable pod by the rental deadline independently of SSH or training, while retaining its separate network volume; save its confirmation. Runpod documents that a network-volume pod is terminated rather than stopped. Bound child processes with an external timeout as well as their internal limit. Keep TensorBoard private through SSH, provide the owner connection details, and collect every run's monitoring events.

After completion or failure, preserve hashes, manifests, every scheduled artifact and operation logs. Retrieve and verify the archive before terminating the pod; remove its billable storage after retrieval. If the watchdog terminates compute first, recover the separate network volume within the spending envelope. Do not destroy the only artifact copy to meet a timer. Stop compute early when finished; the four-hour ceiling is not a target runtime. No automatic campaign restart.

The storage choice was corrected before provisioning: the current console and [Runpod documentation](https://docs.runpod.io/storage/network-volumes) say container data is wiped on stop, while network volumes survive pod termination. EU-RO-1 reports high CPU availability. Persistent storage protects the only artifact copy if the watchdog fires; see [pod lifecycle rules](https://docs.runpod.io/pods/manage-pods).

## Execution and retained evidence

```bash
python -m scripts.train_holdem --plan configs/holdem/branching-online-first.json \
  --seed 719 --out results/branching-first-719
python -m scripts.train_holdem --plan configs/holdem/branching-online-second.json \
  --seed 719 --out results/branching-second-719
```

Repeat for 727 and 733 in separate processes. Use the existing TensorBoard sidecar across all six roots. Recovery verification re-exports/re-evaluates final saved boundaries without additional optimizer steps; compare saved bytes/outcomes. Fresh-process next-training-step equivalence is established by the integration tests. Hash-check all intermediate artifacts, and retain their compatible source/dependency environment for future model comparisons. Publish compact measurements and the final decision in this PR once the run finishes; large artifacts remain outside Git.
