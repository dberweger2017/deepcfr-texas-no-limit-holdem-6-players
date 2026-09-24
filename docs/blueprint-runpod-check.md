# One bounded blueprint scaling and learning check

Use the hash-pinned M4 seed-`2026092402` checkpoint at iteration 8,733 from the [M4 slice](reports/blueprint-m4-slice-v1.md). This is one paid measurement of worker scaling, large-policy export memory, and an early playing signal. It does not start a second seed or change the training abstraction. The source checkpoint SHA-256 is `94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a`.

## Admission and budget

The signed-in RunPod console on 2026-09-24 showed **$18.07** balance and **$0.44/hour** for the 3 GHz memory-optimized 8-vCPU/64-GB CPU pod. Its 20 GB container disk showed **$0.003/hour**, for approximately **$0.443/hour** before any transfer fees. Use one official Ubuntu CPU pod with that disk and no retained network volume. Recheck the complete live quote before provisioning. Cap total pod lifetime at **four hours** and the test at **$3 including disk and transfer**; stop earlier when work and retrieval finish. The displayed compute and disk rates total about $1.77 over four hours, leaving margin for transfer. Do not enable automatic top-ups. If the live total or available capacity no longer fits, stop and revise before renting.

Keep the pod under active supervision and set a hard four-hour termination deadline from its creation. Use a provider-level cutoff for the specific pod if one is available; an SSH or Python timeout alone does not end billing. Do not leave the pod running unattended without provider control. The disposable container disk is lost on termination, so retrieve and hash-verify every artifact first. If a cutoff interrupts retrieval, retain the failure record and recover what the provider still permits within the cap. No credentials or checkpoint bytes belong in Git.

## Fixed measurements

1. Install the reviewed Git revision and Python 3.11 with the repository's pinned `pokers` engine and CPU dependencies, using the [known-good CPU setup](runpod-multistreet-ops.md#remote-setup-and-launch). Run `tests/test_blueprint.py` and `tests/test_blueprint_scale.py` before loading the large checkpoint. Transfer the 252,235,464-byte M4 checkpoint and verify its SHA-256 before loading it. The source is retained on the M4 at `results/blueprint-m4-slice-v1-continued/checkpoint.json.gz`; transfer it over SSH to the ignored `results/blueprint-runpod-source/checkpoint.json.gz` path on the pod.
2. Run `scripts.check_blueprint_scale worker` from the **same** source checkpoint for 64 complete iterations with 1, 2 and 4 workers, in separate processes. Run 8 workers only if the 4-worker conservative parent-plus-worker RSS remains below 45 GiB and the pod has at least 10 GiB physical memory headroom. Keep each arm's output and any failure. Compare completed checkpoint hashes exactly; record total wall time including load/save, step nodes per second, worker RSS sums and Linux cgroup peak. Do not select a faster arm by poker result.
3. Run `scripts.check_blueprint_scale learning` on the original iteration-8,733 checkpoint. Export **current and average** strategies, record export time and peak RSS, and compare each with an empty-table blueprint that uses the same uniform abstract action menu. The [frozen arena plan](../configs/blueprint/runpod-check-v1.json) uses 128 fresh validation blocks, six seat rotations per block and the existing three-style opponent pool. It produces 1,536 hands per strategy comparison. Record legality, BB/100 paired difference and interval, preflop card probe and inference latency. This is an exploratory warning check, not v0.5 confirmation.

An exact worker hash mismatch, invalid action, non-finite update, failed checkpoint load, or resource guard failure blocks the longer campaign. A decisive loss to the empty-table control would direct investigation before AWS. An inconclusive 128-block result or a poor early style-pool estimate is retained as evidence, not a trigger for repeated small tuning runs. The full two-seed campaign still needs its own declared work target, budget and fresh final evaluation.

## Runner commands

From the reviewed repository checkout with the source checkpoint at `results/blueprint-runpod-source/checkpoint.json.gz`:

```bash
.venv/bin/python -m scripts.check_blueprint_scale worker \
  --checkpoint results/blueprint-runpod-source/checkpoint.json.gz \
  --expected-sha256 94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a \
  --out results/blueprint-runpod-worker-1 --workers 1 --steps 64 --max-rss-gib 48
```

Repeat with fresh output directories and `--workers 2`, then `4`, and conditionally `8`. Each arm must start from the same source path. The 48 GiB conservative RSS stop is a second guard; also watch the pod's actual memory and its 64 GB allocation.

```bash
.venv/bin/python -m scripts.check_blueprint_scale learning \
  --checkpoint results/blueprint-runpod-source/checkpoint.json.gz \
  --expected-sha256 94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a \
  --plan configs/blueprint/runpod-check-v1.json \
  --out results/blueprint-runpod-learning --max-rss-gib 48
```

Copy the source, all arm outputs, both exports, arenas, logs, quote, balance observations, and pod lifecycle record back to an ignored local artifact directory. Verify the two exported model hashes, every completed worker checkpoint hash and the source hash locally before terminating the pod. Record the final displayed balance and confirm the pod and storage show no continuing charge. Commit only a compact report and the next campaign decision.
