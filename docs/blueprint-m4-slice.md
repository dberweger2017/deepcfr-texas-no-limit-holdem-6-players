# One-hour M4 blueprint slice

Use the [frozen plan](../configs/blueprint/m4-slice-v1.json) for one useful six-player 100 BB training slice. It keeps the pilot abstraction, raises the independent roots to four per seat per iteration, and sets high iteration/entry targets so the operational wall/RAM guards choose the first stopping point. It is a resource and coverage measurement, not a strength test. If the abstraction and sampler remain fixed, the retained checkpoint starts training seed `2026092402` for the longer campaign.

On the 16 GiB M4, run from a clean committed revision with one worker:

```bash
.venv/bin/python -m scripts.train_blueprint \
  --plan configs/blueprint/m4-slice-v1.json \
  --out results/blueprint-m4-slice-v1 \
  --workers 1 \
  --checkpoint-seconds 900 \
  --max-wall-seconds 3600 \
  --max-rss-gib 10 \
  --min-free-gib 30
```

The M4 reported 93 GiB free disk before launch on 2026-09-24. A normal cap stop exits 2 with a resumable checkpoint and no final arena export. The live process can exceed a guard within one iteration because limits are checked at iteration boundaries; the per-iteration node and 900-second limits provide additional bounds. If a guard trips, preserve the last checkpoint and report the reason. Do not silently reduce the work target or increase the cap. If the 10,000-iteration plan unexpectedly finishes first, the runner exports and performs only its two-block arena smoke; that is too small to claim strength.

Inspect iteration throughput and entry growth in early and late windows, trained/fallback decisions by street, checkpoint size/save time, peak RSS, free disk, and any invalid/numerical failures. Project memory and elapsed time for a longer two-seed run. If the projected table remains within M4 headroom and runtime is acceptable, continue on M4; otherwise use a bounded RunPod memory/worker test. Fresh scripted-pool validation and the v0.5 strength gate remain later steps in the [training roadmap](blueprint-training-roadmap.md).
