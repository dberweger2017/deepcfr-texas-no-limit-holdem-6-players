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

## Measured entry-cap amendment

The first timed checkpoint reached iteration 2,333 with 1,611,738 entries, about 1.23 GB peak RSS, a 69,766,280-byte checkpoint, and a 12.27-second save. The original two-million-entry ceiling would end the slice before the one-hour wall cap despite ample measured RAM. A clean signal stop at iteration 2,522 retained 1,740,153 entries and a hash-pinned checkpoint. No invalid or numerical failure occurred.

The [amended plan](../configs/blueprint/m4-slice-v1-entry-cap-amendment.json) raises **only** `trainer.max_entries` from 2 million to 8 million; it keeps the seed, abstraction, sampling, node/time limits, and evaluation plan fixed. The runner accepts an increased entry ceiling on resume and continues the same policy state. Use a fresh artifact directory and about 43 more minutes of wall time, preserving the original total hour budget:

```bash
.venv/bin/python -m scripts.train_blueprint \
  --plan configs/blueprint/m4-slice-v1-entry-cap-amendment.json \
  --resume results/blueprint-m4-slice-v1/checkpoint.json.gz \
  --out results/blueprint-m4-slice-v1-continued \
  --workers 1 \
  --checkpoint-seconds 900 \
  --max-wall-seconds 2580 \
  --max-rss-gib 10 \
  --min-free-gib 30
```

This amendment responds to a measured limit, not an early poker result. Preserve both artifact directories and include both phases in the final resource report.

The completed measurements and retained checkpoint hashes are in the [M4 slice report](reports/blueprint-m4-slice-v1.md).
