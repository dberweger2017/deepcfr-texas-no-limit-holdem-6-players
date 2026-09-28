# HU20 native-reopening resource preflight

All six outcome-free 1M-node runs completed at source `613539d152a5d9abf563a12756cd79001b4f556c`. Every deliberately cancelled iteration retained the prior checkpoint byte for byte, and the next complete iteration matched after resume. No failed traversal was retried or omitted. The 240 inference/LBR timing hands suppress all profit fields; their native replay is part of the final audit. These measurements establish resource feasibility, not strategic improvement.

| Arm | Resource seed | Complete nodes | Outer iterations | Entries | Outer p99 / max seconds | Training elapsed seconds | Peak RSS GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 2026093011 | 1,000,229 | 3,583 | 80,950 | 0.0339 / 0.0443 | 71.27 | 0.228 |
| A | 2026093012 | 1,000,272 | 3,684 | 80,661 | 0.0324 / 0.0515 | 72.75 | 0.228 |
| A | 2026093013 | 1,000,162 | 3,753 | 80,057 | 0.0323 / 0.0438 | 72.56 | 0.226 |
| B | 2026093011 | 1,000,389 | 2,626 | 118,978 | 0.0516 / 0.0666 | 77.81 | 0.331 |
| B | 2026093012 | 1,000,135 | 2,658 | 117,849 | 0.0551 / 0.0920 | 77.33 | 0.329 |
| B | 2026093013 | 1,000,339 | 2,690 | 117,942 | 0.0538 / 0.0826 | 76.78 | 0.329 |

The maximum over **all six** runs is 0.092 seconds per completed outer iteration; the earlier first-pair update's 0.067-second maximum covered only that pair. The uncapped arm has more entries and fewer completed outer iterations at equal nodes. Swap did not grow. Checkpoint writes cost 0.55–0.85 seconds and exports 0.39–0.66 seconds at 1M nodes. Street work, decisions/updates after two and after more than two raises, new/revisited keys, native engine identity, cancelled work and validation replay overhead are retained in the [resource summary](hu20-native-reopening-preflight-artifacts/resource-summary.json).

## Frozen main decision, before main outcomes

Choose **20M complete nodes per arm**, three paired fresh seeds, fixed 2M/5M/10M/20M checkpoints and final-current C. Choose **512 two-position LBR blocks per final target**, four future samples and five soft seconds. Cheap fixed attackers use 4,096 blocks at all four checkpoints plus the three original final HU references. The original six-opponent panel uses 512 blocks per final target/opponent. Total planned confirmation hands: **1,148,928**. Both arms receive identical attacker contracts and schedules; the target uses its own menu.

The conservative forecast allows twice the slowest complete-outer cost per node, 25% LBR/entry-growth overhead, and export/density/cheap-suite/audit allowances. It projects 3.91 hours training plus 4.14 hours reserved evaluation/reporting, 8.05 hours total after the decision, and 8.52 GiB peak. Larger 1,024/2,048 LBR counts fail the remaining-time gate. The forecast is not a tail guarantee: exceeding a guard stops and retains the attempt.

The original deadline is **2026-09-29T04:37:43.706907+00:00** (06:37:43 Madrid), ten hours after preflight began. It is never reset. Limits: 10.5-GiB RSS, 8-GiB free disk, 0.5-GiB swap growth, one heavy process at a time. Training ends early enough to retain its frozen 14,914-second evaluation/report reserve.

Prior #114 checkpoint-contrast variance suggests a **38.9-BB/100** 97.5% half-width at 512 blocks. This is a rough scale, not A/B power assurance, and may not resolve the fixed 10-BB/100 non-inferiority margin. An unresolved safeguard stays inconclusive. No margin or count changes after outcomes.

The [frozen plan](../../configs/blueprint/hu20-native-reopening-m4.json) records every root, count, limit, reference-model hash and resource-decision hash. Common independent public observations are constructed from fixed uniform/passive/repeated-raise paths before any main training. Diagnostics compare the same observations under each arm's keys; hash intersections are not a coverage measure.

## Retention

Raw preflight and main artifacts stay on `ssh m4` under `/Users/dberweger/Local/hu20-native-reopening-ab/results/hu20-native-reopening-m4-20260928`. Original preflight checksums and supervisor/resource records are preserved separately when main begins. Final inventory is generated after child logs close. Retrieval example:

```sh
scp -r m4:/Users/dberweger/Local/hu20-native-reopening-ab/results/hu20-native-reopening-m4-20260928 ./
```

Existing HU20/TP20 models, reports and demos are preserved. Draft PR #115 is dependent on draft #114; no merge, promotion, rental or follow-on campaign.
