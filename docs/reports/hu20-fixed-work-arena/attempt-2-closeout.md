# HU20 fixed-work arena: attempt 2 closeout

Attempt 2 closeout: 82,944/82,944 completed hands independently replayed. All three pods' full archives and member hashes were verified on the M1 before termination; get-pod 404 and complete list-pods receipts confirm removal.

Base+search minus base, BB/100, exploratory paired 95% Student-t intervals conditional on the three frozen lineages:

| Panel | Primary | Sensitivity (exclude defect-affected pairs) | Complete / flagged / incomplete lineage pairs |
| --- | --- | --- | --- |
| uniform | 5.79 [-20.44, 32.03]; n=256 | 5.79 [-20.44, 32.03]; n=256 | 768 / 0 / 0 |
| passive | -21.94 [-55.87, 11.99]; n=256 | -21.94 [-55.87, 11.99]; n=256 | 768 / 0 / 0 |
| minraise-cap2 | -15.89 [-49.77, 18.00]; n=256 | -15.89 [-49.77, 18.00]; n=256 | 768 / 0 / 0 |
| pressure-cap2 | -0.72 [-25.20, 23.77]; n=256 | -0.72 [-25.20, 23.77]; n=256 | 768 / 0 / 0 |
| tight_passive | 2.54 [-2.65, 7.73]; n=256 | 2.54 [-2.65, 7.73]; n=256 | 768 / 0 / 0 |
| loose_passive | -5.34 [-26.88, 16.21]; n=256 | -5.34 [-26.88, 16.21]; n=256 | 768 / 0 / 0 |
| tight_aggressive | 0.72 [-4.79, 6.22]; n=256 | 0.72 [-4.79, 6.22]; n=256 | 768 / 0 / 0 |
| loose_aggressive | 3.26 [-14.93, 21.44]; n=256 | 3.26 [-14.93, 21.44]; n=256 | 768 / 0 / 0 |
| pot_pressure | 15.62 [-0.66, 31.91]; n=256 | 15.62 [-0.66, 31.91]; n=256 | 768 / 0 / 0 |
| train_pressure | 7.68 [-11.71, 27.08]; n=256 | 7.68 [-11.71, 27.08]; n=256 | 768 / 0 / 0 |
| native-pressure | 25.82 [14.48, 37.17]; n=2048 | 25.82 [14.48, 37.17]; n=2048 | 6144 / 0 / 0 |
| selective-stackoff | -10.35 [-18.89, -1.82]; n=256 | -10.35 [-18.89, -1.82]; n=256 | 768 / 0 / 0 |
| lbr | 39.35 [30.43, 48.26]; n=2048 | 39.26 [30.34, 48.18]; n=2047 | 6144 / 1 / 0 |

Per-lineage contrasts retain every eligible completed pair. The three-lineage contrast requires a shared block ID in all three lineages; the JSON report records every omitted joint block. Crash-missing pairs enter neither analysis; recovered complete hands enter both unless defect-affected.

Recorded per-decision defects: 1; fallbacks/conditioning gaps including interrupted-hand decisions: 1. Resume-log entries: 23. Full defect and recovery lists are retained in `attempt-2-results.json`.

Per-host decision timing (seconds; descriptive):

| Host | Decisions | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: |
| 688f951f39b4 | 7130 | 28.909415620227808 | 38.06436364962952 | 77.114698151825 |
| e0de5970c901 | 7401 | 28.866921368055046 | 38.316375178983435 | 76.08844638313167 |
| 2226815b1ed0 | 11717 | 19.749638466630127 | 30.400333865843717 | 64.36701077688485 |

Flagged/excluded pairs by pod:

```json
{
  "primary": {
    "gdyfqg9817qme0": {
      "planned": 5958,
      "complete": 5958,
      "incomplete": 0,
      "defect_affected_complete": 1,
      "included": 5958,
      "excluded_defect": 0
    },
    "ne7ui0na5wd27u": {
      "planned": 5958,
      "complete": 5958,
      "incomplete": 0,
      "defect_affected_complete": 0,
      "included": 5958,
      "excluded_defect": 0
    },
    "yig5a8bfutpxjg": {
      "planned": 8820,
      "complete": 8820,
      "incomplete": 0,
      "defect_affected_complete": 0,
      "included": 8820,
      "excluded_defect": 0
    }
  },
  "sensitivity": {
    "gdyfqg9817qme0": {
      "planned": 5958,
      "complete": 5958,
      "incomplete": 0,
      "defect_affected_complete": 1,
      "included": 5957,
      "excluded_defect": 1
    },
    "ne7ui0na5wd27u": {
      "planned": 5958,
      "complete": 5958,
      "incomplete": 0,
      "defect_affected_complete": 0,
      "included": 5958,
      "excluded_defect": 0
    },
    "yig5a8bfutpxjg": {
      "planned": 8820,
      "complete": 8820,
      "incomplete": 0,
      "defect_affected_complete": 0,
      "included": 8820,
      "excluded_defect": 0
    }
  }
}
```

Spend record:

```json
{
  "pods": [
    {
      "id": "gdyfqg9817qme0",
      "hours": 20.38043513139089,
      "usd": 2.8192935265090733
    },
    {
      "id": "ne7ui0na5wd27u",
      "hours": 20.276720498336687,
      "usd": 2.804946335603242
    },
    {
      "id": "yig5a8bfutpxjg",
      "hours": 18.854675688611138,
      "usd": 6.5677120315328805
    }
  ],
  "basis": "provisioning wall clock \u00d7 readback compute and disk rates; estimate, not settled invoice",
  "historical_charge_usd": 0.670942,
  "fleet_usd": 12.191951893645196
}
```

These are the frozen descriptive contrasts and predeclared defect sensitivity. No model promotion, release claim, or new experiment follows. Attempt 1 remains archived and contributes no observations.

The primary contains **20,736 complete lineage pairs / 82,944 hands**. Sensitivity contains **20,735 pairs / 82,940 hands**. Its three-lineage LBR average drops shared block 1281 across the three lineages; other lineages retain that eligible pair in their individual contrasts. No crash-missing pair remains after partition recovery. The flagged hand is seed 2026093001, search/LBR/block 1281/rotation 1 (`zero_support_opponent`). There were zero timeout fallbacks and zero turn-conditioning gaps. LBR completed every requested four-batch evaluation; its 1,428 elapsed soft overruns are descriptive only.

Five automatic partition restarts occurred: four on worker 0 and one on worker 3, all from the enforced swap-growth guard. The 23 resume-log entries describe recovered streams across those five restarts, not 23 crashes or additional unique hands. The hand coordinates and settlement replay establish unique completed coverage.

Host completion and shutdown:

| Pod / GPU | Workers / hands | Recorded crashes | Launch to last completed search decision (h) | Deleted UTC |
| --- | --- | ---: | ---: | --- |
| `gdyfqg9817qme0` / RTX 3070 | [0, 1] / 23,832 | 4 | 9.700 | 17:20:21 |
| `ne7ui0na5wd27u` / RTX 3070 | [2, 3] / 23,832 | 1 | 9.365 | 17:14:12 |
| `yig5a8bfutpxjg` / RTX 4090 | [4, 5, 6] / 35,280 | 0 | 8.077 | 15:48:56 |

The decision spans use controller timestamps from the 07:19 UTC launch; they are not exact last-hand finish times. Worker elapsed seconds reset on resume and are retained separately. Timing includes live and hypothetical search decisions. Controllers were still running at evidence capture; all of their workers were complete before deletion. Host IDs in the latency table map in order to these three pods.

Provisioning-to-termination compute plus disk estimate for the three pods is **$12.191952**. It includes overnight holding/setup and both attempts. Previously recorded historical rental charge is separately **$0.670942**, giving **$12.862894** estimated cumulative recorded spend. Current overlapping posted billing is **$11.929676** (3070 buckets through 17:00 UTC, 4090 through 16:00 UTC); final 3070 partial buckets are not yet posted. Posted amounts are not added again to the clock estimate. No settled invoice or current account balance is claimed. Fleet ongoing burn is zero.

[Compact result and operational receipts](attempt-2-results.json) retain per-lineage/position contrasts, absolute arms, whole-hand partitions, coverage, tails, host exclusions, billing and source identities. Raw block arrays, controller events, decision records, failed streams, retained solver profiles, input bundle and three exact source snapshots remain in the research ZIP and M1 originals. Approved science `249a79b` and launched operational head `8632ccf` are distinguished. The default-off `continue_on_defect` and `--resume` changes were reviewed; complete retained-hand replay passed. This was an agent source review, not an independent review. Closeout tools `046c472` passed [full CI](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/actions/runs/37485084849) before live use; local retrieval was first tested against copied incremental evidence.

Full research ZIP: `attempt-2-full-closeout-M1-20261006.zip`, 5,645,638,363 bytes; SHA256 `cac0d2a766a8d0276820662112b68bb12ffe3688b8597faa0f63d43a829b3a0e`; 1,433 top-level member files verified. Nested full pod archives retain 125,156 verified evidence members. Destination: [PR-166 Research-Cloud folder](https://drive.google.com/drive/folders/1iVCttvjpYo8X4jD9C3tcflLyT_Y9QBqO). Native cloud upload confirmation is pending; M1 originals and all older archives are preserved. A separate publication seal retains the subsequent CI, PR and wake-disable receipts.

Owner-authorized merge validation: integrated current `main` while preserving its newer roadmap, research storage rules and cleanup receipts. Conflicts were confined to ROADMAP.md and RESULTS_INDEX.md; the executable arena and frozen result bytes remain unchanged. The current tree passes 135 focused tests across turn-search strict/continue behavior, partition recovery, arena operations, verified per-pod closeout, paired sensitivity and LBR batch completion. No open GitHub review thread remains. Full CI on the integrated candidate remains the merge gate.
