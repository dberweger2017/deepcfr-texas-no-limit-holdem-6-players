# HU20 cross-board pooling diagnostic

Status: completed. 40/40 common three-export boards; retained weight 100.0%.

Primary held-out readout: **trainer/coverage consistent**.

| Group | Blueprint BB | Per-root v1 BB | Held-out v1 BB | Held-out equity50 BB | In-sample v1 BB | In-sample equity50 BB |
|---|---:|---:|---:|---:|---:|---:|
| pooled | 1.4194 [1.3310, 1.5161] | 0.4728 [0.4365, 0.5100] | 0.6567 [0.6273, 0.6868] | 0.3874 [0.3582, 0.4190] | 0.6063 [0.5787, 0.6331] | 0.3609 [0.3318, 0.3918] |
| 2026093001 | 1.4313 [1.3220, 1.5472] | 0.4733 [0.4360, 0.5120] | 0.6670 [0.6375, 0.6972] | 0.3863 [0.3563, 0.4188] | 0.6071 [0.5812, 0.6340] | 0.3587 [0.3289, 0.3908] |
| 2026093002 | 1.3248 [1.2436, 1.4148] | 0.4667 [0.4298, 0.5030] | 0.6496 [0.6169, 0.6833] | 0.3967 [0.3633, 0.4340] | 0.6018 [0.5731, 0.6287] | 0.3693 [0.3365, 0.4041] |
| 2026093003 | 1.5019 [1.4061, 1.6098] | 0.4783 [0.4411, 0.5168] | 0.6533 [0.6179, 0.6898] | 0.3792 [0.3474, 0.4151] | 0.6100 [0.5773, 0.6431] | 0.3547 [0.3236, 0.3908] |
| seat-0 | 1.2307 [1.1335, 1.3326] | 0.4584 [0.4162, 0.4999] | 0.6133 [0.5789, 0.6492] | 0.3287 [0.3008, 0.3581] | 0.5713 [0.5390, 0.6043] | 0.2930 [0.2727, 0.3136] |
| seat-1 | 1.6080 [1.4862, 1.7491] | 0.4871 [0.4553, 0.5202] | 0.7000 [0.6451, 0.7588] | 0.4462 [0.3915, 0.5056] | 0.6413 [0.5901, 0.6942] | 0.4288 [0.3748, 0.4855] |

Covered-context D uses the held-out strategy on covered keys and the per-root witness on absent/zero-mass keys. This is a hybrid sensitivity, not conditional EV or a causal decomposition. The primary D is descriptive only if either fold/lineage exceeds 5% missing target decision reach on either street. Detailed coverage and sensitivity intervals are in summary.json.

Intervals condition on the fitted pooled policies/codebook; they omit fitting uncertainty. Signed placement, ratios, pot-percent intervals, every seat result, exclusion, failure, resource/clock admission and hashes are retained in summary.json.

Cost evidence: `{"actual_cost_usd": 0, "host": "owner M4", "reason": "no rental", "unused_runpod_budget_usd": 5}`.

These are conditional turn/river feasible witnesses on the seed-1-selected limped/check-through public line, not a verdict on raised pots. Ranges are taken as given; preflop errors and flop strategy are excluded. Projections are not abstraction equilibria or lower bounds. Primary policies and codebooks fit only the opposite frozen half; in-sample losses remain secondary. Absent or zero-mass training keys use uniform probabilities, with raw per-street fallback reach coverage retained. This does not establish coverage of all boards or full-game strength.

The solver-free companion reports diagnostic board diversity, not unretained historical training occupancy. No training, promotion or automatic merge.
