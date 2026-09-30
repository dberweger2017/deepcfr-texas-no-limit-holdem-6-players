# Stability-gated B100M posterior audit on M4

Status: **scientific_stop**.
Reason: stability five-case posterior stability gate failed; primary values prohibited

Merged #126: `7d31a39c80cc72deb772336ce251f3c4bdcd46c9`; scientific source: `0d3a5cf1c5ba763ec0db149795650b2773236855`.
Immutable UTC start/cutoff: 2026-09-30T14:31:55.584733+00:00 / 2026-10-01T00:01:55.584733+00:00.
Real-clock gate: passed; initial stability: stopped; main stability: pending.

## Frozen stability findings

| Rank / street / seed / position | 4-vs-4 TV | 4-vs-16 TV | ESS ratios | 16 mass on four-zero support | Pass |
| --- | --- | --- | --- | --- | --- |
| 0086ab6bc713 / flop / 2026093001 / big_blind | 0.011619, 0.010861, 0.012140 | 0.010359, 0.008589, 0.008704 | 0.994243, 0.996590, 0.995044 | 0.005618, 0.004627, 0.005221 | True |
| 0101cbbe10d9 / preflop / 2026093002 / big_blind | 0.003476, 0.004676, 0.004888 | 0.003359, 0.003379, 0.005090 | 0.999485, 0.999179, 0.998934 | 0.000000, 0.000000, 0.000000 | True |
| 04cd69f4398f / turn / 2026093002 / button | 0.154020, 0.181168, 0.165172 | 0.142112, 0.141494, 0.141236 | 0.920944, 0.919346, 0.968533 | 0.067262, 0.062007, 0.050156 | True |
| 276c2b1a6f40 / river / 2026093002 / button | 0.092675, 0.106880, 0.095507 | 0.092127, 0.084098, 0.091786 | 0.965027, 0.970931, 0.966495 | 0.023948, 0.016769, 0.026384 | True |
| 0232d9bf7b38 / flop / 2026093003 / button | 0.373808, 0.383473, 0.373933 | 0.340645, 0.331440, 0.323459 | 0.747381, 0.749193, 0.750650 | 0.213008, 0.223106, 0.192536 | False |

Failures at `0232d9bf7b3858bf06a2bf384cc442da9ea7dd9dc49ed3b54b00c7a292058132`: four-0: TV versus 16 > 0.15; four-0: higher mass on zero support > 0.10; four-1: TV versus 16 > 0.15; four-1: higher mass on zero support > 0.10; four-2: TV versus 16 > 0.15; four-2: higher mass on zero support > 0.10; four-0/four-1: TV > 0.20; four-0/four-2: TV > 0.20; four-1/four-2: TV > 0.20.


## Conditional values and controls

Primary values: pending, 0/24 decisions.
Suit likelihood/value controls: pending / pending. River references: pending.
Pending conditional values are not zero gaps or negative findings. A failed gate prohibits the primary values.

## Cost, verification and next measurement

Scientific elapsed: 2.509 h versus the 8.66 h full-work projection.
Committed likelihood rows: 409,892; primary worlds: 0; suit worlds: 0.
Peak aggregate owned RSS: 2.776 GiB; maximum swap growth: 0.00 MiB; minimum free disk: 37.079 GiB.
Independent arithmetic: 20 posteriors and 0 held-out summaries; 100 input hashes unchanged.
Coordinator recoveries: 0; failed durable rows: 0.

**Exactly one recommended next experiment:** One prospectively specified larger-likelihood stability assessment of the same five frozen decisions, with independent higher-count references and an outcome-free M4 cost preflight. Determine a sufficient likelihood budget before another conditional-value or training experiment; do not reuse this attempt to choose favorable coordinates or seeds.

M4 remains the default. No paid host was used or quoted; current credit is unknown.

## Interpretation limits

- Stratified local diagnostics, not exact exploitability or a decomposition of -73.14 BB/100.
- Held-out intervals condition on estimated posterior; posterior-estimation uncertainty is separate.
- Five stability cases do not certify the other 19; no causal card/history/menu diagnosis follows.
- Empirical zero matches are finite-sample zeros, not proven impossible actions.
- No training, paid host, promotion or release criterion change.

## Retained raw artifacts

M4 root: `/Users/dberweger/Local/hu20-posterior-audit-v2/results/hu20-posterior-audit-v2-m4-20260930`. All raw likelihood/world journals remain here.
Retrieve with `scp -o HostName=100.122.216.94 m4:<absolute-root>/<relative-path> <destination>` and verify against `final-manifest.json`.
