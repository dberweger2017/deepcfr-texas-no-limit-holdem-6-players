# Stability-gated B100M posterior audit on M4

Status: **scientific_stop**.
Reason: stability five-case posterior stability gate failed; primary values prohibited

Merged #126: `7d31a39c80cc72deb772336ce251f3c4bdcd46c9`; scientific source: `0d3a5cf1c5ba763ec0db149795650b2773236855`.
Immutable UTC start/cutoff: 2026-09-30T14:31:55.584733+00:00 / 2026-10-01T00:01:55.584733+00:00.
Real-clock gate: passed; initial stability: stopped; main stability: pending.

## What this establishes

The accelerated executor passed all 336 fresh-process real-clock comparisons
and every recorded attacker prefix had its requested batches. The engineering
speedup is usable on this corpus. The posterior estimator then failed its
prospective accuracy gate: four cases passed and the fifth failed all three
independent-repeat TV comparisons, all three four-versus-16 TV comparisons,
and all three zero-support-mass comparisons. No simulated likelihood row was
failed or soft-limited, and no posterior had zero total evidence.

This is a completed **measurement-validity result**, not a training-quality
verdict. In the failing seed-3 button flop, the low-count ranges disagree
substantially and omit holdings that retain appreciable mass in the higher-count
range. Sixteen samples are a comparison estimate, not an exact posterior. Even
the passing turn case has four-versus-16 TV around 0.141; passing these thresholds
does not prove that estimated action gaps would be insensitive to range error.

The supervisor stopped before main likelihoods, all 4,608 planned primary
worlds, 768 suit worlds and river references. The 24 selected decisions and
prospective 96-world, 48/48 protocol remain unchanged. No held-out gap,
role/seed profit contrast, training-mechanism attribution or playing benefit
is available from this attempt. There are no new playing hands to replay;
native attacker-prefix reconstruction and timer identity passed.

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

### Concrete next measurement and host decision

Recommend **one larger-likelihood stability assessment**, before another value
or training experiment. Keep these same five decisions (including the failure),
all seeds/positions, unchanged native LBR `(4 chance samples, 5 soft seconds)`,
no smoothing, independent new roots, and the existing TV/ESS/support checks.
Freeze its protocol and output paths prospectively; never extend this opened
attempt or reuse its completed simulation IDs as new independent evidence.

A candidate for the new protocol is three independent **16-sample likelihood
estimates per holding/action** against one independent **64-sample comparison**.
These are repetitions of the unchanged LBR action calculation, not an increase
to its internal four-chance-sample budget. This is a resource proposal, not a
claim that 16 or 64 will suffice. All five cases require 14,639 holding/action
coordinates, or 1,639,568 calls at this candidate allocation. First perform an
outcome-free cost/timer preflight, then freeze the allocation, gates, roots and
absolute window before inspecting new likelihoods. If this candidate remains
unstable, report that result; do not automatically raise counts or run values.

The executed stability phase took 8,905.99 seconds for 409,892 calls
(**46.02 calls/sec** including loading/checkpoint work). Constant-throughput
projection for that candidate is **9.90 hours**, or **12.37 hours with 25%
headroom**, before a separately measured verification allowance. Larger counts
may change cache hit rates and checkpoint cost, so the prospective preflight
must replace this linear estimate. A new window requires separate owner approval;
the present audit finished early and its deadline has not been extended.

Use **one M4 worker**, with the existing 10.5-GiB aggregate RSS, 0.5-GiB
swap-growth, 8-GiB free-disk and AC guards. Measured peak was 2.776 GiB;
cache growth at the larger allocation still needs the guard. Retain per-ID
journals, fsync/checkpoint at least every 15 minutes and every coordinate,
immutable source/model hashes, and verified missing-ID-only recovery. There
are no training arms, traversal nodes or new learned tables in this measurement.

**M4 is the recommended host.** No paid compute was used in #128. The separately
completed [CPU training parity pilot in draft #129](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/129) validates trainer
state/resume across Linux and macOS; it does not certify real-clock LBR behavior
on Linux or speedup from parallel likelihood workers. A paid posterior run would
need its own live quote and timer/parity/throughput evidence. No rental or new
scientific execution is launched by this report. Evidence remains insufficient
to choose a training intervention.

## Interpretation limits

- Stratified local diagnostics, not exact exploitability or a decomposition of -73.14 BB/100.
- Held-out intervals condition on estimated posterior; posterior-estimation uncertainty is separate.
- Five stability cases do not certify the other 19; no causal card/history/menu diagnosis follows.
- Empirical zero matches are finite-sample zeros, not proven impossible actions.
- No training, paid host, promotion or release criterion change.

## Retained raw artifacts

M4 root: `/Users/dberweger/Local/hu20-posterior-audit-v2/results/hu20-posterior-audit-v2-m4-20260930`. All raw likelihood/world journals remain here.
The original 262-file seal was verified after all phase/coordinator logs
closed. A separate publication check rehashed all **262 files and 100 inputs**,
independently recalculated all five TV/ESS/zero-support comparisons, and
confirmed that unexecuted science directories are absent. It did not rerun
any likelihood or value calculation. The sidecar is outside the immutable seal.

Compact [machine report](hu20-posterior-audit-v2-artifacts/report.json),
[final inventory](hu20-posterior-audit-v2-artifacts/hu20-posterior-audit-v2-m4-20260930-final-manifest.json),
[seal verification](hu20-posterior-audit-v2-artifacts/hu20-posterior-audit-v2-m4-20260930-seal-verification.json),
and [publication verification](hu20-posterior-audit-v2-artifacts/hu20-posterior-audit-v2-m4-20260930-publication-verification.json)
are retained in this PR. The 109,155,132 inventoried bytes remain on M4,
including raw likelihood journals, posterior weights and resource samples.

Final manifest SHA-256:
`9ec297b0eb856900da8a5a41c17c8a91feb614a5d7dd648e7c4e72951ca0359d`.
Machine report SHA-256:
`112aab9aed5786bf625156b9b6890d1d693937a5d3cfd0b6cf15be8597f336ab`.

```sh
scp -o HostName=100.122.216.94 -o BatchMode=yes -r \
  m4:/Users/dberweger/Local/hu20-posterior-audit-v2/results/hu20-posterior-audit-v2-m4-20260930 \
  /path/to/archive/
scp -o HostName=100.122.216.94 -o BatchMode=yes \
  m4:/Users/dberweger/Local/hu20-posterior-audit-v2/results/hu20-posterior-audit-v2-m4-20260930-final-manifest.json \
  /path/to/archive/
shasum -a 256 /path/to/archive/hu20-posterior-audit-v2-m4-20260930-final-manifest.json
```

Match each retrieved file's size/SHA-256 against its original absolute-path
entry in the manifest. Keep any large archive verification on M4 or another
authorized heavy-compute host. The pinned three B100M checkpoint/current-policy
pairs and raw-hand inputs are recorded in `verified-inputs.json`; all remained
unchanged. Older models, human demos, reports and UI/service files are preserved.

The coordinator scientific stop finished at 17:02:28 UTC; wrapper/seal finished
at **17:02:46 UTC (19:02:46 Madrid)**, well before the unchanged hard cutoff.
M4 has been released. Focused M4 tests passed **24/24**; final publication CI
status is tracked in the PR. No implementation or experiment setting changed
in this reporting commit.
