# Exact diagnostic LBR ranker on M4

**Result:** exact ranking and all 336 real-case checks pass. The larger frozen
workload runs **7.82× faster per LBR call and 5.90× faster including startup**.
Recommend the optional ranker for a new stability-gated posterior comparison
on one M4 worker: **8.66-hour projection, 9.5-hour safety window**, for owner
approval. The scientific audit has not started.

PR #126 is an engineering experiment. It generates hypothetical-attacker
decision vectors for equivalence/timing, but no combined target-perspective
posterior, target conditional-value audit or playing-return measurement.
The saved B100M models, original LBR, engine settlement and production defaults
are unchanged. No training or paid compute was run.

## Implementation and provenance

The [protocol](../hu20-exact-lbr-ranker-protocol.md) was committed at `a7f9ea7`
before candidate timings. Baseline main was
`a2053bbeea8a5e5170a1d83bf5d440684f82283d`. The corrected runtime source is
`c4f16184f7c982b539338b78cfab23d004b1ac9a`; the reporting source is separately
recorded in the machine-readable report and manifest.

One original pure Python seven-card ranker uses multiplicities and per-suit
bit masks, with an internally generated 8,192-entry straight table and the
same 8,192-entry tuple-key LRU limit. No third-party code, table or dependency
was added. Five/six-card inputs delegate to the original evaluator. The
optional `RankedCachedLocalBestResponse` binds the unchanged native
`choose_action` code object into private globals, replacing only its ranking
function. Each instance binds the active suit/deck context. It uses #121's
same immutable-source query cache; native LBR remains available and default.

All three original B100M policy/checkpoint identities, the model roster,
retained raw curve archives, the 17-file #121 seal and #123's sealed
result/attempt hashes were verified before execution and rechecked after.
The actual raw root is #117's `curve` directory, not #116's differently named
primary archive. Input hashes and public-prefix identities are retained.

## Correctness

- **34/34 M4 focused tests passed**, including legal-menu/zero-evidence
  behavior, exact ties and whole-comparison-batch timer/RNG tests.
- **100,000 distinct seven-card hands plus 20 explicit boundary fixtures**
  matched the independent best-five-subset evaluator. There were also
  **3,552 global-suit and 1,024 input-order comparisons**, all passing.
  This is substantial bounded coverage, not exhaustive seven-card enumeration.
- **336/336 fixed-work real LBR cases passed**: chosen action and exact
  raise-to, ordered menus, requested/completed batches, range weights/support,
  zero-evidence records, RNG state and every action value at absolute
  `1e-10`-chip tolerance. Repeats and coupled-suit controls passed.
- Fresh original/candidate 336-case output digests matched the fixed-work
  oracle `650d2635648ef455c3be2b4ab0742c9d9df0a9b93a6160adef7ef6cb6976c1ac`.

The generated rank corpus digest is
`3649d91ac05866f6ac707b8334ee6d42bbdffe3d776c18fe106ff17b1044b6e6`.
The #121 corpus and #119 selected coordinates/seed definitions did not change.
Information-safe hypothetical holdings/runouts are legitimate ranker inputs;
played hidden opponent cards and future deck order are not evaluator inputs.

## Timing and limits

The comparison uses fresh, uninstrumented sequential processes: shared-query
cache with original ranking, then the same cache with exact ranking. Timings
include every fixed call; they are not selected by speed or action result.
Cold microbenchmarks clear each cache and use all 100,000 distinct sets;
the following 4,096 repeated calls deliberately hit that cache.

| Fresh workload | Original wall | Candidate wall | Speedup |
| --- | ---: | ---: | ---: |
| 100k distinct rank calls | 4.614 s | 0.375 s | **12.30×** |
| 336 complete LBR calls | 66.304 s | 11.224 s | **5.91×** |
| 336 including startup/guards/output | 104.765 s | 51.620 s | **2.03×** |
| 3,591 first + 672 incremental LBR calls | 1,089.765 s | 139.416 s | **7.82×** |
| Larger workload including startup/guards/output | 1,144.775 s | 193.871 s | **5.90×** |

Distinct-rank CPU time was 4.519 versus 0.284 seconds (**15.90×**); rank-pass
wall time includes periodic guard overhead. The 4,096 warm LRU hits took
0.284 versus 0.295 milliseconds: there is no meaningful cache-hit speed gain.
The 336-case descriptive 95% decision-cluster speedup interval is
**[5.40, 6.47]×**. It is not a repeated-machine or full-workload interval.
Larger call CPU time was 1,089.698 versus 139.416 seconds. Three model loads
took 37.540 versus 38.352 seconds; preprocessing took 0.239 versus 0.246 s.

Query-cache contents, hits and misses match exactly: 29,992–30,680 entries
per seed, **99.12–99.18%** hit rate. Natural rank-cache counters finish at
1,699,876 hits / 21,412,844 misses for the reference and
1,678,109 / 21,398,525 for the diagnostic replacement, with 8,192 entries
in both. Those counters are cumulative across seeds and have the differing
engine-versus-diagnostic scope described below.

| Selected-target stratum | Full one-sample calls | Candidate first mean | Incremental mean |
| --- | ---: | ---: | ---: |
| Preflop | 3,675 | 52.045 ms | 51.927 ms |
| Flop | 12,972 | 32.657 ms | 31.959 ms |
| Turn | 18,630 | 28.726 ms | 27.632 ms |
| River | 22,770 | 27.661 ms | 25.684 ms |

The actual attacker-prefix inventory is different: **24,427 preflop,
19,580 flop, 11,070 turn and 2,970 river** calls, still totaling 58,047.
For example, actual river-prefix ranking has little marginal benefit because
of fixed-board rank reuse, consistent with the negligible warm-hit gain;
per-street rank-cache hits were not separately measured. The actual-prefix
timings are retained, not substituted for the prospectively specified strata.

Microbenchmark and LBR speedups answer different questions. Reference cache
statistics include other unchanged engine/disclosure uses of `hand_value`;
candidate cache statistics count the replacement's diagnostic path. Neither
ranking-cache telemetry nor query-cache hits imply fewer sampled holdings or
different RNG work. Measured output equivalence is the acceptance check.

The primary extrapolation retains #121's selected-target-street strata. A
river target decision can contain earlier-street attacker prefixes; the report
also reconstructs the actual attacker-prefix street inventory and timing
separately. Startup/model loading, first samples and independent incremental
samples are reported separately. Additional-sample projection uses at least
the first-sample mean in each stratum.

These are one machine and one declared timing order. The descriptive
decision-cluster bootstrap covers variation across the 24 selected decisions,
not future host load, cache growth or repeated benchmarking uncertainty.
The full likelihood workload and subsequent value/control work were not run.

## Timer semantics and future attacker identity

Controlled clocks passed both already-expired preparation and crossing the
deadline after the second complete batch: both executors perform at least one
whole comparison batch and stop only between batches, with identical cards
and RNG progression. Production's five-second soft rule is unchanged.

Every fresh 336-case and larger 4,263-call comparison matched actions,
ordered menus, completed work, range/zero-evidence records, RNG progression
and action values. **Zero limited calls, zero over-soft calls and zero
original/candidate differences** occurred. The larger maximum call wall time
was 0.551 seconds original versus 0.144 seconds candidate. Its fresh output
digest is `9a7636deb5a5378c2ebcd1b1a345a272e64e87127b07436dfb18e8706173c278`
for both executors.

Finite-corpus equivalence does not prove unconditional equality for a
host-dependent time-limited attacker. The proposed scientific audit therefore
includes real-clock original/candidate prefix checks before posterior work,
verifies original recorded completion and stops before values on a mismatch
or limited attacker. A faster executor cannot silently redefine the original
attacker being modeled.

## Retained attempts and resources

The initial wrapper put a log in the source checkout and failed the clean-tree
check before tests, model loads or the engineering clock. Its startup log is
retained. Attempt 2 at `645bb86` completed, but review found an import-time
`DECK` snapshot in the optional production executor. Its fixed-clock helper
rebound globals and therefore bypassed that production-context defect.
An actual-executor regression failed on the old source as expected.

The correction binds active context per instance and changes only that actual
instance's clock for fixed-work validation. The corrected attempt reran the
same counts/corpus/seeds with a fresh output path. It also stops the repeated
rank timer before digest bookkeeping, which was included in the earlier
microtiming. Attempt 2 and its hashes are preserved; none of its timings enters
the final performance claim. No candidate or parameter sweep followed.

One immutable engineering clock began at `1790769277.5442078` and ends at
`1790783677.5442078` (September 30, 15:54:37 UTC / 17:54:37 Madrid).
The corrected attempt reuses that exact clock. All heavy work stayed on M4;
M1 only edited, used Git and transferred compact records. One heavy child ran
at a time, with AC/caffeinate and external owned-job/RSS/swap/disk supervision.
The quarter-second old-source regression is retained separately and was not
a model-loaded benchmark; it has no claim of second-by-second resource sampling.

The retained 3,314 external active-child samples across original and corrected
attempts show **2.99 GiB** peak aggregate owned RSS, **no swap growth** from
the original 761.38-MiB baseline, and **37.45 GiB** minimum free disk. The
corrected benchmark finished 3,909 seconds after the original start. M4 is
Apple M4 / 16 GiB, macOS 26.6 ARM64, Python 3.11.14; the installed native
engine binary hash is recorded. Full posterior cache growth remains unmeasured.

Final verification also exposed a reporting-only accounting defect: idle
`child=0` included the system process tree in boundary/report guard samples.
Active benchmark samples used actual child PIDs and correctly tracked only
the owned job. The fix excludes that sentinel, with **2/2 additional M4 guard
tests passing**. The first report, its 60-file seal and verification are
retained verbatim. A fresh publication directory contains the corrected
report: 94-MiB reporter peak, about 97-MiB guarded aggregate. No benchmark,
gameplay, clock, counts or timing/cost fields were rerun or changed.

The final **64-file inventory and all 32 inputs** were independently rehashed
on M4 after benchmark/wrapper logs closed. The corrected report hash is
`ea9ea795e68a30e13bea81a9020347ed15380324af54b5ea43e91449db19f52c`;
manifest hash is
`c82f6808163e23b7588f6753147d4933b56bb3b3249abec11356f41b18e70fe6`.
Compact transfers were checked against those seals. Large raw attempt rows
and resource logs remain on M4.

## Recommendation and next experiment

**Use the optional exact-ranker shared-cache executor for the proposed audit,
subject to its additional original-attacker timer checks.** It passed the
declared correctness gate and saved substantial complete-workload time.
The full 58,047-call one-sample estimate is **0.494 hours**, versus 3.676 hours
for the original ranker in this fresh run. The historical minimum main-only
four-sample/value/control design projects to 3.49 hours, versus 19.40 hours.
That minimum omits the new stability work and is not the approval budget.

| Proposed component | Calls/worlds | Candidate hours before headroom |
| --- | ---: | ---: |
| Main four-sample likelihood | 232,188 calls | 1.978 |
| Three 4-sample repeats + 16-sample stability, five decisions | 409,892 calls | 3.584 |
| Four-sample coupled-suit likelihood, four decisions | 49,908 calls | 0.434 |
| Two ranges × 96 worlds × 24 decisions | 4,608 worlds | 0.480 historical proxy |
| Suit worlds / river references / report | 768 suit worlds + bounded references | 0.333 historical reserve |
| Original/candidate real-clock prefix checks | Prespecified holdings/seeds | 0.067 allowance |
| Five phase loads/preprocessing | Three models per phase | 0.054 allowance |
| Total before / after 1.25× headroom | 691,988 likelihood calls | **6.929 / 8.661 h** |

Keeping the original ranker with all new checks would project to **57.69 h**;
it is an alternative requiring a revised approval if timer faithfulness fails,
not prohibited by a ten-hour cutoff. An eight-sample main likelihood variant
would project to **11.13 h** with the candidate, but is **not** the recommended
design. Four samples are not an accuracy certificate: the stability gate is
mandatory. These are extrapolations, not measured full runs or promises.
The historical 1,728-second value proxy and 1,200-second controls/report
reserve, 240-second timer allowance, five-load estimate and full cache growth
remain uncertain; 1.25× headroom does not establish their coverage.

**Recommend M4, one worker, a new 9.5-hour absolute safety window** from its
first heavy scientific work, with the last 30 minutes reserved for reporting
and sealing. Expected full cost is about 8 h 40 min; stop scientific work at
start + 9 h and never extend or reset the clock. Freeze actual UTC/Madrid
timestamps at an owner-approved start. Early stability/timer failure stops
before values, so ten hours is neither a workload target nor a feasibility
rule. Checkpoint atomic likelihood/world shards at every coordinate and at
least every 15 minutes; verify hashes and resume only missing deterministic
IDs within the same deadline. Detached M4 supervision must survive SSH loss.

This fits a single measured M4 worker without rental/transfer or cross-platform
uncertainty. Larger cache residency is guarded, not assumed to remain at
benchmark memory. M4 availability must be coordinated before starting.
No material need for a paid pilot is demonstrated. No live RunPod quote,
worker scaling or current balance was verified; balance is unknown. Paid
compute would require a separate specific quote, authorized one-worker pilot
and measured parity/scaling/recovery/shutdown plan. No GPU assumption or
eight-worker configuration is adopted.

The [proposed scientific protocol](../hu20-posterior-audit-v2-proposal.md)
retains all 24 outcome-blind coordinates and all model identities. Each range
gets 96 worlds: 48 select an action, 48 independently estimate its paired gain
against the saved policy. Uniform/posterior comparison is paired, with suit
controls and bounded independent terminal-river reference checks.

Before values, five prospectively selected decisions cover every street,
all three seeds and both positions. Three independent four-sample likelihood
repetitions plus one 16-sample estimate add 28 samples on this subset; the
main four-sample estimate is checked against them too. Raw match integers and
denominators, finite-sample zeros, support, ESS and total variation are retained.
Failure of the stated stability/timer gate stops before values, without
smoothing, selecting new coordinates or raising counts. Passing five cases
does not certify all 24. Held-out intervals condition on an estimated
posterior and cover rollout uncertainty, not all posterior-estimation error.

**One next scientific experiment:** the stability-gated paired posterior
comparison, for owner approval. Persistent local gaps would identify
vulnerabilities under the declared attacker/range, not prove a particular
training or abstraction remedy. An inconclusive result does not authorize
continuation. No posterior audit or new training has started.

Claude v7's M1 training profile remains unreplicated: the temporary profiling
scripts were absent, and its overlapping cumulative shares are not a cost
sum. The separate [observation-reuse handoff](../hu20-observation-reuse-handoff.md)
targets reuse for the same immutable Hand and acting seat while preserving
legal-action validation. It requires fixed-work keys/regrets/averages/visits,
RNG and byte-identical deterministic checkpoint/resume evidence before a
mature-table M4 benchmark. It changes no trainer in this PR.

## Reproduction and retrieval

Compact records: [report](hu20-exact-lbr-ranker-artifacts/m4/publication-final/report.json),
[manifest](hu20-exact-lbr-ranker-artifacts/m4/publication-final/manifest.json),
[final verification](hu20-exact-lbr-ranker-artifacts/m4/publication-verification.json),
[input identities](hu20-exact-lbr-ranker-artifacts/m4/verified-inputs.json), and
[transfer hashes](hu20-exact-lbr-ranker-artifacts/compact-transfer-hashes.json).
The [proposed selection](hu20-posterior-audit-v2-proposed-selection.json) fixes
all stability/suit/river coordinates and RNG roots; no scientific run exists.

Do not rerun sealed timing attempts to chase performance. For an authorized
engineering reproduction, use a fresh root and the frozen supervisor command
recorded in `supervisor.json`; replay the same corpus and source/seed/counts.
This would be a new engineering attempt, not a reset of the retained clock.

Retrieve the entire archive, including earlier attempts and the first seal:

```sh
mkdir -p /tmp/pr126-archive
rsync -a -e 'ssh -o HostName=100.122.216.94 -o BatchMode=yes' \
  m4:/Users/dberweger/Local/hu20-exact-lbr-ranker-pr126/results/ \
  /tmp/pr126-archive/results/
rsync -a -e 'ssh -o HostName=100.122.216.94 -o BatchMode=yes' \
  'm4:/Users/dberweger/Local/hu20-exact-ranker-pr126-*' \
  /tmp/pr126-archive/external/
rsync -a -e 'ssh -o HostName=100.122.216.94 -o BatchMode=yes' \
  m4:/Users/dberweger/Local/hu20-exact-lbr-ranker-pr126/docs/reports/hu20-posterior-audit-v2-proposed-selection.json \
  /tmp/pr126-archive/hu20-posterior-audit-v2-proposed-selection.json
```

The manifest uses exact absolute M4 paths. Map its clone `results/` prefix to
`/tmp/pr126-archive/results/`, external `/Users/dberweger/Local/` files to
`/tmp/pr126-archive/external/`, and the proposal JSON to the final transfer.
Verify each recorded byte count and SHA-256 on M4 or another capable host;
do not run a large archive verification on the travelling M1. Underlying
policy/checkpoint/raw-input locations and hashes are listed in
`verified-inputs.json` and remain in their established #116/#117/#121 roots.

The final 64-file manifest includes the original report/manifest and its
verification, corrected guard-test log and all retained benchmark attempts.
The final verification record is a separate post-seal attestation with hash
`a5fc1f1d015b8ba1486fd14440e17a6d02b80e344d49c74e9c02e1893158f64f`.
No frozen parent result, old attempt or deadline is overwritten.
