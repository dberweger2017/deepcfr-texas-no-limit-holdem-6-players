# Acting-observation reuse on M4

## Result and recommendation

**Exact equivalence passed; recommend the narrow optimization for review.**
[PR #132](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/132)
remains draft and unmerged. The owner explicitly requires no automatic merge.
The [protocol](../trainer-observation-reuse-protocol.md) was committed before
heavy execution. Merged main was pulled at `7d74b6c82445c7865c3969f3757d9b64a7d2bccb`;
the executed implementation/harness was frozen at
`50326afcd4776308054e0c9efce8681de1e877eb`.

One private cache reuses only the acting seat's observation on the same
immutable concrete Hand with no prior-hand context. New instances, branches,
replaced histories, other seats and prior private records do not share it.
`Hand.apply`, full legal-action validation and the pinned native engine are
unchanged. The original control executes the literal merged-base observation
method; it shares the candidate class's unused empty cache slot, isolating the
replay-reuse effect rather than comparing unrelated trainer rewrites.

| Starting state | Original nodes/s, three runs | Candidate nodes/s, three runs | Median training speedup | Median whole-process speedup |
| --- | --- | --- | ---: | ---: |
| From zero, fixed seed 2026093011 | 16,984 / 16,932 / 16,963 | 20,932 / 20,784 / 20,993 | **1.233×** | **1.215×** |
| B100M, fixed seed 2026093001 | 16,539 / 16,513 / 16,640 | 20,335 / 20,554 / 20,502 | **1.232×** | **1.117×** |

This is approximately **23% more training throughput**, not Claude's
unreplicated 22× microbenchmark. The mature whole-process gain is about 12%:
load/save/export cost remains significant. All three fixed order-balanced
pairs are reported; three repeated timing measurements are not independent
training seeds or evidence of playing-strength improvement.

## Mature work and overhead

The unchanged B100M parent is seed 2026093001, iteration 246,212,
1,496,914 entries and **100,000,029 actual lifetime complete nodes**. Its
checkpoint SHA-256 is
`b560669df702057b9df72c90495195a60741e1c6c4b42603cc7c330e2615f64a`.
Its configuration, seed, iteration weights, K1, native reopening, current
extraction and original input bytes were preserved.

Each performance path added **1,000,569 complete nodes**, ending at iteration
248,467 and **1,503,132 entries**: 6,218 new keys. Small-table paths completed
1,000,389 nodes, iteration 2,626 and 118,978 entries. Overshoot preserves whole
iterations; no traversal was discarded. All paths then executed the same
extra next-iteration recovery check. Repeats, trace prefixes and resumed
suffixes overlap; they are engineering replays, not extra independent learning.

| Mature operation | Original range | Candidate range |
| --- | ---: | ---: |
| Load original checkpoint | 5.72–5.75 s | 5.72–5.76 s |
| Midpoint checkpoint save | 9.62–9.71 s | 9.62–9.73 s |
| Final checkpoint save | 9.73–9.80 s | 9.63–9.74 s |
| Current-policy export | 7.12–7.14 s | 7.05–7.15 s |
| Next-iteration checkpoint save | 9.65–9.84 s | 9.61–9.68 s |
| Benchmark path including hashes | 104.8–105.2 s | 93.0–93.4 s |

Whole-process ratios use supervisor start/finish durations, including Python
startup and completion polling (approximately one-second resolution). Internal
benchmark durations exclude imports before its timer. Training throughput
excludes saving/exporting/hashing. The whole process deliberately includes
recovery and hash verification; a long campaign with less frequent saves will
have a different overhead fraction.

## Correctness and recovery

- **40/40 focused M4 tests** passed: reuse boundaries, invalid-action rejection,
  private-history ownership, hidden-information observations, native reopening,
  generated chip accounting, trainer/checkpoint and replication checks.
- Both 100k-node traces have identical hashes for **every returned observation,
  ordered menu/key and action RNG seed/state/choice**. Mature trace completed
  100,063 nodes; the small trace completed 100,128.
- All original/candidate and repeated performance runs have **byte-identical
  full checkpoints/current exports/next checkpoints**, complete non-timing
  iteration records, keys, regrets, strategy sums, visits and next-root RNG
  hashes. No float tolerance or unexplained normalization was used.
- All four fresh-process midpoint resumes have byte-identical reload, final,
  current and next artifacts. Every resumed iteration equals the corresponding
  uninterrupted suffix, independently checked.
- **35/35 execution phases** completed without failure or guard violation.
  Independent publication verification passed **82 checks**, including all
  **212 original archive files**, source-body/action-validation invariance,
  input hash, repeat/resume work and resource guards.

### Retained measurement/reporting defects

The literal original method has a copied globals namespace; its observe-local
replays bypassed the replay-counter monkeypatch. Observation/key/menu/RNG
traces were instrumented directly and remain valid. Raw counters are retained.
Every successful original observe invokes replay exactly once, so the complete
original replay count is derived from counted transition replays plus original
observe calls, without rerunning or modifying training:

| Trace | Original complete replays, derived | Candidate replays, directly counted | Replays avoided |
| --- | ---: | ---: | ---: |
| Small | 243,253 | 143,388 | 99,865 |
| Mature | 243,124 | 143,120 | 100,004 |

The first publication verifier failed on an external-script import path after
its archive checks. Its log, empty output directory and supervisor attempt are
retained. A self-contained reporting-only correction passed in attempt 2 under
the unchanged deadline. No benchmark/traversal was repeated, and no experiment
setting or scientific result changed.

Publication CI initially passed one full-suite shard and failed one new control
test because the shallow checkout lacked Git object `7d74b6c`. The test now
supplies the literal historical observe method offline and additionally checks
its entire AST body against the unchanged uncached method. Assertions were
retained; the benchmark still obtains its control from real Git. **41 focused
M4 tests passed** on repair source `9750ba565fc4e4253960ab4daf780fa7ec1e0644` in
an isolated checkout, with the original engineering deadline and no benchmark
rerun. [Repair campaign](observation-reuse-artifacts/ci-repair-campaign.json)
and [test log](observation-reuse-artifacts/ci-repair-tests.log) are supplemental
CI records; the original sealed runtime/outcomes remain unchanged. Final
publication CI runs again.

## Resources and limits

One M4 child at a time, Python **3.11.14**, pinned engine **5db20e3**, AC and
caffeinate. Native build/environment hashes are in the compact summary.
The benchmark lasted **23 minutes 59.75 seconds** within its original four-hour
safety ceiling; the verification also finished under that same ceiling.
Highest timed-process lifetime RSS including load/save/export was **2.04 GiB**;
five-second sampled aggregate owned RSS peaked at **2.03 GiB**. Sampled swap
growth was **zero**, minimum free disk **34.32 GiB**, and all power samples were
AC. Sampling can miss short aggregate peaks; process lifetime RSS covers its
own peaks separately. No parallel trainer scalability is inferred.

This establishes the fixed-work engineering effect on one small seed and one
representative mature lineage. It does not establish later table growth,
500M throughput, three-worker scaling or stronger poker play. No original
model was replaced or promoted, and no poker evaluation ran.

## Archives and reproducibility

Compact [summary](observation-reuse-artifacts/summary.json),
[independent verification](observation-reuse-artifacts/verification.json) and
[publication manifest](observation-reuse-artifacts/observation-reuse-publication-manifest-20260930.json)
were transferred with matching M4/local hashes. Publication manifest SHA-256:
`0cbe7746f4cdddde09537ef9216b21515f23270d3632a70aade06a3d4acf031a`.
It includes the successful raw archive, both publication attempts, closed
supervisor logs and external reporting scripts/log. The original benchmark
manifest is separately retained; no failed attempt disappeared.

Large artifacts remain in:
`/Users/dberweger/Local/trainer-observation-reuse-20260930/results/`.
Retrieve the exact result sets without loading policies on M1:

```sh
ssh -o HostName=100.122.216.94 -o BatchMode=yes m4 \
  'cd /Users/dberweger/Local/trainer-observation-reuse-20260930 && tar -cf - results/observation-reuse-m4-20260930 results/observation-reuse-m4-20260930-manifest.json results/observation-reuse-verification-20260930 results/observation-reuse-verification-supervisor-20260930 results/observation-reuse-verification-attempt-2-20260930 results/observation-reuse-verification-supervisor-attempt-2-20260930 results/observation-reuse-publication-manifest-20260930.json' \
  > observation-reuse-m4-20260930.tar
```

Retrieve manifest-listed external `/tmp/doctor-research-report-observation-reuse*.py`
and `/Users/dberweger/Local/observation-reuse-m4-20260930.log` separately with
`scp` using the same SSH options. Verify all manifest members on M4 or another
authorized compute host; M1 remains lightweight. Commands in the sealed
`jobs.json` reproduce each original/candidate/trace/resume path. The original
input checkpoint stays at the #116 path in the frozen protocol.

## Separate next task

Recommend using this validated candidate in the mature-table RunPod pilot,
with the exact frozen source recorded even while this PR awaits owner review.
The proposed **$2 total cap** covers three CPU-only configurations, each with
5M additional nodes and fresh-process midpoint recovery. Live catalog rates
are $0.080/h for CPU3 general-purpose 2 vCPU/8 GB, $0.092/h for CPU5
general-purpose 2 vCPU/8 GB, and $0.130/h for CPU5 memory-optimized 2 vCPU/16 GB.
Approval is pending; no rental has started. That pilot gets a separate focused
PR and measures Linux/M4 state parity and mature economics. The eventual
150M/200M/300M/500M three-lineage plan remains conditional and unexecuted.
