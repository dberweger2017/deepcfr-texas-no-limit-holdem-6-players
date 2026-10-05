# HU20 calibration-02 clean stop and final-stage forecast

**Stopped under the [owner-approved budget amendment](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/148#issuecomment-5967372814); awaiting a further budget decision.** Calibration-02 retained 238 screen rows (all 128 native rows and 110 reduced-menu rows) and zero final rows. Eighteen reduced-menu screen rows remain unattempted. No setting is qualified. The owner-authorized SIGTERM is retained as an incomplete attempt; worker 18889, sidecar 18890 and their native children exited. No resource guard failed and swap remained 901,513,216 bytes.

All 1,752 manifest members / 1,614,849,743 bytes passed independent size/SHA checks in the M1 copy. [Stop provenance](hu20-turn-search-artifacts/calibration-02-owner-stop.json), [verification](hu20-turn-search-artifacts/calibration-02-owner-stop-verification.json) and the [complete retained screen curve](hu20-turn-search-artifacts/calibration-02-retained-screen.jsonl) are published. The running source remains 4105419 and original settings remain unchanged. Calibration-01 is separate; none of its timings enter this forecast.

The charged attempt consumed 6,312.987 seconds. The append-preserving journal totals **12,053.465 seconds (3.348 hours)** of the original 86,400 seconds, leaving **20.652 hours**. Part A, builds, pilots, failures and verification are included. The heartbeat is PAUSED and no replacement worker has launched.

## Timing-only forecast

The [machine-readable forecast](hu20-turn-search-artifacts/calibration-02-final-budget-forecast.json) ranks the complete native screen by cold p95 then median, separately for each floor. It reads no residuals, returns or quality outcomes. Six threads without compression is fastest for both floors; six threads with compression is second. All failures remain in cold timing samples. At 30 seconds, the linear screen scaling retains 25/50/100 iterations; at 120 seconds it retains 25/50/100/200/400.

| Finalists per floor | 30-second final play + quality | Conditional 120-second final play + quality | River reserve | Required remaining time |
| --- | ---: | ---: | ---: | ---: |
| 2 | 21.418 h | 94.852 h | 3.000 h | 119.271 h |
| 1 | 10.576 h | 46.835 h | 3.000 h | 60.410 h |

The conditional 120-second column is included in full when planning the fallback case. Quality evaluation is a second native request: it repeats fixed solve work, checks the exported matrices, and evaluates full-native deviations under reference/search laws. Its measured receipt time is additional to cold play time.

| Floor | Threads | Compression | Cold p95 at 100 iterations | Cold median | Additional quality receipt p95 |
| --- | ---: | --- | ---: | ---: | ---: |
| 0 | 6 | off | 27.354 s | 19.523 s | 23.055 s |
| 0.01 | 6 | off | 27.322 s | 19.540 s | 23.014 s |
| 0 | 6 | on | 27.673 s | 20.475 s | 23.981 s |
| 0.01 | 6 | on | 27.669 s | 20.552 s | 23.952 s |

Each cold estimate includes eight screen roots; additional quality timing has seven completed play requests because one timed out. This is a linear p95 planning estimate, not a runtime upper bound. Timeout censoring and full-root variation may increase cost. Preparation outside receipts, the eighteen remaining reduced-menu screen rows, any new 120-second screen, verification and build overhead are **excluded**, making the estimate favorable to fitting the budget. All candidate timing cells remain in the JSON.

Even an optimistic counterfactual sharing both seats under both floors and ignoring all quality evaluation requires **23.777 hours**, including the three-hour river reserve, for one finalist plus the conditional 120-second final. That already exceeds the remaining 20.652 hours. This is a planning comparison, not a guaranteed minimum runtime.

## Seat independence and proposed next decision

At epsilon zero, an unlocked turn root constructs the same request for both bot seats. The fixture test checks exact request equality and all exported holding matrices. At epsilon 0.01, flooring applies only to the opponent's history factors, so the bot seat can change the range weights. A holding-dependent action-likelihood fixture verifies that these unlocked requests differ. Exact request comparison and absence of locks are therefore mandatory before sharing; floor 0.01 is conservatively forecast as separate. The tests use the deterministic fixture solver; production seat sharing has not been activated or verified against every native reference root.

The [new immutable proposal](../../configs/diagnostics/hu20-turn-search-calibration-budget-proposal-03.json) is explicitly `forecast-blocked`, cannot launch in the production runner, and does not overwrite calibration-02. No reduced roots, lineages, gates, reference laws, river strata or arena counts are proposed. Fifty-eight search/campaign/protocol/reference tests pass; with the imported exact-turn/flop diagnostics, 83 tests pass after integration of main's published #145 evidence; the only merge conflict was the roadmap, and both task records are preserved.

**Recommendation for owner decision:** authorize only the 30-second final with one timing-selected finalist per floor, all roots/lineages/seats and 25/50/100 iterations. Its estimated play + quality + three-hour river reserve is **13.576 hours**, leaving about **7.076 hours** for uncounted overhead. If it yields no qualifier, pause for a separate decision about the 120-second campaign rather than admitting that campaign now. This recommendation changes the previously automatic research fallback and therefore needs approval. Alternatively revise the iteration coverage or cumulative budget prospectively. No paid allocation is included.

The owner amendment explicitly says: “If the forecast still can't fit, stop and ask rather than shrinking roots or lineages.” Consequently the task and recurring continuation remain paused. After a new decision, implement exact sharing/retained-screen resume, test it, freeze amended settings, refresh resource admission and push before any final outcomes. Preserve all existing evidence and the original #145 binary. Sampled rivers and the measured RunPod quote remain pending.
