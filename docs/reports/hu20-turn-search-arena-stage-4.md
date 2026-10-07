# HU20 turn-search arena stage 4

**Stopped at the approved first-500 checkpoint:** 86/500 live search decisions timed out (17.2%, above 5%). The frozen protocol is incomplete: 13,560/82,944 complete hands retained. No training or arena job remains running, and full base-versus-base+search contrasts are unavailable. [Final results on #166](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166#issuecomment-5999110011).

The owner approved dropping the RTX 5080 `vvc7cmrepn7rtt` and running six independent worker coordinates on four transferred pods. The 5080 termination and not-found readback are retained. No new pod was rented for this approved layout.

| Pod | GPU | Actual quota CPUs | Actual RAM | Coordinates | Solver threads per worker |
| --- | --- | ---: | ---: | --- | ---: |
| m5pxmipuqyjtoo | RTX 3070 | 18.7 | 24 GB | 0, 1 | 6 |
| mislfsw9a0bmva | RTX 3070 | 18.7 | 24 GB | 2, 3 | 6 |
| 6oqjxlfdjk0bzp | RTX 3080 Ti | 13.6 | 30 GB | 4 | 6 |
| xtu3jr3utxqasx | RTX 3080 Ti | 10.2 | 46 GB | 5 | 6 |

All hosts have 40 GB container disks, admitted against measured free space and reduced retention. Both 3070s passed admission, 80/80 Linux checks, checker/Mac parity with zero strategy/value differences, 96/96 native request replay with maximum profile difference zero, and retention checks. The owner then authorized starting those hosts while proxy-only hosts finish setup. Coordinates 0–3 started at 16:14 UTC on October 5. First hands are base-arm evaluation; no training, completion or strength claim.

Scientific source remains `6a5ff44ec3e5d313be594d177244c3579081ff00`. [CI run 37322306683](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/actions/runs/37322306683) was rechecked successful. All 82,944 hands, 13 panels, three lineages and `request-sha256-lowest-1pct-v1` remain frozen. Every hand/request/receipt/response and generated profile SHA/size is retained; only the predeclared outcome-blind hash sample retains full profile bodies.

The original four-host quote at the owner's decision was $21.22 maximum/$7.29 expected. Before first dispatch, elapsed setup refreshed this to **$21.70 maximum/$7.77 expected** under the same measured p95 × 1.5 / mean-block rules, fleet rates and all reserves. The agreed 30-minute sleep charge is excluded from the cap, not from actual provider billing. The $25 hard ceiling and $21 dispatch stop remain. Global and per-host checkpoints stop at >5% fallbacks after 500 search decisions, with subsequent cumulative/trailing checks unchanged.

The authenticated controller lives on the first 3070. M4 supervises that controller and actual pod state, with no allocation or scientific restart capability. Before any arena dispatch, the zero-event preflight controller was upgraded to support the owner's partial-start instruction; its original empty journal and explicit upgrade receipt remain retained. Operational files are separately SHA-bound to preserve the approved scientific source. An M1 sleep resumes monitoring of these existing processes.

Every joining host must pass its own actual admission, parity and retention before its reserved coordinate is issued. At completion or a stop, owned writers are cancelled/frozen, sampled bodies replayed, all retained evidence packed in lossless ≤1 GB chunks, and every archive and member hash stream-verified before termination. Original manifests remain authoritative. Final list-pods must confirm removal; historical exited pods `43z4itur3hwnyv`, `cl0riravggku4r`, `xu414eguzakxfr`, `k9rdph2fwhym87` are protected.

Evidence locations are indexed in RESULTS_INDEX.md. [ETA comment](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166#issuecomment-5998252962) and the subsequent start comment on #166 record live progress. The stopped result is recorded below; the incomplete protocol cannot support the planned strength contrasts. Any stopped protocol remains incomplete and requires owner review before a new scientific run.

## Stopped result and verified closeout

Both 3080 Ti hosts also passed actual-host admission, checker/Mac parity, 96/96 exact replay and retention before joining their reserved coordinates around 16:49–16:50 UTC. They completed base-arm hands but reached no live search decisions before the global stop. The controller stopped at exactly 500 completed live decisions with no issued/inflight decisions remaining; the original stop reason is preserved.

| Pod | Exact live decisions | Timeout fallbacks | Fallback rate | p95 latency in retained complete hands |
| --- | ---: | ---: | ---: | ---: |
| m5pxmipuqyjtoo | 239 | 46 | 19.25% | 30.510 s |
| mislfsw9a0bmva | 261 | 40 | 15.33% | 30.482 s |
| 6oqjxlfdjk0bzp | 0 | 0 | unavailable | unavailable |
| xtu3jr3utxqasx | 0 | 0 | unavailable | unavailable |

Latency telemetry covers 497 decisions in retained complete hands; the exact controller counts above include three decisions in interrupted hands. Workers 0–5 retained 2,446 / 2,440 / 2,445 / 2,452 / 1,832 / 1,945 complete hands respectively. All partials and negative outcomes remain retained. Observed solver processes requested six solver threads (seven OS threads including main); this run does not establish CPU contention as the cause of the timeouts.

Every retained compressed chunk and every archived member was stream-verified by SHA-256 and size. The original manifests and retrieval verification receipts are retained on M4. All five owned pods were terminated, each termination was read back, and final list-pods confirmed no active pods. The four protected historical exited pods were untouched. The Macs retain compressed evidence without expanding unsampled full profiles.

Full provisioning spending is estimated at **$1.587550 gross** ($1.538851 compute plus $0.048700 conservative disk), or **$1.068661 against the owner's cap** after the agreed $0.518889 sleep exclusion. Available posted billing buckets total **$0.957235 through 16:00 UTC**; the last 16:00–17:00 UTC bucket was not posted at closeout. These estimates are not a settled invoice. The quote's unused storage/pilot contingencies are not actual spend.

A local final-report import-path error occurred after evidence verification and termination. Its original failure record is preserved; the reporting import was corrected and the final report posted without restarting any pod or science. No rental, retune or further arena run is authorized by this closeout. Next: owner review of the timeout evidence before a new prospectively defined host/timing proposal and paid-run quote.

## Archive and billing refresh

The final-report import-path fix is covered by a fresh-process regression that loads the approved scientific audit from a checkout separate from the operational package. All 15 operational checks pass. Both raw evidence and reporting failure receipts remain unchanged.

The verified research ZIPs are now at `~/Local/Research-Cloud/PR-166-HU20-turn-search-arena/`: `stage-4-closed-M4-20261005.zip` (541,375,924 bytes, SHA-256 `d4ff034cd19d6e6fd8db39313ceba16ff76031d86bc5bc8b74dfea87a87defc6`, 659 verified members), and `stage-4-closed-M1-20261005.zip` (6,372,654 bytes, SHA-256 `aee43e7c031cf5b2cac8b7f67b6ee6367e95fc196d11227fb4dbf11482de8777`, 74 verified members). Archive/member hashes were reverified at the destination; task credential copies and bytecode caches are excluded. Original files remain preserved.

Fresh MCP billing now includes the final 16:00–17:00 UTC bucket: **$1.170743 posted gross / $0.651854 cap-counted** after the agreed sleep credit. This replaces the earlier incomplete posted sum, while preserving the separate conservative provisioning estimate. A fresh list-pods again confirms no active pods. The 13,560 stopped-run hands are historical evidence only and will not be reused in an amended arena.
