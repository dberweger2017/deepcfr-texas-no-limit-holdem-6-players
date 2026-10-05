# HU20 turn-search arena stage 4

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

Evidence locations are indexed in RESULTS_INDEX.md. [ETA comment](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166#issuecomment-5998252962) and the subsequent start comment on #166 record live progress. Final contrasts, per-host latency/fallbacks and actual spend remain pending complete verified evidence. Any stopped protocol remains incomplete and requires owner review before a new scientific run.
