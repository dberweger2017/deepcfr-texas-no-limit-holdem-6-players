# HU20 turn-search arena: formal RunPod quote

**Superseded by [revision 2](hu20-turn-search-arena-quote-02.md): fixed 1% profile retention, compressed retrieval, $6.03 mean / $16.25 maximum. This original quote is historical, not the approval target.**

**Stage 3 of PR #166; awaiting explicit owner approval in chat.** Recommend **three RTX 3090 community pods, three workers × six solver threads per pod**, for the unchanged 82,944-hand arena. The quote is **$17.75 for production and closeout; $18.25 including a conservative $0.50 pilot allowance**, within the approved **$25 campaign ceiling**. No pod was created and no arena was started while preparing this quote.

The [gate amendment](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/148#issuecomment-5993910394) and [staged approval](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/148#issuecomment-5994292377) authorize the completed river validation, paid pilot, and ceiling. They still require approval of this quote and pod selection before provisioning production. [Stage 1](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166#issuecomment-5994590997) passed 32/32 rivers; [stage 2](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166#issuecomment-5995438052) passed 96/96 bit-identical Linux request replays and terminated its pod. Original strict and absolute relaxed calibration failures remain unchanged. The separately recorded owner-amended qualification selects native menu, six threads, uncompressed, 50 iterations, epsilon zero, with a 30-second play deadline.

## Layout and contention

The pilot's four six-thread workers demanded 24 solver threads against a 23.8-CPU cgroup quota, before Python preparation, parsing and LBR work. It recorded 184 completed requests, 42 timeout requests and **86/417 live turn/river search decisions falling back (20.62%)**, including follow-on decisions reusing a failed solve. Its reported CPU throttling was 998/13,812 periods. The timed-out 9,369-node limped-turn request completes in 17.9 seconds on M4; timing alone supports contention as the working diagnosis, rather than proving every timeout has the same cause.

Three workers demand **18 solver threads**, leaving **5.8 quota CPUs (24.37%)** for orchestration and load variation. Keep six threads so the selected numerical configuration and existing parity remain applicable. Set OMP/MKL/OpenBLAS/PyTorch orchestration pools to one thread; run one native solve per worker at a time. No concurrent setup, builds, parity or replay while production workers solve. Each worker has a **12-GiB process-family cap**; three caps total 36 GiB. Actual pod admission requires x86_64 Linux, at least **23.8 quota CPUs**, **60 GB host RAM**, fresh disk/RSS checks, and resource isolation. Catalog GPU stock does not establish a host CPU quota or promise three suitable pods.

**This reduced layout is proposed, not re-measured.** The cost calculation assumes no per-worker speedup: it retains the four-worker pilot's measured block times, including timeouts, and divides work across nine total workers. Relative to four workers on the same number of pods, this gives each worker 4/3 as many blocks. The lower fallback rate is an assumption to test under the stop rule, not a claimed result. Recovering successful continuations can change later play and timing; the 1.5× margin and stop/deadline controls bound exposure, but eight timing blocks cannot establish a distributional upper bound.

## Measurements and arithmetic

Input: M1 `~/Local/hu20-turn-search-pilot-20261005/evidence/timing`. All 24 retained hand files match their worker manifests. Pool each panel's eight blocks across all four workers. Every joint block contains **12 hands: both arms × two rotations × three lineages**, with preparation, parsing, cache behavior, all solver requests and speculative LBR work included. Only coordinates, timing and resource/fallback telemetry enter sizing; payoffs are excluded. The repository's quantile convention `sorted(values)[int(.95 * len(values))]` selects the maximum for eight blocks; no averaging of worker p95 values.

| Panel | Final joint blocks | Measured p95 seconds / joint block |
|---|---:|---:|
| uniform | 256 | 209.817 |
| passive | 256 | 193.864 |
| minraise-cap2 | 256 | 60.512 |
| pressure-cap2 | 256 | 0.522 |
| tight_passive | 256 | 0.574 |
| loose_passive | 256 | 60.861 |
| tight_aggressive | 256 | 61.078 |
| loose_aggressive | 256 | 31.686 |
| pot_pressure | 256 | 64.770 |
| train_pressure | 256 | 86.712 |
| native-pressure | 2,048 | 29.172 |
| selective-stackoff | 256 | 0.589 |
| lbr | 2,048 | 63.839 |

LBR and native pressure retain 2,048 blocks each; the other eleven retain 256 each. Total **6,912 blocks, 82,944 hands, 41,472 search-arm hands**, paired deals/coupled action streams and all three B500M average lineages. The separate pilot root is 202610051801; the arena root is 202610020803.

Using `scripts/quote_hu20_search_arena.py`, the balanced nine-worker upper forecast is:

- Total measured-p95 work: **387,858.896 worker-seconds**.
- Largest worker: `sum(ceil(panel_blocks / 9) × panel_p95)` = **43,565.097 seconds**.
- Add every reserve below, then multiply the entire sum by **1.5** and round up: `ceil(1.5 × (43,565.097 + 10,500))` = **81,098 seconds = 22.5272 hours per pod/worker**.
- Price the **three pods**, each shared by three workers. Do not charge the full pod rate nine times.

| Reserve, seconds per worker / pod wall-clock envelope | Seconds | Purpose |
|---|---:|---|
| Setup, input transfer, dependency installation and build | 1,800 | Includes failed setup/build allowance |
| Actual-pod parity and selected-settings checks | 900 | Require every actual production host to pass |
| Independent replay verification | 3,600 | Replay every completed hand; retain invalid/partial evidence |
| Retrieval and independent size/SHA256 verification | 3,600 | Retrieve hands, requests, responses, logs, manifests and failures |
| Shutdown | 600 | Confirm termination of only newly created campaign pods |
| **Total before 1.5×** | **10,500** | All phases are included in the priced pod lifetime |

## Live offer, storage and alternatives

Catalog source: **RunPod MCP `https://mcp.getrunpod.io/`, server 4.0.0**, authenticated read-only `get-gpu-type` and `get-cpu-type` calls with `include=[AVAILABILITY]`, `product=[POD]`, community filter/count one for 3090 and 32 vCPUs for CPU5. Timestamped raw replies are [3090](hu20-turn-search-arena-quote/mcp-3090-offer.json) and [CPU5](hu20-turn-search-arena-quote/mcp-cpu5-offer.json). Quote timestamp (one second after the latest read): **2026-10-05T13:37:34.121442+00:00**. Availability is a stock category, not a reservation or quantity guarantee. Re-read price/stock and verify the admitted total rate after approval, before allocating the next pod; a different layout/rate needs a revised quote within the ceiling and owner approval.

| Host/layout | Live compute price per pod | Live stock | Priced maximum wall hours per pod | Production + closeout | With $0.50 pilot allowance |
|---|---:|---|---:|---:|---:|
| **3 × RTX 3090 community; 3 workers × 6 threads/pod** | **$0.22/h** | **LOW** | **22.5272** | **$17.75** | **$18.25** |
| 2 × CPU5 compute `cpu5c`, 32 vCPU/64 GB; 4 workers × 6 threads/pod | $1.12/h (`32 × $0.035/vCPU/h`) | HIGH | 24.5761 | $57.42 | $57.92 |

Storage is priced separately: **200 GB total pod-local disk per pod** (20 GB container + 180 GB volume), at [RunPod's running-disk price](https://www.runpod.io/pricing) of **$0.10/GB/month**. A conservative 720-hour divisor gives **$0.027778/pod-hour**, so the admitted total 3090 rate must be at most **$0.247778/h**. The pilot timing tree retains 2,140,348,050 bytes; linear expansion to the complete arena gives 142.25 GB across all pods, or 47.42 GB/pod before extra margin. 200 GB/pod allows inputs, builds and larger successful profiles; preserve a 20-GB free-disk floor and stop on exhaustion. No extra network volume is proposed.

Calculation: `ceil_cent(3 × (81,098 / 3,600) × ($0.22 + $0.02777778) + $1 storage contingency)` = **$17.75**, then add **$0.50 pilot allowance** = **$18.25**. Pilot compute was previously estimated at $0.17; the scoped live billing read still returns no settled rows, so this quote conservatively carries its entire approved allowance. The $1 storage contingency also covers temporary retained storage during closeout. The **$6.75 gap to $25** is available for recorded failures/overruns; it is not permission to retry or retune automatically.

CPU5 uses exactly the same measured per-worker p95, with no assumed CPU5 speedup. It leaves eight of 32 vCPUs outside its 24 solver threads. Its quote **exceeds the approved ceiling by $32.92** and is **not an automatic fallback**. If suitable 3090 stock is unavailable, stop and bring a revised measured quote or a separate CPU5 budget decision; do not shrink the scientific arena, substitute CPU5, or spend $57.92 under a $25 approval.

## Predeclared stops and launch conditions

1. **Fallback gate:** at the first **500 live turn/river search decisions globally**, halt if more than **25** use base fallback (**26/500 = 5.2%** triggers). Count every failed-search base decision, including cached/follow-on failures, once per live decision. Exclude ordinary preflop/flop blueprint queries and hypothetical LBR probes. Apply the same test independently on each pod after its first 500 decisions. Thereafter check every additional 100 decisions, both cumulative and over the trailing 500. Any rate **strictly greater than 5%** halts the whole campaign. Pause dispatch at checkpoints while aggregating all in-flight telemetry; do not dilute the count with successful speculative probes. Record any cancellation overshoot.
2. **Hard correctness/resource stops:** any invalid action/hand, parity failure, turn-conditioning gap, RSS/disk guard, unexpected process sharing, or a stuck solver that fails deadline cancellation stops the relevant work and prevents further dispatch. Ordinary 30-second solver timeout fallbacks are counted by rule 1; they do not alone trigger this hard stop. The required zero conditioning-gap tolerance and 30-second selected deadline remain intact.
3. **Spend/deadline stop:** count wall-clock compute plus provisioned storage, pilot allowance, and every failed attempt. Stop dispatch when the campaign's conservative charge reaches **$21**, preserving **$4** for retrieval, verification and termination within **$25**. Each pod's all-inclusive deadline is **81,098 seconds from provisioning**; production stops before the 7,800-second replay/retrieval/shutdown reserves (plus 1.5× headroom) are consumed. Setup/parity also count against this lifetime. If retrieval is forecast to exceed its reserve, stop production early. No automatic extra pod, restart or changed settings.
4. **Stopped runs:** cancel owned workers, preserve all complete/partial blocks and failure evidence, retrieve with exact-size/SHA verification, then terminate only the pods created for this campaign. A partial arena remains incomplete; completed subsets cannot replace its planned denominator. Any retry, concurrency change or retuning requires an owner decision and retains the original failure.
5. **Before production:** explicit owner chat approval must name this quote, 3090 selection and stop rule. Stage 4 must implement and locally validate the telemetry/spend/checkpoint monitor, perform fresh actual-host admission and parity, and bind an approval document to the final quote/plan hashes. This report and its JSON remain `owner_approved: false`; they cannot authorize launch. A short paid re-measure is not authorized by this quote preparation.

Leave **43z4itur3hwnyv, cl0riravggku4r, xu414eguzakxfr, k9rdph2fwhym87** untouched. No merge, promotion, paid allocation or arena result is implied by stage 3.

## Reproduction

The [complete quote JSON](hu20-turn-search-arena-quote/quote.json) binds original Part A and calibration digests, the separately amended qualification, the candidate arena plan, every timing input size/hash, per-block samples, offers, layout, reserves and stop rules. The candidate plan remains quote evidence, not a paid approval. [Reproduction script](hu20-turn-search-arena-quote/reproduce.py):

```sh
PYTHONPATH=. python docs/reports/hu20-turn-search-arena-quote/reproduce.py \
  --pilot ~/Local/hu20-turn-search-pilot-20261005/evidence/timing \
  --out /tmp/hu20-quote-reproduction
```

It replays the historical catalog timestamp for reproducibility; launch requires fresh reads. The quote helper now prices shared CPU workers once per pod and refuses CPU oversubscription, insufficient memory/disk, missing/stale offers and unpriced reserves. Its existing one-worker-per-pod behavior is retained.

Validation: **36 campaign/protocol tests pass**, including shared-pod charging and rejection of oversubscribed CPU, insufficient memory/disk, absent stock and GPU workloads. Independent local reproduction yields byte-identical quote JSON; all 24 timing hand files pass exact-size/SHA256 checks against the pilot manifests. No solver was run and no paid resource was mutated during stage 3.
