# HU20 arena quote revision 2: bounded profile retention

**Awaiting explicit owner approval in chat.** This revision supersedes the [original $18.25 quote](hu20-turn-search-arena-quote.md), preserving its 3 RTX 3090 community pods × 3 workers × 6 threads, full 82,944-hand arena, six-thread search configuration, 30-second deadline, fallback stop and $25 hard ceiling.

**Revised campaign cost: $6.03 expected (mean blocks), $16.25 maximum (p95 blocks × 1.5), both including $0.50 pilot allowance.** A mean-block forecast with the same 1.5× contingency would be $8.29. These remain estimates from eight outcome-blind blocks per panel; reduced concurrency has not been re-measured. No pod was created or arena started during revision.

**Fresh RunPod MCP stock is NONE for the $0.22/hour RTX 3090 community offer; CPU5 stock is HIGH at $1.12/hour for 32 vCPUs.** Quote catalog timestamp: 2026-10-05T13:53:18.892022+00:00; [raw 3090 read](hu20-turn-search-arena-quote-02/mcp-3090-offer.json), [raw CPU5 read](hu20-turn-search-arena-quote-02/mcp-cpu5-offer.json). This is a conditional 3090 price quote, not a claim of allocatable capacity. Approval does not override stock or actual-host admission. Wait for suitable stock at the approved rate; do not substitute CPU5 automatically.

## Fixed evidence retention

Retain **every hand file, request, receipt, response, log, manifest, failure and partial**, and the **SHA-256 plus exact byte length of every generated profile**. Profile sampling is fixed before production:

```python
request_hash = SHA256(exact_final_request_json_bytes).hexdigest()
keep_full_profile = int(request_hash, 16) < (2**256 // 100)
```

Rule ID: **`request-sha256-lowest-1pct-v1`**. The final serialized request is hashed after allocation admission and before starting its solver. Identical request bytes receive identical treatment. This selects one percent of hash space; the realized solve/body count may differ from one percent. Do not vary the predicate, rehash with a salt, retry for selection, force a minimum sample, or add outcome-selected profiles. Failures/partial profiles receive the same predicate. A request that generated no profile records absence, rather than inventing a hash.

A solve may create its full dump transiently to parse the policy matrices. After the native process exits and live matrices are in memory, hash the complete generated body and put its SHA/size, request SHA, selection flag and retention disposition in its receipt. **Flush/fsync the receipt and directory before unlinking only an unsampled `profile.jsonl`.** If durable recording fails, retain the body and halt; never delete it first. Current manifests list extant bodies only; the retained receipts explicitly account for omitted bodies. Sampled bodies remain byte-identical. Every hand, request, receipt and response remains available, including on a failed block.

Apply this to **all new production-pod solves**. Actual-pod parity uses full matrices transiently for comparison, then applies the same request-hash rule before export. The arena runner accepts the rule through the quote-bound plan; ordinary calibration/pilot/parity behavior defaults to full retention until its comparison/retention wrapper is explicitly applied. Stage 4 must validate parity finalization and export inventory coverage before launch. **Existing M1/M4 pilot, calibration and river artifacts are unchanged.** No old evidence was deleted while measuring this revision.

## Measured size, disk and retrieval

The timing directory contains **2,140,348,050 bytes**, of which **1,780,444,876 bytes / 83.18%** are 196 profile bodies. The full pilot directory contains 2,970,072,796 bytes, of which 2,520,587,401 / 84.87% are profiles. These are exact measurements of the named directories, rather than the earlier approximate 88% estimate. [Profile inventory](hu20-turn-search-arena-quote-02/profile-inventory.json) binds every retained request's SHA and every generated profile's SHA/size against the original solve manifests.

There are 214 retained requests in the timing directory. The fixed predicate selects two requests, one with a generated **7,442,403-byte profile**, one without a generated body. Applying the policy without deleting anything leaves **367,345,577 bytes**. A streamed, lossless gzip tar measurement (level 6, no scratch archive written) produces **12,964,232 bytes across 1,333 retained files in 2.73 seconds** on M1; [compression measurement](hu20-turn-search-arena-quote-02/compression-measurement.json).

For sizing, use `82,944 / 1,248 = 66.461538` hand expansion and a full **1% of profile bytes**, instead of assuming the pilot's unusually small selected-body share repeats:

- Raw retained forecast: `(359,903,174 + .01 × 1,780,444,876) × 66.461538` = **25.10 GB total**, approximately **8.37 GB per 3090 pod**. The observed fixed pilot sample instead projects to 24.41 GB.
- Lossless archive projection from the fixed sample: **0.862 GB total**. Treat compression and sample-size variation as uncertain; reserve an **up-to-4-GB total compressed retrieval envelope**, including new receipt metadata and setup/parity artifacts, not merely the point estimate.
- **3090 disk: 60 GB per pod**, comprising **20 GB container + 40 GB volume**, reduced from 200 GB. A 2× raw-evidence allowance is 16.74 GB per pod; inputs/build scratch (4 GB), transient profiles, archive output and a **10-GB free-disk floor** fit the volume. CPU5's two-pod alternative has more evidence per pod, so price **80 GB/pod (20 + 60)** there.

Retrieve in **lossless compressed chunks no larger than 1 GB**. Preserve their exact archive hashes and a manifest of each original member's logical name, uncompressed size and SHA. Independently hash archives on arrival and **stream-decompress/hash every retained member without expanding the entire dataset to Mac disk**. Keep original byte identities even when compression wraps requests or profile bodies. No request/hand/receipt/response is discarded to achieve these sizes.

Current free space measured read-only: **M1 13 GiB; M4 20 GiB**. A <=4-GB retained archive plus a <=1-GB transfer window and a 5-GiB free floor fits either Mac; the 25.10-GB raw tree does not. Do not require duplicate raw expansions or a full set of unsampled profiles on either Mac. Archive verification confirms receipt hashes and that every generated profile has a SHA record; only sampled profile bodies can be independently byte-verified later. **All-hand native replay remains complete; numerical full-profile audit is limited to the fixed sample.**

If the compressed envelope, disk free floor or projected retrieval time is exceeded, stop production early, preserve retained artifacts and finish verified closeout within the paid reserves. Never thin the sample or discard other evidence to fit. Fresh destination admission remains required before provisioning.

## Timing and pricing

Retain the original contended pilot timings, including timeouts; no unmeasured per-worker speedup or profile-write savings is credited. Every joint block is both arms × two rotations × three lineages. Mean/p95 come from all eight blocks per panel; payoffs are excluded. P95 uses the existing upper quantile convention, selecting the maximum of eight.

| Panel | Final blocks | Mean seconds/block | P95 seconds/block |
|---|---:|---:|---:|
| uniform | 256 | 65.038 | 209.817 |
| passive | 256 | 90.680 | 193.864 |
| minraise-cap2 | 256 | 29.414 | 60.512 |
| pressure-cap2 | 256 | 0.177 | 0.522 |
| tight_passive | 256 | 0.100 | 0.574 |
| loose_passive | 256 | 7.803 | 60.861 |
| tight_aggressive | 256 | 9.877 | 61.078 |
| loose_aggressive | 256 | 9.222 | 31.686 |
| pot_pressure | 256 | 14.464 | 64.770 |
| train_pressure | 256 | 21.355 | 86.712 |
| native-pressure | 2,048 | 6.750 | 29.172 |
| selective-stackoff | 256 | 0.163 | 0.589 |
| lbr | 2,048 | 30.400 | 63.839 |

Each pod's wall envelope includes all worker reserves:

| Phase | Revised seconds | Earlier seconds |
|---|---:|---:|
| Setup/transfer/build | 1,800 | 1,800 |
| Actual-pod parity | 900 | 900 |
| Independent all-hand replay | 3,600 | 3,600 |
| Retrieval and independent archive/member hashing | **900** | **3,600** |
| Shutdown | 600 | 600 |
| Retention finalization and lossless archive creation | **300** | 0 |
| **Total before maximum headroom** | **8,100** | **10,500** |

The retrieval reduction follows the 0.862-GB compressed forecast and <=4-GB envelope, rather than transferring the former 142-GB raw projection. Its 900 seconds still includes hashing every retained member. The separate 300-second reserve covers retention finalization/archiving; existing per-solve profile hashing and dumping time remains in measured blocks. Maximum applies **1.5× to all block work and every reserve**.

For nine workers:

- P95 largest-worker work = `sum(ceil(blocks/9) × panel_p95)` = **43,565.097 s**. Maximum lifetime `ceil(1.5 × (43,565.097 + 8,100))` = **77,498 s / 21.5272 h per pod**.
- Mean largest-worker work = **15,670.640 s**. Expected lifetime `ceil(15,670.640 + 8,100)` = **23,771 s / 6.6031 h per pod**. Expected uses the same phase reserves and prices, but excludes maximum's 50% contingency. With 1.5× contingency on mean work/reserves, the corresponding budget is $8.29.

| Layout | Compute + disk rate/pod | Current stock | Expected mean-block campaign cost | Maximum p95 × 1.5 campaign cost |
|---|---:|---|---:|---:|
| **3 × RTX 3090 community; 3 workers × 6 threads; 60 GB/pod** | **$0.228333/h** | **NONE** | **$6.03** | **$16.25** |
| 2 × CPU5 `cpu5c`, 32 vCPU; 4 workers × 6 threads; 80 GB/pod | $1.131111/h | HIGH | $17.56 | **$54.84** |

[Running storage pricing](https://www.runpod.io/pricing) remains **$0.10/GB/month**, using the same conservative 720-hour divisor: 60 GB = $0.008333/pod-hour; 80 GB = $0.011111. Both columns add the same **$1 storage contingency** and **$0.50 pilot allowance**. Round production/closeout costs up to cents, then add the pilot allowance. Maximum: `ceil_cent(3 × 77,498/3,600 × (.22 + 60 × .10/720) + 1) + .50 = $16.25`. Expected substitutes 23,771 seconds and gives $6.03.

CPU5's lower mean cost **does not authorize it**: its $54.84 maximum exceeds the approved $25 ceiling. All prices are conditional on admitted actual hosts, bounded CPU/RAM and refreshed stock. The quote helper now permits an explicitly marked **price-only** forecast for stock NONE and records `stock_available: false`; it never marks approval true.

## Stops and approval

The original **>5% fallback stop after the first 500 live search decisions** remains: 26/500 triggers globally; independently apply it per pod at 500, then every additional 100 decisions for cumulative and trailing-500 rates. Include cached/follow-on base fallbacks; exclude preflop/flop blueprint decisions and speculative probes. Keep correctness, zero-conditioning-gap, deadline and RSS guards unchanged.

Stop dispatch at a **$21 conservative campaign charge**, preserving **$4** within the **$25 hard ceiling**. Each 3090 pod's all-inclusive lifetime is **77,498 seconds**; production stops before the 5,100-second replay/retrieval/shutdown reserves plus 1.5× headroom are consumed. Record all failures, cancellation overshoot and spend; no automatic restart, retune or alternative pod.

Approval requested in chat: **this revision-2 retention rule, three 3090 community pods with 60 GB each, $16.25 maximum quote/$25 hard ceiling, and the existing fallback stop**. No allocation until explicit approval, fresh suitable stock/resources, validated spend/fallback monitoring and actual-pod parity/retention checks. Old exited pods **43z4itur3hwnyv, cl0riravggku4r, xu414eguzakxfr, k9rdph2fwhym87** remain untouched.

## Validation and reproduction

**69 focused campaign/protocol/search tests pass**, including sample boundary, sampled/unsampled success and failure, preserved live matrices, receipt/response/request hashes, manifest coverage and refusal to delete before receipt fsync. Quote tests cover mean pricing and a price-only stock-NONE forecast that remains unapproved. This stage did not run a solver or delete a profile from existing evidence; fake solver tests used temporary fixtures.

[Quote JSON](hu20-turn-search-arena-quote-02/quote.json) binds the new retention policy into the candidate plan and timing hash; [reproduction](hu20-turn-search-arena-quote-02/reproduce.py) checks every original generated profile against its manifest and reproduces both mean and maximum costs. Run from the repository root:

```sh
PYTHONPATH=. python docs/reports/hu20-turn-search-arena-quote-02/reproduce.py \
  --pilot ~/Local/hu20-turn-search-pilot-20261005/evidence/timing \
  --out /tmp/hu20-quote-revision-02
```

The catalog timestamp is replayed for reproducibility; allocation requires fresh reads. The original quote/evidence remains retained as superseded history.
