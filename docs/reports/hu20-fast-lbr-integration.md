# B100M native/fast LBR integration on M1

I ran the [frozen eight-block protocol](../hu20-lbr-integration-smoke-protocol.md)
against the existing v0.4 first-seed B100M inference export. Compressed bytes were
verified **before** loading against
`4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf`.
The policy was loaded once, shared read-only, and its HU20 native-reopening schema,
two seats, current extraction and original lineage checked. No trainer checkpoint
was loaded. The worker has exited; no service or model remains resident.

## Result

All **16 native/fast hand pairs** pass, including both target positions. Each
executor completed the same native hands: **32 LBR decisions**, comprising
18 preflop, 11 flop, 2 turn and 1 river. Every requested comparison batch completed
(LBRConfig: four chance samples, five-second soft limit; exact river uses one batch).
The longest native decision took 0.731 seconds.

For every pair I checked:

- identical full action sequence and **exactly equal** per-action value arrays;
- identical sample counts, chosen actions and semantic telemetry;
- identical posterior holdings/weights/processed-state digests at each decision;
- identical chance RNG digests after each decision and literal final RNG states;
- identical target action RNG state, target lookup counts and final chips;
- identical public-event digests, zero-sum settlement and independent native replay.

The actual target made 36 trained decisions and zero fallback decisions per
executor. These are actual-action lookups, not hypothetical range-query coverage.
Target whole-hand result is −35.50 BB over 16 hands for both executors; this tiny
integration sample estimates neither playing strength nor expected win rate.

| M1 single-worker timing | Native | Ranked + cached |
| --- | ---: | ---: |
| Algorithm wall time for 16 hands, excluding replay | 12.931 s | 3.125 s |
| Time inside 32 LBR decisions | 12.902 s | 3.098 s |
| Whole algorithm ratio | | **4.137×** |

One policy load took **5.357 s**. Process peak RSS was **1.753 GiB** (1,882,374,144
bytes), including the load and both executors; it is not a separate per-executor
memory comparison. Native always ran first at each coordinate. Hand-local caches
and both rank caches reset before each hand; the fast probability cache is scoped
to the same immutable source and discarded between hands. This smoke does not
measure large shared-cache reuse. Small-run timings depend on M1/system activity
and order; no confidence interval or general speedup guarantee is implied.

## Evidence and use

[Raw complete hands](hu20-fast-lbr-integration-artifacts/hands.jsonl) retain exact
seeds, actions, value arrays, timings, RNG states and native outcome digests.
[Summary](hu20-fast-lbr-integration-artifacts/summary.json) records every pair,
model/game identity and implementation source
`f110feae898c16fef71dc14aaf5d1f688b8a1153`.
[Checksums](hu20-fast-lbr-integration-artifacts/manifest.json) pin both outputs.

Generated-policy runner checks plus existing cache/ranker/guard checks pass:
**37 focused tests**. The new runner test exercises a complete native hand and
rejects changed actions, values, RNG, outcomes and incomplete batch evidence.
Existing tests cover information access, suit coupling and native batch semantics;
this integration smoke does not re-prove #126.

M1 / Python 3.11.15 / NumPy 1.26.4 / maintained native engine revision
`5db20e3d5d6862b32a7402035c1340b622d3b005`. No M4 computation, paid compute,
training, policy modification or campaign changes.

```sh
python -m pytest tests/diagnostics/test_fast_lbr_smoke.py \
  tests/test_cached_lbr.py tests/test_exact_lbr_ranker.py tests/test_exact_ranker_guard.py -q
python -m scripts.smoke_hu20_fast_lbr --policy /path/to/verified-B100M-export.json.gz \
  --out results/hu20-fast-lbr-integration-reproduction
```

Use a new output directory. The protocol and seed schedule are fixed; errors and
incomplete comparisons stay visible rather than triggering easier replacement
seeds. The adapter remains explicitly selected; the native default is unchanged.
