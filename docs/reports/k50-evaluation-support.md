# K50 full-game evaluation support

[PR235](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/235) supports pinned K50 native averages/current exports and checkpoint inspection in Python. This is evaluation engineering after #232's bench pass; it makes no full-game strength claim, changes no default/release, and leaves `native/hu20-trainer/` untouched.

## Explicit table ownership

```python
from pathlib import Path
from src.blueprint.abstraction import HU20_EQUITY_SCHEMA
from src.blueprint.artifact import load_training
from src.blueprint.average import AveragePolicy

tables = Path('results/k50-tables')  # hash-verified restoration of #163
checkpoint = load_training(Path('results/checkpoint.gz'), card_buckets=tables)
average = AveragePolicy(Path('results/average.gz'), expected_sha256,
                        expected_schema=HU20_EQUITY_SCHEMA, card_buckets=tables)
```

Both readers require the explicit tables, validate all recorded table SHA256s and the entire game/schema identity against the pins, and reject missing, changed or v1-mislabeled tables/models. Tables are read-only `np.memmap` mappings. Each policy owns its reference and passes it into `information_key`; existing process registration for parity/bench fixture builders cannot change a loaded policy. Worker processes should independently load the same table paths, sharing file-backed pages, rather than pickle entire policy/table objects.

HU100 K50 uses `HU100_EQUITY_SCHEMA` and its existing HU100 average format. The native HU200 schema follows the same identity/format mapping, although step 4's scope and real smoke are HU20. HU20/HU100 synthetic and native-export tests cover the requested games.

K50 checkpoint loading is **evaluation only** in Python, including opponent-sampled native averages. `step` and Python checkpoint saves refuse these objects so they cannot mix native average accumulators with Python traverser-reach updates. Resume with the native trainer. The existing v1 opponent-sampled default resume refusal stays intact; `evaluation_only=True` explicitly permits read-only v1 checkpoint inspection.

## Evaluation paths and information boundary

An arena `Checkpoint` or saved plan model declares `abstraction` and `card_buckets` for K50. Existing v1 specs retain their inferred schema and omit tables. For example, use the existing HU20 average format with:

```json
{
  "name": "k50-candidate",
  "path": "results/average.gz",
  "sha256": "<exact export SHA256>",
  "format": "holdem-hu20-stored-cfr-average-diagnostic-v1",
  "abstraction": "hu20-native-reopening-ordered-history-equity-k50-v1",
  "card_buckets": "results/k50-tables"
}
```

The registry pins/loads both models before dealing; table paths are explicit external dependencies in reproduction plans, not copied into every model snapshot. Restore #163's files at those paths before reproduction. Fingerprints record the schema and table identities.

`scripts.evaluate_hu20_v041_arena.load` and the direct duplicate harness accept the same explicit `abstraction`/`card_buckets` fields in their existing model specs (plus the retained seed, iteration, byte-count and checkpoint-provenance pins). Target trace keys now use that policy's schema, rather than a hardcoded v1 key. The rival independently queries its own distribution. Both receive only their acting seat's immutable legal observation.

Bounded LBR and its shared/exact-ranker variants query the target's distribution with public-history replay and a hypothetical target holding. That interface now reaches the policy-owned K50 tables automatically; it never obtains the actual hidden deal. Native pressure remains the unchanged public reactive policy. Use the same external resource guard for real-table evaluation and audits, not just a per-worker peak check.

**Not yet supported:** HU20 turn search, rollout search blueprint ranges and the web runtime refuse K50 at entry with clear errors. They are unnecessary for the proposed step 4 confirmation. No K50 model is added to the web release catalog, and HU100 action translation remains v1-only.

## Validation and smoke

The focused source/fixture suite passed **106 tests, 2 optional skips**. Synthetic tables come from `tests/native_equity_tables.py`; production readers have no unpinned API. Tests substitute pins only within fixtures. Actual native HU20/HU100 single-iteration exports contain exactly the independently generated Python K50 traverser keys, with unchanged menus. Tests also cover both schemas' refusal cases, memory-map ownership, hypothetical LBR queries, hidden-card invariance, mixed K50/v1 registration/direct replay and early search/web refusal. Optional real-table test skips unless `K50_REAL_TABLES` names restored pinned tables.

Independent source and smoke-supervisor review found no actionable findings. [Review record](k50-evaluation-support-artifacts/source-review.json).

The owner-authorized **M1 correctness smoke passed** at source `d97d6c7`: 20,000,204 native nodes, 59,105 iterations and 1,693,042 keys, using the opponent-sampled average. It played **256 direct duplicate hands against v0.4.2, 8 bounded-LBR hands and 16 native-pressure hands**. All 280 hands/1,361 actions independently replayed, including legal bounds, public event digests and final chip accounting. Rust/Python real-table parity checked every decision with **zero mismatched hands**, and all 670 target trace keys matched. LBR made postflop hypothetical queries on flop and turn. The optional pinned-real-table fixture also passed. [Smoke summary](k50-evaluation-support-artifacts/smoke-summary.json), [replay audit](k50-evaluation-support-artifacts/replay-audit.json).

The existing every-row native export auditor verified checkpoint totals/current regrets and average normalization on all 1,693,042 rows: 1,192,181 positive-mass keys and 500,861 zero-mass keys. This checks stored state, not reconstructed historical accumulator increments. [Export audit](k50-evaluation-support-artifacts/export-audit.json). No smoke returns are presented as strength evidence.

| M1 measurement | Result |
| --- | ---: |
| Native training command, including checkpoint write | 27.07 s |
| K50 checkpoint load | 11.90 s |
| K50 average load | 10.09 s |
| v0.4.2 average load | 40.66 s |
| 280-hand evaluation | 3.64 s |
| Whole-family smoke peak RSS | 1.50 GiB |
| Whole-family audit/archive peak RSS | 1.38 GiB |

Both one-use supervisors enforced 7 GiB whole-family RSS, normal pressure/15% ongoing free memory, 8 GiB entry headroom, 512 MiB growth from the original fixed swap baseline/2 GiB total swap, 16 GiB free disk, AC and 30-minute caps. Across both runs pressure stayed normal, free memory stayed at least 51%, swap stayed zero, and disk stayed above 78.65 GiB; both owned sessions ended empty. The smoke supervisor took 649.96 s including initial FileProvider table hydration and release retrieval; audit/archive took 56.69 s. No other process was stopped. [Resource receipts](k50-evaluation-support-artifacts/resources.json).

The immutable member-hashed ZIP is uploaded to the designated [PR235 Research-Cloud folder](https://drive.google.com/drive/folders/1KOsHYBJ2QOPQOgfGItDaZFlJSN9wCQk4). Local whole/member verification and separate native/connected-cloud upload acceptance are recorded in [RESULTS_INDEX.md](../../RESULTS_INDEX.md), with source/helpers, model hashes and restoration commands. Only this worktree's three extracted table copies (1,384,031,032 bytes) were removed after exact path/size/mtime/hash and no-open-handle checks; outputs, release download and synced files remain. [Cleanup receipt](k50-evaluation-support-artifacts/cleanup-receipt.json).

CI initially exposed an older uniform fixture missing the policy-owned trace-key method. Its explicit v1 method repairs the fixture; production source and the archived smoke source are unchanged. The failed run is retained at [38136754797](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/actions/runs/38136754797); complete green checks remain mandatory before merge.

## Next step

The [full-game proposal](../hu20-k50-full-game-proposal.md) specifies three K50 lineages, matched-visit reporting and a fixed learning curve/plateau endpoint, fresh primary direct play against v0.4.2, unchanged safeguard margins, counts and an extrapolated M4 quote. It is posted for the owner's go; no M4 use or full-game campaign is authorized here.
