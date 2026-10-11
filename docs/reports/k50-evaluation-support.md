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

The owner-authorized M1 correctness smoke is in progress at this report revision: fixed 20M native nodes, opponent-sampled average, 256 direct duplicate hands against v0.4.2, 8 bounded-LBR hands and 16 native-pressure hands, with every action/settlement replayed. No smoke return will be presented as strength evidence. The one-use supervisor caps whole-family RSS at 7 GiB, normal pressure/15% ongoing free memory, 8 GiB entry headroom, 512 MiB fixed-baseline swap growth/2 GiB total swap, 16 GiB free disk, AC and 30 minutes. It stops only its owned worker session; source/restoration/partials and logs will be archived.

## Next step

The [full-game proposal](../hu20-k50-full-game-proposal.md) specifies three K50 lineages, matched-visit reporting and a fixed learning curve/plateau endpoint, fresh primary direct play against v0.4.2, unchanged safeguard margins, counts and an extrapolated M4 quote. It is posted for the owner's go; no M4 use or full-game campaign is authorized here.
