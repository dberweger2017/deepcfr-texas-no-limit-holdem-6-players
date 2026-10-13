# HU200 paired evaluation harness

[PR263](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/263) adds internal duplicate head-to-head matches and a paired scripted panel at **200 BB**, without changing the HU100 streams, the game rules or a model's inference. This is engineering qualification and cost measurement, not a campaign or a playing-strength result. The owner decides the merge and the subsequent model comparison.

## Matches and replay

`scripts.evaluate_hu100_direct` now takes `--stack-bb 200`. A direct block plays one fixed physical deal twice, rotating the evaluated identity through both seats. Both players have separate seeded sampler instances; their schema, menu and K50 tables belong to their respective policy. v1/K50 card keys and menu v1/v2 can be mixed in either direction. The game resets to 20,000 chips per seat, blinds 50/100, with no rake or ante; translation is off.

`scripts.evaluate_hu200_panel` evaluates the **same two pinned averages** against each script. Each block plays candidate and baseline in each seat, four hands altogether. Both arms start fresh private instances with the same logical-role action streams and physical deal. The paired comparison is the candidate-minus-baseline block average; hands and seat rotations are not independent samples. Different actions can consume different subsequent random draws. The nominal interval is a Student-t interval over independent blocks. The panel additionally reports absolute rates for each arm and a Bonferroni interval over its seven model-difference contrasts. Predeclare which contrasts are primary before a campaign. The inherited `bonferroni_three_ci` field is historical; the panel's family correction is `bonferroni_seven_ci`.

The first five scripts are #236's unchanged `random`, `check_call`, `tight_aggressive`, `loose_aggressive`, and `pot_pressure`. That last name means **1.5× pot**, as in #251. `half_pot_pressure` and `matched_pot_pressure` are its size controls: `Style(0.40, 0.60, 0.95, 0.20, numerator, 2)` with numerator 1 and 2 respectively. Thresholds, bluffing, raise frequencies and the two random draws per decision are identical. Bets use a fraction of the current pot; raises use the pot after calling, added to the outstanding wager. Integer rounding, minimum raises and all-in caps apply. These controls test sizing; they do not model Slumbot's strategy.

Both commands stream public hand events and private evaluator decision traces, then replay every decision and settlement independently. Replay reconstructs each model's menu/key with its own schema and newly opened, hash-verified tables; scripted actions are regenerated from their own public observation and seed. A second execution must reproduce all hands and decisions exactly. Model bytes, sizes, entries, iterations and recovery-parent SHA256 are validated. Invalid actions, mismatches or extra/missing records stop the run; output roots must be fresh. Seeds, opposing private cards and evaluator labels never enter a policy observation.

## Cache and pinned inputs

HU200 commands require `--policy-cache` at an explicit nonsynced path, for example `results/policy-cache`. The registry forwards it to `AveragePolicy(..., cache=...)`. Each cache folder is keyed by canonical model SHA256, schema and cache format; all arrays and their manifest are verified before mapping read-only. Model and table pins are still checked on every load. Missing caches are built once; for an 85M-entry model this first parse is several minutes, excluded from the warm-cache projections below. A corrupt cache fails rather than silently rebuilding. Research-Cloud, GoogleDrive, CloudStorage and Mobile Documents paths, including aliases resolved through symlinks, are refused.

Prepare `planning/hu200-pair.json` as an array of two specs, **candidate first, reference second**, with distinct names. Required fields are `name`, `path`, `sha256`, `format`, `abstraction`, `bytes`, `entries`, `iteration`, and `source_checkpoint_sha256`. Every K50 spec also has its own `card_buckets` directory. Do not infer entry counts from `max_entries` or from a target-named file. Build the pins from the full audit and exported header, for example:

```sh
mkdir -p planning
export HU200_CANDIDATE=/absolute/path/to/candidate/average.gz
export HU200_CANDIDATE_AUDIT=/absolute/path/to/candidate/audit.json
export HU200_CANDIDATE_TABLES=/absolute/path/to/candidate/tables
export HU200_REFERENCE=/absolute/path/to/reference/average.gz
export HU200_REFERENCE_AUDIT=/absolute/path/to/reference/audit.json
export HU200_REFERENCE_TABLES=/absolute/path/to/reference/tables
python - <<'PY'
import gzip, json, os
from pathlib import Path
from src.policies.files import file_hash
from src.blueprint.abstraction import EQUITY_SCHEMAS, STACK_BY_SCHEMA
specs = []
for role in ('CANDIDATE', 'REFERENCE'):
    path = Path(os.environ['HU200_' + role])
    audit = json.loads(Path(os.environ['HU200_' + role + '_AUDIT']).read_text())
    checked = audit.get('audit', audit)  # Flat #250 or wrapped HU100-style audit.
    assert audit['status'] == checked['status'] == 'verified'
    assert checked['all_nodes_verified'] == checked['counts']['entries']
    pin = next(v for k, v in audit['files'].items()
               if k == 'average' or Path(k).name == 'average.gz')
    assert path.stat().st_size == pin['bytes']
    assert file_hash(path) == pin['sha256'] == checked['average_sha256']
    with gzip.open(path, 'rt') as stream:
        meta = json.loads(stream.readline())
    header = meta['checkpoint_header']; schema = header['abstraction']
    assert STACK_BY_SCHEMA[schema] == 20000
    assert meta['source_checkpoint_sha256'] == checked['checkpoint_sha256']
    nodes = audit.get('actual_nodes', audit.get('native_state', {}).get('completed_nodes'))
    assert header['native_state']['completed_nodes'] == nodes
    spec = dict(name='HU200-' + role.lower(), path=str(path), sha256=pin['sha256'],
        bytes=pin['bytes'], format=meta['format'], abstraction=schema,
        entries=checked['counts']['entries'], iteration=header['iteration'],
        source_checkpoint_sha256=checked['checkpoint_sha256'])
    if schema in EQUITY_SCHEMAS:
        spec['card_buckets'] = os.environ['HU200_' + role + '_TABLES']
    specs.append(spec)
with Path('planning/hu200-pair.json').open('x') as stream:
    json.dump(specs, stream, indent=2)
    stream.write('\n')
PY
```

The paths above must identify the two chosen, audited models. v1 card policies ignore the corresponding TABLES variable. The #250 terminal is **752,593,175 nodes**, despite its `1000000000.average.gz` filename. Its average SHA256 is `60bd37c979e018f11f136370f468cc41b8e1435550531707e89d45f28709ec7c`; exact tables and restoration are indexed under [PR250](../../RESULTS_INDEX.md#pr250-first-hu200-k50-model--october-13-2026).

On a clean checkout containing this change, with the research dependencies installed, these are the fixed-size evaluation commands. They do not start training, external services or resource rentals. They include primary play, independent replay and full deterministic reproduction. Run on the M4 only when the owner has made it available and admitted the selected models/resources:

```sh
python -m scripts.evaluate_hu100_direct \
  --stack-bb 200 --specs planning/hu200-pair.json \
  --policy-cache results/policy-cache --blocks 262144 \
  --root 202610140702 --rung hu200-candidate-vs-reference \
  --source "$(git rev-parse HEAD)" --out results/hu200-direct-20261014

python -m scripts.evaluate_hu200_panel \
  --specs planning/hu200-pair.json --policy-cache results/policy-cache \
  --blocks 16384 --root 202610140703 \
  --source "$(git rev-parse HEAD)" --out results/hu200-panel-20261014
```

Outputs: `play/{hands,decisions}.jsonl.gz`, `play/inputs.json`, `play/complete.json`, `replay.json`, `reproduction/` and `costs.json`; the panel nests these under each opponent. Positive direct `bb_per_100` favors the first model; panel `bb_per_100` is its paired gain over the second, with absolute script scores under `arms`. An optional separate cost pilot uses `--pilot` and fresh roots/output folders, for example 128 direct blocks with root `202610140700` and 32 panel blocks with root `202610140701`. Pilot replay outputs variance and coverage, with no mean, interval or improvement label. Keep pilots separate from confirmation data; freeze campaign budgets and primaries before final play. Confidence intervals condition on the selected model pair and do not estimate variation across training seeds.

## Qualification and HU100 byte identity

A clean main at `ac19e29abb89b90312ebab4bf68036c3964df482` and the implementation commit `f0b3d46f7d3b2dec95b6bba38bb9d6f1f2bf6047` ran the same retained HU100 fixture files, 32 blocks, root `202610130701`, rung `identity`: **64 hands /289 decisions**. SHA256s of the decompressed exact bytes agree:

| Stream | SHA256 on both revisions |
|---|---|
| Hands | `e0b6f3f138ff0737d12142b3054391be244e7e4848944c1b7a3e3682978dcbf6` |
| Decisions | `1b901eff25b5e8e61000e48dc2c35aae3295b681c5f2e08137eb3665ea54e22c` |
| Canonical public event traces, one per hand | `23d0fcb4c556e9f4e9d258261ad83ae82a63fe6822a42e1b8ac4e7ed3af9cb59` |

[Receipt](hu200-evaluation-harness-artifacts/hu100-byte-identity.json). Gzip container timestamps vary in the existing writer; the unchanged claim covers every hand, decision and event byte, not container timestamps or cost metadata. A permanent regression pins the first two hashes.

The focused suite passes 64 tests with three existing native-binary-dependent checks skipped locally; after final replay validation edits the core suite passes 52 with the same three skips. New tests exercise all 16 ordered HU200 schema/menu pairs, cold cache reproduction, K50 scripted-panel replay, size-only controls, private stream coupling, wrong-depth refusal and corrupted trace rejection. No native source or trainer change. Separate tiny committed-source smokes replay/reproduce 4 direct blocks and 2 blocks per script using small fixture averages.

## Measured costs and projections

The owner's **M1 only** ran a fixed 128-block direct self-match of #250's K50 average under two logical policy names, and 32 blocks per script pairing that average with a tiny uniform fixture. There were **352 real-model pilot blocks**, 1,152 primary hands and 1,152 reproduction hands; every primary hand was independently replayed. No scores, winning directions or hand outcomes are reported or interpreted. Source, hash pins, timing receipts and variance-only [projections](hu200-evaluation-harness-artifacts/cost-projections.json) are retained.

The guarded worker, including fixture checks, finished in **62.64 s**; 0.5-second sampling found peak RSS **1,033,715,712 bytes (0.963 GiB)**, below the 6-GiB own-process cap, with normal system pressure throughout. Cached direct loading of both policy aliases plus validation took **16.66 s**, reused validation **2.57 s**. The canonical average was hash-checked; tables and all existing cache files were read only, with unchanged sizes/mtime after the run. No M4, server, Slumbot or external opponent was contacted. The supplied verified cache was used; fixture caches were created under this PR's ignored `results/`.

A direct primary block cost **11.28 ms**, warm reproduction **3.60 ms** and independent audit **15.10 ms/block**. The audit figure includes fixed table hashing divided by only 128 blocks. For **262,144 duplicate blocks**, there are **524,288 primary hands**, their replay and **524,288 reproduction hands**: **2.19 h linear M1-equivalent**, or **4.38 h with a 2× planning allowance** for the M4, giving it no assumed speed credit. Direct duplicate SD was **1,717.81 BB/100 per block**; if it persisted in the actual chosen pair, the nominal 95% half-width would be about **6.58 BB/100** at that count. This self-match variance is a cost/precision diagnostic, not a forecast guaranteed for different models.

For **16,384 blocks per scripted opponent**, each row means **65,536 primary hands**, their replay and **65,536 reproduction hands**. Times include play, replay and reproduction; SDs describe the K50-minus-uniform-fixture pilot contrast, not tomorrow's learned-model pair.

| Opponent | Primary /repeat /audit ms per block | Paired SD BB/100 | Linear minutes | 2× planning minutes |
|---|---:|---:|---:|---:|
| random | 7.37 /4.80 /35.15 | 3,258.81 | 13.09 | 26.18 |
| check_call | 15.30 /9.36 /36.37 | 785.86 | 16.71 | 33.43 |
| tight_aggressive | 3.12 /2.69 /26.37 | 1,773.43 | 8.83 | 17.66 |
| loose_aggressive | 7.25 /4.63 /28.53 | 5,384.26 | 11.08 | 22.15 |
| pot_pressure (1.5×) | 3.43 /2.74 /26.60 | 296.21 | 8.99 | 17.98 |
| half_pot_pressure | 4.78 /3.90 /27.28 | 1,766.36 | 9.86 | 19.72 |
| matched_pot_pressure (1×) | 5.56 /4.35 /35.05 | 1,818.76 | 12.32 | 24.65 |

All seven total **114,688 blocks /458,752 primary hands**, replay and reproduction: **80.89 min linear**, **161.77 min with 2× allowance**. The executable panel evaluates *two* models; a single-model-versus-scripts-only workload would have fewer hands, but is not what these commands do.

Projection formula: `fixed_setup + target_blocks × (primary_play + reproduction_play + audit) / pilot_blocks`; setup is added once per panel. Audit scaling deliberately overcounts its fixed table hashing, providing conservative padding. The 2× margin is advisory, not an upper bound or a new compute authorization. Different models, wider v2 menus, storage speed, page locality and background load can change both cost and variance. The smoke does not establish two-distinct-model full-run RAM residency: the direct names alias one cache, and the panel's reference is tiny. The 85,373,907-entry cache contains **5,115,547,961 bytes** of arrays and K50 tables **1,384,031,032 bytes**. Two distinct caches of that layout plus shared tables have about **10.82 GiB** of backing files, before Python/schedule/output overhead; v2 caches may be larger. Memory mapping permits eviction and shared table pages, but does not guarantee the pilot's small RSS at full scale. Inspect both actual manifests and retain ordinary host headroom before running.

The same linear trace-size estimate is about **416 MB** for direct play plus reproduction and **181 MB** for all scripted traces, excluding inputs/caches/archives. Reserve substantially more for changed trajectories and wider menus. Cache/model/table storage is separate. None of these estimates is a strength result.

## Evidence storage

Raw pilot hands, decision traces, fixtures, setup failure, input provenance and resource samples are archived under `~/Local/Research-Cloud/PR-263-hu200-evaluation-harness/`, with a member manifest and [index/restoration receipt](../../RESULTS_INDEX.md#pr263-hu200-evaluation-harness--october-13-2026). Canonical #250 model/tables/cache stay in their original owner-supplied nonsynced root; they are referenced through the existing restoration index rather than copied. All local originals are retained. A Python import-path setup failure preceded the smoke and made no evaluations; its log/receipt is retained in the bundle.
