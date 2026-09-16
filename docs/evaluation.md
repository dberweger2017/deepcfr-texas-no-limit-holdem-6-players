# Reproducible evaluation arena

The arena runs a declared plan, saves the exact schedule before dealing, and compares a candidate against a baseline on paired deal or session blocks. It receives ordinary observation-based policies; it never gives them its seeds, policy labels, checkpoints, or privileged simulator objects.

The arena includes simple controls, card-aware style opponents, named training/evaluation pools, and frozen standard-network adapters. See [benchmark suites and checkpoint contracts](benchmarks.md) for their configurations and limits. The old `scripts.evaluate_models` command has been retired.

## Run and reproduce

```bash
python -m scripts.run_arena \
  --plan configs/arena/smoke.json --out results/arena-smoke

python -m scripts.run_arena \
  --reproduce results/arena-smoke --out results/arena-replay
```

Output directories must be new; existing experiments are never overwritten. The smoke plan covers four-, five-, and six-player starting configurations, unequal stacks, and bankroll sessions. Its two independent blocks per scenario deliberately do not support confidence intervals.

For the declared sensitivity control:

```bash
python -m scripts.run_arena \
  --plan configs/arena/sensitivity.json --out results/arena-control
```

The [validation protocol and results](reports/arena-validation.md) explain its fixed budget and acceptance rule. This is an easy control against deliberately weak behavior, not a poker-strength benchmark.

A successful run exits 0. A run with a failed hand exits 1 and saves an invalid report. Invalid input, a stale manifest, or a reproduction mismatch exits nonzero. Policy exceptions and illegal actions stop the run at the first failure; no fallback action or skipped hand can turn it into a valid result. A keyboard interruption produces an incomplete, invalid report. Abrupt process termination can leave partial JSONL files; absence of a valid final report means the run did not finish.

## Plan and random streams

A plan declares candidate, baseline, opponent pool, root seed, split, independent block count, decision cap, and named scenarios. Each scenario specifies integer stacks, blinds, denomination, and either `fixed` or `session` mode. See the committed JSON plans for examples.

`stream_seed` derives separate streams for deals, policy sampling, opponent selection, and future training work. Split prefixes reserve non-overlapping 64-bit seed ranges for `train`, `validation`, and `test`. Distinct seeds are not a mathematical guarantee of different card permutations. The separation applies to plans using this API; historical checkpoints with unknown training provenance cannot claim disjoint data.

Adding blocks preserves the earlier blocks. Adding a differently named scenario does not change an existing scenario's random streams. Changing the candidate, baseline, or opponent pool does not change deal or action seeds. Opponents are sampled uniformly with replacement from the declared pool, once per block; each instance owns its own private memory and generator.

The supported controls are:

| Policy | Behavior |
| --- | --- |
| `check_call` | Check when available, otherwise call |
| `fold` | Fold whenever legal, otherwise check; deliberately weak |
| `random` | Existing legal random policy, with its own seeded generator |

These controls have implementation hashes and no weight files, so their manifest `weights_sha256` is explicitly null. Named style policies and explicitly declared checkpoint aliases are also supported; unknown policies fail before dealing. Frozen artifacts are hash-checked and snapshotted before play. [Benchmark documentation](benchmarks.md) defines pool separation, model eligibility, legal decoding, and inference settings.

## Pairing and seat rotation

Each block fixes a deal sequence and an ordered opponent lineup. It runs every seat rotation, in both candidate and baseline arms. The evaluated identity is `player-0`; the other public identities and their assigned policies rotate around it. Private benchmark labels are never used as player identities or hand identifiers.

In fixed-stack mode, the physical button, stacks, and engine deal seed remain the same across rotations and arms. The candidate therefore visits every physical seat, starting stack, and position on that deal. Each rotation starts fresh policies and empty histories. The starting button advances by block index to cover its physical positions as well.

In session mode, each rotation starts a fresh table, policies, and histories, then plays the declared number of consecutive hands. Stacks and owner-specific history carry forward within that session. A busted seat reloads to its configured initial stack before the next hand and waits for the big blind under the session rules. Reloads are separately recorded and excluded from winnings. A player may remain dealt out while waiting, so actual participant counts can differ from the initial table size.

The two arms share the deal-seed sequence and initial action generators. Once session lineups diverge, identical seeds need not assign identical cards to each player. Different decisions also consume random draws differently. Pairing still compares two outcomes driven by the same underlying scheduled randomness; it does not pretend the resulting game paths remain identical.

Policy memory resets between arms, rotations, and blocks. It persists within one session only. Sharing private information across paired repetitions would invalidate both the information boundary and the statistical interpretation.

## Rates and uncertainty

One independent observation is a complete **block**: all seat rotations and, in session mode, all consecutive hands in those rotations. For each arm:

```
block BB/100 = 100 × total evaluated-player net chips
               / (big blind × scheduled table hands in the block)
```

The report averages those block rates. Its paired improvement uses the candidate-minus-baseline rate within each block, then computes a confidence interval across those differences. It never treats the correlated rotations, paired arms, or individual hands within a session as independent observations.

The interval is the standard two-sided Student-t interval for a mean, using the number of independent blocks and their sample standard deviation. See [NIST's definition](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm) and the [SciPy Student-t implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html). It is an approximation for non-normal block outcomes, not an exact guarantee for heavy-tailed poker returns. The initial minimum is 30 blocks; that threshold alone does not establish adequate precision. With fewer blocks or no observed variation, the arena withholds the interval and reports the comparison as inconclusive.

A positive lower endpoint yields `candidate_better`; a negative upper endpoint yields `baseline_better`; otherwise the result is `inconclusive`. Effect sizes and intervals are retained regardless of that label. There is no automatic checkpoint promotion.

Rates are per **scheduled table hand**, including session hands where the evaluated player is waiting. The report separately counts hands actually dealt to that player. This keeps the session denominator fixed and captures the cost of sitting out after a reload; it is not a rate conditional on receiving cards.

Results are separated by predeclared scenario: initial player count, starting-stack vector, and fixed/session mode. JSON also reports the sampled opponent-lineup strata. Those subgroup intervals are descriptive and not adjusted for multiple comparisons. We do not pool incompatible scenarios or post-select on the realized number of session participants, which depends on previous outcomes.

These are fixed-budget intervals, not a sequential testing procedure. Decide the budget and primary comparison before the run. Repeatedly examining one validation/test schedule and extending it until the interval clears zero invalidates that interpretation. Fresh confirmation data and training-seed variation remain necessary for model-promotion claims.

## Saved artifacts

| File | Contents |
| --- | --- |
| `manifest.json` | Plan, schedule hash, source revision and fingerprint, dirty-tree flag, rules/schema versions, policy fingerprints, statistical protocol, Python/platform/package versions, installed engine commit and binary hashes |
| `models/<sha256>.pt` | Exact verified checkpoint bytes, present only for model policies; used for reproduction |
| `schedule.json` | Exact blocks, deal seeds, action seeds, opponent selection seeds and selected lineups |
| `hands.jsonl` | One record per completed or failed attempt: arm/block/rotation/hand, starting configuration, participant identities, net chips, reloads, public event trace, failure detail and record hash |
| `timings.jsonl` | Per-hand wall time and per-policy-call timings, separated from deterministic results |
| `report.json` | Counts, outcome/schedule hashes, rates, intervals, paired conclusions, lineup strata and performance |
| `report.md` | Readable summary with policy names, scenario rates, paired intervals and timing |

The manifest and schedule are privileged **evaluator artifacts**. They contain seeds and policy labels and must not be passed to a playing agent. Public hand traces contain only public disclosures; folded/mucked private cards are not exported. A failed attempt retains the last available public event prefix and its error. Later unattempted hands remain in the scheduled denominator and invalidate the entire report's strength estimates.

Reproduction checks the source fingerprint, rules, policy fingerprints, complete environment, installed engine binary, statistical protocol, and rebuilt schedule before execution. It then compares hand records byte for byte and report contents excluding timings. Revision and dirty-tree status are retained as provenance; a merge or documentation-only change is acceptable if the implementation fingerprint is identical. A rebuilt engine or different environment is deliberately rejected by this strict path, even if it might happen to produce the same results. There is no override that labels such a run reproduced.

The source fingerprint covers tracked and non-ignored Python files and requirements files, including newly added files. Freeze the code before recording an experiment. Compact plans and validation summaries belong in git; raw bundles stay under ignored `results/` or a separate retained artifact location. The arena currently holds report rows in memory and writes full public traces; large-scale streaming summaries and storage compaction should follow profiling.

Timings measure policy calls separately from observation construction and engine work; reported total wall time covers runner execution and per-hand artifact writes. It excludes checkpoint loading, hashing, snapshotting, initial manifest/schedule writes, final report serialization, dependency installation, and startup. Built-in-policy latency is not a GPU inference benchmark. Neither timings nor the human-readable report alone establish poker strength.

The current manifest and schedule schemas are version 2. Version 1 bundles need their original implementation; this change does not preserve their seed derivation or support replaying them with the new schema.
