# Benchmark opponents and frozen checkpoints

The arena now has a fixed set of legal style opponents, versioned evaluation plans, and an inference-only adapter for archived standard networks. These are controls and historical comparisons. They do not establish competitive poker strength or validate the old trainer.

Read [evaluation.md](evaluation.md) for paired schedules, block-level uncertainty, failure handling, and the strict reproduction contract. The [validation report](reports/benchmark-validation.md) records the declared checks and results.

## Opponent pools

`styles-evaluation-v1` contains five card-aware heuristics:

| Policy | Tendency | Raise increment, relative to the pot after calling |
| --- | --- | --- |
| `tight_passive` | Selective continuation, infrequent value raises | Half pot |
| `loose_passive` | Wide continuation, infrequent raises | Half pot |
| `tight_aggressive` | Selective continuation, frequent value raises | Pot |
| `loose_aggressive` | Wider continuation and more bluffs | Pot |
| `pot_pressure` | Frequent raises and larger pressure bets | 1.5 pots |

They use only their own cards, the public board, and the legal betting context. Preflop rank/pair/suit scores and postflop made-hand scores are hand-selection heuristics, **not equity estimates**. They ignore ranges, draws, position, and history; expect exploitable behavior. Integer raise targets are clamped to the current legal bounds, including short all-ins. Each instance owns its action generator.

`training-v1` contains `random`, `check_call`, and a separate `train_pressure` preset. The committed training and evaluation pools have disjoint policy names. The pressure presets share an implementation, so this is not a claim of independent opponent families. A pool records its name, purpose, and exact members in the plan. Training pools require split `train`; evaluation pools require `validation` or `test`. An explicit `opponents` list must match the declared pool.

This enforcement applies to arena plans. The legacy trainer does not yet consume this catalog. Ad hoc plans without a named pool remain supported. Do not relabel evaluation opponents as training opponents or tune against a final-test schedule. Changes to a published pool or benchmark should receive a new version; manifests also pin implementation bytes and resolved members.

## Committed suites

All chip vectors and seeds are in [configs/arena](../configs/arena). Counts include both paired arms and every seat rotation.

| Plan | Coverage | Blocks per scenario | Scheduled hands |
| --- | --- | ---: | ---: |
| `core-v1.json` | Four/five/six players, 100 BB | 128 | 3,840 |
| `depth-v1.json` | Four/five/six players, 20 and 200 BB; six-player unequal stacks | 64 | 4,608 |
| `sessions-v1.json` | Four/five/six initial players, 100 BB, 20-hand bankroll sessions | 32 | 19,200 |
| `opponents-smoke.json` | Unequal stacks at four/five/six players and a short session | 2 | 180 |
| `historical-smoke.json` | Two archived six-player models at 20/100/200 BB | 2 | 72 |
| `training-pool-v1.json` | Example training split and separate pool; no training invocation | 2 | 24 |

The style suites compare `tight_aggressive` with `check_call`. Their labels and sample budgets are controls, not model-promotion criteria. Larger depth/session campaigns are declared but were not run in this PR. Sessions can shrink after busts and blind admission; fixed-size historical networks are excluded from all session plans.

```bash
python -m scripts.run_arena --plan configs/arena/opponents-smoke.json --out results/style-check
python -m scripts.run_arena --reproduce results/style-check --out results/style-replay
python -m scripts.run_arena --plan configs/arena/core-v1.json --out results/core-baseline
```

For a new candidate, copy a plan and declare the candidate, artifact, budget, seeds, primary comparison, and scenario regression limits before execution. Six-player checkpoint plans must contain only six-player fixed-hand scenarios. Train and evaluate separate four-/five-player artifacts until the replacement model explicitly supports variable player counts.

## Frozen standard networks

A plan's `models` list declares policy aliases. Each entry has `name`, `path`, `sha256`, and `format: "legacy-standard-v1"`. The alias can appear as candidate, baseline, or opponent. Use [historical-smoke.json](../configs/arena/historical-smoke.json) as a complete example. Relative model paths resolve from the repository root, not the plan's directory. Absolute paths also work.

The loader reads each artifact once, verifies its full-file SHA-256, and decodes on CPU with `torch.load(..., weights_only=True)`. There is no unrestricted pickle fallback. It validates the standard strategy architecture, finite float32 weights, sizing bounds, player dimensions, and consistent metadata before creating output or dealing. Opponent-modeling checkpoints and unsupported schemas fail explicitly. See the [PyTorch loading documentation](https://docs.pytorch.org/docs/2.14/generated/torch.load.html); the installed Torch 2.5.1 path is covered by our tests.

The adapter keeps the legacy absolute-seat feature layout, including its old normalization. It uses the actual compact seat, without pretending every player occupies the old training seat. Legal logits receive a masked softmax and a private Python sampling stream. The sizing head keeps the saved multiplier bounds, additional-raise convention, and one-table-unit pot floor, then rounds half chips upward and clamps to explicit legal raise-to bounds. This is a documented current inference adapter, not a promise to reproduce the old NumPy sampler's trajectories. Non-finite inputs or outputs fail; no substitute action conceals them.

Policies accept only immutable player observations. Copies may share frozen network tensors, but have separate random streams and no shared private histories or learning state. Tests change hidden cards and undealt streets while preserving the current observation and require identical distributions at four, five, and six players. Read-only use is an application contract, not a sandbox for hostile Python code.

Model evaluation uses one CPU thread with deterministic Torch algorithms. The arena restores prior thread/determinism settings, including after errors, and loading preserves global random streams. Run it in its own process: Torch settings are process-wide, so concurrent threaded training inside that process is unsupported. Cross-device, cross-version, and rebuilt-engine reproduction are outside the strict supported path; see [PyTorch's reproducibility limits](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## Artifact ownership and provenance

Each run snapshots the exact verified checkpoint bytes under `models/<sha256>.pt` inside its output bundle. Reproduction reads those snapshots, never a mutable source path, and rejects missing or altered bytes. Keep the complete bundle in retained artifact storage. Raw results and model binaries stay out of git; do not delete the original archive after a smoke check.

Reports identify artifact hashes, dimensions, adapter encoding/sampling versions, and any recorded iteration and training seed. Missing seeds remain `null`/unknown. Different iterations do not count as independent training runs. Future campaigns must supply independently trained candidates with real seed provenance; synthetic test networks only exercise this metadata contract.

The archived files referenced by `historical-smoke.json` are available in the owner's local `models/standard` archive, not in a fresh checkout. Obtain those exact files from that archive at the paths in the plan, or retain the generated bundle for reproduction; there is no automatic download. The reported hashes are the identity check. CI uses small synthetic checkpoint fixtures so it does not depend on private local archives.

Loading an old checkpoint proves that the adapter can execute its policy under today's rules and information boundary. It does not establish the correctness of its original training game, seat coverage, regret updates, or strategy averaging. Multi-seed training evidence and the small-game solver checks remain ahead of us.
